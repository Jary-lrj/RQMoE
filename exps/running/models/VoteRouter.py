import math
from typing import List, Literal, Optional

import torch
import torch.nn as nn


# ====== 你已有的 VoterRouter 保持不变 ======
class VoterRouter(nn.Module):
    def __init__(self, num_codebooks: int, codebook_size: int, num_experts: int):
        super().__init__()
        self.M = num_codebooks
        self.K = codebook_size
        self.N = num_experts

        self.W_affinity = nn.Parameter(torch.empty(self.M, self.K, self.N))
        nn.init.kaiming_uniform_(self.W_affinity, a=math.sqrt(5))

    def forward(self, soft_probabilities_list: list) -> torch.Tensor:
        """
        参数:
        - soft_probabilities_list: 长度为 M 的列表，每个元素是 [B, K]
        """
        all_probs_flat = torch.cat(soft_probabilities_list, dim=1)
        W_flat = self.W_affinity.view(-1, self.N)
        total_votes = torch.matmul(all_probs_flat, W_flat)

        return total_votes


# ====== 新增：统一的 Router 接口基类（可选，但方便约束）======
class BaseTrajectoryRouter(nn.Module):
    """
    约定：
    - 输入：soft_probabilities_list: List[Tensor[B, K]]，长度 M
    - 输出：Tensor[B, N]
    """

    def __init__(self, num_codebooks: int, codebook_size: int, num_experts: int):
        super().__init__()
        self.M = num_codebooks
        self.K = codebook_size
        self.N = num_experts

    def forward(self, soft_probabilities_list: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


# ====== 新增：Mean 聚合（trajectory mean）======
class MeanRouter(BaseTrajectoryRouter):
    """
    将 M 个子空间的 soft prob 先做 mean 聚合得到 [B, K]，
    然后做 [B,K] -> [B,N] 的线性映射。
    """

    def __init__(self, num_codebooks: int, codebook_size: int, num_experts: int, bias: bool = False):
        super().__init__(num_codebooks, codebook_size, num_experts)
        self.proj = nn.Linear(self.K, self.N, bias=bias)
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, soft_probabilities_list: List[torch.Tensor]) -> torch.Tensor:
        # [B, K] x M -> [B, M, K]
        probs = torch.stack(soft_probabilities_list, dim=1)
        # mean over M -> [B, K]
        pooled = probs.mean(dim=1)
        return self.proj(pooled)


# ====== 新增：Max 聚合（trajectory max）======
class MaxRouter(BaseTrajectoryRouter):
    """
    将 M 个子空间的 soft prob 先做 max 聚合得到 [B, K]，
    然后做 [B,K] -> [B,N] 的线性映射。
    """

    def __init__(self, num_codebooks: int, codebook_size: int, num_experts: int, bias: bool = False):
        super().__init__(num_codebooks, codebook_size, num_experts)
        self.proj = nn.Linear(self.K, self.N, bias=bias)
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, soft_probabilities_list: List[torch.Tensor]) -> torch.Tensor:
        probs = torch.stack(soft_probabilities_list, dim=1)  # [B, M, K]
        pooled = probs.max(dim=1).values  # [B, K]
        return self.proj(pooled)


# ====== 新增：Attention 聚合（trajectory attention over codebooks）======
class AttentionRouter(BaseTrajectoryRouter):
    """
    对每个 codebook 的 [B,K] 用一个 shared 的 K->N 投影得到 [B,N] 的 token，
    再对 M 个 token 做 attention 加权求和，输出 [B,N]。
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        num_experts: int,
        attn_hidden: int = 64,
        bias: bool = False,
        temperature: Optional[float] = None,
    ):
        super().__init__(num_codebooks, codebook_size, num_experts)
        self.temperature = temperature

        # 将每个 codebook 的分布 [B,K] 投影成一个 expert-score token [B,N]
        self.token_proj = nn.Linear(self.K, self.N, bias=bias)
        nn.init.kaiming_uniform_(self.token_proj.weight, a=math.sqrt(5))
        if bias:
            nn.init.zeros_(self.token_proj.bias)

        # 用一个小 MLP 产生每个 token 的 attention logit（对 M 做 softmax）
        self.attn_mlp = nn.Sequential(
            nn.Linear(self.N, attn_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(attn_hidden, 1, bias=True),  # -> [B, 1]
        )
        # 初始化不必太复杂，默认即可

    def forward(self, soft_probabilities_list: List[torch.Tensor]) -> torch.Tensor:
        # probs: [B, M, K]
        probs = torch.stack(soft_probabilities_list, dim=1)

        # token: [B, M, N]
        token = self.token_proj(probs)

        # attn_logits: [B, M, 1] -> squeeze -> [B, M]
        attn_logits = self.attn_mlp(token).squeeze(-1)

        if self.temperature is not None:
            attn_logits = attn_logits / self.temperature

        # attn_weights: [B, M]
        attn_weights = torch.softmax(attn_logits, dim=1)

        # 加权求和：([B,M] -> [B,M,1]) * [B,M,N] -> sum_M -> [B,N]
        out = (attn_weights.unsqueeze(-1) * token).sum(dim=1)
        return out


# ====== 新增：一个 factory，改一个名字/字符串就能切换 ======
RouterType = Literal["vote", "mean", "max", "attn"]


def build_trajectory_router(
    router_type: RouterType,
    num_codebooks: int,
    codebook_size: int,
    num_experts: int,
    **kwargs,
) -> nn.Module:
    """
    用法：
      router = build_trajectory_router("vote", M, K, N)
      router = build_trajectory_router("mean", M, K, N)
      router = build_trajectory_router("max",  M, K, N)
      router = build_trajectory_router("attn", M, K, N, attn_hidden=64, temperature=1.0)
    """
    if router_type == "vote":
        return VoterRouter(num_codebooks, codebook_size, num_experts)

    if router_type == "mean":
        return MeanRouter(num_codebooks, codebook_size, num_experts, **kwargs)

    if router_type == "max":
        return MaxRouter(num_codebooks, codebook_size, num_experts, **kwargs)

    if router_type == "attn":
        return AttentionRouter(num_codebooks, codebook_size, num_experts, **kwargs)

    raise ValueError(f"Unknown router_type: {router_type}")
