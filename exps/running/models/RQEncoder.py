import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

logger = getLogger()


class RQEncoder(nn.Module):
    def __init__(self, num_codebooks: int, codebook_size: int, embed_dim: int):
        super().__init__()
        self.M = num_codebooks
        self.K = codebook_size
        self.D = embed_dim

        # 优化 1: 使用一个大张量存储所有码本，保证显存连续性
        # Shape: [M, K, D]
        self.codebooks = nn.Parameter(torch.empty(self.M, self.K, self.D))

        # 初始化
        nn.init.uniform_(self.codebooks, -1.0 / self.K, 1.0 / self.K)

    def forward(self, x: torch.Tensor, tau: float = 1.0):
        B = x.shape[0]

        # 优化 2: 预先计算所有码本向量的平方和
        # Shape: [M, K] -> 扩展为 [1, M, K] 以便广播
        # 这步操作移到了循环外，利用 GPU 并行能力
        codebook_sq = torch.sum(self.codebooks**2, dim=-1).unsqueeze(0)

        current_residual = x

        all_soft_probabilities = []
        all_residuals_in = []
        all_quantized_vectors = []
        all_indices = []

        # 这里的循环是 RQ 算法逻辑必须的 (自回归)，但内部计算已大幅简化
        for m in range(self.M):
            # 记录输入残差
            all_residuals_in.append(current_residual)

            # 获取当前层的码本: [K, D]
            C_m = self.codebooks[m]

            # 优化 3: 极简距离计算 (移除 residual^2 项)
            # Logits = - (||r||^2 + ||c||^2 - 2rc)
            # 由于 ||r||^2 对 Softmax 无影响，等价于计算: 2rc - ||c||^2

            # [B, D] @ [D, K] -> [B, K]
            interaction = torch.matmul(current_residual, C_m.t())

            # Logits = 2 * interaction - codebook_squared
            logits = 2 * interaction - codebook_sq[:, m, :]
            indices = torch.argmax(logits, dim=-1)
            all_indices.append(indices)

            # Gumbel-Softmax (保持原逻辑)
            y_soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            all_soft_probabilities.append(y_soft)

            # 计算量化向量: [B, K] @ [K, D] -> [B, D]
            c_m = torch.matmul(y_soft, C_m)
            all_quantized_vectors.append(c_m)

            # 更新残差
            current_residual = current_residual - c_m

        # 堆叠结果
        residuals_in = torch.stack(all_residuals_in, dim=1)
        quantized_vectors = torch.stack(all_quantized_vectors, dim=1)
        codebook_indices = torch.stack(all_indices, dim=-1)

        return all_soft_probabilities, residuals_in, quantized_vectors, codebook_indices

    def calculate_vq_loss(self, residuals_in, quantized_vectors, commitment_cost=0.25):
        # 保持原有的损失计算逻辑不变
        sg_residuals_in = residuals_in.detach()
        loss_quant = F.mse_loss(sg_residuals_in, quantized_vectors)

        sg_quantized_vectors = quantized_vectors.detach()
        loss_commit = F.mse_loss(residuals_in, sg_quantized_vectors)

        return loss_quant + commitment_cost * loss_commit
