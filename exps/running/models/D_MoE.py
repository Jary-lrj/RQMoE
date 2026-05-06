import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossExpertCorrelationLoss(nn.Module):
    """
    (保持不变) 去相关 Loss
    """

    def __init__(self, eps=1e-8):
        super(CrossExpertCorrelationLoss, self).__init__()
        self.eps = eps

    def forward(self, expert_outputs):
        loss = 0.0
        num_experts = len(expert_outputs)
        if num_experts < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)

        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                out_i = expert_outputs[i]
                out_j = expert_outputs[j]

                # 1. Centering
                out_i_mean = out_i - out_i.mean(dim=0, keepdim=True)
                out_j_mean = out_j - out_j.mean(dim=0, keepdim=True)

                # 2. Normalization
                out_i_std = out_i_mean.norm(dim=0, keepdim=True) + self.eps
                out_j_std = out_j_mean.norm(dim=0, keepdim=True) + self.eps

                out_i_norm = out_i_mean / out_i_std
                out_j_norm = out_j_mean / out_j_std

                # 3. Correlation Matrix L2 Norm
                corr_matrix = torch.mm(out_i_norm.t(), out_j_norm)
                loss += torch.norm(corr_matrix, p=2)

        return loss


class DMoE_Representation(nn.Module):
    def __init__(
        self,
        input_dim,  # 总输入维度 (num_fields * emb_size)
        expert_output_dim,  # 专家输出维度 (也是最终该模块的输出维度)
        num_experts=3,
        top_k=None,  # Top-K 选项
        expert_hidden_dim=64,  # 专家中间层维度
        gate_hidden_dim=64,
        dropout=0.1,
    ):
        """
        Args:
            expert_output_dim: int, 这是该模块最终输出向量的维度 [Batch, expert_output_dim]
        """
        super(DMoE_Representation, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.expert_output_dim = expert_output_dim
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            self.experts.append(
                nn.Sequential(
                    nn.Linear(input_dim, expert_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden_dim, expert_output_dim),
                    nn.ReLU(),
                )
            )
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, num_experts)
        )
        self.corr_loss_fn = CrossExpertCorrelationLoss()

    def forward(self, x_embeddings):
        """
        Args:
            x_embeddings: [Batch, Fields, Emb] 或 [Batch, Input_Dim]

        Returns:
            fused_embedding: shape [batch_size, expert_output_dim] -> 供外部 deep_predict 使用
            aux_loss: scalar
        """
        # 输入展平处理
        if x_embeddings.dim() == 3:
            batch_size = x_embeddings.size(0)
            inputs_flat = x_embeddings.view(batch_size, -1)
        else:
            inputs_flat = x_embeddings
            batch_size = x_embeddings.size(0)

        gate_logits = self.gate_net(inputs_flat)

        if self.top_k is not None and self.top_k < self.num_experts:
            topk_vals, topk_indices = torch.topk(gate_logits, self.top_k, dim=1)
            topk_weights = F.softmax(topk_vals, dim=1)
            gate_weights = torch.zeros_like(gate_logits)
            gate_weights.scatter_(1, topk_indices, topk_weights)
        else:
            gate_weights = F.softmax(gate_logits, dim=1)

        expert_outputs = []
        for i in range(self.num_experts):
            out = self.experts[i](inputs_flat)
            expert_outputs.append(out)

        aux_loss = self.corr_loss_fn(expert_outputs)
        stacked_outputs = torch.stack(expert_outputs, dim=1)  # (B, N, D_exp)
        weights = gate_weights.unsqueeze(2)  # (B, N, 1)
        fused_embedding = torch.sum(stacked_outputs * weights, dim=1)  # (B, expert_output_dim)

        return fused_embedding, aux_loss


# ==========================================================
# 外部统一调用的示例
# ==========================================================
if __name__ == "__main__":
    BATCH_SIZE = 4
    INPUT_DIM = 20 * 16
    EXPERT_OUT_DIM = 32  # 假设我们要输出 32 维的向量给外部使用

    # 1. 实例化 DMoE Layer
    dmoe_layer = DMoE_Representation(
        input_dim=INPUT_DIM,
        expert_output_dim=EXPERT_OUT_DIM,  # 输出 [B, 32]
        expert_hidden_dim=64,
        num_experts=5,
        top_k=2,
    )

    # 2. 模拟外部的 Deep Predict 模块 (例如一个简单的 MLP Tower)
    external_deep_predict = nn.Sequential(nn.Linear(EXPERT_OUT_DIM, 16), nn.ReLU(), nn.Linear(16, 1))  # 输出 Logits

    # 模拟输入
    dummy_input = torch.randn(BATCH_SIZE, INPUT_DIM)

    # Forward Pass
    # 1. 获取 DMoE 的中间表示 和 辅助 Loss
    moe_output, corr_loss = dmoe_layer(dummy_input)

    print(f"MoE Output Shape: {moe_output.shape}")
    # Output: torch.Size([4, 32]) -> 符合预期 [B, d_out]

    # 2. 外部统一预测
    final_logits = external_deep_predict(moe_output)

    print(f"Final Logits Shape: {final_logits.shape}")
    # Output: torch.Size([4, 1])

    # 3. Loss 计算
    targets = torch.randint(0, 2, (BATCH_SIZE, 1)).float()
    bce_loss = nn.BCEWithLogitsLoss()(final_logits, targets)

    total_loss = bce_loss + 0.1 * corr_loss
    print(f"Total Loss: {total_loss.item():.4f}")
