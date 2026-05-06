import torch
import torch.nn as nn


class CollapseAvoidingFlat(nn.Module):
    """
    Collapse-avoiding Mechanism
    输入/输出：[Batch_Size, Flattened_Dim] (即 num_fields * embed_dim)
    """

    def __init__(self, num_fields, embed_dim):
        super(CollapseAvoidingFlat, self).__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim

    def forward(self, x_flat):
        """
        Args:
            x_flat: [Batch, num_fields * embed_dim] - 展平的 Embedding
        Returns:
            x_filtered_flat: [Batch, num_fields * embed_dim] - 过滤后的展平 Embedding
        """
        batch_size = x_flat.size(0)

        # 1. Reshape: 还原特征维度 [B, N*D] -> [B, N, D]
        # 只有还原了维度，才能计算每个特征独立的模长
        x = x_flat.view(batch_size, self.num_fields, self.embed_dim)

        # 2. 计算模长 (L2 Norm) [cite: 2153-2156]
        # norms: [Batch, Num_Fields]
        norms = torch.norm(x, p=2, dim=-1)

        # 3. 动态阈值 (Dynamic Thresholding) [cite: 2150]
        # 计算整个 Batch 内所有特征的平均模长作为阈值
        # 这里使用了 .mean()，即 paper 中的 "average modulus length within the batch"
        batch_avg_threshold = 0.2 * norms.mean()

        # 4. 生成 Mask [cite: 2150]
        # 如果模长 >= 阈值，保留(1)；否则认为是塌缩特征，丢弃(0)
        # mask: [Batch, Num_Fields, 1]
        mask = (norms >= batch_avg_threshold).float().unsqueeze(-1)

        # 5. 应用过滤
        x_filtered = x * mask

        # 6. Flatten Back: 还原回 [B, N*D] 以适配你的后续网络
        x_filtered_flat = x_filtered.view(batch_size, -1)

        return x_filtered_flat
