import torch
import torch.nn as nn
import torch.nn.functional as F


class HiLoMoELayer(nn.Module):
    """
    Single Layer of Hierarchical LoRA MoE.
    Represents one horizontal layer containing K rank-1 experts.
    """

    def __init__(self, num_experts, input_dim, output_dim, dropout=0.0):
        super(HiLoMoELayer, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim

        # LoRA Vectors: Expert i = u_i * v_i^T
        # V: [K, d_in] - Acts as both the Down-projection and the Router Key
        # U: [K, d_out] - Acts as the Up-projection
        self.V = nn.Parameter(torch.randn(num_experts, input_dim))
        self.U = nn.Parameter(torch.zeros(num_experts, output_dim))  # Zero-U Initialization

        self.dropout = nn.Dropout(dropout)

        # Initialization
        nn.init.xavier_uniform_(self.V)
        # U is already zero-initialized

    def compute_routing_score(self, query, top_k=None):
        """
        Computes routing scores based on query and V (expert weights).
        Eq (2): s^(l) = Softmax(q * V^T / sqrt(d))

        If top_k is not None and 0 < top_k < num_experts, we apply sparse
        top-k gating by masking out non-top-k logits before softmax.
        """
        # query: [Batch, d_in]
        # logits: [Batch, K] = query @ V.T
        logits = F.linear(query, self.V)
        logits = logits / (self.input_dim**0.5)

        # --- NEW: sparse top-k gating on logits ---
        if top_k is not None and top_k > 0 and top_k < self.num_experts:
            # topk_vals: [B, top_k], topk_idx: [B, top_k]
            topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
            # Create a mask with -inf everywhere, then scatter top-k logits
            masked_logits = logits.new_full(logits.size(), float("-inf"))
            masked_logits.scatter_(-1, topk_idx, topk_vals)
            logits = masked_logits
        # 如果 top_k 是 None、<=0 或 >=num_experts，就退化成 dense softmax

        scores = F.softmax(logits, dim=-1)
        return scores, logits

    def update_query(self, query, scores):
        """
        Updates the query for the next layer.
        Eq (2): q^(l+1) = q^(l) + sum(s_k * v_k)
        """
        # scores: [Batch, K]
        # V: [K, d_in]
        # weighted_V: [Batch, d_in] = scores @ V
        weighted_V = F.linear(scores, self.V.t())
        next_query = query + weighted_V
        return next_query

    def forward_expert_computation(self, x, scores):
        """
        Computes the LoRA output efficiently without materializing W.
        Output = x * (U * diag(s) * V)^T = x * V^T * diag(s) * U^T
        """
        # x: [Batch, d_in]
        # scores: [Batch, K]

        # 1. Project down: a = x @ V.T -> [Batch, K]
        a = F.linear(x, self.V)

        # 2. Weight by scores: b = a * scores -> [Batch, K] (Element-wise)
        b = a * scores

        # 3. Project up: y = b @ U -> [Batch, d_out]
        # Note: U is [K, d_out], so standard linear uses U
        y = F.linear(b, self.U.t())

        return self.dropout(y)


class HiLoMoE(nn.Module):
    """
    Hierarchical LoRA MoE Module (The full wrapper).
    Integrates vertical scaling (layers) and horizontal scaling (experts).
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers=2,
        num_experts=5,
        top_k=1,  # sparse gating: how many experts per sample
        dropout=0.1,
    ):
        super(HiLoMoE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k  # --- NEW: 保存 top_k，用于 sparse gating ---

        # Shared Base Weight W^(0) [cite: 149]
        self.W_0 = nn.Linear(input_dim, output_dim, bias=True)

        # Hierarchical Layers
        self.layers = nn.ModuleList(
            [HiLoMoELayer(num_experts, input_dim, output_dim, dropout) for _ in range(num_layers)]
        )

        # Query Projection (if input features need alignment for routing)
        # Assuming input x is already the representation suitable for query
        # Eq implies q^(1) comes from x [cite: 167]
        self.query_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor [Batch, input_dim]
        Returns:
            output: [Batch, output_dim]
            aux_loss: Scalar tensor (Load balance + Z-loss)
        """
        batch_size = x.size(0)

        # 1. Base Computation (Global Expert)
        base_out = self.W_0(x)

        # 2. Hierarchical Routing (Decoupled from Expert Execution)
        # Initial Query q^(1)
        current_query = self.query_proj(x)

        all_layer_scores = []
        all_layer_logits = []

        for layer in self.layers:
            # --- 修改：传入 top_k，实现 sparse routing ---
            scores, logits = layer.compute_routing_score(current_query, top_k=self.top_k)
            all_layer_scores.append(scores)
            all_layer_logits.append(logits)

            # Update query for next layer (Hierarchical)
            current_query = layer.update_query(current_query, scores)

        # 3. Parallel Expert Execution [cite: 128]
        # We sum up the contributions from all layers
        moe_out = 0
        for i, layer in enumerate(self.layers):
            moe_out += layer.forward_expert_computation(x, all_layer_scores[i])

        final_output = base_out + moe_out

        # 4. Compute Auxiliary Losses
        aux_loss = self.compute_aux_loss(all_layer_scores, all_layer_logits)

        return final_output, aux_loss

    def compute_aux_loss(self, all_scores, all_logits):
        """
        Computes Load Balancing Loss and Z-Loss.
        Formula (3) in paper[cite: 204].
        """
        total_aux_loss = 0.0

        for scores, logits in zip(all_scores, all_logits):
            num_experts = scores.size(1)

            # 1. Load Balancing Loss
            # l_lb = N * sum(f_i * P_i) where f_i is fraction of samples assigned, P_i is avg prob
            # For sparse routing, scores 里非 top-k 的位置接近 0。

            # Average probability per expert across batch
            prob_per_expert = torch.mean(scores, dim=0)  # [K]

            # Fraction of samples where expert is 'activated'
            # For soft sparse routing,用 scores 近似
            fraction_per_expert = torch.mean(scores, dim=0)  # [K]

            # Standard Switch Transformer-style auxiliary loss
            l_lb = num_experts * torch.sum(prob_per_expert * fraction_per_expert)

            # 2. Z-Loss (Regularizes logits size)
            # l_z = log^2(sum(exp(logits)))
            log_z = torch.logsumexp(logits, dim=-1)
            l_z = torch.mean(log_z**2)

            total_aux_loss += 0.01 * l_lb + 0.001 * l_z  # Coefficients are typical defaults

        return total_aux_loss


# --- Example Usage ---
if __name__ == "__main__":
    # Parameters
    B, D_in, D_out = 64, 128, 128

    # Initialize Model
    # 例如：每层只激活 top_k=2 个专家
    model = HiLoMoE(input_dim=D_in, output_dim=D_out, num_layers=2, num_experts=5, top_k=2)

    # Dummy Input
    x = torch.randn(B, D_in)

    # Forward Pass
    y_pred, aux_loss = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_pred.shape}")
    print(f"Auxiliary Loss: {aux_loss.item()}")

    # Backward test
    total_loss = y_pred.mean() + aux_loss
    total_loss.backward()
    print("Backward pass successful.")

    # Check Zero-U Init
    print(f"Layer 0 U stats (Should be 0): {model.layers[0].U.abs().sum().item()}")
    print(f"Layer 0 V stats (Should be >0): {model.layers[0].V.abs().sum().item()}")
