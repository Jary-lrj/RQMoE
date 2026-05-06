import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMoE(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_experts: int,
        hidden_sizes=(256,),
        top_k_eval: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        noisy_gating: bool = True,
        noise_eps: float = 1e-2,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.top_k_eval = max(1, min(top_k_eval, num_experts))
        self.noisy_gating = noisy_gating
        self.noise_eps = noise_eps

        self.gate = nn.Linear(input_size, num_experts)
        self.noise_gate = nn.Linear(input_size, num_experts) if noisy_gating else None
        if self.noise_gate is not None:
            nn.init.zeros_(self.noise_gate.weight)
            nn.init.zeros_(self.noise_gate.bias)

        def act(name):
            return {
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "leakyrelu": nn.LeakyReLU(),
                "none": nn.Identity(),
            }.get(name.lower(), nn.ReLU())

        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            layers = []
            dims = [input_size] + list(hidden_sizes) + [output_size]
            for i in range(len(dims) - 1):
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(act(activation))
            self.experts.append(nn.Sequential(*layers))

    def _route_logits(self, x, train: bool):
        logits = self.gate(x)
        if train and self.noisy_gating:
            std = F.softplus(self.noise_gate(x)) + self.noise_eps
            logits = logits + torch.randn_like(logits) * std
        return logits

    def forward(self, x, return_gate: bool = False):
        orig_shape = x.shape
        x = x.view(-1, self.input_size)  # [B, H]
        B = x.size(0)

        if self.training:
            # Dense gating over all experts
            logits = self._route_logits(x, train=True)  # [B, E]
            gates = F.softmax(logits, dim=1)  # [B, E]
            out = torch.zeros(B, self.output_size, device=x.device, dtype=x.dtype)
            for e, expert in enumerate(self.experts):
                y_e = expert(x)  # [B, H_out]
                out += gates[:, e].unsqueeze(1) * y_e
            topk_idx = None
        else:
            # Sparse inference: Top-k (default Top-1)
            logits = self._route_logits(x, train=False)  # [B, E]
            k = self.top_k_eval
            top_logits, top_idx = logits.topk(k, dim=1)  # [B, k]
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(1, top_idx, top_logits)  # 未入选 = -inf
            gates = F.softmax(masked, dim=1)  # 严格稀疏
            out = torch.zeros(B, self.output_size, device=x.device, dtype=x.dtype)
            for e, expert in enumerate(self.experts):
                mask_e = (top_idx == e).any(dim=1)  # [B]
                if mask_e.any():
                    y_e = expert(x[mask_e])  # [B_e, H_out]
                    g_e = gates[mask_e, e].unsqueeze(1)  # [B_e, 1]
                    out[mask_e] += g_e * y_e
            topk_idx = top_idx
            print(torch.mean(logits, dim=0))
            counts = torch.bincount(topk_idx.view(-1), minlength=self.num_experts)
            print("Expert usage:", counts.tolist())

        out = out.view(*orig_shape[:-1], self.output_size)
        if return_gate:
            return out, {"logits": logits, "gates": gates, "topk_idx": topk_idx}
        return out
