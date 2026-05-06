import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_
import json
import os

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import BaseFactorizationMachine


class DeepFM_MoE(ContextRecommender):

    def __init__(self, config, dataset, item_freq_tensor):
        super(DeepFM_MoE, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [config["embedding_size"] * self.num_feature_field] + self.mlp_hidden_size

        self.moe_layers = SparseMoE(
            input_size=size_list[0],
            output_size=size_list[-1],
            num_experts=4,
            top_k_eval=config["top_k"],
            hidden_sizes=[32],
            dropout=self.dropout_prob,
        )
        self.ln = nn.LayerNorm(normalized_shape=config["embedding_size"] * self.num_feature_field)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)  # Linear product to the final score
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        self.aux_lambda = config["aux_lambda"]
        self.eval_stage = None
        self.item_group = None
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def _aux_load_balance(self, meta, num_experts: int):
        gates = meta.get("gates", None)  # [B, E]
        topk_idx = meta.get("topk_idx", None)  # [B, k] or None
        assert gates is not None
        p = gates.mean(dim=0)  # [E]
        if topk_idx is not None and topk_idx.numel() > 0:
            counts = torch.bincount(topk_idx.reshape(-1), minlength=num_experts).float().to(gates.device)
            f = counts / topk_idx.numel()
        else:
            f = p
        aux = num_experts * (p * f).sum()
        return aux

    def forward(self, interaction, return_gate: bool = False):

        deepfm_all_embeddings = self.concat_embed_input_fields(interaction)
        batch_size = deepfm_all_embeddings.shape[0]
        x = deepfm_all_embeddings.view(batch_size, -1)
        try:
            moe_out = self.moe_layers(x, return_gate=return_gate)
        except TypeError:
            moe_out = self.moe_layers(x) if not return_gate else (self.moe_layers(x), None)

        if return_gate:
            moe_feat, meta = moe_out if isinstance(moe_out, tuple) else (moe_out, None)
            y_deep = self.deep_predict_layer(moe_feat)
        else:
            y_deep = self.deep_predict_layer(moe_out)

        y_fm = self.first_order_linear(interaction) + self.fm(deepfm_all_embeddings)
        y = y_deep
        return (y.squeeze(-1), meta) if return_gate else y.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output, meta = self.forward(interaction, return_gate=True)
        bce = self.loss(output, label)
        if self.aux_lambda > 0 and meta is not None and "gates" in meta:
            num_experts = getattr(self.moe_layers, "num_experts", 1)
            aux = self._aux_load_balance(meta, num_experts)
            return bce + self.aux_lambda * aux
        else:
            return bce

    def predict(self, interaction, save_path=None, model_name=None):
        output = self.forward(interaction, return_gate=False)  # 建议测速时直接设为 False
        prediction = self.sigmoid(output)
        return prediction


class SparseMoE(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_experts: int,
        hidden_sizes=(256,),
        top_k_eval: int = 1,
        dropout: float = 0.0,
        activation: str = "leakyrelu",
        noisy_gating: bool = False,
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
                if i < len(dims) - 2:
                    layers.append(act(activation))
            self.experts.append(nn.Sequential(*layers))

    def _route_logits(self, x, train: bool):
        logits = self.gate(x)
        if train and self.noisy_gating:
            std = F.softplus(self.noise_gate(x)) + self.noise_eps
            logits = logits + torch.randn_like(logits) * std
        return logits

    def forward(self, x, return_gate: bool = False):
        B = x.size(0)

        logits = self._route_logits(x, train=self.training)

        k = self.top_k_eval
        top_logits, top_idx = logits.topk(k, dim=1)  # [B, k]

        if self.training:
            gates_full = F.softmax(logits, dim=1)
            gates = torch.zeros_like(logits)
            gates.scatter_(1, top_idx, gates_full.gather(1, top_idx))
            gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-10)
        else:
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits.scatter_(1, top_idx, top_logits)
            gates = F.softmax(masked_logits, dim=1)
        final_output = torch.zeros(B, self.output_size, device=x.device, dtype=x.dtype)

        for i, expert in enumerate(self.experts):
            expert_weight = gates[:, i].unsqueeze(1)
            expert_out = expert(x)
            final_output += expert_weight * expert_out

        if return_gate:
            return final_output, {"logits": logits, "gates": gates, "topk_idx": top_idx}
        return final_output
