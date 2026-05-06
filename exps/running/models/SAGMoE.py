from asyncio.proactor_events import base_events
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_
import json
import os
from dataclasses import dataclass
from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import BaseFactorizationMachine, MLPLayers
from .RQEncoder import RQEncoder
from .VoteRouter import VoterRouter, build_trajectory_router
from .HiLoMoE import HiLoMoE
from .D_MoE import DMoE_Representation
from .Dual_MoE import CollapseAvoidingFlat
from logging import getLogger

logger = getLogger()


class SAGMoE(ContextRecommender):

    def __init__(self, config, dataset, item_freq_tensor):
        super(SAGMoE, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]  # [32, 16]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [config["embedding_size"] * self.num_feature_field] + self.mlp_hidden_size

        self.rq = RQEncoder(
            num_codebooks=config["num_codebooks"],
            codebook_size=config["codebook_size"],
            embed_dim=10,
        )
        self.rq_weight = nn.Parameter(torch.tensor(1.0))
        self.tau = config["tau"]

        self.moe_layers = RQMoE(
            input_size=size_list[0],
            output_size=size_list[-1],
            num_experts=4,
            top_k_eval=config["top_k"],
            hidden_sizes=[32],
            dropout=self.dropout_prob,
            config=config,
        )
        # self.moe_layers = HiLoMoE(input_dim=size_list[0], output_dim=size_list[-1], num_layers=1, num_experts=4)
        # self.moe_layers = DMoE_Representation(
        #     input_dim=size_list[0],
        #     expert_output_dim=size_list[-1],
        #     num_experts=4,
        #     top_k=1,
        #     expert_hidden_dim=32,
        #     dropout=self.dropout_prob,
        # )
        # self.filter = CollapseAvoidingFlat(self.num_feature_field, self.embedding_size)

        # self.mlp_layers = MLPLayers(size_list, self.dropout_prob)

        # Linear product to the final score
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        self.aux_lambda = config["aux_lambda"]
        self.eval_stage = None
        self.item_group = None
        self.dataset_name = dataset.dataset_name
        if item_freq_tensor is not None:
            self.register_buffer("item_freq_tensor", item_freq_tensor)

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

        y_fm = self.first_order_linear(interaction) + self.fm(deepfm_all_embeddings)
        x = deepfm_all_embeddings.view(batch_size, -1)
        # # x = self.filter(x)
        x_item = x[:, 10:20]

        router_indices, residuals_in, quantized_vectors, _ = self.rq(x_item, tau=self.tau)
        loss_rq = self.rq.calculate_vq_loss(residuals_in, quantized_vectors)

        x_item_quantized = quantized_vectors.sum(dim=1)
        item_ids = interaction["item_id"].view(-1).long()
        item_freq = self.item_freq_tensor[item_ids]
        rare_mask = (item_freq <= 10).unsqueeze(1)

        target_item = torch.where(rare_mask, x_item_quantized, x_item)
        delta_item = self.rq_weight * target_item
        scale_factor = 1.0 + self.rq_weight
        x_scaled = x * scale_factor
        diff = self.rq_weight * (x_item_quantized - x_item)
        correction = rare_mask.float() * diff
        x_scaled[:, 10:20] += correction
        x = x_scaled

        moe_feat = self.moe_layers(x, router_indices)

        y_deep = self.deep_predict_layer(moe_feat)
        y = y_deep

        return (y.squeeze(-1), loss_rq)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output, loss_rq = self.forward(interaction, return_gate=False)
        bce = self.loss(output, label)
        return bce + loss_rq

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction)[0])

    # def predict(self, interaction, save_path="gate_stats_test.jsonl", model_name="SAGMoE"):
    #     if model_name is not None:
    #         save_path = f"{model_name}_{self.dataset_name}.jsonl"
    #     output, meta, _ = self.forward(interaction, return_gate=True)
    #     B = output.shape[0]
    #     prediction = self.sigmoid(output)
    #     batch_entropy, max_gate, expert_counts = None, None, None
    #     if meta is not None and "gates" in meta:
    #         gates = meta["gates"]
    #         max_gate, top_idx = torch.max(gates, dim=1)
    #         expert_counts = torch.bincount(top_idx, minlength=gates.size(1))
    #         p = expert_counts.float() / B
    #         batch_entropy = -(p * torch.log(p + 1e-8)).sum()
    #         if getattr(self, "eval_stage", None) == "test":
    #             group_value = self.item_group if self.item_group is not None else "all"
    #             record = {
    #                 "group": f"[{group_value}]",
    #                 "entropy": batch_entropy.item(),
    #                 "expert_counts": expert_counts.detach().cpu().tolist(),
    #             }
    #             os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    #             with open(save_path, "a", encoding="utf-8") as f:
    #                 f.write(json.dumps(record) + "\n")

    #     return prediction


class RQMoE(nn.Module):
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
        config: object = None,
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
        self.config = config
        self.voterouter = build_trajectory_router(
            router_type="vote",
            num_codebooks=self.config["num_codebooks"],
            codebook_size=self.config["codebook_size"],
            num_experts=4,
        )

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

    def forward(self, x, r_idx=None, return_gate: bool = False):
        B = x.size(0)

        # logits = self._route_logits(x, train=self.training)
        logits = self.voterouter(r_idx)

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
