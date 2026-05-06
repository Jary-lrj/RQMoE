# _*_ coding: utf-8 _*_
# @Time   : 2022/8/29
# @Author : Yifan Li
# @Email  : 295435096@qq.com

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.model.loss import RegLoss
from .RQEncoder import RQEncoder
from .VoteRouter import VoterRouter


class SparseMoE(nn.Module):
    """
    Sparse Mixture-of-Experts Layer
    """

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
        self.router = VoterRouter(
            num_codebooks=2,
            codebook_size=128,
            num_experts=num_experts,
        )
        # Gating Network
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

        # Experts: Each expert is an MLP that maps Input -> Hidden -> Output (Scalar)
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

        logits = self._route_logits(x, train=self.training)
        # logits = self.router(r_idx)
        # Top-K Gating
        k = self.top_k_eval
        top_logits, top_idx = logits.topk(k, dim=1)  # [B, k]

        if self.training:
            gates_full = F.softmax(logits, dim=1)
            gates = torch.zeros_like(logits)
            gates.scatter_(1, top_idx, gates_full.gather(1, top_idx))
            # Normalize so active gates sum to 1
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


class DCNV2(ContextRecommender):
    r"""DCNV2 improves the cross network by extending the original weight vector to a matrix,
    significantly improves the expressiveness of DCN. It also introduces the MoE and
    low rank techniques to reduce time cost.
    """

    def __init__(self, config, dataset, item_freq_tensor):
        super(DCNV2, self).__init__(config, dataset)

        # load and compute parameters info
        self.mixed = config["mixed"]
        self.structure = config["structure"]
        self.cross_layer_num = config["cross_layer_num"]
        self.embedding_size = config["embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.reg_weight = config["reg_weight"]
        self.dropout_prob = config["dropout_prob"]

        self.rq = RQEncoder(
            num_codebooks=config["num_codebooks"],
            codebook_size=config["codebook_size"],
            embed_dim=10,
        )
        self.tau = config["tau"]
        # --- SWITCH: Control whether to use MoE for the final prediction layer ---
        # Default to False to keep original behavior unless specified
        self.use_moe_predict = config["use_moe_predict"]

        if self.mixed:
            self.expert_num = config["expert_num"]
            self.low_rank = config["low_rank"]

        self.in_feature_num = self.num_feature_field * self.embedding_size

        # --- Define Cross Layers ---
        if self.mixed:
            self.cross_layer_u = nn.ParameterList(
                nn.Parameter(torch.randn(self.expert_num, self.in_feature_num, self.low_rank))
                for _ in range(self.cross_layer_num)
            )
            self.cross_layer_v = nn.ParameterList(
                nn.Parameter(torch.randn(self.expert_num, self.in_feature_num, self.low_rank))
                for _ in range(self.cross_layer_num)
            )
            self.cross_layer_c = nn.ParameterList(
                nn.Parameter(torch.randn(self.expert_num, self.low_rank, self.low_rank))
                for _ in range(self.cross_layer_num)
            )
            self.gating = nn.ModuleList(nn.Linear(self.in_feature_num, 1) for _ in range(self.expert_num))
        else:
            self.cross_layer_w = nn.ParameterList(
                nn.Parameter(torch.randn(self.in_feature_num, self.in_feature_num)) for _ in range(self.cross_layer_num)
            )

        self.bias = nn.ParameterList(
            nn.Parameter(torch.zeros(self.in_feature_num, 1)) for _ in range(self.cross_layer_num)
        )

        # --- Define Deep Layers (MLP) ---
        mlp_size_list = [self.in_feature_num] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(mlp_size_list, dropout=self.dropout_prob, bn=True)

        # --- Determine Input Size for Prediction Layer ---
        if self.structure == "parallel":
            # Parallel: Concat(CrossOutput, MLPOutput)
            predict_input_size = self.in_feature_num + self.mlp_hidden_size[-1]
        elif self.structure == "stacked":
            # Stacked: MLPOutput
            predict_input_size = self.mlp_hidden_size[-1]
        else:
            raise ValueError("structure must be 'parallel' or 'stacked'")

        # --- Initialize Prediction Layer based on Switch ---
        if self.use_moe_predict:
            # Use SparseMoE as Predict Layer
            moe_num_experts = config["expert_num"]
            moe_top_k = config["top_k"]
            # You can separate config for MoE head vs DCN mixed part if needed
            # Here we reuse the mlp_hidden_size for experts internal structure
            moe_expert_hidden = config["mlp_hidden_size"]

            self.moe_predict_layer = SparseMoE(
                input_size=predict_input_size,
                output_size=1,
                num_experts=moe_num_experts,
                hidden_sizes=moe_expert_hidden,
                top_k_eval=moe_top_k,
                dropout=self.dropout_prob,
            )
        else:
            # Use Standard Linear Layer (Original DNN Predict Layer)
            self.predict_layer = nn.Linear(predict_input_size, 1)

        # define loss and activation functions
        self.reg_loss = RegLoss()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.BCELoss()
        if item_freq_tensor is not None:
            self.register_buffer("item_freq_tensor", item_freq_tensor)
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def cross_network(self, x_0):
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0
        for i in range(self.cross_layer_num):
            xl_w = torch.matmul(self.cross_layer_w[i], x_l)
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x_0, xl_w)
            x_l = xl_dot + x_l
        x_l = x_l.squeeze(dim=2)
        return x_l

    def cross_network_mix(self, x_0):
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0
        for i in range(self.cross_layer_num):
            expert_output_list = []
            gating_output_list = []
            for expert in range(self.expert_num):
                gating_output_list.append(self.gating[expert](x_l.squeeze(dim=2)))
                xl_v = torch.matmul(self.cross_layer_v[i][expert].T, x_l)
                xl_c = self.tanh(xl_v)
                xl_c = torch.matmul(self.cross_layer_c[i][expert], xl_c)
                xl_c = self.tanh(xl_c)
                xl_u = torch.matmul(self.cross_layer_u[i][expert], xl_c)
                xl_dot = xl_u + self.bias[i]
                xl_dot = torch.mul(x_0, xl_dot)
                expert_output_list.append(xl_dot.squeeze(dim=2))
            expert_output = torch.stack(expert_output_list, dim=2)
            gating_output = torch.stack(gating_output_list, dim=1)
            moe_output = torch.matmul(expert_output, self.softmax(gating_output))
            x_l = x_l + moe_output
        x_l = x_l.squeeze(dim=2)
        return x_l

    def forward(self, interaction):
        dcn_all_embeddings = self.concat_embed_input_fields(interaction)
        batch_size = dcn_all_embeddings.shape[0]
        dcn_all_embeddings = dcn_all_embeddings.view(batch_size, -1)
        x = dcn_all_embeddings

        x_item = x[:, 10:20]

        router_indices, residuals_in, quantized_vectors = self.rq(x_item, tau=self.tau)
        loss_rq = self.rq.calculate_vq_loss(residuals_in, quantized_vectors)

        if not hasattr(self, "rq_weight"):
            self.rq_weight = nn.Parameter(torch.tensor(1.0, device=x.device))

        x_rq = x.clone()
        x_item_quantized = quantized_vectors.sum(dim=1)
        x_rq[:, 10:20] = x_item_quantized
        item_ids = interaction["item_id"].view(-1).long()
        item_freq = self.item_freq_tensor[item_ids]
        rare_mask = (item_freq <= 10).unsqueeze(1)
        x_rq[:, 10:20] = torch.where(rare_mask, x_item_quantized, x_item)
        x = x + self.rq_weight * x_rq

        # 1. Compute Deep Output
        if self.structure == "parallel":
            deep_output = self.mlp_layers(x)
        elif self.structure == "stacked":
            # For stacked, deep layers come after cross network, calculated below
            pass

        # 2. Compute Cross Output
        if self.mixed:
            cross_output = self.cross_network_mix(x)
        else:
            cross_output = self.cross_network(x)

        # 3. Combine to get Final Input for Prediction Layer
        if self.structure == "parallel":
            final_input = torch.cat([cross_output, deep_output], dim=-1)
        elif self.structure == "stacked":
            # Stacked: Embed -> Cross -> Deep -> Predict
            deep_output = self.mlp_layers(cross_output)
            final_input = deep_output

        # 4. Final Prediction (Switch Logic)
        if self.use_moe_predict:
            logit = self.moe_predict_layer(final_input)
        else:
            logit = self.predict_layer(final_input)

        output = self.sigmoid(logit)
        return output.squeeze(dim=1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
