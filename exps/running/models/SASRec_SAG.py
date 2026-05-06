import torch
import copy
import math
from torch import nn
from torch.nn import functional as F

from recbole.model.abstract_recommender import SequentialRecommender


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
        self.saved_topk = []

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

    def _estimate_signal_quality(self, x: torch.tensor) -> torch.tensor:
        max_abs = x.abs().max(dim=1, keepdim=True)[0]
        l2 = x.norm(p=2, dim=1, keepdim=True) + 1e-8
        concentration = max_abs / l2
        quality = 1.0 - concentration
        return quality

    def forward(self, x, return_gate: bool = False, force_expert: int = None):
        orig_shape = x.shape
        x = x.view(-1, self.input_size)  # [B, H]
        B = x.size(0)
        individual_expert_outputs = [[] for _ in range(B)]

        if self.training:
            raw_logits = self._route_logits(x, train=True)  # [B, E]
            logits = raw_logits
            k = self.top_k_eval
            top_logits, top_idx = logits.topk(k, dim=1)  # [B, k]
            gates_full = F.softmax(logits, dim=1)  # [B, E]

            # keep only top-k mass, then renormalize (soft-sparse during training)
            sparse_gates = torch.zeros_like(logits)
            sparse_gates.scatter_(1, top_idx, gates_full.gather(1, top_idx))
            sparse_gates = sparse_gates / (sparse_gates.sum(dim=1, keepdim=True) + 1e-10)

            out = torch.zeros(B, self.output_size, device=x.device, dtype=x.dtype)
            for e, expert in enumerate(self.experts):
                mask_e = (top_idx == e).any(dim=1)  # [B]
                if mask_e.any():
                    y_e = expert(x[mask_e])  # [B_e, H_out]
                    g_e = sparse_gates[mask_e, e].unsqueeze(1)  # [B_e, 1]
                    out[mask_e] += g_e * y_e

            gates = sparse_gates  # keep naming consistent for return
            topk_idx = top_idx

        else:
            if force_expert is not None:
                logits = torch.zeros(B, self.num_experts, device=x.device, dtype=x.dtype)
                expert_output = self.experts[force_expert](x)  # [B, output_size]
                gates = torch.zeros(B, self.num_experts, device=x.device, dtype=x.dtype)
                gates[:, force_expert] = 1.0
                topk_idx = torch.full((B, 1), force_expert, device=x.device, dtype=torch.long)
                for i in range(B):
                    individual_expert_outputs[i].append({"expert_id": force_expert, "output": expert_output[i]})
                out = expert_output
            else:
                logits = self._route_logits(x, train=False)  # [B, E]
                k = self.top_k_eval
                top_logits, top_idx = logits.topk(k, dim=1)  # [B, k]
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(1, top_idx, top_logits)  # non-topk = -inf
                gates = F.softmax(masked, dim=1)  # strictly sparse

                out = torch.zeros(B, self.output_size, device=x.device, dtype=x.dtype)
                for e, expert in enumerate(self.experts):
                    mask_e = (top_idx == e).any(dim=1)  # [B]
                    if mask_e.any():
                        selected_indices = mask_e.nonzero(as_tuple=True)[0]
                        y_e = expert(x[mask_e])  # [B_e, H_out]
                        g_e = gates[mask_e, e].unsqueeze(1)  # [B_e, 1]
                        out[mask_e] += g_e * y_e
                        for i, original_idx in enumerate(selected_indices):
                            individual_expert_outputs[original_idx.item()].append({"expert_id": e, "output": y_e[i]})
                topk_idx = top_idx

        out = out.view(*orig_shape[:-1], self.output_size)
        if return_gate:
            gate_info = {"logits": logits, "gates": gates, "topk_idx": topk_idx}
            if not self.training:
                gate_info["individual_outputs"] = individual_expert_outputs
            return out, gate_info
        return out


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = SparseMoE(
            input_size=hidden_size,
            output_size=hidden_size,
            num_experts=1,
            hidden_size=intermediate_size,
            dropout=hidden_dropout_prob,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config["inner_size"]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
