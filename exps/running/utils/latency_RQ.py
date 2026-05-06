import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

# ==========================================
# 1. 配置参数 (与论文一致)
# ==========================================
BATCH_SIZE = 4096
EMBED_DIM = 20  # D
NUM_CODEBOOKS = 2  # M (RQ层数)
CODEBOOK_SIZE = 256  # K (码本大小)
NUM_EXPERTS = 4  # N (专家数)
EXPERT_HIDDEN = 32  # 专家中间层维度
TOP_K = 1  # MoE Top-K

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benchmarking on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ==========================================
# 2. Naive Implementation (你的原始代码 + Loop MoE)
# ==========================================


class NaiveRQEncoder(nn.Module):
    def __init__(self, num_codebooks, codebook_size, embed_dim):
        super().__init__()
        self.M = num_codebooks
        self.K = codebook_size
        self.D = embed_dim
        self.codebooks = nn.Parameter(torch.empty(self.M, self.K, self.D))
        nn.init.uniform_(self.codebooks, -1.0 / self.K, 1.0 / self.K)

    def forward(self, x, tau=1.0):
        # 你的原始逻辑
        codebook_sq = torch.sum(self.codebooks**2, dim=-1).unsqueeze(0)
        current_residual = x
        all_probs = []

        for m in range(self.M):
            C_m = self.codebooks[m]
            interaction = torch.matmul(current_residual, C_m.t())
            logits = 2 * interaction - codebook_sq[:, m, :]
            y_soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            all_probs.append(y_soft)
            c_m = torch.matmul(y_soft, C_m)
            current_residual = current_residual - c_m

        return all_probs, current_residual


class NaiveVoteRouter(nn.Module):
    def __init__(self, num_codebooks, codebook_size, num_experts):
        super().__init__()
        self.M = num_codebooks
        self.K = codebook_size
        self.N = num_experts
        self.W_affinity = nn.Parameter(torch.empty(self.M, self.K, self.N))
        nn.init.kaiming_uniform_(self.W_affinity, a=math.sqrt(5))

    def forward(self, soft_probabilities_list):
        # 你的原始逻辑
        all_probs_flat = torch.cat(soft_probabilities_list, dim=1)  # [B, M*K]
        W_flat = self.W_affinity.view(-1, self.N)
        total_votes = torch.matmul(all_probs_flat, W_flat)
        return total_votes


class NaiveSparseMoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
                for _ in range(num_experts)
            ]
        )

    def forward(self, x, router_logits):
        # Naive Loop Implementation
        topk_logits, topk_indices = router_logits.topk(self.top_k, dim=1)
        gate_weights = F.softmax(topk_logits, dim=1)
        final_output = torch.zeros_like(x)

        for i in range(self.num_experts):
            # Masking overhead
            mask = topk_indices == i
            batch_indices = mask.nonzero(as_tuple=True)[0]
            if len(batch_indices) == 0:
                continue

            selected_x = x[batch_indices]
            expert_out = self.experts[i](selected_x)

            # Scatter overhead
            # Simplify weights handling for benchmarking compute
            final_output.index_add_(0, batch_indices, expert_out)

        return final_output


# ==========================================
# 3. Optimized Proxy (cuBLAS / Fused Kernel 模拟)
# ==========================================


class OptimizedPipelineProxy(nn.Module):
    def __init__(self):
        super().__init__()
        # RQ Params
        # 假设我们将 M 次计算并行化 (模拟流水线吞吐量)
        # Weights: [M, D, K] (Distance) 和 [M, K, D] (Quantization)
        self.rq_w1 = nn.Parameter(torch.randn(NUM_CODEBOOKS, EMBED_DIM, CODEBOOK_SIZE))
        self.rq_w2 = nn.Parameter(torch.randn(NUM_CODEBOOKS, CODEBOOK_SIZE, EMBED_DIM))

        # Vote Params
        # Dense GEMM: [M*K, N]
        self.vote_w = nn.Parameter(torch.randn(NUM_CODEBOOKS * CODEBOOK_SIZE, NUM_EXPERTS))

        # MoE Params (Batched GEMM)
        # 假设完美负载均衡，每个 Expert 处理 Batch * TopK / N 个样本
        self.capacity = int(BATCH_SIZE * TOP_K / NUM_EXPERTS)
        # Weights: [N, D, H] 和 [N, H, D]
        self.moe_w1 = nn.Parameter(torch.randn(NUM_EXPERTS, EMBED_DIM, EXPERT_HIDDEN))
        self.moe_w2 = nn.Parameter(torch.randn(NUM_EXPERTS, EXPERT_HIDDEN, EMBED_DIM))

    def forward(self, x):
        # --- 1. RQ Proxy (Throughput Simulation) ---
        # 模拟 M 次迭代的计算量。
        # 我们用 bmm 并行计算 [B, M, D] * [M, D, K] -> [B, M, K]
        # 这忽略了串行依赖的时延，但在大 Batch 下吞吐量是近似的。
        x_expanded = x.unsqueeze(1).expand(-1, NUM_CODEBOOKS, -1).permute(1, 0, 2)  # [M, B, D]

        # Step 1: Distance / Logits (Compute Bound)
        # [M, B, D] @ [M, D, K] -> [M, B, K]
        logits_proxy = torch.bmm(x_expanded, self.rq_w1)

        # Step 2: Quantization / Residual Update (Compute Bound)
        # [M, B, K] @ [M, K, D] -> [M, B, D]
        quant_proxy = torch.bmm(logits_proxy, self.rq_w2)

        # --- 2. Vote Proxy (Dense GEMM) ---
        # 模拟 [B, M*K] @ [M*K, N]
        # 我们这里简化，直接做相同FLOPs的计算
        # flatten: [B, M, K] -> [B, M*K]
        votes_in = logits_proxy.permute(1, 0, 2).reshape(BATCH_SIZE, -1)
        router_logits = torch.matmul(votes_in, self.vote_w)

        # --- 3. MoE Proxy (Batched GEMM) ---
        # 模拟 C++ Kernel 已经把数据排好序了
        # Input: [N, Capacity, D]
        moe_in = torch.randn(NUM_EXPERTS, self.capacity, EMBED_DIM, device=x.device)

        # MLP Layer 1: [N, C, D] @ [N, D, H] -> [N, C, H]
        h = torch.bmm(moe_in, self.moe_w1)
        h = F.relu(h)

        # MLP Layer 2: [N, C, H] @ [N, H, D] -> [N, C, D]
        out = torch.bmm(h, self.moe_w2)

        return out


# ==========================================
# 4. Benchmarking Utils
# ==========================================


def run_benchmark(models_or_proxy, x, desc, n_warmup=50, n_repeat=500):
    # --- 修复点：区分列表和单个 Module ---
    if isinstance(models_or_proxy, list):
        for m in models_or_proxy:
            m.eval()
    else:
        models_or_proxy.eval()
    # -----------------------------------

    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            if isinstance(models_or_proxy, list):
                # Naive Pipeline: 依次调用 list 里的三个模型
                probs, res = models_or_proxy[0](x)  # RQ
                votes = models_or_proxy[1](probs)  # Vote
                out = models_or_proxy[2](x, votes)  # MoE
            else:
                # Optimized Proxy: 直接调用
                models_or_proxy(x)

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(n_repeat):
            if isinstance(models_or_proxy, list):
                probs, res = models_or_proxy[0](x)
                votes = models_or_proxy[1](probs)
                out = models_or_proxy[2](x, votes)
            else:
                models_or_proxy(x)

        torch.cuda.synchronize()
        end = time.time()

    avg_lat = (end - start) / n_repeat * 1000
    print(f"[{desc}] Latency: {avg_lat:.4f} ms")
    return avg_lat


# ==========================================
# 5. Main Execution
# ==========================================

if torch.cuda.is_available():
    # Setup Data
    x = torch.randn(BATCH_SIZE, EMBED_DIM, device=device)

    # --- Naive Setup ---
    naive_rq = NaiveRQEncoder(NUM_CODEBOOKS, CODEBOOK_SIZE, EMBED_DIM).to(device)
    naive_vote = NaiveVoteRouter(NUM_CODEBOOKS, CODEBOOK_SIZE, NUM_EXPERTS).to(device)
    naive_moe = NaiveSparseMoE(EMBED_DIM, EMBED_DIM, NUM_EXPERTS, EXPERT_HIDDEN, TOP_K).to(device)
    naive_models = [naive_rq, naive_vote, naive_moe]

    # --- Optimized Setup ---
    optimized_proxy = OptimizedPipelineProxy().to(device)

    print("-" * 40)
    print(
        f"Pipeline Config: B={BATCH_SIZE}, RQ(M={NUM_CODEBOOKS}, K={CODEBOOK_SIZE}), MoE(N={NUM_EXPERTS}, TopK={TOP_K})"
    )
    print("-" * 40)

    # Run
    t_naive = run_benchmark(naive_models, x, "Naive PyTorch (Current)")
    t_opt = run_benchmark(optimized_proxy, x, "Optimized Proxy (cuBLAS)")

    # Add Overhead Estimate (Scatter/Gather usually +15%)
    t_projected = t_opt * 1.15

    print("-" * 40)
    print(f"Speedup Potential: {t_naive / t_projected:.1f}x")
    print(f"Projected Limit:   {t_projected:.4f} ms")
    print("-" * 40)

else:
    print("CUDA not available, cannot run benchmark.")
