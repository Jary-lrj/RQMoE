import torch
import time
import numpy as np

# -----------------------------------------------------------------------------
# 设置参数 (保持和你论文一致)
# -----------------------------------------------------------------------------
BATCH_SIZE = 4096
INPUT_DIM = 20  # 假设 Embedding 维度
HIDDEN_DIM = 32  # Expert 内部维度
OUTPUT_DIM = 16
NUM_EXPERTS = 4  # 或你的实际数量
TOP_K = 1  # 你的 Top-K

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# 1. 模拟 Naive Python Loop (你的现状)
# -----------------------------------------------------------------------------
class NaiveMoELayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(INPUT_DIM, HIDDEN_DIM), torch.nn.ReLU(), torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
                )
                for _ in range(NUM_EXPERTS)
            ]
        )

    def forward(self, x, indices):
        # x: [B, Dim], indices: [B, TopK]
        # 这是一个简化的模拟，只测计算瓶颈
        out = torch.zeros(BATCH_SIZE, OUTPUT_DIM, device=device)
        for i in range(NUM_EXPERTS):
            # 找出分配给 expert i 的样本 (模拟 mask overhead)
            mask = indices == i
            batch_idx, k_idx = mask.nonzero(as_tuple=True)
            if len(batch_idx) == 0:
                continue

            selected = x[batch_idx]
            expert_out = self.experts[i](selected)
            out.index_add_(0, batch_idx, expert_out)  # 模拟聚合
        return out


# -----------------------------------------------------------------------------
# 2. 模拟 C++/cuBLAS Optimized (Batched GEMM)
# -----------------------------------------------------------------------------
# 原理：假设通过 C++ Kernel 极其高效地把数据搬运好了，
# 我们将计算重塑为 [Num_Experts, Capacity, Dim] 的批量矩阵乘法。
# 这是硬件计算能力的理论上限。
class OptimizedProxyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 将所有 Expert 权重堆叠: [Num_Experts, In, Out]
        self.w1 = torch.nn.Parameter(torch.randn(NUM_EXPERTS, INPUT_DIM, HIDDEN_DIM))
        self.b1 = torch.nn.Parameter(torch.randn(NUM_EXPERTS, 1, HIDDEN_DIM))
        self.w2 = torch.nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN_DIM, OUTPUT_DIM))
        self.b2 = torch.nn.Parameter(torch.randn(NUM_EXPERTS, 1, OUTPUT_DIM))

    def forward(self, x_sorted):
        # x_sorted: [Num_Experts, Avg_Capacity, Input_Dim]
        # 假设完美负载均衡：每个 Expert 处理 (B * TopK / Num_Experts) 个样本
        # 1. First Layer GEMM (cuBLAS optimized bmm)
        # [E, C, In] @ [E, In, H] -> [E, C, H]
        h = torch.bmm(x_sorted, self.w1) + self.b1
        h = torch.relu(h)

        # 2. Second Layer GEMM
        # [E, C, H] @ [E, H, Out] -> [E, C, Out]
        out = torch.bmm(h, self.w2) + self.b2
        return out


# -----------------------------------------------------------------------------
# Benchmark 函数
# -----------------------------------------------------------------------------
def benchmark(model, input_data, n_warmup=50, n_repeat=1000, desc=""):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            model(*input_data)
        torch.cuda.synchronize()

        # Timing
        start = time.time()
        for _ in range(n_repeat):
            model(*input_data)
        torch.cuda.synchronize()
        end = time.time()

    avg_latency = (end - start) / n_repeat * 1000  # ms
    print(f"{desc}: {avg_latency:.4f} ms")
    return avg_latency


# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    print(f"Benchmarking on {torch.cuda.get_device_name(0)}...")

    # 准备数据
    x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
    # 随机路由索引
    indices = torch.randint(0, NUM_EXPERTS, (BATCH_SIZE, TOP_K), device=device)

    # 1. 测 Naive Loop
    naive_model = NaiveMoELayer().to(device)
    t_naive = benchmark(naive_model, (x, indices), desc="Naive PyTorch (Loop)")

    # 2. 测 Optimized Proxy
    # 假设每个 Expert 分到的数据量一样 (Ideal Load Balancing)
    avg_capacity = int((BATCH_SIZE * TOP_K) / NUM_EXPERTS)
    # 构造对应形状的输入，模拟已经 Gather 好的数据
    x_batched = torch.randn(NUM_EXPERTS, avg_capacity, INPUT_DIM, device=device)

    opt_model = OptimizedProxyLayer().to(device)
    t_opt = benchmark(opt_model, (x_batched,), desc="Optimized Proxy (cuBLAS BMM)")

    # 3. 估算 Scatter/Gather 开销 (通常假设占 20%)
    t_scatter_gather = t_opt * 0.2
    t_projected = t_opt + t_scatter_gather

    print("-" * 30)
    print(f"Current Latency:   {t_naive:.4f} ms")
    print(f"Projected Latency: {t_projected:.4f} ms (with C++ Kernel)")
    print(f"Speedup Potential: {t_naive / t_projected:.1f}x")
else:
    print("No GPU detected. Cannot run cuBLAS benchmark.")
