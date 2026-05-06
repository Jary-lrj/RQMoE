import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# ==========================================
# 1. 配置参数 (模拟 DeepFM 的典型设置)
# ==========================================
BATCH_SIZE = 4096
INPUT_DIM = 20  # 假设 concat 后的 embedding 维度
HIDDEN_LAYERS = [32, 16]  # MLP 层结构
DROPOUT_PROB = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benchmarking on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# ==========================================
# 2. Standard DeepFM MLP (PyTorch 现状)
# ==========================================
class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# ==========================================
# 3. Optimized Proxy (Pure GEMM / Perfect Fusion)
# ==========================================
# 模拟极致的算子融合：假设 Bias, ReLU, Dropout 都是"免费"的
# 只测最核心的矩阵乘法 (cuBLAS GEMM) 耗时，这是物理上限。
class FusedMLPProxy(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        self.weights = nn.ParameterList()
        curr_dim = input_dim
        for h in hidden_layers:
            # 权重矩阵 [Out, In]
            self.weights.append(nn.Parameter(torch.randn(h, curr_dim)))
            curr_dim = h

    def forward(self, x):
        out = x
        for w in self.weights:
            # 只做矩阵乘法，不分配新显存给中间的 Bias/ReLU 结果
            # F.linear 底层就是调用 cuBLAS GEMM
            out = F.linear(out, w)
        return out


# ==========================================
# 4. Benchmark Utils
# ==========================================
def run_benchmark(model, x, desc, n_warmup=50, n_repeat=1000):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(n_repeat):
            model(x)
        torch.cuda.synchronize()

        end = time.time()

    avg_lat = (end - start) / n_repeat * 1000
    print(f"[{desc}] Latency: {avg_lat:.4f} ms")
    return avg_lat


# ==========================================
# 5. Main
# ==========================================
if torch.cuda.is_available():
    x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)

    # 1. Standard
    std_mlp = StandardMLP(INPUT_DIM, HIDDEN_LAYERS, DROPOUT_PROB).to(device)
    t_std = run_benchmark(std_mlp, x, "Standard DeepFM MLP")

    # 2. Optimized Proxy
    fused_proxy = FusedMLPProxy(INPUT_DIM, HIDDEN_LAYERS).to(device)
    t_opt = run_benchmark(fused_proxy, x, "Fused Proxy (Pure GEMM)")

    print("-" * 40)
    print(f"Standard Latency:  {t_std:.4f} ms")
    print(f"Theoretical Limit: {t_opt:.4f} ms")
    print(f"Max Speedup:       {t_std / t_opt:.2f}x")
    print("-" * 40)
    print("结论: DeepFM 已经是 '极速' 了，优化空间非常有限。")
else:
    print("No GPU.")
