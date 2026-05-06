from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from models.SAGMoE import SAGMoE
from recbole.data.dataloader.general_dataloader import NegSampleEvalDataLoader
from recbole.sampler import Sampler
from recbole.model.context_aware_recommender import DeepFM, DeepFM_MoE
from models.Switch import DeepFM_MoE
from models.DCNv2 import DCNV2
from recbole.model.sequential_recommender import DIN
import numpy as np
import torch
from plot_frequency import plot_frequency_log_log
import os
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    init_seed,
    set_color,
    get_flops,
)
import time
import argparse


def build_item_freq_map(train_dataset):
    item_num = train_dataset.item_num
    train_items = train_dataset.inter_feat["item_id"].numpy()
    freq = np.bincount(train_items, minlength=item_num)
    return freq


def build_user_freq_map(train_dataset):
    user_num = train_dataset.user_num
    train_users = train_dataset.inter_feat["user_id"].numpy()
    freq = np.bincount(train_users, minlength=user_num)
    return freq


def split_test_interactions_by_freq(test_dataset, freq_map):
    test_items = test_dataset.inter_feat["item_id"].numpy()  # [N_test]
    test_freq = freq_map[test_items]  # [N_test]

    mask_ultra = test_freq <= 1
    mask_cold = (test_freq >= 2) & (test_freq <= 10)
    mask_mid = (test_freq >= 11) & (test_freq <= 100)
    mask_hot = test_freq > 100

    idx_ultra = np.where(mask_ultra)[0]
    idx_cold = np.where(mask_cold)[0]
    idx_mid = np.where(mask_mid)[0]
    idx_hot = np.where(mask_hot)[0]

    inter_all = test_dataset.inter_feat
    inter_ultra = inter_all[mask_ultra]
    inter_cold = inter_all[mask_cold]
    inter_mid = inter_all[mask_mid]
    inter_hot = inter_all[mask_hot]

    return (
        inter_hot,
        inter_mid,
        inter_cold,
        inter_ultra,
        test_items[idx_hot],
        test_items[idx_mid],
        test_items[idx_cold],
        test_items[idx_ultra],
    )


def make_eval_loader_from_interaction(config, base_dataset, inter_subset, sampler, name="test-subset"):
    sub_dataset = base_dataset.copy(inter_subset)
    eval_loader = NegSampleEvalDataLoader(config, sub_dataset, sampler=sampler, shuffle=False)
    return eval_loader


def get_item_embeddings(model, dataset, id_list, device="cuda") -> torch.Tensor:
    emb_weight = getattr(getattr(getattr(model, "token_embedding_table"), "embedding"), "weight", None)
    if emb_weight is None:
        raise AttributeError("未找到 embedding 权重。期望路径: model.token_embedding_table.embedding.weight")
    emb_weight = emb_weight.to(device)
    ids = torch.as_tensor(id_list, dtype=torch.long, device=device)
    base_indices = ids
    offset = int(dataset.user_num) + 1
    real_indices = base_indices + offset
    emb_subset = torch.index_select(emb_weight, 0, real_indices)  # [M, emb_dim]
    return emb_subset


def get_svd_values(embeddings):
    X = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    if len(s) < embeddings.shape[1]:
        s = np.pad(s, (0, embeddings.shape[1] - len(s)), constant_values=0)

    return s.reshape(1, -1)


def get_mean_l2_norm(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)
    return np.mean(norms)


def get_eval_metrics_by_group(model, trainer, results, item_group, eval_data, embeddings, logger):
    model.item_group = item_group
    results[item_group] = trainer.evaluate(eval_data)
    # logger.info(f"[{item_group}] {results[item_group]}")
    # logger.info(f"Mean L2 Norm: {get_mean_l2_norm(embeddings)}")
    # logger.info(f"Singular Values: {get_svd_values(embeddings)}")


def evaluate_by_frequency_buckets(config, train_data, test_data, dataset, model, trainer, logger):
    """
    根据物品频率将测试集分桶并评估

    Args:
        config: 配置对象
        train_data: 训练数据
        test_data: 测试数据
        dataset: 数据集对象
        model: 模型对象
        trainer: 训练器对象
        logger: 日志记录器

    Returns:
        dict: 包含各分桶评估结果的字典
    """
    # 构建物品频率映射
    item_freq = build_item_freq_map(train_data._dataset)

    # 根据频率分割测试交互
    (
        inter_hot,
        inter_mid,
        inter_cold,
        inter_ultra,
        idx_hot,
        idx_mid,
        idx_cold,
        idx_ultra,
    ) = split_test_interactions_by_freq(test_data._dataset, item_freq)
    logger.info(
        f"[Test Split] Hot={len(inter_hot)}, Mid={len(inter_mid)}, Cold={len(inter_cold)}, Ultra-Cold={len(inter_ultra)}"
    )

    # 模型评估阶段设置
    model.eval_stage = "test"
    results = {}

    # 创建采样器
    sampler = Sampler(phases="test", datasets=dataset, distribution="uniform")
    sampler = sampler.set_phase("test")

    # 为每个分桶创建评估加载器
    eval_hot = make_eval_loader_from_interaction(config, test_data._dataset, inter_hot, sampler, name="test-hot")
    eval_mid = make_eval_loader_from_interaction(config, test_data._dataset, inter_mid, sampler, name="test-mid")
    eval_cold = make_eval_loader_from_interaction(config, test_data._dataset, inter_cold, sampler, name="test-cold")
    eval_ultra = make_eval_loader_from_interaction(config, test_data._dataset, inter_ultra, sampler, name="test-ultra")

    usage_stats = {}
    save_dir = "./analysis_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 评估各个分桶
    if len(inter_hot) > 0:
        hot_embedding = get_item_embeddings(model, dataset, idx_hot)
        with torch.no_grad():
            _, _, _, hot_indices = model.rq(hot_embedding)
            usage_stats["hot"] = hot_indices.cpu()  # 转到 CPU 省显存
        get_eval_metrics_by_group(model, trainer, results, "Hot", eval_hot, hot_embedding, logger)
    if len(inter_mid) > 0:
        mid_embedding = get_item_embeddings(model, dataset, idx_mid)
        with torch.no_grad():
            _, _, _, mid_indices = model.rq(mid_embedding)
            usage_stats["mid"] = mid_indices.cpu()
        get_eval_metrics_by_group(model, trainer, results, "Mid", eval_mid, mid_embedding, logger)
    if len(inter_cold) > 0:
        cold_embedding = get_item_embeddings(model, dataset, idx_cold)
        with torch.no_grad():
            _, _, _, cold_indices = model.rq(cold_embedding)
            usage_stats["cold"] = cold_indices.cpu()
        get_eval_metrics_by_group(model, trainer, results, "Cold", eval_cold, cold_embedding, logger)
    if len(inter_ultra) > 0:
        ultra_embedding = get_item_embeddings(model, dataset, idx_ultra)
        with torch.no_grad():
            _, _, _, ultra_indices = model.rq(ultra_embedding)
            usage_stats["ultra"] = ultra_indices.cpu()
        get_eval_metrics_by_group(model, trainer, results, "Ultra-Cold", eval_ultra, ultra_embedding, logger)
    torch.save(usage_stats, os.path.join(save_dir, "codebook_usage_stats.pt"))
    logger.info(f"Codebook usage stats saved to {save_dir}/codebook_usage_stats.pt")
    return results


def get_model_class(model_name):

    model_map = {"DeepFM": DeepFM, "DeepFM_MoE": DeepFM_MoE, "SAGMoE": SAGMoE, "DCNV2": DCNV2}

    model_name = model_name.strip()
    if model_name not in model_map:
        raise ValueError(f"Unavailable model: {model_name}." f"Available models: {list(model_map.keys())}")

    return model_map[model_name]


import torch
import numpy as np


def measure_latency_and_qps(model, eval_data, device="cuda", warmup=5):
    """
    严格按照你的范式：传入 DataLoader，自动取出第0个Batch，
    并基于这个固定Batch进行高精度的 Latency 和 QPS 测量。
    """
    model.eval()

    # --- 1. 数据准备 (Data Preparation) ---
    # 从 DataLoader 中取出第 0 个 batch
    # 注意：eval_data 是一个 iterable，用 next(iter()) 取出第一个
    first_batch_raw = next(iter(eval_data))

    # 严格保留你代码里的逻辑：batch = batch[0]
    # (通常用于处理 list/tuple 格式的 batch，如 [features, labels])
    if isinstance(first_batch_raw, (list, tuple)):
        input_batch = first_batch_raw[0]
    else:
        input_batch = first_batch_raw

    # 关键步骤：提前将数据移动到 GPU
    # 这样循环测量时，测的才是纯粹的模型计算时间，不包含 PCIe 传输时间
    input_batch = input_batch.to(device)

    # 获取 Batch Size，这对计算 QPS 至关重要
    # 假设 input_batch 是 Tensor，直接取 shape[0]
    current_batch_size = 4096

    # 为了获得统计学意义上的稳定结果，建议循环次数多一点 (比如 1000 次)
    # 虽然参数没传进来，我们在函数内部定义一个合理的重复次数
    n_repeats = 1000

    print(f"Start measuring... Fixed Batch Size: {current_batch_size}")

    # --- 2. 预热 (Warm-up) ---
    # 唤醒 GPU，分配显存，编译 Kernel
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.predict(input_batch)

        # 确保预热彻底完成
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    # --- 3. 正式测量 (Measurement using CUDA Events) ---
    latencies = []

    # 使用 PyTorch 官方推荐的 GPU 计时器
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(n_repeats):
            # 记录开始时刻
            starter.record()

            # 执行模型预测 (因为输入已经在 GPU 上，这里只测计算)
            _ = model.predict(input_batch)

            # 记录结束时刻
            ender.record()

            # 同步：等待 GPU 跑完这一轮
            if device.startswith("cuda"):
                torch.cuda.synchronize()

            # 计算这一轮的耗时 (ms)
            latencies.append(starter.elapsed_time(ender))

    # --- 4. 统计与输出 (Statistics) ---
    latencies = np.array(latencies)
    mean_latency = latencies.mean()  # ms per batch
    p90_latency = np.percentile(latencies, 90)
    p99_latency = np.percentile(latencies, 99)

    # 计算 QPS (Throughput)
    # 公式：Batch Size / (平均耗时秒数)
    qps = current_batch_size / (mean_latency / 1000)

    print("-" * 30)
    print(f"Metrics over {n_repeats} repeats (Pure Compute):")
    print(f"Mean Latency: {mean_latency:.4f} ms/batch")
    print(f"P90 Latency : {p90_latency:.4f} ms")
    print(f"P99 Latency : {p99_latency:.4f} ms")
    print(f"Throughput  : {qps:.2f} samples/s")
    print("-" * 30)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="SAGMoE",
        choices=["DeepFM", "DeepFM_MoE", "SAGMoE", "DCNV2"],
        help="Select model: DeepFM, DeepFM_MoE, or SAGMoE (default: SAGMoE)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="beauty.yaml",
        help="Configuration file path (default: beauty.yaml)",
    )
    parser.add_argument("--seed", type=int, default="42", help="Random seed (default: 42)")
    args = parser.parse_args()

    # Get model class
    ModelClass = get_model_class(args.model)

    # configurations initialization
    config = Config(model=ModelClass, config_file_list=[args.config])

    # init random seed
    init_seed(args.seed, config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)
    logger.info(f"Using model: {args.model}")

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    if dataset.dataset_name == "avazu":
        item_freq_tensor = None
    else:
        item_freq = build_item_freq_map(train_data._dataset)
        item_freq_tensor = torch.tensor(item_freq, dtype=torch.float32)

    # model loading and initialization
    if args.model == "DeepFM":
        model = ModelClass(config, train_data.dataset).to(config["device"])
    else:
        model = ModelClass(config, train_data.dataset, item_freq_tensor).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)

    # # 分桶评估
    results = evaluate_by_frequency_buckets(config, train_data, test_data, dataset, model, trainer, logger)

    # model.eval_stage = "test"
    # overall_result = trainer.evaluate(test_data)
    # logger.info(f"[Overall] {overall_result}")

    # measure_latency_and_qps(model, test_data)

    # # 打印分桶评估结果
    # print("\n=== Grouped Test Results ===")
    # for k, v in results.items():
    #     print(k, v)
    # print("Overall", overall_result)
