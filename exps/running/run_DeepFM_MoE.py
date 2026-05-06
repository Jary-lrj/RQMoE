from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from models.SAGMoE import SAGMoE
from recbole.data.dataloader.general_dataloader import NegSampleEvalDataLoader
from recbole.sampler import Sampler
from recbole.model.context_aware_recommender import DeepFM_MoE, DeepFM
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
    return emb_subset.detach().cpu().numpy()


def get_svd_values(embeddings):
    X = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    if len(s) < embeddings.shape[1]:
        s = np.pad(s, (0, embeddings.shape[1] - len(s)), constant_values=0)

    return s.reshape(1, -1)


def get_mean_l2_norm(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)
    return np.mean(norms)


def get_eval_metrics_by_group(model, trainer, results, item_group, eval_data, embeddings):
    model.item_group = item_group
    results[item_group] = trainer.evaluate(eval_data)
    logger.info(f"[{item_group}] {results[item_group]}")
    logger.info(f"Mean L2 Norm: {get_mean_l2_norm(embeddings)}")
    logger.info(f"Singular Values: {get_svd_values(embeddings)}")


if __name__ == "__main__":

    # configurations initialization
    config = Config(model=DeepFM, config_file_list=["beauty.yaml"])

    # init random seed
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    item_freq = build_item_freq_map(train_data._dataset)
    item_freq_tensor = torch.tensor(item_freq, dtype=torch.float32)
    user_freq = build_user_freq_map(train_data._dataset)
    user_freq_tensor = torch.tensor(user_freq, dtype=torch.float32)

    inter_hot, inter_mid, inter_cold, inter_ultra, idx_hot, idx_mid, idx_cold, idx_ultra = (
        split_test_interactions_by_freq(test_data._dataset, item_freq)
    )
    logger.info(
        f"[Test Split] Hot={len(inter_hot)}, Mid={len(inter_mid)}, Cold={len(inter_cold)}, Ultra-Cold={len(inter_ultra)}"
    )

    # model loading and initialization
    model = DeepFM(config, train_data.dataset).to(config["device"])
    # model = DeepFM_MoE(config, train_data.dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)

    # test_result = trainer.evaluate(test_data)
    # logger.info(test_result)

    # # model evaluation
    model.eval_stage = "test"
    results = {}

    sampler = Sampler(phases="test", datasets=dataset, distribution="uniform")
    sampler = sampler.set_phase("test")
    eval_hot = make_eval_loader_from_interaction(config, test_data._dataset, inter_hot, sampler, name="test-hot")
    eval_mid = make_eval_loader_from_interaction(config, test_data._dataset, inter_mid, sampler, name="test-mid")
    eval_cold = make_eval_loader_from_interaction(config, test_data._dataset, inter_cold, sampler, name="test-cold")
    eval_ultra = make_eval_loader_from_interaction(config, test_data._dataset, inter_ultra, sampler, name="test-ultra")

    if len(inter_hot) > 0:
        hot_embedding = get_item_embeddings(model, dataset, idx_hot)
        get_eval_metrics_by_group(model, trainer, results, "Hot", eval_hot, hot_embedding)
    if len(inter_mid) > 0:
        mid_embedding = get_item_embeddings(model, dataset, idx_mid)
        get_eval_metrics_by_group(model, trainer, results, "Mid", eval_mid, mid_embedding)
    if len(inter_cold) > 0:
        cold_embedding = get_item_embeddings(model, dataset, idx_cold)
        get_eval_metrics_by_group(model, trainer, results, "Cold", eval_cold, cold_embedding)
    if len(inter_ultra) > 0:
        ultra_embedding = get_item_embeddings(model, dataset, idx_ultra)
        get_eval_metrics_by_group(model, trainer, results, "Ultra-Cold", eval_ultra, ultra_embedding)

    model.eval_stage = None
    t1 = time.time()
    overall_result = trainer.evaluate(test_data)
    t2 = time.time()
    logger.info(f"[Overall] {overall_result}, {(t2-t1)*1000/len(test_data):.2f} ms/batch.")

    # print("\n=== Grouped Test Results ===")
    # for k, v in results.items():
    #     print(k, v)
    # print("Overall", overall_result)
