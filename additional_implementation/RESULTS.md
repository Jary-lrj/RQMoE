# Controlled Embedding → Routing Intervention：实验记录

## 1. 实验状态

本报告记录 `seed=42` 的 controlled intervention。每个数据集只训练一份
8-expert、Top-k=4 checkpoint，随后冻结全部参数，在同一 checkpoint、同一测试
划分上切换推理时的 `k=1` 与 `k=4`。

| Dataset | Experts | Training Top-k | Seed | Device | Status |
|---|---:|---:|---:|---|---|
| ML-1M | 8 | 4 | 42 | NVIDIA A100-SXM4-80GB, GPU 2 | completed |
| Beauty | 8 | 4 | 42 | NVIDIA A100-SXM4-80GB, GPU 2 | completed |
| Avazu | 8 | 4 | 42 | NVIDIA A100-SXM4-80GB, GPU 2 | completed |

软件环境：Python 3.10.20、PyTorch 2.4.0+cu121、RecBole 1.2.1、
NumPy 1.26.4、pandas 2.3.3、scikit-learn 1.7.2。

## 2. 数据来源与 RecBole 处理

数据采用 RecBole 官方数据生态提供的 atomic files：

- [RecBole dataset download](https://recbole.io/docs/user_guide/data/dataset_download.html)
- [Running a new dataset](https://recbole.io/docs/user_guide/usage/running_new_dataset.html)
- [Atomic files](https://recbole.io/docs/user_guide/data/atomic_files.html)
- [RecBole URL map](https://github.com/RUCAIBox/RecBole/blob/master/recbole/properties/dataset/url.yaml)
- [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets)
- [Official Google Drive mirror](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj?usp=sharing)

RecBole 的 S3 端点在本次下载时返回 HTTP 403，因此使用官方 Google Drive
镜像；同类现象见 [RecBole issue #2218](https://github.com/RUCAIBox/RecBole/issues/2218)。
三个 ZIP 均通过归档完整性检查。Beauty 目录由 `Amazon_Beauty` 对齐为原项目
YAML 使用的 `beauty`，atomic file 内容未改写。

| Dataset | Atomic files | Interactions (excluding header) |
|---|---|---:|
| ML-1M | `.inter`, `.item`, `.user` | 1,000,209 |
| Beauty | `.inter`, `.item` | 2,023,070 |
| Avazu | `.inter` | 40,428,967 |

字段加载、标签和划分由原项目 YAML 驱动：ML-1M 与 Beauty 使用
`rating >= 3` 生成标签；三者均使用 seed-42 random 8:1:1 split，Beauty 额外按
user 分组。训练和测试通过 RecBole 的 `create_dataset`、`data_preparation`、
`Trainer.fit` 和 whole-split `Trainer.evaluate` 完成。

Avazu 的原论文 YAML 与 RecBole context-aware 性能基准配置存在差异。本次实验
保留原论文配置以保证 checkpoint 和预测路径一致：`timestamp`、`banner_pos`
未列入 `numerical_features`，atomic 文件中的高基数 `item_id` 作为普通 token
字段保留。[官方性能基准](https://github.com/RUCAIBox/RecBole/blob/master/asset/time_test_result/Context-aware_recommendation.md)
采用不同数据规模和数值字段设置；因此 Avazu 结果应标注为
original-paper-config reproduction。

Avazu 训练通过 RecBole 原生 `save_dataset` 缓存保存完成 token remapping 的
Dataset 对象。缓存只影响重复加载时间；字段、过滤、标签、划分和模型配置保持一致。

## 3. 干预协议

1. 从最多 200,000 个训练样本拟合 router input 的 uncentered SVD basis；测试集
   不参与 basis 拟合。保留秩为 `r=5`。
2. Tail collapse 对第 `r+1...d` 个谱方向乘以
   `gamma ∈ {1,.75,.5,.25,0}`。
3. Reverse intervention 使用 `strength ∈ {0,.25,.5,.75,1}` 渐进 flatten 谱。
4. 每次变换后逐样本恢复到原 representation 的 L2 norm。
5. pre-hook 只替换 `moe_layers.gate` 的输入；experts 始终接收原 representation，
   expert、router、aggregation 和预测头参数全部冻结。
6. 每个 condition 使用相同 checkpoint 和测试样本计算
   `Delta_scale = AUC(k=4) - AUC(k=1)`。

## 4. 指标口径

- `effective_rank`：以奇异值归一化后计算的谱熵有效秩。
- `full_softmax_logit_entropy`：Top-k 之前、全部 8 个 router logits 的逐样本
  softmax entropy；反映 logits 的样本级尖锐度。
- `active_topk_weight_entropy`：实际入选 Top-k 权重在每个样本内的 entropy；
  `k=1` 定义为 0。
- `top1_load_entropy` / `top1_load_cv_squared`：在整个测试集聚合 argmax expert
  计数后的利用率熵与不均衡度，是本报告判断 hard load concentration 的主指标。
- `gate_weight_load_entropy`：在整个测试集聚合稀疏 gate weights 后的利用率熵。

样本级 entropy 与跨样本 load entropy 描述不同现象。本文将两者分列报告。

## 5. Checkpoint 结果

| Dataset | Best epoch index | Valid AUC | Valid LogLoss | Test AUC (k=4) | Test LogLoss |
|---|---:|---:|---:|---:|---:|
| ML-1M | 15 | 0.8529 | 0.3206 | 0.8480 | 0.3235 |
| Beauty | 0 | 0.6872 | 0.3474 | 0.6915 | 0.3608 |
| Avazu | 0 | 0.7858 | 0.3765 | 0.7861 | 0.3759 |

## 6. Tail-collapse 结果

### ML-1M

| gamma | Eff. rank | Full-soft H | Active-k4 H | Top-1 load H | Load CV² | AUC k=1 | AUC k=4 | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 50.4489 | 0.937070 | 0.960374 | 0.986514 | 0.054998 | 0.823610 | 0.848022 | 0.024412 |
| 0.75 | 45.6720 | 0.937162 | 0.960796 | 0.976631 | 0.089024 | 0.825427 | 0.847752 | 0.022324 |
| 0.50 | 38.0298 | 0.937139 | 0.961262 | 0.957233 | 0.147491 | 0.825179 | 0.846809 | 0.021630 |
| 0.25 | 25.0707 | 0.937128 | 0.961798 | 0.929669 | 0.216153 | 0.822279 | 0.845056 | 0.022778 |
| 0.00 | 4.8395 | 0.937499 | 0.962541 | 0.911612 | 0.251701 | 0.815910 | 0.841675 | 0.025765 |

### Beauty

| gamma | Eff. rank | Full-soft H | Active-k4 H | Top-1 load H | Load CV² | AUC k=1 | AUC k=4 | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 8.3137 | 0.998969 | 0.999475 | 0.613898 | 1.444611 | 0.689181 | 0.691499 | 0.002318 |
| 0.75 | 7.2882 | 0.998963 | 0.999471 | 0.610692 | 1.449729 | 0.689406 | 0.691517 | 0.002111 |
| 0.50 | 6.0924 | 0.998957 | 0.999467 | 0.607166 | 1.454888 | 0.689546 | 0.691500 | 0.001954 |
| 0.25 | 4.6967 | 0.998953 | 0.999464 | 0.604752 | 1.458040 | 0.689580 | 0.691507 | 0.001927 |
| 0.00 | 2.9568 | 0.998952 | 0.999462 | 0.603928 | 1.459536 | 0.689507 | 0.691524 | 0.002017 |

### Avazu

| gamma | Eff. rank | Full-soft H | Active-k4 H | Top-1 load H | Load CV² | AUC k=1 | AUC k=4 | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 97.3857 | 0.887630 | 0.928815 | 0.937444 | 0.220327 | 0.774546 | 0.786065 | 0.011520 |
| 0.75 | 90.7541 | 0.877809 | 0.925330 | 0.932438 | 0.242225 | 0.775085 | 0.785935 | 0.010850 |
| 0.50 | 79.1245 | 0.866219 | 0.921988 | 0.921148 | 0.289522 | 0.775797 | 0.785412 | 0.009615 |
| 0.25 | 55.2762 | 0.855547 | 0.920511 | 0.904421 | 0.363897 | 0.775748 | 0.784181 | 0.008433 |
| 0.00 | 4.8662 | 0.854354 | 0.924318 | 0.883410 | 0.474921 | 0.773546 | 0.781964 | 0.008418 |

## 7. Reverse spectral-flattening 结果

下表保留 baseline、`strength=.5` 与最强 flatten；完整 sweep 见 raw CSV。

| Dataset | Strength | Eff. rank | Full-soft H | Active-k4 H | Top-1 load H | Load CV² | AUC k=1 | AUC k=4 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ML-1M | 0.00 | 50.4489 | 0.937070 | 0.960374 | 0.986514 | 0.054998 | 0.823610 | 0.848022 | 0.024412 |
| ML-1M | 0.50 | 69.8273 | 0.945562 | 0.965746 | 0.973213 | 0.121165 | 0.819334 | 0.847232 | 0.027897 |
| ML-1M | 1.00 | 77.8722 | 0.967755 | 0.979936 | 0.893135 | 0.510043 | 0.815857 | 0.844726 | 0.028869 |
| Beauty | 0.00 | 8.3137 | 0.998969 | 0.999475 | 0.613898 | 1.444611 | 0.689181 | 0.691499 | 0.002318 |
| Beauty | 0.50 | 15.2067 | 0.999143 | 0.999549 | 0.678137 | 1.390888 | 0.685951 | 0.690646 | 0.004695 |
| Beauty | 1.00 | 18.7534 | 0.999316 | 0.999592 | 0.535765 | 2.919015 | 0.683184 | 0.685862 | 0.002678 |
| Avazu | 0.00 | 97.3857 | 0.887630 | 0.928815 | 0.937444 | 0.220327 | 0.774546 | 0.786065 | 0.011520 |
| Avazu | 0.50 | 136.1925 | 0.925632 | 0.949172 | 0.911114 | 0.355075 | 0.772106 | 0.785358 | 0.013252 |
| Avazu | 1.00 | 153.8936 | 0.960343 | 0.972994 | 0.792251 | 0.948087 | 0.770557 | 0.782689 | 0.012132 |

## 8. 当前结论

ML-1M 的中等干预 `gamma: 1→.5` 使 effective rank 下降 24.62%、Top-1
load entropy 下降 2.97%、`Delta_scale` 下降 11.40%。Beauty 的
`gamma: 1→.25` 对应下降 43.51%、1.49% 和 16.85%。这两段结果为
“rank reduction → aggregate hard-load concentration → smaller Top-k delta”提供局部、
单 seed 的方向性证据。

Avazu 给出最清楚的 forward intervention：从 `gamma=1→0`，effective rank
单调下降 95.00%，Top-1 load entropy 单调下降 5.76%，load CV² 上升 115.55%，
`Delta_scale` 单调下降 26.92%。其中 `gamma=1→.5` 已使四项分别变化
-18.75%、-1.74%、+31.41% 和 -16.53%。Avazu 的 full-softmax entropy 同时
下降 3.75%，说明该数据集的 aggregate load concentration 还伴随 pre-Top-k
样本级分布变尖。

ML-1M 与 Beauty 的完整剂量曲线没有形成单调规律。ML-1M 在 `gamma=0` 时 `Delta_scale` 回升至
0.025765，高于 identity 5.54%；Beauty 也在端点小幅回升。collapse 过程中
full-softmax entropy 与 active-Top-k entropy 基本持平或轻微上升，因此当前证据
针对跨样本 expert utilization，尚未支持样本级 routing distribution 变尖。

Reverse intervention 同样呈 mixed evidence。Beauty 在 `strength=.5` 时 effective
rank 与 Top-1 load entropy 上升、load CV² 下降；ML-1M 和 Avazu 的 hard-load
balance 恶化。Avazu `strength=.5` 将 effective rank 提高 39.85%，full-softmax
和 active-k4 entropy 分别提高 4.28% 与 2.19%，`Delta_scale` 提高 15.04%；同时
Top-1 load entropy 下降 2.81%、load CV² 上升 61.16%。即样本内权重变平与跨样本
expert utilization 集中可以同时出现。

三个数据集在 flatten 后的 `Delta_scale` 增长都伴随 `AUC(k=1)` 下降，
`AUC(k=4)` 也未恢复到 identity 以上。该结果不能作为绝对性能恢复或一致的
hard-load recovery 证据。综合三套单-seed 结果，forward tail-collapse 对完整链条
提供了中等强度区间的一致方向性支持，Avazu 还给出全 sweep 单调支持；reverse
intervention 尚未形成跨数据集一致证据。

## 9. 有效性检查与限制

- `gamma=1` 与 flatten `strength=0` 的结果逐列一致。
- k=1/k=4 的 checkpoint、split、sample count、representation spectrum 和
  pre-Top-k logits 指标通过硬一致性检查。
- ML-1M、Beauty 与 Avazu 的逐样本 norm restoration 最大相对误差分别为
  `2.50e-7`、`2.51e-7`、`2.51e-7`；zero-norm count 均为 0。
- 当前只有一个 seed。论文主张前应补足五 seeds，并报告 paired effect 与置信区间。
- Beauty 与 Avazu 的最佳 epoch index 均为 0，后续 epochs 的 validation loss
  快速恶化；两者的单-seed 证据需要在重复实验中验证稳定性。
- shared checkpoint 在 Top-k=4 下训练，k=1 是 inference-time routing sensitivity；
  它没有复现分别训练 k=1 与 k=4 的 training-time scaling law。
- 干预对象是每个样本拼接后的 router representation。它没有单独定位 item table
  或 tail-item subgroup，论文中应准确限定 causal object。
- 旧版 `Switch.py` 计算 FM term 后将最终 logit 设为 `y_deep`；当前预测路径实际为
  embedding → MoE → head。本实验在训练和所有干预条件中完整保留该行为。

## 10. 结果文件

- `results/ml-1m_e8_seed42_shared_r5/`
- `results/beauty_e8_seed42_shared_r5/`
- `results/avazu_e8_seed42_shared_r5/`

每个完成目录包含 `condition_results.csv`、`condition_results.jsonl`、
`delta_scale.csv` 和 `run_config.json`；训练 checkpoint、resolved config 与日志分别
保存在 `checkpoints/` 和 `logs/`。
