# Controlled embedding-to-routing intervention

This directory implements the controlled experiment requested for the causal
chain

\[
\operatorname{effective\ rank}(H)\downarrow
\Longrightarrow H(\text{routing})\downarrow
\Longrightarrow \Delta_{\text{scale}}\downarrow,
\qquad
\Delta_{\text{scale}}=\operatorname{AUC}_{k=4}-\operatorname{AUC}_{k=1}.
\]

The intervention runner is evaluation-only. It loads a frozen standard
`DeepFM_MoE` checkpoint and installs a temporary pre-hook on
`model.moe_layers.gate`. Consequently:

- the router receives the intervened representation;
- experts receive the original representation;
- the legacy forward behavior and prediction head are unchanged;
- every model parameter remains frozen.

The experiment targets the standard dense-input MoE router in
`exps/running/models/Switch.py`. RQ-MoE's VoteRouter consumes quantization
trajectories, so applying the same intervention there would test a different
claim.

The loader intentionally accepts only checkpoints with the
`models.Switch.DeepFM_MoE` parameter layout, including checkpoints produced by
`train_shared_checkpoint` and historical `run_SAG.py --model DeepFM_MoE`
runs. Checkpoints produced by the separate root-level `scaling_failure.py` use
another model class and parameter layout. The runner fails early when those
state keys are supplied. It also compares the checkpoint's saved RecBole
config with the supplied YAML and seed for the dataset, schema,
split/evaluation settings, preprocessing, and model dimensions. Checkpoints
without recorded Top-k metadata are rejected so the shared protocol cannot
silently evaluate a checkpoint with an unknown training route.

## Interventions

### Norm-preserving rank collapse

A fixed feature-space basis is fitted once on a calibration split through the
global second moment of the router inputs. For
`retained_rank = r` and `gamma` in `[0, 1]`, the router sees

\[
H_\gamma = H V\,\operatorname{diag}
(\underbrace{1,\ldots,1}_{r},
 \underbrace{\gamma,\ldots,\gamma}_{d-r})V^\top.
\]

Every transformed row is then rescaled to the original row's L2 norm. Thus the
intervention changes spectral rank while controlling representation
magnitude. `gamma=1` returns the original tensor directly, giving an exact
identity baseline.

The default uses the uncentered `H=UΣVᵀ` specified above. Pass
`--center-spectrum` only when a centered-spectrum variant is desired.

### Norm-preserving spectral flattening

The reverse intervention amplifies weak fitted spectral directions by

\[
g_i = \min\left[g_{\max},
\left(\frac{\sigma_1}
{\max(\sigma_i,\tau\sigma_1)}\right)^\eta\right],
\]

where `eta` is `--flatten-strengths`, `tau` is
`--spectral-floor-ratio`, and `g_max` is `--max-flatten-gain`. Per-row norms
are restored after the transform. The spectral floor and gain cap prevent
near-zero singular directions from being amplified without bound.

## Recommended protocol

The strictest controlled comparison uses one checkpoint trained with Top-k=4:

```bash
cd /home/liruijie/RQMoE
python -m additional_implementation.run_controlled_intervention \
  --config exps/running/beauty.yaml \
  --shared-checkpoint saved/DeepFM_MoE-CHECKPOINT.pth \
  --seed 42 \
  --retained-rank 5 \
  --gammas 1 0.75 0.5 0.25 0 \
  --flatten-strengths 0.25 0.5 0.75 1 \
  --output-dir additional_implementation/results/beauty_seed42_shared
```

This evaluates the same frozen encoder, experts, router weights, and
prediction head with inference Top-k set to 1 and 4. The resulting quantity is
an inference-time Top-k delta with training trajectory fully controlled.
The runner reads the training Top-k from the checkpoint, records it, and
rejects a shared checkpoint whose recorded training Top-k is not 4.

### Training the shared checkpoint from RecBole atomic files

`train_shared_checkpoint` is the minimal training path for this protocol. It
uses RecBole's standard `create_dataset`, `data_preparation`, `Trainer.fit`, and
whole-test `Trainer.evaluate` flow. It never calls `run_SAG.py`'s
frequency-bucket evaluator.

Place an already-downloaded atomic dataset in RecBole's standard layout:

```text
/path/to/atomic-data/
└── ml-1m/
    ├── ml-1m.inter
    ├── ml-1m.item
    └── ml-1m.user
```

Then train one seed with:

```bash
cd /home/liruijie/RQMoE
python -m additional_implementation.train_shared_checkpoint \
  --dataset ml-1m \
  --data-path /path/to/atomic-data \
  --num-experts 8 \
  --seed 42
```

For large atomic files that will be evaluated repeatedly, add
`--cache-dataset`. This uses RecBole's native filtered-dataset cache inside the
checkpoint directory; it changes loading time only and is recorded in the
resolved training config.

The command requires `<data-path>/<dataset>/<dataset>.inter` to exist before
RecBole is invoked, so a misspelled path cannot trigger RecBole's automatic
dataset downloader. The default YAML is for atomic files containing a numeric
`rating` column and creates labels at `rating >= 3`. For a dataset with an
existing binary label, pass a dataset-specific YAML with `--config`, set
`LABEL_FIELD`, and set `threshold: null`.

Training always writes `top_k: 4`, `num_experts` (8 by default, matching the
manuscript), and the actual CLI seed into both the checkpoint's RecBole config
and a `training_config.yaml` beside the checkpoint. This additional runner
overrides the restored legacy class's hard-coded four-expert constructor
without modifying the original model file.
Use the printed paths for the controlled evaluation:

```bash
python -m additional_implementation.run_controlled_intervention \
  --config additional_implementation/checkpoints/ml-1m/experts_8/seed_42/training_config.yaml \
  --shared-checkpoint additional_implementation/checkpoints/ml-1m/experts_8/seed_42/DeepFM_MoE-CHECKPOINT.pth \
  --seed 42 \
  --retained-rank 5 \
  --output-dir additional_implementation/results/ml-1m_seed42_shared
```

The generated config records the resolved atomic-data directory. This also
ensures that a local `ml-100k` directory is used instead of RecBole's bundled
example copy.

To reproduce the paper's separately trained scaling comparison, provide the
paired checkpoints:

```bash
python -m additional_implementation.run_controlled_intervention \
  --config exps/running/beauty.yaml \
  --checkpoint-k1 saved/DeepFM_MoE-K1-CHECKPOINT.pth \
  --checkpoint-k4 saved/DeepFM_MoE-K4-CHECKPOINT.pth \
  --seed 42 \
  --retained-rank 5 \
  --output-dir additional_implementation/results/beauty_seed42_paired
```

Run each seed separately so that `delta_scale` remains paired by seed. The
paired-checkpoint protocol retains the original training-time scaling
definition, while encoder and expert weights differ between k=1 and k=4.

`--seed` is required and must equal both the resolved YAML seed and the
checkpoint seed. The new training entry writes that seed into its resolved
RecBole config. Historical `run_SAG.py` checkpoints whose CLI seed differed
from their saved YAML require a new, unambiguous training run before this
strict paired evaluation. Results record both the runtime split seed and the
checkpoint's saved-config seed.

The fitted basis uses `train` by default and never uses test examples. Use
`--calibration-split valid` for a held-out calibration basis. At most 200,000
calibration examples are used by default; change this with
`--max-calibration-samples`.

## Recorded measurements

`condition_results.jsonl` and `condition_results.csv` contain one row for each
intervention and Top-k value:

- exact, unrounded AUC and log loss;
- effective rank
  `exp(-sum_i p_i log p_i)`, where `p_i = sigma_i / sum_j sigma_j`;
- full singular-value spectrum of the actual post-intervention router inputs;
- normalized entropy of hard top-1 expert loads, the primary load metric and
  directly comparable across k=1 and k=4;
- aggregate sparse gate-weight load entropy, CV-squared, and maximum load
  ratio;
- selected-slot counts and entropy for completeness; with four experts and
  k=4 this count-based statistic is mechanically uniform and must not be used
  as evidence of routing recovery;
- mean per-sample entropy of the full softmax over all router logits, measured
  before Top-k and labeled `full_softmax_logit_entropy`;
- mean per-sample entropy of the normalized weights of the active Top-k
  experts, labeled `active_topk_weight_entropy` (defined as zero for k=1);
- maximum relative L2-norm error and zero-norm sample count.

All spectra and expert loads are aggregated over the complete evaluation set
before the metrics are computed. No per-batch effective ranks or entropies are
averaged.

`delta_scale.csv` pairs the k=1 and k=4 rows and reports
`AUC_k4 - AUC_k1` together with their effective-rank and routing diagnostics.
`run_config.json` records the intervention settings used for the run.

The legacy `Switch.py` forward pass computes an FM term but assigns the final
logit from `y_deep` alone. This experiment preserves that checkpoint behavior
exactly; the evaluated prediction path is therefore the embedding-to-MoE path
used by the original code.

## Interpretation of the reverse intervention

Successful post-hoc spectral flattening recovery would be strong evidence.
Failure is inconclusive because a frozen router was optimized for the original
coordinate spectrum and may not adapt immediately to flattened inputs. A
training-time flattening study or a frozen-encoder/frozen-expert experiment
that retrains only the router can be added later if the post-hoc result is
negative.
