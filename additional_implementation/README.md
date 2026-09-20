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

The implementation is evaluation-only. It loads a frozen standard
`DeepFM_MoE` checkpoint and installs a temporary pre-hook on
`model.moe_layers.gate`. Consequently:

- the router receives the intervened representation;
- experts receive the original representation;
- the FM branch and prediction head are unchanged;
- every model parameter remains frozen.

The experiment targets the standard dense-input MoE router in
`exps/running/models/Switch.py`. RQ-MoE's VoteRouter consumes quantization
trajectories, so applying the same intervention there would test a different
claim.

The loader intentionally accepts only `DeepFM_MoE` checkpoints produced by
`exps/running/run_SAG.py --model DeepFM_MoE`. Checkpoints produced by the
separate root-level `scaling_failure.py` use another model class and parameter
layout. The runner fails early when those state keys are supplied. It also
compares the checkpoint's saved RecBole config with the supplied YAML and seed
for the dataset, schema, split/evaluation settings, and model dimensions.

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

`--seed` is required and must equal the CLI `--seed` passed to the historical
`run_SAG.py` training command. That script calls `init_seed(args.seed, ...)`
without writing the value back into the RecBole Config, so the seed stored in
an old checkpoint may only reflect the YAML file. Results record both the
runtime split seed and the checkpoint's saved-config seed.

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
- mean per-sample full-router probability entropy;
- maximum relative L2-norm error and zero-norm sample count.

All spectra and expert loads are aggregated over the complete evaluation set
before the metrics are computed. No per-batch effective ranks or entropies are
averaged.

`delta_scale.csv` pairs the k=1 and k=4 rows and reports
`AUC_k4 - AUC_k1` together with their effective-rank and routing diagnostics.
`run_config.json` records the intervention settings used for the run.

## Interpretation of the reverse intervention

Successful post-hoc spectral flattening recovery would be strong evidence.
Failure is inconclusive because a frozen router was optimized for the original
coordinate spectrum and may not adapt immediately to flattened inputs. A
training-time flattening study or a frozen-encoder/frozen-expert experiment
that retrains only the router can be added later if the post-hoc result is
negative.
