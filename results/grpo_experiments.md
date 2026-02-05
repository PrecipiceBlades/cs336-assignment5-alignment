# GRPO Training Experiments on MATH

## Overview

This document describes experiments with Generalized Reward Proximal Policy Optimization (GRPO) for training a math reasoning model on the MATH dataset. We compare different loss types and normalization approaches.

## Model and Dataset

- **Base Model**: Qwen2.5-Math-1.5B
- **Training Data**: MATH train split (7,500 examples)
- **Validation Data**: MATH validation split (5,000 examples)
- **Prompt Format**: r1_zero prompt (chain-of-thought with <think>...</think> <answer>...</answer> format)

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| n_grpo_steps | 100 |
| learning_rate | 1e-5 |
| rollout_batch_size | 64 |
| group_size | 8 |
| train_batch_size | 64 |
| gradient_accumulation_steps | 32 |
| micro_batch_size | 2 |
| epochs_per_rollout_batch | 1 (on-policy) |
| max_grad_norm | 1.0 |
| sampling_temperature | 1.0 |
| sampling_max_tokens | 2048 |
| eval_interval | 10 steps |

---

## Experiment 1: Effect of Baselining

### reinforce_with_baseline vs no_baseline

| Step | reinforce_with_baseline | no_baseline |
|------|------------------------|-------------|
| 10   | 23.86% | 28.72% |
| 20   | 29.04% | 32.88% |
| 30   | 32.82% | 33.30% |
| 40   | 36.22% | 34.84% |
| 50   | 36.58% | 35.70% |
| 60   | 37.62% | 36.72% |
| 70   | 37.60% | 37.16% |
| 80   | 37.96% | 35.76% |
| 90   | 39.60% | 36.98% |
| 100  | 40.56% | 38.04% |
| **Final** | **40.60%** | **38.30%** |

### Key Findings

1. **Early Training**: no_baseline starts faster (28.72% vs 23.86% at step 10)
2. **Final Performance**: reinforce_with_baseline achieves higher final accuracy (40.60% vs 38.30%)
3. **Training Stability**: no_baseline shows more variance (step 80 dropped to 35.76%)
4. **Conclusion**: reinforce_with_baseline is the better choice for stable, high-performance training

---

## Experiment 2: Effect of Length Normalization and Std Normalization

Comparing masked_mean (per-sequence normalization) vs masked_normalize (constant normalizer)
with and without standard deviation normalization.

### Full Results Table

| Step | masked_mean + std_true | masked_normalize + std_true | masked_mean + std_false | masked_normalize + std_false |
|------|------------------------|-----------------------------|--------------------------|-----------------------------|
| 10   | 27.64% | 26.78% | 26.54% | 24.50% |
| 20   | 33.40% | 31.22% | 29.60% | 29.72% |
| 30   | 37.26% | 33.92% | 33.46% | 34.86% |
| 40   | 39.76% | 36.66% | 35.74% | 38.56% |
| 50   | 41.26% | 37.98% | 37.62% | 39.92% |
| 60   | 42.78% | 38.06% | 37.32% | 40.36% |
| 70   | 42.96% | 38.32% | 37.40% | 40.08% |
| 80   | 43.38% | 38.28% | 37.76% | 41.26% |
| 90   | 43.10% | 40.08% | 38.06% | 41.82% |
| 100  | 43.36% | - | 38.40% | 42.28% |
| **Final** | **43.70%** | **40.10%** | **38.86%** | **41.88%** |

---

## Summary and Conclusions

### Final Rankings (by Final Validation Accuracy):

1. **masked_mean + std_true**: 43.70% (BEST)
2. **masked_normalize + std_false**: 41.88%
3. **masked_normalize + std_true**: 40.10%
4. **masked_mean + std_false**: 38.86%

### Key Findings:

1. **Best Configuration**: masked_mean with use_std_normalization=True achieves the highest accuracy (43.70%)

2. **Effect of Length Normalization**:
   - With std_true: masked_mean (43.70%) outperforms masked_normalize (40.10%) by 3.6 percentage points
   - With std_false: masked_normalize (41.88%) outperforms masked_mean (38.86%) by 3.0 percentage points
   - Length normalization choice interacts with std normalization

3. **Effect of Std Normalization**:
   - With masked_mean: std_true (43.70%) >> std_false (38.86%), improvement of 4.84 pp
   - With masked_normalize: std_false (41.88%) > std_true (40.10%), improvement of 1.78 pp
   - Std normalization helps masked_mean but slightly hurts masked_normalize

4. **Training Stability**:
   - masked_mean + std_true shows the most stable improvement throughout training
   - masked_mean + std_false shows some plateauing around step 60-70
   - masked_normalize + std_false shows consistent improvement throughout

5. **Conclusion**: The combination of per-sequence length normalization (masked_mean) with standard deviation normalization (use_std_normalization=True) provides the best results for GRPO training on math reasoning tasks.
