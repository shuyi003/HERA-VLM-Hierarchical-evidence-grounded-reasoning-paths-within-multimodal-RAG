# Reward specification

The batch entry point `compute_score` implements the evidence grounding and hierarchical reasoning path expressions in manuscript Sections 3.3.3–3.3.4 (Eqs. 8–16).

## Total and answer reward

`R = R_accuracy + R_ground + R_path + R_format`.

All four outer weights are `1.0`. Answer accuracy combines `0.10 EM + 0.10 token_F1 + 0.80 semantic`, using sentence embeddings and text normalization. Output-format scoring checks the required tags, reasoning steps, and evidence syntax.

The grounding formula permits a maximum of **1.1**, because both bonuses may be 1. Thus the total has range **[0, 4.1]** with the default weights. There is no clipping or averaging of the total. `require_valid_weight_sum=false` permits unit outer weights.

## Evidence grounding: Eqs. 8–10

For a predicted box and its matched reference box:

`u = area(intersection) / area(union)`.

```text
S_h-IoU(u) = 0    if u <= 0.1
             u    if 0.1 < u < 0.5
             1    if u >= 0.5

R_ground = 0.9 S_h-IoU + 0.1 (S_cov + S_merge)
```

An IoU of `0.5` receives a hybrid IoU score of `1`. The intermediate reward is the original IoU, not a linear rescaling between thresholds. There is no additional coordinate-validity reward or extra-box penalty.

### Operational definitions for bonuses and multiple boxes

The coverage and merged-evidence bonuses use the following configurable implementation defaults.

| Setting | Default | Meaning |
| --- | --- | --- |
| `grounding_coverage_threshold` | 0.95 | Minimum fraction of a gold box covered by a prediction |
| `grounding_coverage_max_area_ratio` | 1.25 | Maximum predicted/gold area ratio for the coverage bonus |
| `grounding_merged_iou_threshold` | 0.50 | Minimum overlap with a merged gold region |
| `grounding_merge_max_gap` | 0.02 | Maximum separation in each normalized image axis for adjacent gold regions |

For multiple evidence boxes:

1. Match predictions and gold regions greedily by descending IoU, with each region used at most once. Apply the hybrid function **to each pair**, then average over all gold regions. Unmatched gold regions contribute zero.
2. `S_cov` is the fraction of gold regions for which a same-page prediction satisfies both the coverage threshold and area-ratio limit. A prediction may cover more than one gold region.
3. Group adjacent gold regions on the same page using connected components. Two regions are adjacent when the maximum of their horizontal and vertical gaps is at most `grounding_merge_max_gap`; gaps use coordinates normalized by the corresponding page dimensions. Each group must contain at least two regions.
4. Enclose each group in a bounding box. `S_merge` is the fraction of these merged regions with a one-to-one prediction match meeting the merged IoU threshold. A singleton or absence of adjacent groups receives no merged bonus.

Groups never combine different pages. Both bonuses may be awarded simultaneously. A sample with one exact predicted/gold box receives `0.9 + 0.1 = 1.0`; qualifying simultaneous bonuses can produce `1.1`.

## Hierarchical reasoning path: Eqs. 11–16

```text
G_accuracy = 1[R_accuracy >= 0.6]
v_i        = 1[ColQwen2_similarity(statement_i, crop_i) >= 0.5]
V_step     = sum(v_i) / N

j_i        = argmax_j semantic_similarity(statement_i, reference_statement_j)
level_i    = gold_steps[j_i].hierarchy_level
V_path     = sum(1[level_(i+1) >= level_i]) / (N - 1), for N >= 2

R_path     = G_accuracy * (0.4 V_step + 0.6 V_path)
```

ColQwen2 crop/text similarity retains the implementation's cosine between mean-pooled, L2-normalized multi-vector embeddings. It is not the application's ColPali retrieval score. A similarity equal to `0.5` passes.

Step validity is the **fraction of predicted steps**, not an all-or-nothing minimum over the whole path. Missing citations, missing page images, and invalid crops count as failed steps in the denominator. When a step cites multiple regions, every citation must pass; the step is counted once.

Reference steps must carry an explicit integer `hierarchy_level`:

- `1`: section;
- `2`: subsection;
- `3`: detailed evidence.

Matching uses the same semantic similarity function as answer scoring, with no exact-match, token-F1, or bounding-box term added to the matching score. Reference-list position is not a substitute for a hierarchy label. Ties select the first highest-scoring reference. Missing or invalid labels raise an error for order evaluation rather than silently being inferred.

`[1, 2, 3]` has order validity `1`; `[2, 1, 3]` has validity `0.5`; `[3, 2, 1]` has validity `0`. Equal consecutive labels are permitted. There is no hierarchy-coverage bonus and no box-overlap penalty.

## Data and diagnostics

See `examples/reward_input.json` and `examples/rl_sample.json`. Add `hierarchy_level` to every annotated gold step; do not automatically label steps by their list positions. Keep `image_paths` accessible to all reward workers. Pixel coordinates and 1-based indices in input-image order remain the defaults.

EasyR1 receives numeric metrics including `overall`, `answer_accuracy`, `final_grounding`, `evidence_path`, `output_format`, `grounding_hybrid_iou`, both bonus scores, `step_validity`, `step_order_score`, and `step_accuracy_gate`. Per-sample failures produce `reward_error=1` and zero rewards. Invalid global settings raise before sample processing.

Use `configs/rl/full_reward.yaml` for these equations. Pass the reward arguments listed in that configuration to `compute_score`.
