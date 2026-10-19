# Reward specification

This document describes `compute_score` in the supplied full-reward Python file. Entry-point defaults take precedence over different helper-function defaults or comments.

## Total and answer reward

`R = 1.0 A + 1.0 G + 1.0 P + 1.0 F` in `[0, 4]`.

Each component remains in `[0, 1]`. The total is not clipped to 1 or divided by 4. The four aspect weights default to `1.0`, and `require_valid_weight_sum=false` allows this unnormalized sum. The optional sum-to-one check can still be enabled for normalized custom weights. Internal component weights below are unchanged.

`A = 0.10 EM + 0.10 token_F1 + 0.80 semantic`.

Answer text is extracted from `<answer>`. Text normalization lowercases, removes punctuation and English articles, and collapses whitespace. SentenceTransformer cosine is mapped with `(cosine + 1) / 2` and clamped. The default encoder is `sentence-transformers/all-MiniLM-L6-v2`. The original order-only YAML uses different answer weights (`0.10/0.30/0.60`); those are not the full Python entry-point defaults.

## Final evidence grounding

`G = 0.50 IoU_match + 0.05 coverage + 0.05 IoU_merged + 0.30 soft_IoU + 0.10 validity`.

Matched IoU uses one-to-one matching; coverage allows one predicted box to cover multiple gold boxes. Gold boxes are enclosed by a merged box per image for the merged term. The soft term uses the maximum of matched and merged IoU, with lower/upper thresholds `0.10/0.50`. Validity checks normalized coordinates. These terms are scores, not all binary indicators. The source file also retains an older strict-grounding helper, but `_score_one` calls `final_grounding_score`.

## Evidence path

For answerable examples with reasoning/evidence pairs:

`P = gate × (support_pass + overlap_pass + order_score) / 3`.

- `gate = 1[A >= 0.50]`.
- `support_pass = 1[min(step_crop_text_cosines) >= 0.45]` when every pair can be cropped and scored; otherwise zero.
- `overlap_pass = 1[max_pairwise_bbox_IoU <= 0.50]` using the entry-point default.
- Step support uses mean-pooled, normalized ColQwen2 embeddings and cosine similarity, not the application's ColPali retrieval score.
- Each predicted step is matched to a gold step with `0.70 × text_score + 0.30 × bbox_score`. The text matching score itself combines exact match, token F1, and semantic similarity.
- `order_score = 0.70 × monotonic_transition_fraction + 0.30 × coverage`.
- `coverage = distinct_matched_positions / min(number_of_gold_steps, number_of_predicted_steps)`.

A single matched step can get full monotonic and coverage scores even when the gold path is longer. Full hierarchy completion is logged separately; it is not an extra multiplier. Missing images zero the support term but do not necessarily zero the other path terms. Missing gold steps affect the order term. Thus a high reward does not prove a complete or correct reasoning path.

The original helper docstring says it does not compare intermediate text to gold text. That statement only describes crop support: the actual order-alignment code does compare predicted and annotated step text.

## Format and unanswerable examples

Format requires exactly one `<think>`, `<answer>`, and `<evidence>` block in that order, nonempty contents, and at least one parsed numbered step. Answerable examples additionally need parseable boxes in reasoning and final evidence. It does not enforce a box on every individual step.

For `is_no_answer=true`, answer scoring uses the implementation's recognized no-answer strings, grounding rewards omission of boxes, and the path branch rewards omission of reasoning/box pairs without the normal accuracy gate. A nonempty `<evidence>None</evidence>` and a numbered unanswerable explanation can satisfy the format rule.

## Runtime and diagnostics

EasyR1 supplies a batch of dictionaries with `response` and `ground_truth`; the function returns one dictionary per sample with `overall`, the four component scores, and detailed metrics. Store image paths inside the ground-truth payload so the reward worker can load crops independently of rollout image decoding.

The source catches per-sample exceptions and returns zero reward with `reward_error=1`. Monitor that metric and worker logs before interpreting low scores. The Python default is `semantic_local_files_only=true`; the new full-reward template explicitly sets it to `false` for initial model acquisition. For offline training, pre-cache the encoder and set it back to `true`.

The reference order-only YAML refers to an unavailable `conreg_four_aspect_reward_order_only_light.py`. It uses different grounding argument names and documents `P = gate × binary_order`. It must not be paired with the full reward function unchanged.
