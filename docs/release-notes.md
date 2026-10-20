# Release preparation notes

## Supplied implementation

- `reward_functions/conreg_four_aspect_reward_colqwen2_stepwise_with_order.py` comes from the supplied full reward file. Its grounding and path components now implement manuscript Eqs. 8–16, as described in the update below. An unused module-global SentenceTransformer initialization was removed; the existing lazy cache remains, so importing the module no longer downloads this encoder unnecessarily.
- `app/app_revised_weighted_rerank_single_or_multiple.py` comes from the supplied PDF QA app. The personal cloud-drive checkpoint path was replaced with `checkpoints/hera-vlm-rl`. Gradio sharing is now opt-in via `GRADIO_SHARE=true`, and launch debugging is disabled by default.
- `configs/rl/order_only_reference.yaml` preserves the supplied order-only configuration structure with portable paths. Its missing reward file is not fabricated or replaced by the full implementation. The original checkpoint was a continuation checkpoint; this portable reference uses a placeholder SFT checkpoint instead.

The original source files outside this release directory were not modified.

## Newly prepared materials

The READMEs, documentation, SFT/export templates, full-reward GRPO template, prompt, synthetic JSON examples, dependency inventories, and validation scripts were created for repository publication. They must not be described as the exact artifacts used to produce prior results.

The full reward configuration derives reward arguments from the actual `compute_score` signature. Its training/rollout settings come from the supplied order-only YAML. Model acquisition is enabled for the sentence encoder by setting `semantic_local_files_only=false`.

## Not included

Training or evaluation datasets; model weights; the order-only reward implementation; original SFT and prompt configurations; exact framework revisions; a tested environment lockfile; measured performance; or a publication citation. The existing repository’s MIT license is retained. No unrelated files from the surrounding workspace are part of this release.

## Preparation checks (2026-10-19)

Passed: Python AST syntax checks; JSON example parsing; full-reward keyword and weight checks; local Markdown link checks; parsing of all four YAML files; dependency-free format, evidence parsing, no-answer, IoU, and soft-threshold checks. At initial publication, an AST comparison confirmed every reward function body was unchanged from the supplied source.

The smoke check exposed that quoted JSON `"image_index"` keys are not reliably recognized by the existing text parser. Model-response examples and the new prompt therefore use the application's original `Image index: N, box 2d: [...]` notation. Ground-truth payloads remain ordinary JSON objects. This is a formatting adjustment, not a parser change.

Not run: complete reward-module import with its ML dependencies, neural similarity inference, GPU SFT/GRPO, model export, or the Gradio end-to-end workflow. Dependency inventories are not lockfiles.

## Unit-weight update (2026-10-19)

At the repository owner’s request, all four top-level aspect weights are now `1.0` in Python defaults and the full-reward configuration. The sum-to-one requirement is disabled by default, and the total no longer clamps to 1: four perfect aspects produce 4. Component scoring and internal component weights remain unchanged. Documentation and regression checks were updated together.

## Manuscript reward update (2026-10-20)

- Grounding now uses the piecewise hybrid IoU and `0.9 H + 0.1 (coverage + merged)` instead of the previous five-component mixture. The upper boundary includes IoU exactly 0.5. Both bonuses can coexist, so grounding can reach 1.1 and the unit-weight total can reach 4.1.
- Path reward now uses the 0.6 answer gate, per-step validity at ColQwen2 similarity 0.5, and 0.4/0.6 evidence/order weights. Semantic-only alignment assigns explicit annotated hierarchy levels; the overlap penalty, text/bbox matching mixture, and hierarchy-coverage term were removed.
- Numerical bonus criteria, multi-box aggregation, multi-citation steps, and paths with fewer than two steps are documented implementation conventions because the excerpt does not fully specify these cases.
- Updated the full-reward YAML, data examples, validation script, and reward documentation. Removed the Chinese README and the requested introductory note.
- PyTorch imports are deferred to neural helper functions, allowing the actual module and batch entry point to be exercised with Pillow and mocked encoders. Regression tests cover equations, boundaries, missing data, and batch behavior. Neural checkpoint inference and GPU training were not run for this change.
