# Training and data preparation

## Environment and provenance

Use separate environments for LLaMA-Factory, EasyR1, and the demo. Follow the [LLaMA-Factory installation instructions](https://github.com/hiyouga/LLaMA-Factory) and [EasyR1 installation instructions](https://github.com/hiyouga/EasyR1), then record the exact commits and `pip freeze` output used in your experiment. The original environment versions were not supplied; these instructions do not promise compatibility with every upstream revision.

The reward needs Transformers exports `ColQwen2ForRetrieval` and `ColQwen2Processor`; the demo needs `ColPaliForRetrieval`, `ColPaliProcessor`, and `Qwen2_5_VLForConditionalGeneration`. Verify their availability in the corresponding environment before downloading models. Review `requirements-reward.txt` against EasyR1's dependency constraints before adding packages.

All project-relative commands below assume the HERA-VLM root as the working directory. Training is intended for an appropriately provisioned CUDA environment. A one-GPU configuration does not establish a minimum VRAM requirement: actor, rollout, reference, and ColQwen2 reward inference all consume memory.

## 1. SFT data

Use the [official LLaMA-Factory dataset schema](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md). `examples/sft_sample.json` demonstrates ShareGPT-style multimodal data; its image path, question, answer, and boxes are synthetic placeholders.

Prepare `data/sft/train.json` using real annotations and copy the registration:

```bash
mkdir -p data/sft
cp examples/dataset_info.json data/sft/dataset_info.json
```

Each `<image>` token must correspond to one image in `images`. Use absolute image paths valid on the training machine. Label reasoning steps with `Step 1:`, `Step 2:`, etc. Keep image indices, bounding boxes, and image dimensions consistent. Split by source document when appropriate to avoid leaking pages from the same document across train and evaluation sets.

## 2. Train and export SFT

Activate the environment where LLaMA-Factory is installed:

```bash
llamafactory-cli train configs/sft/train_lora.yaml
llamafactory-cli export configs/sft/export_lora.yaml
```

These are newly proposed LoRA templates, not recovered experiment settings. Adjust batch size, image resolution, cutoff length, epoch count, and learning rate to your data and hardware. Confirm `qwen2_vl` is the Qwen2.5-VL template supported by your installed LLaMA-Factory revision.

The exported directory must include the full model, tokenizer, and processor. The full-reward RL template points to `checkpoints/hera-vlm-sft-merged`.

## 3. RL data

Create `data/rl/train.parquet` and `data/rl/valid.parquet`. Each row has:

| Field | Type | Meaning |
| --- | --- | --- |
| `problem` | string | Question; the prompt template adds output instructions |
| `answer` | JSON string | Ground-truth payload consumed by the reward function |
| `images` | list of strings | Ordered local image paths supported by the configured EasyR1 data loader |

`examples/rl_sample.json` illustrates one row. Do not use this single synthetic example as a training dataset. The ground-truth JSON contains:

- `gold_answer`: reference answer.
- `gold_evidence`: list of `{image_index, bbox}` objects.
- `gold_steps`: ordered `{step_index, text, bboxes}` objects.
- `image_sizes`: `{image_index, width, height}` records.
- `image_paths`: ordered image file paths accessible to every reward worker.
- `is_no_answer`: boolean, not a string.
- `pred_coordinate_space` and `gold_coordinate_space`: `pixel` by default.

Use 1-based indices relative to `images`; keep them distinct from original PDF page numbers. The pixel coordinate system must match the actual stored image dimensions. When changing render DPI or resizing images, transform annotations accordingly. Ensure the selected EasyR1 revision passes the `answer` string to the reward as `ground_truth` and supports the chosen image representation.

## 4. Run GRPO

Activate the EasyR1 environment. Install the framework following its official instructions; if using its source checkout, make `verl` importable:

```bash
export EASYR1_ROOT=/absolute/path/to/EasyR1
export PYTHONPATH="$EASYR1_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m verl.trainer.main config=configs/rl/full_reward.yaml
```

This launcher follows the [official EasyR1 Qwen2.5-VL example](https://github.com/hiyouga/EasyR1/blob/main/examples/qwen2_5_vl_7b_geo3k_grpo.sh). Verify the selected revision supports the supplied YAML fields, including actor LoRA settings and `reward_function_kwargs`.

The new configuration retains the supplied template's rollout settings: four samples per prompt, temperature 0.9, top-p 0.95, GRPO, KL coefficient 0.01, and actor LoRA rank 8. It changes the checkpoint path, prompt path, reward module, and reward arguments to connect the SFT output with the full reward. It uses a local file logger by default. The full reward can consume substantially more memory than the order-only ablation.

The new prompt uses EasyR1's `content` template variable. It is a replacement template; the original prompt file was not provided. Compare it with your original SFT prompt before experimental replication.

For offline training, pre-download the required models, set local model paths as appropriate, and enable `semantic_local_files_only`. Run validation first and inspect `reward_error`, component scores, image loading, and generated examples. A missing model or invalid ground-truth field may produce zero rewards instead of stopping the worker.

## 5. Export and serve

Use the checkpoint export/merging procedure for your installed EasyR1 version. A raw sharded actor checkpoint or LoRA-only directory is not sufficient for the application. Point `QWEN_MODEL_PATH` to the full Hugging Face checkpoint containing processor/tokenizer files.

The app defaults to three pages in multiple-page mode while RL rollout limits images to two. Its `QWEN_MIN_PIXELS=3136` also differs from the RL template's `min_pixels=262144`. Match inference and training preprocessing for controlled evaluations; these settings were not silently unified in this release.

## Evaluation status

No real dataset, run logs, held-out predictions, or benchmark measurements were supplied. Report answer EM/F1, evidence IoU, path-order and crop-support diagnostics, format rate, and latency only after evaluating a fixed held-out set. Compare SFT-only, full-reward GRPO, and order-only GRPO using documented data splits and settings; do not infer gains from reward formulas alone.
