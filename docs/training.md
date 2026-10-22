# Training and data preparation

## Environment setup

Use separate environments for LLaMA-Factory, EasyR1, and the demo. Follow the [LLaMA-Factory installation instructions](https://github.com/hiyouga/LLaMA-Factory) and [EasyR1 installation instructions](https://github.com/hiyouga/EasyR1), then record the exact commits and `pip freeze` output used in your experiment.

The reward needs Transformers exports `ColQwen2ForRetrieval` and `ColQwen2Processor`; the demo needs `ColPaliForRetrieval`, `ColPaliProcessor`, and `Qwen2_5_VLForConditionalGeneration`. Verify their availability in the corresponding environment before downloading models. Review `requirements-reward.txt` against EasyR1's dependency constraints before adding packages.

All project-relative commands below assume the HERA-VLM root as the working directory. Use a CUDA environment with sufficient GPU memory for the actor, rollout, reference model, and ColQwen2 reward inference.

## 1. SFT data

Use the [official LLaMA-Factory dataset schema](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md). `examples/sft_sample.json` demonstrates the ShareGPT-style multimodal data format. Populate the image paths, questions, answers, and boxes with your annotations.

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

Adjust batch size, image resolution, cutoff length, epoch count, and learning rate to your data and hardware. Confirm `qwen2_vl` is the Qwen2.5-VL template supported by your installed LLaMA-Factory revision.

The exported directory must include the full model, tokenizer, and processor. The full-reward RL configuration points to `checkpoints/hera-vlm-sft-merged`.

## 3. RL data

Create `data/rl/train.parquet` and `data/rl/valid.parquet`. Each row has:

| Field | Type | Meaning |
| --- | --- | --- |
| `problem` | string | Question; the prompt template adds output instructions |
| `answer` | JSON string | Ground-truth payload consumed by the reward function |
| `images` | list of strings | Ordered local image paths supported by the configured EasyR1 data loader |

`examples/rl_sample.json` shows the row structure for preparing your training data. The ground-truth JSON contains:

- `gold_answer`: reference answer.
- `gold_evidence`: list of `{image_index, bbox}` objects.
- `gold_steps`: ordered `{step_index, hierarchy_level, text, bboxes}` objects; explicit integer hierarchy levels are `1` (section), `2` (subsection), and `3` (detailed evidence).
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

The full-reward configuration follows manuscript Eqs. 8–16 with a 0.6 answer gate, a 0.5 step-alignment threshold, and 0.4/0.6 step/order weights. Coverage/adjacency thresholds are documented implementation defaults in [the reward specification](reward.md). Add explicit `hierarchy_level` annotations to existing datasets before using it.

The configuration uses four samples per prompt, temperature 0.9, top-p 0.95, GRPO, KL coefficient 0.01, and actor LoRA rank 8. It connects the SFT checkpoint to the structured prompt and full reward function, and uses a local file logger by default.

The prompt uses the `content` template variable. Keep the output structure consistent between SFT and RL.

For offline training, pre-download the required models, set local model paths as appropriate, and enable `semantic_local_files_only`. Before training, check `reward_error`, component scores, image loading, and generated examples. A missing model or invalid ground-truth field may produce zero rewards instead of stopping the worker.

## 5. Export and serve

Use the checkpoint export/merging procedure for your installed EasyR1 version. A raw sharded actor checkpoint or LoRA-only directory is not sufficient for the application. Point `QWEN_MODEL_PATH` to the full Hugging Face checkpoint containing processor/tokenizer files.

The app defaults to three pages in multiple-page mode while RL rollout limits images to two. Its `QWEN_MIN_PIXELS=3136` also differs from the RL template's `min_pixels=262144`. Align these preprocessing settings for controlled evaluations.
