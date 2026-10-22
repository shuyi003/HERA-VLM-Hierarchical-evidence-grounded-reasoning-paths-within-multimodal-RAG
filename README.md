# HERA-VLM

**Hierarchical evidence-grounded reasoning paths within multimodal RAG**

[Training and data](docs/training.md) · [Reward specification](docs/reward.md)

## Overview

```mermaid
flowchart LR
  A[Annotated document images] --> B[SFT]
  B --> C[Merged Qwen2.5-VL checkpoint]
  C --> D[GRPO]
  E[Four-aspect reward] --> D
  D --> F[Adapted VLM]
  G[PDF + question] --> H[ColPali page retrieval]
  H --> I[Weighted visual + text reranking]
  I --> J[Selected page images]
  J --> F
  F --> K[Answer + reasoning steps + evidence boxes]
```

The training reward uses **ColQwen2** to assess reasoning-step/image-region alignment. The application uses **ColPali** for document-page retrieval. These are separate components.

## What is included

- A batch reward function for answer accuracy, final evidence grounding, stepwise attribution, and output format.
- A GRPO configuration for the four-aspect reward function.
- SFT and LoRA export configurations, with data-format examples.
- A Gradio PDF QA application with single-page and multiple-page inference, ranking inspection, and evidence visualization.

```text
app/                 PDF retrieval, reranking, inference, and visualization
configs/sft/         SFT and LoRA export configurations
configs/rl/          Full-reward GRPO configuration
reward_functions/    Manuscript-aligned four-aspect reward implementation
prompts/             Structured-output prompt template
examples/            Data-format examples
scripts/             Code checks and reward test launcher
tests/               Equation and batch integration tests
docs/               Training guide and reward specification
```

## Training: SFT → GRPO

1. Prepare image/question/answer samples with numbered reasoning steps and evidence boxes.
2. Run SFT then export a merged checkpoint.
3. Use that checkpoint to initialize GRPO and optimize the four-aspect reward.
4. Export the selected RL checkpoint in Hugging Face format, including its processor, and load it in the application.

`configs/rl/full_reward.yaml` starts from an SFT checkpoint and uses the full reward implementation. See [training instructions](docs/training.md) for commands and data preparation.

## Four-aspect reward

With the full reward function's entry-point defaults:

$$
R = R_{\mathrm{answer}} + R_{\mathrm{ground}} + R_{\mathrm{path}} + R_{\mathrm{format}}.
$$

| Component | What it checks |
| --- | --- |
| Answer | Exact match, token F1, and sentence-embedding similarity |
| Grounding | Hybrid IoU (0 below/equal to 0.1, raw IoU between thresholds, 1 at/above 0.5), plus coverage and merged-evidence bonuses |
| Path | Answer gate ≥ 0.6, then 0.4 × proportion of valid ColQwen2-aligned steps + 0.6 × proportion of non-decreasing hierarchy transitions |
| Format | Exactly one ordered `<think>`, `<answer>`, and `<evidence>` block, numbered steps, and parseable boxes for answerable examples |

Each reference step carries a `hierarchy_level` of `1` (section), `2` (subsection), or `3` (detailed evidence). Predicted steps are matched to reference steps by text semantic similarity. Step evidence passes at ColQwen2 similarity ≥ `0.5`; the order score is the fraction of adjacent non-decreasing level transitions. See [the equations and implementation conventions](docs/reward.md).

Example output:

```xml
<think>
1. Locate the section "". {Image index: 1, box 2d: [100, 200, 800, 300]}
2. Read the specified width. {Image index: 1, box 2d: [100, 400, 800, 500]}
</think>
<answer>The required width is 1.2 m.</answer>
<evidence>[{Image index: 1, box 2d: [100, 400, 800, 500]}]</evidence>
```

Coordinates are `[x1, y1, x2, y2]` pixels by default. Image indices are 1-based positions in the model's input image list, not necessarily original PDF page numbers.

## PDF QA demo

Use a separate environment for the application. Install a CUDA-compatible PyTorch build and the dependencies listed below.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
export QWEN_MODEL_PATH=/absolute/path/to/merged-or-full-rl-checkpoint
python app/app_revised_weighted_rerank_single_or_multiple.py
```

Use a full or merged checkpoint containing model weights, tokenizer, and processor assets. CUDA inference uses bitsandbytes 4-bit quantization for Qwen. Allocate additional memory for ColPali and the text encoder.

The app renders PDF pages at 200 DPI, retrieves five candidates by default, and reranks them. Text similarity uses `BAAI/bge-base-en-v1.5` over extracted PDF text. Single-page inference defaults to one page; multiple-page inference defaults to three. Scanned PDFs without extractable text receive a placeholder in the text reranker.

| Environment variable | Default / purpose |
| --- | --- |
| `QWEN_MODEL_PATH` | `checkpoints/hera-vlm-rl`; local full/merged model directory |
| `COLPALI_MODEL_ID` | `vidore/colpali-v1.2-hf` |
| `TEXT_MODEL_ID` | `BAAI/bge-base-en-v1.5` |
| `GRADIO_SHARE` | `false`; set to `true` to request a public Gradio link |

## Development checks

```bash
python scripts/validate_release.py
# Equation and batch tests (requires Pillow):
python scripts/smoke_reward.py
```

## Acknowledgements

Built around [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [ColPali](https://huggingface.co/vidore/colpali-v1.2-hf), and [ColQwen2](https://huggingface.co/vidore/colqwen2-v1.0-hf). Framework usage follows their official documentation; dependencies and model assets retain their respective licenses. This repository is licensed under the [MIT License](LICENSE).
