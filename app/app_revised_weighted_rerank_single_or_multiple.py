import gc
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Union

import fitz  # PyMuPDF
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    ColPaliForRetrieval,
    ColPaliProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
import re
import textwrap


try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


# ============================================================
# Configuration
# ============================================================
COLPALI_MODEL_ID = os.getenv("COLPALI_MODEL_ID", "vidore/colpali-v1.2-hf")
TEXT_MODEL_ID = os.getenv("TEXT_MODEL_ID", "BAAI/bge-base-en-v1.5")


QWEN_MODEL_PATH = os.getenv(
    "QWEN_MODEL_PATH",
    "checkpoints/hera-vlm-rl",
)


DEFAULT_COLPALI_TOP_K = 5
DEFAULT_SINGLE_PAGE_TOP_K = 1
DEFAULT_MULTIPLE_PAGE_TOP_K = 3

# Weighted reranking
COLPALI_WEIGHT = 0.80
SEMANTIC_WEIGHT = 0.20

# PDF rasterization resolution.
PDF_RENDER_DPI = 200

MAX_PAGE_TEXT_CHARS = 10000

# ============================================================
# Qwen2.5-VL image processor
# Keep exactly the same settings as training.
# ============================================================

QWEN_MIN_PIXELS = 3136
QWEN_MAX_PIXELS = 12845056
#QWEN_MIN_PIXELS = 200704
#QWEN_MAX_PIXELS = 786432

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# ============================================================
# Global model cache
# ============================================================
_colpali_model = None
_colpali_processor = None
_text_model = None
_qwen_model = None
_qwen_processor = None


# ============================================================
# Model loaders
# ============================================================
def load_colpali():
    global _colpali_model, _colpali_processor

    if _colpali_model is None:
        _colpali_processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL_ID)
        _colpali_model = ColPaliForRetrieval.from_pretrained(
            COLPALI_MODEL_ID,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        ).eval()

    return _colpali_model, _colpali_processor


def load_text_model():
    global _text_model

    if _text_model is None:
        _text_model = SentenceTransformer(TEXT_MODEL_ID, device=DEVICE)

    return _text_model


def load_qwen():
    global _qwen_model, _qwen_processor

    # --------------------------------------------------------
    # Return cached model if already loaded
    # --------------------------------------------------------
    if _qwen_model is not None:
        return _qwen_model, _qwen_processor

    print("\n" + "=" * 70)
    print("Loading merged / full adapted Qwen checkpoint")
    print("=" * 70)
    print("Checkpoint:", repr(QWEN_MODEL_PATH))

    # --------------------------------------------------------
    # Validate checkpoint path
    # --------------------------------------------------------
    if not QWEN_MODEL_PATH:
        raise ValueError(
            "QWEN_MODEL_PATH is None or empty."
        )

    if not os.path.exists(QWEN_MODEL_PATH):
        raise FileNotFoundError(
            f"Qwen checkpoint not found:\n"
            f"{QWEN_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # 4-bit quantization for GPU inference
    # --------------------------------------------------------
    quant_config = None

    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # --------------------------------------------------------
    # Load the FULL adapted model
    # --------------------------------------------------------
    _qwen_model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            QWEN_MODEL_PATH,
            torch_dtype=DTYPE,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True,
        )
        .eval()
    )

    # --------------------------------------------------------
    # Load processor from the SAME adapted checkpoint
    # --------------------------------------------------------
    _qwen_processor = AutoProcessor.from_pretrained(
        QWEN_MODEL_PATH,
        min_pixels=QWEN_MIN_PIXELS,
        max_pixels=QWEN_MAX_PIXELS,
        trust_remote_code=True,
    )

    print("Model loaded successfully.")
    print(
        "Model config path:",
        getattr(
            _qwen_model.config,
            "_name_or_path",
            "Unknown",
        ),
    )
    print("=" * 70)

    return _qwen_model, _qwen_processor


# ============================================================
# PDF processing
# ============================================================
def render_pdf(pdf_path: str, dpi: int = PDF_RENDER_DPI):
    """
    Returns:
        page_images: list[PIL.Image]
        page_texts:  list[str]
    """
    doc = fitz.open(pdf_path)
    page_images = []
    page_texts = []

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc[page_index]

        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_images.append(image)

        text = page.get_text("text") or ""
        text = " ".join(text.split())
        page_texts.append(text[:MAX_PAGE_TEXT_CHARS])

    doc.close()
    return page_images, page_texts


# ============================================================
# Stage 1: ColPali visual page retrieval
# ============================================================
@torch.inference_mode()
def colpali_retrieve(
    question: str,
    page_images: List[Image.Image],
    top_k: int = DEFAULT_COLPALI_TOP_K,
    image_batch_size: int = 4,
):
    model, processor = load_colpali()

    query_inputs = processor(text=[question]).to(model.device)
    query_embeddings = model(**query_inputs).embeddings

    scored_pages = []

    for start in range(0, len(page_images), image_batch_size):
        batch = page_images[start : start + image_batch_size]
        image_inputs = processor(images=batch).to(model.device)
        image_embeddings = model(**image_inputs).embeddings

        scores = processor.score_retrieval(
            query_embeddings,
            image_embeddings,
        )[0]

        scores = scores.detach().float().cpu().numpy().tolist()

        for offset, score in enumerate(scores):
            scored_pages.append(
                {
                    "page_index": start + offset,
                    "page_number": start + offset + 1,
                    "colpali_score": float(score),
                }
            )

        del image_inputs, image_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scored_pages.sort(key=lambda x: x["colpali_score"], reverse=True)
    return scored_pages[: min(top_k, len(scored_pages))]


# ============================================================
# Stage 2: semantic textual-feature reranking
# ============================================================
def _minmax_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmax, vmin):
        return np.ones_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def semantic_text_rerank(
    question: str,
    candidates: List[dict],
    page_texts: List[str],
    colpali_weight: float = COLPALI_WEIGHT,
    semantic_weight: float = SEMANTIC_WEIGHT,
):
    """
    Weighted reranking of ColPali top-k candidates.

    final_score =
        0.8 * normalized_colpali_score
      + 0.2 * normalized_semantic_score
    """
    if not candidates:
        return []

    if not np.isclose(colpali_weight + semantic_weight, 1.0):
        raise ValueError(
            "colpali_weight + semantic_weight must equal 1.0."
        )

    model = load_text_model()

    candidate_texts = []
    valid_candidates = []

    for item in candidates:
        text = page_texts[item["page_index"]].strip()
        if not text:
            text = "[NO EXTRACTABLE PDF TEXT]"
        candidate_texts.append(text)
        valid_candidates.append(item.copy())

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    page_embeddings = model.encode(
        candidate_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    semantic_scores = page_embeddings @ question_embedding

    colpali_scores = np.asarray(
        [item["colpali_score"] for item in valid_candidates],
        dtype=np.float32,
    )

    colpali_norm = _minmax_normalize(colpali_scores)
    semantic_norm = _minmax_normalize(semantic_scores)

    for item, sem, col_n, sem_n in zip(
        valid_candidates,
        semantic_scores,
        colpali_norm,
        semantic_norm,
    ):
        item["semantic_score"] = float(sem)
        item["colpali_score_normalized"] = float(col_n)
        item["semantic_score_normalized"] = float(sem_n)
        item["final_score"] = float(
            colpali_weight * col_n
            + semantic_weight * sem_n
        )

    valid_candidates.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    for rank, item in enumerate(valid_candidates, start=1):
        item["rerank_rank"] = rank

    return valid_candidates


# ============================================================
# Stage 3: adapted Qwen inference on one or multiple reranked pages
# ============================================================
@torch.inference_mode()
def qwen_answer(
    question: str,
    images: List[Image.Image],
    page_numbers: List[int] = None,
    max_new_tokens: int = 1024,
):
    """Run Qwen2.5-VL inference on multiple retrieved document pages.

    Image index 1 corresponds to images[0], image index 2 to images[1], etc.
    page_numbers stores the original PDF page numbers for prompt context only.
    Bounding boxes remain local pixel coordinates of the corresponding image.
    """
    model, processor = load_qwen()

    if not images:
        raise ValueError("At least one image is required for Qwen inference.")

    if page_numbers is None:
        page_numbers = list(range(1, len(images) + 1))

    if len(page_numbers) != len(images):
        raise ValueError("page_numbers and images must have the same length.")

    original_images = []
    page_mapping_lines = []
    for image_index, (image, page_number) in enumerate(
        zip(images, page_numbers), start=1
    ):
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Qwen input image {image_index} must be a PIL.Image.Image."
            )
        rgb_image = image.convert("RGB")
        original_images.append(rgb_image)
        page_mapping_lines.append(
            f"- Image index {image_index} = retrieved PDF page {page_number}"
        )

    page_mapping = "\n".join(page_mapping_lines)

    is_multi_page = len(original_images) > 1
    if is_multi_page:
        input_description = (
            "multiple retrieved construction regulatory document page images"
        )
        analysis_instruction = (
            "analyze all supplied images jointly"
        )
        source_instruction = (
            "Use only information visible in the supplied images."
        )
        context_instruction = (
            "Consider all supplied images jointly before answering. Evidence may "
            "appear in one image or be distributed across multiple images."
        )
    else:
        input_description = (
            "one retrieved construction regulatory document page image"
        )
        analysis_instruction = (
            "analyze the supplied image"
        )
        source_instruction = (
            "Use only information visible in the supplied image."
        )
        context_instruction = (
            "Base the reasoning and answer on the single supplied image."
        )

    task_prompt = f"""Task Description:

Given {input_description} and a relevant question,
{analysis_instruction}, identify the visual evidence required to answer
the question, and provide a concise evidence-grounded response.

Retrieved image-to-page mapping:
{page_mapping}

Requirements:

1. {source_instruction}

2. {context_instruction}

3. Identify the relevant visual evidence before answering the question.

4. Every bounding box MUST use the following exact format:

- Image index: N, box 2d: [x1, y1, x2, y2]

where N is the 1-based index of the supplied image containing that evidence.

5. Bounding boxes MUST use pixel coordinates of the ORIGINAL corresponding supplied image.
   Do not use normalized coordinates, processor-grid coordinates, or PDF coordinates.

6. If evidence is located on different images, use the correct image index for each box.

7. You MUST return ALL THREE XML sections below.

8. The <answer> and <evidence> tags MUST NOT be omitted.

9. Do NOT write any text before <think> or after </evidence>.

10. Do NOT place the final answer outside the <answer> tags.

11. Even if bounding boxes have already appeared in <think>, repeat the final
supporting bounding boxes inside <evidence>.

12. Follow this exact output format:

<think>
Step 1: ...
- Image index: 1, box 2d: [x1, y1, x2, y2]

Step 2: ...
- Image index: 2, box 2d: [x1, y1, x2, y2]
</think>
<answer>
Concise final answer only.
</answer>
<evidence>
- Image index: 1, box 2d: [x1, y1, x2, y2]
- Image index: 2, box 2d: [x1, y1, x2, y2]
</evidence>

IMPORTANT:
Your response is invalid if <answer>...</answer> or
<evidence>...</evidence> is missing.

Question:
{question}
"""

    # Qwen2.5-VL supports multiple image content items in one user turn.
    content = [
        {"type": "image", "image": image}
        for image in original_images
    ]
    content.append({"type": "text", "text": task_prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    for image_index, image in enumerate(original_images, start=1):
        original_width, original_height = image.size
        print(
            f"[Qwen input image {image_index}] original PIL size: "
            f"{original_width} x {original_height}"
        )

    inputs = processor(
        text=[text],
        images=original_images,
        padding=True,
        return_tensors="pt",
    )

    # Qwen2.5-VL returns one image_grid_thw entry per image.
    if "image_grid_thw" in inputs:
        patch_size = getattr(
            processor.image_processor,
            "patch_size",
            14,
        )
        grids = inputs["image_grid_thw"].detach().cpu().tolist()
        for image_index, (grid, image) in enumerate(
            zip(grids, original_images), start=1
        ):
            _, grid_h, grid_w = grid
            processed_height = int(grid_h) * int(patch_size)
            processed_width = int(grid_w) * int(patch_size)
            original_width, original_height = image.size
            print(
                f"[Qwen processor image {image_index}] effective vision size: "
                f"{processed_width} x {processed_height}"
            )
            print(
                f"[Qwen processor image {image_index}] scale ratio: "
                f"x={processed_width / original_width:.4f}, "
                f"y={processed_height / original_height:.4f}"
            )

    model_device = next(model.parameters()).device
    inputs = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    trimmed_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]

    response = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response.strip()


# ============================================================
# BBOX drawing
# ============================================================

def extract_tag_content(
    text: str,
    tag_name: str,
):
    """
    Extract all contents enclosed by a given XML-like tag.

    Example:
        <think>...</think>
        <evidence>...</evidence>
    """
    if not isinstance(text, str):
        return []

    pattern = rf"""
        <{re.escape(tag_name)}\b[^>]*>
        (.*?)
        </{re.escape(tag_name)}>
    """

    matches = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )

    return [
        match.strip()
        for match in matches
        if match.strip()
    ]

Number = Union[int, float]


def _convert_number(value: str) -> Number:
    numeric_value = float(value)

    if numeric_value.is_integer():
        return int(numeric_value)

    return numeric_value


def _remove_duplicate_bboxes(
    boxes,
    decimal_places=6,
):
    unique_boxes = []
    seen = set()

    for item in boxes:
        image_index = int(
            item["image_index"]
        )

        bbox = item["bbox"]

        bbox_key = tuple(
            round(
                float(value),
                decimal_places,
            )
            for value in bbox
        )

        key = (
            image_index,
            bbox_key,
        )

        if key in seen:
            continue

        seen.add(key)
        unique_boxes.append(item)

    return unique_boxes


def extract_bboxes_from_text(
    text,
    default_image_index=1,
    remove_duplicates=True,
):
    if not isinstance(text, str) or not text.strip():
        return []

    normalized_text = (
        text
        .replace("：", ":")
        .replace("；", ";")
        .replace("，", ",")
        .replace("［", "[")
        .replace("］", "]")
        .replace("【", "[")
        .replace("】", "]")
        .replace("–", "-")
        .replace("—", "-")
    )

    number_pattern = r"""
        [-+]?
        (?:
            \d+(?:\.\d*)?
            |
            \.\d+
        )
        (?:[eE][-+]?\d+)?
    """

    token_pattern = re.compile(
        rf"""
        (?P<image_token>
            (?:
                image\s*index
                |
                image_index
                |
                image\s*id
                |
                image
            )
            \s*
            (?:[:=]|number\s*[:=]?)?
            \s*
            (?P<image_index>\d+)
        )
        |
        (?P<bbox_token>
            (?:
                box\s*2d
                |
                box_2d
                |
                bbox
                |
                bounding\s*box
                |
                bounding_box
            )
            \s*
            (?:[:=])?
            \s*
            \[
                \s*(?P<x1>{number_pattern})\s*,
                \s*(?P<y1>{number_pattern})\s*,
                \s*(?P<x2>{number_pattern})\s*,
                \s*(?P<y2>{number_pattern})\s*
            \]
        )
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    extracted_boxes = []

    current_image_index = (
        default_image_index
    )

    for match in token_pattern.finditer(
        normalized_text
    ):

        if (
            match.group("image_token")
            is not None
        ):
            current_image_index = int(
                match.group(
                    "image_index"
                )
            )
            continue

        if (
            match.group("bbox_token")
            is not None
        ):
            bbox = [
                _convert_number(
                    match.group("x1")
                ),
                _convert_number(
                    match.group("y1")
                ),
                _convert_number(
                    match.group("x2")
                ),
                _convert_number(
                    match.group("y2")
                ),
            ]

            extracted_boxes.append(
                {
                    "image_index":
                        current_image_index,
                    "bbox":
                        bbox,
                }
            )

    if remove_duplicates:
        extracted_boxes = (
            _remove_duplicate_bboxes(
                extracted_boxes
            )
        )

    return extracted_boxes

def extract_response_bboxes(
    model_response,
):
    """
    Extract boxes separately from
    <think> and <evidence>.
    """

    think_contents = extract_tag_content(
        model_response,
        "think",
    )

    evidence_contents = extract_tag_content(
        model_response,
        "evidence",
    )

    think_boxes = []

    for content in think_contents:
        think_boxes.extend(
            extract_bboxes_from_text(
                text=content,
                default_image_index=1,
                remove_duplicates=False,
            )
        )

    evidence_boxes = []

    for content in evidence_contents:
        evidence_boxes.extend(
            extract_bboxes_from_text(
                text=content,
                default_image_index=1,
                remove_duplicates=False,
            )
        )

    return {
        "think": _remove_duplicate_bboxes(
            think_boxes
        ),
        "evidence": _remove_duplicate_bboxes(
            evidence_boxes
        ),
    }

def prepare_bbox(
    bbox,
    image_width,
    image_height,
    coordinate_mode="pixel",
):
    x1, y1, x2, y2 = bbox

    if coordinate_mode == "pixel":
        pass

    elif coordinate_mode == "normalized_1000":
        x1 = x1 / 1000.0 * image_width
        x2 = x2 / 1000.0 * image_width
        y1 = y1 / 1000.0 * image_height
        y2 = y2 / 1000.0 * image_height

    elif coordinate_mode == "normalized_1":
        x1 = x1 * image_width
        x2 = x2 * image_width
        y1 = y1 * image_height
        y2 = y2 * image_height

    else:
        raise ValueError(
            "coordinate_mode must be "
            "'pixel', 'normalized_1000', "
            "or 'normalized_1'."
        )

    x_min = min(x1, x2)
    x_max = max(x1, x2)

    y_min = min(y1, y2)
    y_max = max(y1, y2)

    x_min = max(
        0,
        min(x_min, image_width - 1),
    )

    x_max = max(
        0,
        min(x_max, image_width - 1),
    )

    y_min = max(
        0,
        min(y_min, image_height - 1),
    )

    y_max = max(
        0,
        min(y_max, image_height - 1),
    )

    return [
        x_min,
        y_min,
        x_max,
        y_max,
    ]

def visualize_response_bboxes(
    image: Image.Image,
    model_response: str,
    coordinate_mode="pixel",
    show_think=True,
    show_evidence=True,
    target_image_index=1,
):
    """
    Draw model-generated bounding boxes directly
    on a PIL image for Gradio visualization.

    Think:
        blue solid box

    Evidence:
        red box

    Returns
    -------
    annotated_image : PIL.Image
    bbox_summary : dict
    """

    if image is None:
        return None, {
            "think": [],
            "evidence": [],
        }

    annotated_image = (
        image
        .convert("RGB")
        .copy()
    )

    draw = ImageDraw.Draw(
        annotated_image
    )

    image_width, image_height = (
        annotated_image.size
    )

    parsed_boxes = (
        extract_response_bboxes(
            model_response
        )
    )

    think_boxes = parsed_boxes["think"]
    evidence_boxes = parsed_boxes["evidence"]

    # --------------------------------------------------------
    # Line width based on page resolution
    # --------------------------------------------------------
    line_width = max(
        3,
        int(
            min(
                image_width,
                image_height,
            )
            * 0.0025
        ),
    )

    # Colors
    think_color = (
        31,
        78,
        121,
    )

    evidence_color = (
        176,
        58,
        46,
    )

    # --------------------------------------------------------
    # Draw THINK boxes
    # --------------------------------------------------------
    if show_think:

        current_think_boxes = [
            item
            for item in think_boxes
            if int(
                item["image_index"]
            ) == int(target_image_index)
        ]

        for index, item in enumerate(
            current_think_boxes,
            start=1,
        ):

            x1, y1, x2, y2 = (
                prepare_bbox(
                    item["bbox"],
                    image_width,
                    image_height,
                    coordinate_mode,
                )
            )

            coordinates = [
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
            ]

            draw.rectangle(
                coordinates,
                outline=think_color,
                width=line_width,
            )

            draw_bbox_label(
                draw=draw,
                xy=(
                    coordinates[0],
                    coordinates[1],
                ),
                text=f"Think {index}",
                color=think_color,
                image_width=image_width,
                image_height=image_height,
            )

    # --------------------------------------------------------
    # Draw EVIDENCE boxes
    # --------------------------------------------------------
    if show_evidence:

        current_evidence_boxes = [
            item
            for item in evidence_boxes
            if int(
                item["image_index"]
            ) == int(target_image_index)
        ]

        for index, item in enumerate(
            current_evidence_boxes,
            start=1,
        ):

            x1, y1, x2, y2 = (
                prepare_bbox(
                    item["bbox"],
                    image_width,
                    image_height,
                    coordinate_mode,
                )
            )

            coordinates = [
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
            ]

            # Slightly thicker evidence box
            draw.rectangle(
                coordinates,
                outline=evidence_color,
                width=line_width + 1,
            )

            draw_bbox_label(
                draw=draw,
                xy=(
                    coordinates[0],
                    coordinates[1],
                ),
                text=f"Evidence {index}",
                color=evidence_color,
                image_width=image_width,
                image_height=image_height,
            )

    return (
        annotated_image,
        parsed_boxes,
    )

def draw_bbox_label(
    draw,
    xy,
    text,
    color,
    image_width,
    image_height,
):
    """
    Draw a readable bbox label.
    """

    x, y = xy

    try:
        font = ImageFont.truetype(
            "DejaVuSans.ttf",
            size=max(
                14,
                int(
                    min(
                        image_width,
                        image_height,
                    )
                    * 0.012
                ),
            ),
        )
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox(
        (x, y),
        text,
        font=font,
    )

    text_width = (
        bbox[2] - bbox[0]
    )

    text_height = (
        bbox[3] - bbox[1]
    )

    padding = 4

    label_y = max(
        0,
        y - text_height - 2 * padding,
    )

    label_x = min(
        max(0, x),
        max(
            0,
            image_width
            - text_width
            - 2 * padding,
        ),
    )

    draw.rectangle(
        [
            label_x,
            label_y,
            label_x
            + text_width
            + 2 * padding,
            label_y
            + text_height
            + 2 * padding,
        ],
        fill=color,
    )

    draw.text(
        (
            label_x + padding,
            label_y + padding,
        ),
        text,
        fill="white",
        font=font,
    )

# ============================================================
# UI helpers
# ============================================================
def build_ranking_table(
    colpali_results,
    reranked_results,
):
    """
    Build a concise retrieval table.

    Columns:
        ColPali Rank
        Target Page
        ColPali Score
        Semantic Score
        Final Rank
    """

    rerank_lookup = {
        item["page_number"]: item
        for item in reranked_results
    }

    rows = []

    for initial_rank, item in enumerate(
        colpali_results,
        start=1,
    ):
        updated = rerank_lookup[
            item["page_number"]
        ]

        rows.append(
            [
                initial_rank,
                item["page_number"],
                round(
                    item["colpali_score"],
                    4,
                ),
                round(
                    updated["semantic_score"],
                    4,
                ),
                updated["rerank_rank"],
            ]
        )

    return rows


def run_pipeline(
    pdf_file,
    question,
    inference_mode,
    colpali_top_k,
    max_new_tokens,
):
    if pdf_file is None:
        raise gr.Error("Please upload a PDF file.")

    if not question or not question.strip():
        raise gr.Error("Please enter a question.")

    if inference_mode not in {"Single page", "Multiple pages"}:
        raise gr.Error("Please select a valid inference mode.")

    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name

    # 1. PDF -> page images + native PDF text
    page_images, page_texts = render_pdf(pdf_path)

    if not page_images:
        raise gr.Error("No pages were found in the PDF.")

    # Final number of pages passed to Qwen.
    if inference_mode == "Single page":
        final_top_k = DEFAULT_SINGLE_PAGE_TOP_K
    else:
        final_top_k = DEFAULT_MULTIPLE_PAGE_TOP_K

    final_top_k = min(final_top_k, len(page_images))

    # Retrieval must return at least as many candidates as Qwen will receive.
    requested_retrieval_k = int(colpali_top_k)
    effective_retrieval_k = min(
        max(requested_retrieval_k, final_top_k),
        len(page_images),
    )

    # 2. ColPali retrieve visually relevant pages
    colpali_results = colpali_retrieve(
        question=question.strip(),
        page_images=page_images,
        top_k=effective_retrieval_k,
    )

    # 3. Text semantic feature reranking
    reranked_results = semantic_text_rerank(
        question=question.strip(),
        candidates=colpali_results,
        page_texts=page_texts,
    )

    if not reranked_results:
        raise gr.Error("No pages remained after retrieval and reranking.")

    final_top_k = min(final_top_k, len(reranked_results))
    top_results = reranked_results[:final_top_k]
    top_images = [
        page_images[item["page_index"]]
        for item in top_results
    ]
    top_page_numbers = [
        item["page_number"]
        for item in top_results
    ]

    # 4. Qwen inference
    # Single page mode -> one image.
    # Multiple pages mode -> Top-3 images jointly.
    response = qwen_answer(
        question=question.strip(),
        images=top_images,
        page_numbers=top_page_numbers,
        max_new_tokens=int(max_new_tokens),
    )

    # 5. Visualize predicted BBOXes on the corresponding input image.
    annotated_images = []
    for image_index, image in enumerate(top_images, start=1):
        annotated_image, _ = visualize_response_bboxes(
            image=image,
            model_response=response,
            coordinate_mode="pixel",
            show_think=True,
            show_evidence=True,
            target_image_index=image_index,
        )
        annotated_images.append(annotated_image)

    ranking_table = build_ranking_table(
        colpali_results,
        reranked_results,
    )

    summary_lines = [
        f"**Inference mode:** {inference_mode}  ",
        f"**Qwen input:** Top-{final_top_k} reranked page"
        f"{'s' if final_top_k > 1 else ''} / {len(page_images)} total pages  ",
    ]
    for image_index, item in enumerate(top_results, start=1):
        summary_lines.append(
            f"**Image {image_index}:** PDF page {item['page_number']} | "
            f"ColPali {item['colpali_score']:.4f} | "
            f"Semantic {item['semantic_score']:.4f} | "
            f"Final rank {item['rerank_rank']}  "
        )
    retrieval_summary = "\n".join(summary_lines)

    # One Gallery component is used for both modes.
    # Single-page mode simply returns a one-item gallery.
    selected_gallery = [
        (image, f"Image {idx} · PDF page {page_num}")
        for idx, (image, page_num) in enumerate(
            zip(top_images, top_page_numbers), start=1
        )
    ]
    bbox_gallery = [
        (image, f"Image {idx} · PDF page {page_num}")
        for idx, (image, page_num) in enumerate(
            zip(annotated_images, top_page_numbers), start=1
        )
    ]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        selected_gallery,
        bbox_gallery,
        retrieval_summary,
        ranking_table,
        response,
    )


# ============================================================
# Gradio Web UI
# ============================================================
CSS = """
#title {
    text-align: center;
    margin-bottom: 4px;
}

#subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 20px;
}

.answer-box textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}

/* Hide Gradio's default file-upload instruction text */
.upload-box .wrap,
.upload-box .center,
.upload-box .file-preview-holder + div {
    font-size: 0 !important;
}

/* Keep the upload icon/button visible */
.upload-box button,
.upload-box svg {
    font-size: initial !important;
}
"""

with gr.Blocks(css=CSS, title="ConReg-Chat Retrieval Interface") as demo:
    gr.Markdown("# HERA-VLM: Hierarchical evidence-grounded reasoning paths within multimodal RAG", elem_id="title")
    gr.Markdown(
        "Inference stage",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="1. Upload regulatory file",
                file_types=[".pdf"],
                type="filepath",
                elem_classes=["upload-box"],
            )

            question_input = gr.Textbox(
                label="2. Input question",
                placeholder="e.g., What percentage of the agreement rate is payable?",
                lines=4,
            )

            inference_mode_input = gr.Radio(
                choices=["Single page", "Multiple pages"],
                value="Single page",
                label="3. Inference mode",
                info=(
                    "Single page sends the Top-1 reranked page to Qwen; "
                    "Multiple pages sends the Top-3 reranked pages jointly."
                ),
            )

            with gr.Accordion("Retrieval settings", open=False):
                top_k_input = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=DEFAULT_COLPALI_TOP_K,
                    step=1,
                    label="Target pages (Top-k)",
                )
                max_tokens_input = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    value=1024,
                    step=128,
                    label="Max new tokens",
                )

            run_button = gr.Button(
                "Run the framework",
                variant="primary",
            )

        #with gr.Column(scale=1):
        #    selected_page = gr.Image(
        #        label="Top-1 evidence page after reranking",
        #        type="pil",
        #    )
        #    retrieval_summary = gr.Markdown("No retrieval result yet.")
        with gr.Column(scale=1):

            with gr.Tabs():

                with gr.Tab("Retrieved pages"):
                    selected_page = gr.Gallery(
                        label="Reranked page(s) sent to Qwen",
                        columns=3,
                        rows=1,
                        object_fit="contain",
                        height="auto",
                    )

                with gr.Tab("Grounding evidence"):
                    bbox_visualization = gr.Gallery(
                        label="Predicted bounding boxes on input page(s)",
                        columns=3,
                        rows=1,
                        object_fit="contain",
                        height="auto",
                    )

                retrieval_summary = gr.Markdown(
                    "No retrieval result yet."
                )

    gr.Markdown("Retrieval and reranking trace")
    ranking_table = gr.Dataframe(
        headers=[
            "ColPali Rank",
            "Target Page",
            "ColPali Score",
            "Semantic Score",
            "Final Rank",
        ],

        datatype=[
            "number",
            "number",
            "number",
            "number",
            "number",
        ],
        interactive=False,
        wrap=True,
    )

    gr.Markdown("Response")
    answer_output = gr.Textbox(
        label="Model response",
        lines=14,
        elem_classes=["answer-box"],
    )

    run_button.click(
        fn=run_pipeline,
        inputs=[
            pdf_input,
            question_input,
            inference_mode_input,
            top_k_input,
            max_tokens_input,
        ],
        outputs=[
            selected_page,
            bbox_visualization,
            retrieval_summary,
            ranking_table,
            answer_output,
        ],
    )


if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        debug=False,
    )
