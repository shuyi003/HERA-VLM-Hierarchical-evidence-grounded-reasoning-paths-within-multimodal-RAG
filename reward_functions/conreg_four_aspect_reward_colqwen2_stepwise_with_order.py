"""Four-aspect ConReg reward for EasyR1 GRPO.

The design mirrors conreg_reward_plugin.py but exposes EasyR1's required
`compute_score(reward_input, **kwargs)` entry point.

Four aspects:
1. answer_accuracy
2. final_grounding
3. stepwise_attribution
   - step/evidence semantic faithfulness
   - bbox anti-duplication
   - hierarchical reasoning order
4. output_format
"""

from __future__ import annotations

import json
import math
import re
import string
from difflib import SequenceMatcher
from collections import Counter
from typing import Any, Literal
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

CoordinateSpace = Literal["auto", "pixel", "normalized", "qwen_1000"]



# EasyR1 reward metadata
REWARD_NAME = "conreg_four_aspect"
REWARD_TYPE = "batch"


# ============================================================
# 1. Patterns
# ============================================================

THINK_PATTERN = re.compile(
    r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL
)
ANSWER_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL
)
EVIDENCE_PATTERN = re.compile(
    r"<evidence>\s*(.*?)\s*</evidence>", re.IGNORECASE | re.DOTALL
)
FULL_FORMAT_PATTERN = re.compile(
    r"^\s*"
    r"<think>\s*.+?\s*</think>\s*"
    r"<answer>\s*.+?\s*</answer>\s*"
    r"<evidence>\s*.+?\s*</evidence>"
    r"\s*$",
    re.IGNORECASE | re.DOTALL,
)

BBOX_COORD_PATTERN = re.compile(
    r"""
    (?:
        box\s*2\s*d
        | box[_\s-]*2d
        | bbox[_\s-]*2d
        | bbox
        | bounding\s*box
        | coordinates?
    )?
    \s*(?::|=)?\s*
    [\[\(\{]\s*
    (?P<x1>-?\d+(?:\.\d+)?)\s*[,，]\s*
    (?P<y1>-?\d+(?:\.\d+)?)\s*[,，]\s*
    (?P<x2>-?\d+(?:\.\d+)?)\s*[,，]\s*
    (?P<y2>-?\d+(?:\.\d+)?)\s*
    [\]\)\}]
    """,
    re.IGNORECASE | re.VERBOSE,
)

IMAGE_INDEX_PATTERN = re.compile(
    r"""
    \bimage
    (?:\s*[_-]?\s*(?:index|id|number|no\.?))?
    \s*(?::|=|\#)?\s*
    (?P<image_index>\d+)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

OBJECT_BLOCK_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)


STEP_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Step\s*)?(\d+)[\.\):]\s*"
    r"(.*?)(?=\n\s*(?:Step\s*)?\d+[\.\):]\s*|\Z)",
    re.IGNORECASE | re.DOTALL,
)

ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)

NO_ANSWER_FORMS = {
    "no answer",
    "insufficient to answer",
    "insufficient information",
    "cannot be answered",
}


# ============================================================
# 2. General helpers
# ============================================================

def clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def extract_tag(text: str, pattern: re.Pattern) -> str:
    match = pattern.search(str(text or ""))
    return match.group(1).strip() if match else ""


def load_ground_truth(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise TypeError("Decoded ground_truth must be a JSON object.")
        return parsed
    raise TypeError("ground_truth must be a dict or JSON string.")

import numbers


def validate_easyr1_metrics(
    metrics: dict[str, Any],
) -> dict[str, float]:
    """
    Ensure that every reward metric returned to EasyR1 is a finite
    scalar number.
    """
    validated: dict[str, float] = {}
    invalid: dict[str, str] = {}

    for key, value in metrics.items():
        if isinstance(value, bool):
            validated[key] = float(value)
            continue

        if isinstance(value, numbers.Real):
            numeric_value = float(value)

            if math.isfinite(numeric_value):
                validated[key] = numeric_value
            else:
                invalid[key] = (
                    f"non-finite {numeric_value!r}"
                )
        else:
            invalid[key] = (
                f"{type(value).__name__}: {value!r}"
            )

    if invalid:
        raise TypeError(
            "EasyR1 reward metrics must all be finite scalar "
            f"numbers. Invalid fields: {invalid}"
        )

    return validated


# ============================================================
# 3. Answer and reasoning-text accuracy
# ============================================================

# The embedding model is loaded lazily and cached once per reward worker.
_SEMANTIC_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def normalize_answer(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("brick work", "brickwork")
    text = ARTICLES.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match_score(prediction: str, reference: str) -> float:
    """Normalized exact match."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(reference)
    return float(bool(pred_norm) and pred_norm == gold_norm)


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-words token F1 after answer normalization."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(reference).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _character_ngram_counts(
    text: str,
    n: int = 3,
) -> Counter[str]:
    normalized = normalize_answer(text)
    padded = f"  {normalized}  "

    if not normalized:
        return Counter()

    if len(padded) < n:
        return Counter({padded: 1})

    return Counter(
        padded[index:index + n]
        for index in range(len(padded) - n + 1)
    )


def _counter_cosine(
    left: Counter[str],
    right: Counter[str],
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    dot = sum(
        value * right.get(key, 0)
        for key, value in left.items()
    )
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm <= 0 or right_norm <= 0:
        return 0.0

    return clamp(dot / (left_norm * right_norm))


def lexical_semantic_fallback(
    prediction: str,
    reference: str,
) -> float:
    """
    Dependency-free semantic fallback.

    This is not a neural embedding score. It combines sequence similarity and
    character n-gram cosine similarity, making it more tolerant of punctuation,
    inflection, and moderate rephrasing than strict token overlap.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(reference)

    if not pred_norm and not gold_norm:
        return 1.0
    if not pred_norm or not gold_norm:
        return 0.0

    sequence_score = SequenceMatcher(
        None,
        pred_norm,
        gold_norm,
        autojunk=False,
    ).ratio()

    char_ngram_score = _counter_cosine(
        _character_ngram_counts(pred_norm),
        _character_ngram_counts(gold_norm),
    )

    return clamp(
        0.50 * sequence_score
        + 0.50 * char_ngram_score
    )


def _load_sentence_transformer(
    model_name_or_path: str,
    device: str,
    local_files_only: bool,
) -> Any:
    cache_key = (model_name_or_path, device)

    if cache_key in _SEMANTIC_MODEL_CACHE:
        return _SEMANTIC_MODEL_CACHE[cache_key]

    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(
            model_name_or_path,
            device=device,
            local_files_only=local_files_only,
        )
    except TypeError:
        # Compatibility with older sentence-transformers releases that do not
        # expose local_files_only in SentenceTransformer.__init__.
        model = SentenceTransformer(
            model_name_or_path,
            device=device,
        )

    _SEMANTIC_MODEL_CACHE[cache_key] = model
    return model


def semantic_similarity_score(
    prediction: str,
    reference: str,
    *,
    backend: str = "sentence_transformers",
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    local_files_only: bool = True,
) -> tuple[float, str]:
    """
    Compute semantic similarity in [0, 1].

    Backends:
        auto:
            Try SentenceTransformer embeddings, then fall back safely.
        sentence_transformers:
            Require SentenceTransformer embeddings; errors are propagated.
        lexical_fallback:
            Dependency-free sequence + character n-gram similarity.

    Cosine similarity from SentenceTransformer is mapped from [-1, 1] to
    [0, 1]. With normalized embeddings, ordinary similar sentences generally
    remain in the upper portion of this range.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(reference)

    if not pred_norm and not gold_norm:
        return 1.0, "empty_match"
    if not pred_norm or not gold_norm:
        return 0.0, "empty_mismatch"

    valid_backends = {
        "auto",
        "sentence_transformers",
        "lexical_fallback",
    }
    if backend not in valid_backends:
        raise ValueError(
            f"Unsupported semantic similarity backend: {backend}. "
            f"Expected one of {sorted(valid_backends)}."
        )

    if backend in {"auto", "sentence_transformers"}:
        try:
            model = _load_sentence_transformer(
                model_name_or_path=model_name_or_path,
                device=device,
                local_files_only=local_files_only,
            )
            embeddings = model.encode(
                [prediction, reference],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            cosine = float(embeddings[0] @ embeddings[1])
            mapped = clamp((cosine + 1.0) / 2.0)
            return mapped, "sentence_transformers"
        except Exception:
            if backend == "sentence_transformers":
                raise

    return lexical_semantic_fallback(
        prediction,
        reference,
    ), "lexical_fallback"


def validate_three_component_weights(
    exact_weight: float,
    token_f1_weight: float,
    semantic_weight: float,
    *,
    name: str,
) -> tuple[float, float, float]:
    weights = (
        float(exact_weight),
        float(token_f1_weight),
        float(semantic_weight),
    )

    if any(weight < 0 for weight in weights):
        raise ValueError(
            f"{name} component weights must be non-negative: {weights}"
        )

    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"{name} component weights must sum to 1.0, "
            f"received {weight_sum:.6f}"
        )

    return weights


def three_aspect_text_score(
    prediction: str,
    reference: str,
    *,
    exact_weight: float = 0.10,
    token_f1_weight: float = 0.10,
    semantic_weight: float = 0.80,
    semantic_backend: str = "sentence_transformers",
    semantic_model_name_or_path: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
    semantic_device: str = "cpu",
    semantic_local_files_only: bool = True,
) -> tuple[float, dict[str, Any]]:
    """
    Combine three complementary text-comparison aspects:

    1. normalized exact match
    2. token F1
    3. semantic similarity

    An exact match is always assigned a final text score of 1.0. Otherwise,
    the configured weighted combination is used.
    """
    exact_weight, token_f1_weight, semantic_weight = (
        validate_three_component_weights(
            exact_weight,
            token_f1_weight,
            semantic_weight,
            name="Text accuracy",
        )
    )

    exact = exact_match_score(prediction, reference)
    token = token_f1(prediction, reference)
    semantic, semantic_backend_used = semantic_similarity_score(
        prediction,
        reference,
        backend=semantic_backend,
        model_name_or_path=semantic_model_name_or_path,
        device=semantic_device,
        local_files_only=semantic_local_files_only,
    )

    if exact >= 1.0:
        combined = 1.0
    else:
        combined = clamp(
            exact_weight * exact
            + token_f1_weight * token
            + semantic_weight * semantic
        )

    return combined, {
        "exact_match": exact,
        "token_f1": token,
        "semantic_similarity": semantic,
        "semantic_backend_used": semantic_backend_used,
    }


def is_no_answer(text: str) -> bool:
    return normalize_answer(text) in {
        normalize_answer(item) for item in NO_ANSWER_FORMS
    }


def answer_accuracy_score(
    prediction: str,
    reference: str,
    gold_is_no_answer: bool,
    *,
    exact_weight: float = 0.20,
    token_f1_weight: float = 0.30,
    semantic_weight: float = 0.50,
    semantic_backend: str = "sentence_transformers",
    semantic_model_name_or_path: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
    semantic_device: str = "cpu",
    semantic_local_files_only: bool = True,
) -> tuple[float, dict[str, Any]]:
    """
    Score answer accuracy using exact match, token F1, and semantic similarity.
    """
    if gold_is_no_answer:
        score = 1.0 if is_no_answer(prediction) else 0.0
        return score, {
            "answer_exact_match": score,
            "answer_token_f1": score,
            "answer_semantic_similarity": score,
            "answer_semantic_backend": "no_answer_rule",
        }

    if is_no_answer(prediction):
        return 0.0, {
            "answer_exact_match": 0.0,
            "answer_token_f1": 0.0,
            "answer_semantic_similarity": 0.0,
            "answer_semantic_backend": "predicted_no_answer",
        }

    score, details = three_aspect_text_score(
        prediction,
        reference,
        exact_weight=exact_weight,
        token_f1_weight=token_f1_weight,
        semantic_weight=semantic_weight,
        semantic_backend=semantic_backend,
        semantic_model_name_or_path=semantic_model_name_or_path,
        semantic_device=semantic_device,
        semantic_local_files_only=semantic_local_files_only,
    )

    return score, {
        "answer_exact_match": float(details["exact_match"]),
        "answer_token_f1": float(details["token_f1"]),
        "answer_semantic_similarity": float(
            details["semantic_similarity"]
        ),
        "answer_semantic_backend": str(
            details["semantic_backend_used"]
        ),
    }


# ============================================================
# 4. Bounding-box parsing and scoring
# ============================================================

def parse_bboxes(
    text: str,
    default_image_index: int | None = None,
) -> list[dict[str, Any]]:
    """
    Parse JSON-like and plain-text bbox expressions.

    Accepted examples:
        Image index: 1, box 2d: [10, 20, 100, 200]
        Image 0 bbox [10, 20, 100, 200]
        bbox_2d: [10, 20, 100, 200]
        [10, 20, 100, 200]

    Missing image indices are preserved as None unless an explicit
    default_image_index is provided.
    """
    text = str(text or "")
    boxes: list[dict[str, Any]] = []
    matched_spans: list[tuple[int, int]] = []

    for block_match in OBJECT_BLOCK_PATTERN.finditer(text):
        block = block_match.group(0)
        bbox_match = BBOX_COORD_PATTERN.search(block)

        if bbox_match is None:
            continue

        image_match = IMAGE_INDEX_PATTERN.search(block)
        image_index = (
            int(image_match.group("image_index"))
            if image_match is not None
            else default_image_index
        )

        boxes.append(
            {
                "image_index": image_index,
                "bbox": [
                    float(bbox_match.group(name))
                    for name in ("x1", "y1", "x2", "y2")
                ],
            }
        )
        matched_spans.append(block_match.span())

    for bbox_match in BBOX_COORD_PATTERN.finditer(text):
        bbox_start, bbox_end = bbox_match.span()

        already_parsed = any(
            start <= bbox_start and bbox_end <= end
            for start, end in matched_spans
        )
        if already_parsed:
            continue

        context_start = max(0, bbox_start - 180)
        context_end = min(len(text), bbox_end + 180)
        context = text[context_start:context_end]
        relative_bbox_start = bbox_start - context_start

        image_matches = list(IMAGE_INDEX_PATTERN.finditer(context))

        if image_matches:
            closest = min(
                image_matches,
                key=lambda m: abs(m.start() - relative_bbox_start),
            )
            image_index = int(closest.group("image_index"))
        else:
            image_index = default_image_index

        boxes.append(
            {
                "image_index": image_index,
                "bbox": [
                    float(bbox_match.group(name))
                    for name in ("x1", "y1", "x2", "y2")
                ],
            }
        )

    return boxes


def normalize_bbox_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    image_index = item.get("image_index")
    if image_index is None:
        image_index = item.get("Image_index")
    if image_index is None:
        image_index = item.get("image")
    if image_index is None:
        image_index = item.get("image_id")

    bbox = item.get("bbox")
    if bbox is None:
        bbox = item.get("bbox_2d")
    if bbox is None:
        bbox = item.get("box_2d")
    if bbox is None:
        bbox = item.get("coordinates")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        parsed_index = None if image_index is None else int(image_index)
        parsed_bbox = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(value) for value in parsed_bbox):
        return None

    return {
        "image_index": parsed_index,
        "bbox": parsed_bbox,
    }


def normalize_bbox_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, dict):
        value = [value]

    if not isinstance(value, (list, tuple)):
        return []

    normalized: list[dict[str, Any]] = []

    for item in value:
        parsed = normalize_bbox_item(item)
        if parsed is not None:
            normalized.append(parsed)

    return normalized


def normalize_image_sizes(value: Any) -> dict[int, tuple[float, float]]:
    """
    Accepted forms:
        [{"image_index": 1, "width": 1700, "height": 2200}]
        {"image_index": 1, "width": 1700, "height": 2200}
        {1: [1700, 2200]}
    """
    if isinstance(value, dict):
        if value and all(
            isinstance(v, (list, tuple)) and len(v) == 2
            for v in value.values()
        ):
            result: dict[int, tuple[float, float]] = {}

            for key, pair in value.items():
                try:
                    index = int(key)
                    width = float(pair[0])
                    height = float(pair[1])
                except (TypeError, ValueError):
                    continue

                if width > 0 and height > 0:
                    result[index] = (width, height)

            return result

        value = [value]

    if not isinstance(value, (list, tuple)):
        return {}

    result: dict[int, tuple[float, float]] = {}

    for position, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue

        raw_index = item.get(
            "image_index",
            item.get("Image_index", position),
        )
        raw_width = item.get("width", item.get("image_width"))
        raw_height = item.get("height", item.get("image_height"))

        try:
            index = int(raw_index)
            width = float(raw_width)
            height = float(raw_height)
        except (TypeError, ValueError):
            continue

        if width > 0 and height > 0:
            result[index] = (width, height)

    return result


# ============================================================
# 3. Relaxed image-index handling
# ============================================================

def reconcile_image_indices(
    boxes: Any,
    image_sizes: Any,
    *,
    allow_zero_based_index: bool = True,
    assign_missing_index_for_single_image: bool = True,
) -> list[dict[str, Any]]:
    """
    Relax image-index constraints.

    Single-image sample:
      - missing index -> the only valid image index
      - 0, 1, or the valid image index are treated as equivalent

    Multi-image sample:
      - preserve valid indices
      - optionally map zero-based 0..N-1 to the sorted valid indices
      - missing indices remain None
    """
    boxes = normalize_bbox_list(boxes)
    size_map = normalize_image_sizes(image_sizes)
    valid_indices = sorted(size_map.keys())

    if not valid_indices:
        for item in boxes:
            if allow_zero_based_index and item["image_index"] == 0:
                item["image_index"] = 1
        return boxes

    if len(valid_indices) == 1:
        only_index = valid_indices[0]

        for item in boxes:
            idx = item["image_index"]

            if (
                idx is None
                and assign_missing_index_for_single_image
            ) or idx in {0, 1, only_index}:
                item["image_index"] = only_index

        return boxes

    if allow_zero_based_index:
        zero_based_to_valid = {
            position: valid_index
            for position, valid_index in enumerate(valid_indices)
        }

        for item in boxes:
            idx = item["image_index"]

            if idx not in size_map and idx in zero_based_to_valid:
                item["image_index"] = zero_based_to_valid[idx]

    return boxes


def image_indices_compatible_strict(
    pred_index: int | None,
    gold_index: int | None,
    *,
    # number_of_images: int,
    allow_missing_image_index: bool = True,
) -> bool:
    if pred_index == gold_index:
        return True

    # if number_of_images <= 1:
    #     return True

    image_count = get_image_count(image_sizes)

    if image_count <= 1:
        return True

    if allow_missing_image_index and (
        pred_index is None or gold_index is None
    ):
        return True

    return False

from typing import Any


def get_image_count(image_sizes: Any) -> int:
    """
    Infer the number of images from image_sizes.

    Supported examples:

    [
        {"image_index": 1, "width": 1700, "height": 2200},
        {"image_index": 2, "width": 1700, "height": 2200},
    ]

    or

    {
        1: {"width": 1700, "height": 2200},
        2: {"width": 1700, "height": 2200},
    }
    """
    if isinstance(image_sizes, (list, tuple)):
        return len(image_sizes)

    if isinstance(image_sizes, dict):
        # A single image may itself be represented as:
        # {"width": 1700, "height": 2200}
        if "width" in image_sizes and "height" in image_sizes:
            return 1

        return len(image_sizes)

    return 0


def normalize_image_index(
    image_index: Any,
    image_count: int,
    *,
    index_base: int = 1,
) -> int | None:
    """
    Normalize an image index.

    index_base=1:
        1, 2, ..., image_count are valid.

    index_base=0:
        0, 1, ..., image_count-1 are converted to 1-based indices.
    """
    if image_index is None:
        return None

    try:
        index = int(image_index)
    except (TypeError, ValueError):
        return None

    if index_base == 0:
        if 0 <= index < image_count:
            return index + 1
        return None

    if index_base == 1:
        if 1 <= index <= image_count:
            return index
        return None

    raise ValueError(
        f"index_base must be 0 or 1, received {index_base}."
    )


def image_indices_compatible(
    pred_image_index: int | None,
    gold_image_index: int | None,
    *,
    image_sizes: Any = None,
    allow_missing_image_index: bool = True,
    pred_index_base: int = 1,
    gold_index_base: int = 1,
) -> bool:
    """
    Check whether predicted and gold boxes refer to the same image.

    Rules
    -----
    1. For multiple images, explicit matching indices are required.
    2. For a single image, None, 0, and 1 may be reconciled when
       allow_missing_image_index=True.
    3. Optional 0-based indices can be converted to the internal
       1-based representation.
    """
    image_count = get_image_count(image_sizes)

    # If image_sizes is unavailable, only exact explicit equality is safe.
    if image_count <= 0:
        if pred_image_index is None or gold_image_index is None:
            return (
                allow_missing_image_index
                and pred_image_index is None
                and gold_image_index is None
            )

        try:
            return int(pred_image_index) == int(gold_image_index)
        except (TypeError, ValueError):
            return False

    # Single-image case can safely reconcile missing indices.
    if image_count == 1 and allow_missing_image_index:
        valid_single_indices = {None, 0, 1}

        if (
            pred_image_index in valid_single_indices
            and gold_image_index in valid_single_indices
        ):
            return True

    # For multiple images, a missing index is ambiguous.
    if pred_image_index is None or gold_image_index is None:
        return False

    pred_normalized = normalize_image_index(
        pred_image_index,
        image_count,
        index_base=pred_index_base,
    )
    gold_normalized = normalize_image_index(
        gold_image_index,
        image_count,
        index_base=gold_index_base,
    )

    if pred_normalized is None or gold_normalized is None:
        return False

    return pred_normalized == gold_normalized

# ============================================================
# 4. Coordinate-space conversion
# ============================================================

def infer_coordinate_space(
    box: list[float],
    width: float,
    height: float,
) -> CoordinateSpace:
    """
    Infer coordinate space conservatively.

    Explicit metadata is strongly preferred over auto inference.
    """
    max_abs = max(abs(value) for value in box)

    if max_abs <= 1.5:
        return "normalized"

    if max_abs <= 1000.0 and max(width, height) > 1200.0:
        return "qwen_1000"

    return "pixel"


def convert_bbox_to_normalized(
    box: list[float],
    width: float,
    height: float,
    coordinate_space: CoordinateSpace,
) -> list[float]:
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    mode = (
        infer_coordinate_space(box, width, height)
        if coordinate_space == "auto"
        else coordinate_space
    )

    x1, y1, x2, y2 = box

    if mode == "normalized":
        result = [x1, y1, x2, y2]
    elif mode == "qwen_1000":
        result = [
            x1 / 1000.0,
            y1 / 1000.0,
            x2 / 1000.0,
            y2 / 1000.0,
        ]
    elif mode == "pixel":
        result = [
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
        ]
    else:
        raise ValueError(f"Unsupported coordinate space: {mode}")

    return [float(value) for value in result]


def _resolve_image_size(
    item: dict[str, Any],
    size_map: dict[int, tuple[float, float]],
) -> tuple[float, float] | None:
    image_index = item.get("image_index")

    if image_index in size_map:
        return size_map[image_index]

    if len(size_map) == 1:
        return next(iter(size_map.values()))

    return None


def normalize_boxes_to_common_space(
    boxes: Any,
    image_sizes: Any,
    *,
    coordinate_space: CoordinateSpace = "auto",
    allow_zero_based_index: bool = True,
) -> list[dict[str, Any]]:
    """
    Convert boxes into a common normalized [0, 1] coordinate space.
    """
    size_map = normalize_image_sizes(image_sizes)

    boxes = reconcile_image_indices(
        boxes,
        image_sizes,
        allow_zero_based_index=allow_zero_based_index,
    )

    normalized: list[dict[str, Any]] = []

    for item in boxes:
        resolved_size = _resolve_image_size(item, size_map)

        if resolved_size is None:
            if coordinate_space == "normalized":
                normalized_box = list(item["bbox"])
            else:
                continue
        else:
            width, height = resolved_size
            normalized_box = convert_bbox_to_normalized(
                item["bbox"],
                width,
                height,
                coordinate_space,
            )

        normalized.append(
            {
                "image_index": item["image_index"],
                "bbox": normalized_box,
            }
        )

    return normalized


# ============================================================
# 5. IoU and matching
# ============================================================

def bbox_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(box_a) + bbox_area(box_b) - intersection

    return clamp(intersection / union) if union > 0 else 0.0

def bbox_intersection(
    box_a: list[float],
    box_b: list[float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    return (
        max(0.0, ix2 - ix1)
        * max(0.0, iy2 - iy1)
    )


def bbox_gold_coverage(
    predicted_box: list[float],
    gold_box: list[float],
) -> float:
    """
    Fraction of the gold box covered by the predicted box.

    coverage = intersection / gold area
    """
    gold_area = bbox_area(gold_box)

    if gold_area <= 0:
        return 0.0

    intersection = bbox_intersection(
        predicted_box,
        gold_box,
    )

    return clamp(intersection / gold_area)

def matched_bbox_iou(
    predicted: Any,
    gold: Any,
    image_sizes: Any,
    *,
    pred_coordinate_space: CoordinateSpace = "auto",
    gold_coordinate_space: CoordinateSpace = "pixel",
    allow_missing_image_index: bool = True,
    extra_box_penalty_weight: float = 0.03,
    max_extra_box_penalty: float = 0.15,
) -> tuple[float, dict[str, float]]:
    """
    Continuous one-to-one IoU reward.

    - Prediction and gold are first converted to normalized [0, 1].
    - Missing and zero-based image indices are handled more flexibly.
    - Extra predicted boxes receive only a small penalty.
    """
    size_map = normalize_image_sizes(image_sizes)

    predicted = normalize_boxes_to_common_space(
        predicted,
        image_sizes,
        coordinate_space=pred_coordinate_space,
    )
    gold = normalize_boxes_to_common_space(
        gold,
        image_sizes,
        coordinate_space=gold_coordinate_space,
    )

    if not predicted and not gold:
        return 1.0, {
            "grounding_mean_iou": 1.0,
            "grounding_extra_box_penalty": 0.0,
            "grounding_matched_boxes": 0.0,
            "grounding_predicted_boxes": 0.0,
            "grounding_gold_boxes": 0.0,
        }

    if not predicted or not gold:
        return 0.0, {
            "grounding_mean_iou": 0.0,
            "grounding_extra_box_penalty": 0.0,
            "grounding_matched_boxes": 0.0,
            "grounding_predicted_boxes": float(len(predicted)),
            "grounding_gold_boxes": float(len(gold)),
        }

    candidates: list[tuple[float, int, int]] = []
    number_of_images = len(size_map)

    for pred_index, pred_item in enumerate(predicted):
        for gold_index, gold_item in enumerate(gold):
            if not image_indices_compatible(
                pred_item["image_index"],
                gold_item["image_index"],
                image_sizes=image_sizes,
                # number_of_images=number_of_images,
                allow_missing_image_index=allow_missing_image_index,
                pred_index_base=1,
                gold_index_base=1,
            ):
                continue

            candidates.append(
                (
                    bbox_iou(
                        pred_item["bbox"],
                        gold_item["bbox"],
                    ),
                    pred_index,
                    gold_index,
                )
            )

    candidates.sort(key=lambda item: item[0], reverse=True)

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matched_ious: list[float] = []

    for iou, pred_index, gold_index in candidates:
        if pred_index in used_pred or gold_index in used_gold:
            continue

        used_pred.add(pred_index)
        used_gold.add(gold_index)
        matched_ious.append(iou)

    mean_iou = sum(matched_ious) / len(gold)

    extra_boxes = max(0, len(predicted) - len(gold))
    extra_penalty = min(
        max_extra_box_penalty,
        extra_box_penalty_weight * extra_boxes,
    )

    score = clamp(mean_iou - extra_penalty)

    return score, {
        "grounding_mean_iou": float(mean_iou),
        "grounding_extra_box_penalty": float(extra_penalty),
        "grounding_matched_boxes": float(len(matched_ious)),
        "grounding_predicted_boxes": float(len(predicted)),
        "grounding_gold_boxes": float(len(gold)),
    }

def matched_gold_coverage(
    predicted: Any,
    gold: Any,
    image_sizes: Any,
    *,
    pred_coordinate_space: CoordinateSpace = "pixel",
    gold_coordinate_space: CoordinateSpace = "pixel",
    allow_missing_image_index: bool = True,
) -> tuple[float, dict[str, float]]:
    """
    For every gold box, find the predicted box that covers it best.

    One predicted box is allowed to cover multiple gold boxes.
    This is suitable when the gold evidence is split into several nearby
    regions but the model returns one enclosing region.
    """
    predicted_normalized = normalize_boxes_to_common_space(
        predicted,
        image_sizes,
        coordinate_space=pred_coordinate_space,
    )

    gold_normalized = normalize_boxes_to_common_space(
        gold,
        image_sizes,
        coordinate_space=gold_coordinate_space,
    )

    if not gold_normalized:
        score = 1.0 if not predicted_normalized else 0.0
        return score, {
            "grounding_gold_coverage": score,
            "grounding_mean_gold_coverage": score,
        }

    if not predicted_normalized:
        return 0.0, {
            "grounding_gold_coverage": 0.0,
            "grounding_mean_gold_coverage": 0.0,
        }

    best_coverages = []

    for gold_item in gold_normalized:
        best_coverage = 0.0

        for pred_item in predicted_normalized:
            if not image_indices_compatible(
                pred_item["image_index"],
                gold_item["image_index"],
                image_sizes=image_sizes,
                allow_missing_image_index=allow_missing_image_index,
                pred_index_base=1,
                gold_index_base=1,
            ):
                continue

            coverage = bbox_gold_coverage(
                pred_item["bbox"],
                gold_item["bbox"],
            )

            best_coverage = max(
                best_coverage,
                coverage,
            )

        best_coverages.append(best_coverage)

    mean_coverage = (
        sum(best_coverages)
        / len(best_coverages)
    )

    return clamp(mean_coverage), {
        "grounding_gold_coverage": clamp(mean_coverage),
        "grounding_mean_gold_coverage": clamp(mean_coverage),
    }

def matched_bbox_iou_at_threshold(
    predicted: Any,
    gold: Any,
    image_sizes: Any,
    *,
    iou_threshold: float = 0.5,
    pred_coordinate_space: CoordinateSpace = "pixel",
    gold_coordinate_space: CoordinateSpace = "pixel",
    allow_missing_image_index: bool = True,
) -> tuple[float, int, int]:
    """
    IoU@threshold after coordinate normalization and relaxed index matching.
    """
    size_map = normalize_image_sizes(image_sizes)

    predicted = normalize_boxes_to_common_space(
        predicted,
        image_sizes,
        coordinate_space=pred_coordinate_space,
    )
    gold = normalize_boxes_to_common_space(
        gold,
        image_sizes,
        coordinate_space=gold_coordinate_space,
    )

    if not gold:
        return (1.0 if not predicted else 0.0), 0, 0

    if not predicted:
        return 0.0, 0, len(gold)

    candidates: list[tuple[float, int, int]] = []
    number_of_images = len(size_map)

    for pred_index, pred_item in enumerate(predicted):
        for gold_index, gold_item in enumerate(gold):
            if not image_indices_compatible(
                pred_item["image_index"],
                gold_item["image_index"],
                image_sizes=image_sizes,
            #   number_of_images=number_of_images,
                allow_missing_image_index=allow_missing_image_index,
                pred_index_base=1,
                gold_index_base=1,
            ):
                continue

            iou = bbox_iou(
                pred_item["bbox"],
                gold_item["bbox"],
            )

            if iou >= iou_threshold:
                candidates.append((iou, pred_index, gold_index))

    candidates.sort(key=lambda item: item[0], reverse=True)

    used_pred: set[int] = set()
    used_gold: set[int] = set()

    for _, pred_index, gold_index in candidates:
        if pred_index in used_pred or gold_index in used_gold:
            continue

        used_pred.add(pred_index)
        used_gold.add(gold_index)

    matched_count = len(used_gold)
    return clamp(matched_count / len(gold)), matched_count, len(gold)

def soft_iou_reward(
    iou: float,
    low_threshold: float = 0.10,
    high_threshold: float = 0.50,
) -> float:
    """
    0 reward below low_threshold.
    Linear increase between low_threshold and high_threshold.
    Full reward at or above high_threshold.
    """
    if iou <= low_threshold:
        return 0.0

    if iou >= high_threshold:
        return 1.0

    return clamp(
        (iou - low_threshold)
        / (high_threshold - low_threshold)
    )

def union_enclosing_bbox(
    boxes: Any,
) -> list[dict[str, Any]]:
    """
    Build one enclosing bbox for all gold boxes on each image.
    """
    boxes = normalize_bbox_list(boxes)

    grouped: dict[int | None, list[list[float]]] = {}

    for item in boxes:
        image_index = item.get("image_index")
        grouped.setdefault(
            image_index,
            [],
        ).append(item["bbox"])

    merged = []

    for image_index, image_boxes in grouped.items():
        x1 = min(box[0] for box in image_boxes)
        y1 = min(box[1] for box in image_boxes)
        x2 = max(box[2] for box in image_boxes)
        y2 = max(box[3] for box in image_boxes)

        merged.append(
            {
                "image_index": image_index,
                "bbox": [x1, y1, x2, y2],
            }
        )

    return merged


# ============================================================
# 6. Validity and diagnostics
# ============================================================

def normalized_bbox_validity_score(
    boxes: Any,
    image_sizes: Any,
    *,
    coordinate_space: CoordinateSpace,
) -> float:
    boxes = normalize_boxes_to_common_space(
        boxes,
        image_sizes,
        coordinate_space=coordinate_space,
    )

    if not boxes:
        return 0.0

    valid_count = 0

    for item in boxes:
        x1, y1, x2, y2 = item["bbox"]

        is_valid = (
            0.0 <= x1 < x2 <= 1.0
            and 0.0 <= y1 < y2 <= 1.0
        )
        valid_count += int(is_valid)

    return valid_count / len(boxes)


# ============================================================
# 7. Recommended dense grounding reward
# ============================================================

def final_grounding_score_strict(
    predicted_boxes: Any,
    gold_boxes: Any,
    image_sizes: Any,
    gold_is_no_answer: bool,
    *,
    pred_coordinate_space: CoordinateSpace = "auto",
    gold_coordinate_space: CoordinateSpace = "pixel",
    allow_missing_image_index: bool = True,
    continuous_iou_weight: float = 0.80,
    threshold_iou_weight: float = 0.10,
    validity_weight: float = 0.10,
    iou_threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """
    Dense ConReg grounding reward.

    Normal sample:
        0.80 * continuous matched IoU
      + 0.10 * IoU@0.5 match rate
      + 0.10 * bbox validity

    No-answer sample:
        1.0 only when no bbox is predicted.

    Prediction and gold are always compared in normalized [0, 1].
    """
    weight_sum = (
        continuous_iou_weight
        + threshold_iou_weight
        + validity_weight
    )

    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            "Grounding component weights must sum to 1.0, "
            f"received {weight_sum:.6f}."
        )

    predicted_raw = reconcile_image_indices(
        predicted_boxes,
        image_sizes,
    )
    gold_raw = reconcile_image_indices(
        gold_boxes,
        image_sizes,
    )

    if gold_is_no_answer:
        score = 1.0 if not predicted_raw else 0.0

        return score, {
            "grounding_continuous_iou": score,
            "grounding_iou_at_threshold": score,
            "grounding_validity": score,
            "grounding_parse_success": float(not predicted_raw),
            "grounding_predicted_boxes": float(len(predicted_raw)),
            "grounding_gold_boxes": float(len(gold_raw)),
        }

    continuous_score, continuous_details = matched_bbox_iou(
        predicted_raw,
        gold_raw,
        image_sizes,
        pred_coordinate_space=pred_coordinate_space,
        gold_coordinate_space=gold_coordinate_space,
        allow_missing_image_index=allow_missing_image_index,
    )

    threshold_score, matched_count, gold_count = (
        matched_bbox_iou_at_threshold(
            predicted_raw,
            gold_raw,
            image_sizes,
            iou_threshold=iou_threshold,
            pred_coordinate_space=pred_coordinate_space,
            gold_coordinate_space=gold_coordinate_space,
            allow_missing_image_index=allow_missing_image_index,
        )
    )

    validity = normalized_bbox_validity_score(
        predicted_raw,
        image_sizes,
        coordinate_space=pred_coordinate_space,
    )

    parse_success = float(bool(predicted_raw))

    score = clamp(
        continuous_iou_weight * continuous_score
        + threshold_iou_weight * threshold_score
        + validity_weight * validity
    )

    return score, {
        "grounding_continuous_iou": float(continuous_score),
        "grounding_iou_at_threshold": float(threshold_score),
        "grounding_validity": float(validity),
        "grounding_parse_success": parse_success,
        "grounding_matched_boxes_at_threshold": float(matched_count),
        "grounding_gold_boxes": float(gold_count),
        **continuous_details,
    }

def final_grounding_score(
    predicted_boxes: Any,
    gold_boxes: Any,
    image_sizes: Any,
    gold_is_no_answer: bool,
    *,
    pred_coordinate_space: CoordinateSpace = "pixel",
    gold_coordinate_space: CoordinateSpace = "pixel",
    allow_missing_image_index: bool = True,

    continuous_iou_weight: float = 0.50,
    gold_coverage_weight: float = 0.05,
    merged_gold_iou_weight: float = 0.05,
    soft_threshold_weight: float = 0.30,
    validity_weight: float = 0.10,

    soft_iou_low: float = 0.10,
    soft_iou_high: float = 0.50,
) -> tuple[float, dict[str, float]]:
    """
    Grounding reward designed for document evidence localization.

    It supports:
    - one predicted box matching multiple nearby gold regions;
    - larger predicted boxes that correctly contain gold evidence;
    - dense reward below IoU 0.5;
    - merged evidence regions.
    """
    weights = [
        continuous_iou_weight,
        gold_coverage_weight,
        merged_gold_iou_weight,
        soft_threshold_weight,
        validity_weight,
    ]

    if any(weight < 0 for weight in weights):
        raise ValueError(
            f"Grounding weights must be non-negative: {weights}"
        )

    weight_sum = sum(weights)

    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            "Grounding component weights must sum to 1.0, "
            f"received {weight_sum:.6f}."
        )

    predicted_raw = reconcile_image_indices(
        predicted_boxes,
        image_sizes,
    )

    gold_raw = reconcile_image_indices(
        gold_boxes,
        image_sizes,
    )

    if gold_is_no_answer:
        score = 1.0 if not predicted_raw else 0.0

        return score, {
            "grounding_continuous_iou": score,
            "grounding_gold_coverage": score,
            "grounding_merged_gold_iou": score,
            "grounding_soft_iou": score,
            "grounding_validity": score,
            "grounding_parse_success": float(
                not predicted_raw
            ),
            "grounding_predicted_boxes": float(
                len(predicted_raw)
            ),
            "grounding_gold_boxes": float(
                len(gold_raw)
            ),
        }

    if not predicted_raw or not gold_raw:
        return 0.0, {
            "grounding_continuous_iou": 0.0,
            "grounding_gold_coverage": 0.0,
            "grounding_merged_gold_iou": 0.0,
            "grounding_soft_iou": 0.0,
            "grounding_validity": 0.0,
            "grounding_parse_success": float(
                bool(predicted_raw)
            ),
            "grounding_predicted_boxes": float(
                len(predicted_raw)
            ),
            "grounding_gold_boxes": float(
                len(gold_raw)
            ),
        }

    # 1. Standard one-to-one continuous IoU.
    continuous_score, continuous_details = matched_bbox_iou(
        predicted_raw,
        gold_raw,
        image_sizes,
        pred_coordinate_space=pred_coordinate_space,
        gold_coordinate_space=gold_coordinate_space,
        allow_missing_image_index=allow_missing_image_index,
    )

    # 2. Every gold box may be covered by the same predicted box.
    gold_coverage_score, coverage_details = matched_gold_coverage(
        predicted_raw,
        gold_raw,
        image_sizes,
        pred_coordinate_space=pred_coordinate_space,
        gold_coordinate_space=gold_coordinate_space,
        allow_missing_image_index=allow_missing_image_index,
    )

    # 3. Merge nearby gold regions into one enclosing evidence region.
    merged_gold = union_enclosing_bbox(
        gold_raw
    )

    merged_gold_iou_score, merged_details = matched_bbox_iou(
        predicted_raw,
        merged_gold,
        image_sizes,
        pred_coordinate_space=pred_coordinate_space,
        gold_coordinate_space=gold_coordinate_space,
        allow_missing_image_index=allow_missing_image_index,
    )

    # 4. Soft threshold reward based on the best available localization IoU.
    representative_iou = max(
        continuous_score,
        merged_gold_iou_score,
    )

    soft_threshold_score = soft_iou_reward(
        representative_iou,
        low_threshold=soft_iou_low,
        high_threshold=soft_iou_high,
    )

    # 5. Coordinate validity.
    validity = normalized_bbox_validity_score(
        predicted_raw,
        image_sizes,
        coordinate_space=pred_coordinate_space,
    )

    parse_success = float(
        bool(predicted_raw)
    )

    score = clamp(
        continuous_iou_weight
        * continuous_score

        + gold_coverage_weight
        * gold_coverage_score

        + merged_gold_iou_weight
        * merged_gold_iou_score

        + soft_threshold_weight
        * soft_threshold_score

        + validity_weight
        * validity
    )

    return score, {
        "grounding_continuous_iou": float(
            continuous_score
        ),
        "grounding_gold_coverage": float(
            gold_coverage_score
        ),
        "grounding_merged_gold_iou": float(
            merged_gold_iou_score
        ),
        "grounding_soft_iou": float(
            soft_threshold_score
        ),
        "grounding_validity": float(
            validity
        ),
        "grounding_parse_success": parse_success,
        "grounding_predicted_boxes": float(
            len(predicted_raw)
        ),
        "grounding_gold_boxes": float(
            len(gold_raw)
        ),
        **continuous_details,
        **coverage_details,
        **{
            f"merged_{key}": float(value)
            for key, value in merged_details.items()
        },
    }


# ============================================================
# 5. Stepwise evidence-path reward with ColQwen2
# ============================================================

_COLQWEN2_CACHE: dict[tuple[str, str, str, bool], tuple[Any, Any]] = {}


def parse_predicted_steps(completion_text: str) -> list[dict[str, Any]]:
    think_text = extract_tag(completion_text, THINK_PATTERN)
    steps = []
    for match in STEP_PATTERN.finditer(think_text):
        raw_step = match.group(2).strip()
        step_text = BBOX_COORD_PATTERN.sub(" ", raw_step)
        step_text = IMAGE_INDEX_PATTERN.sub(" ", step_text)
        step_text = re.sub(r"\{\s*[,;]?\s*\}", " ", step_text)
        step_text = re.sub(r"\s+", " ", step_text).strip(" -,;{}")
        steps.append({
            "step_index": int(match.group(1)),
            "text": step_text,
            "bboxes": parse_bboxes(raw_step),
        })
    return steps


def _normalize_path_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(x) for x in value if isinstance(x, (str, Path))]


def resolve_reward_image_paths(
    reward_input: dict[str, Any],
    ground_truth: dict[str, Any],
) -> list[str]:
    for source in (ground_truth, reward_input):
        for key in ("image_paths", "used_images", "images", "image_path"):
            paths = _normalize_path_sequence(source.get(key))
            if paths:
                return paths
    return []


def load_reward_page_images(image_paths: list[str]) -> dict[int, Image.Image]:
    pages = {}
    for index, path in enumerate(image_paths, start=1):
        file_path = Path(path)
        if not file_path.exists():
            continue
        try:
            with Image.open(file_path) as image:
                pages[index] = image.convert("RGB").copy()
        except Exception:
            continue
    return pages


def derive_image_sizes_from_pages(
    page_images: dict[int, Image.Image],
) -> list[dict[str, Any]]:
    result = []
    for image_index, image in sorted(page_images.items()):
        width, height = image.size
        result.append({
            "image_index": image_index,
            "width": width,
            "height": height,
        })
    return result


def convert_bbox_to_pixel(
    box: list[float],
    width: float,
    height: float,
    coordinate_space: CoordinateSpace,
) -> list[float]:
    mode = (
        infer_coordinate_space(box, width, height)
        if coordinate_space == "auto"
        else coordinate_space
    )
    x1, y1, x2, y2 = box
    if mode == "pixel":
        return [float(x1), float(y1), float(x2), float(y2)]
    if mode == "normalized":
        return [
            float(x1 * width),
            float(y1 * height),
            float(x2 * width),
            float(y2 * height),
        ]
    if mode == "qwen_1000":
        return [
            float(x1 / 1000.0 * width),
            float(y1 / 1000.0 * height),
            float(x2 / 1000.0 * width),
            float(y2 / 1000.0 * height),
        ]
    raise ValueError(f"Unsupported coordinate space: {mode}")


def crop_step_bbox(
    page_image: Image.Image,
    bbox: list[float],
    coordinate_space: CoordinateSpace,
) -> Image.Image | None:
    width, height = page_image.size
    x1, y1, x2, y2 = convert_bbox_to_pixel(
        bbox, width, height, coordinate_space
    )
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return page_image.crop((
        int(math.floor(x1)),
        int(math.floor(y1)),
        int(math.ceil(x2)),
        int(math.ceil(y2)),
    ))


def _resolve_colqwen2_dtype(dtype_name: str, device: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    key = str(dtype_name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported ColQwen2 dtype: {dtype_name}")
    dtype = mapping[key]
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    return dtype


def load_colqwen2_retriever(
    model_name_or_path: str,
    *,
    device: str = "cuda",
    dtype_name: str = "bfloat16",
    load_in_4bit: bool = False,
) -> tuple[Any, Any]:
    actual_device = str(device)
    if actual_device.startswith("cuda") and not torch.cuda.is_available():
        actual_device = "cpu"

    key = (
        str(model_name_or_path),
        actual_device,
        str(dtype_name),
        bool(load_in_4bit),
    )
    if key in _COLQWEN2_CACHE:
        return _COLQWEN2_CACHE[key]

    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

    torch_dtype = _resolve_colqwen2_dtype(dtype_name, actual_device)
    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}

    if load_in_4bit:
        if actual_device == "cpu":
            raise ValueError("4-bit ColQwen2 loading requires CUDA.")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        kwargs["device_map"] = "auto"
    elif actual_device.startswith("cuda"):
        kwargs["device_map"] = "auto"

    retriever = ColQwen2ForRetrieval.from_pretrained(
        model_name_or_path, **kwargs
    ).eval()
    if not actual_device.startswith("cuda") and not load_in_4bit:
        retriever.to(actual_device)

    processor = ColQwen2Processor.from_pretrained(model_name_or_path)
    _COLQWEN2_CACHE[key] = (retriever, processor)
    return retriever, processor


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
    return batch


def _mean_pool_and_normalize(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    embeddings = embeddings.float()
    if (
        attention_mask is not None
        and embeddings.ndim == 3
        and attention_mask.ndim == 2
        and attention_mask.shape[:2] == embeddings.shape[:2]
    ):
        mask = attention_mask.to(embeddings.device).float().unsqueeze(-1)
        pooled = (
            (embeddings * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)
        )
    else:
        pooled = embeddings.mean(dim=1)
    return F.normalize(pooled, p=2, dim=-1)


@torch.inference_mode()
def colqwen2_step_similarities(
    crops: list[Image.Image],
    reasoning_texts: list[str],
    *,
    model_name_or_path: str,
    device: str,
    dtype_name: str,
    load_in_4bit: bool,
    batch_size: int = 4,
) -> list[float]:
    """
    Paper-style step similarity:
        cos(Norm(enc_img(crop(B_t))), Norm(enc_txt(r_t)))

    ColQwen2 emits multi-vector embeddings. Here they are mean-pooled and
    L2-normalized to obtain one vector for the crop and one for the text.
    """
    if len(crops) != len(reasoning_texts):
        raise ValueError("crops and reasoning_texts must have equal lengths.")
    if not crops:
        return []

    model, processor = load_colqwen2_retriever(
        model_name_or_path,
        device=device,
        dtype_name=dtype_name,
        load_in_4bit=load_in_4bit,
    )
    model_device = _model_device(model)
    batch_size = max(1, int(batch_size))
    similarities = []

    for start in range(0, len(crops), batch_size):
        crop_batch = crops[start:start + batch_size]
        text_batch = reasoning_texts[start:start + batch_size]

        image_inputs = processor(
            images=crop_batch, return_tensors="pt"
        )
        text_inputs = processor(
            text=text_batch, padding=True, return_tensors="pt"
        )

        image_inputs = _move_batch_to_device(image_inputs, model_device)
        text_inputs = _move_batch_to_device(text_inputs, model_device)

        image_embeddings = model(**image_inputs).embeddings
        text_embeddings = model(**text_inputs).embeddings

        image_mask = image_inputs.get("attention_mask")
        text_mask = text_inputs.get("attention_mask")

        image_vector = _mean_pool_and_normalize(
            image_embeddings, image_mask
        )
        text_vector = _mean_pool_and_normalize(
            text_embeddings, text_mask
        )

        cosine = (image_vector * text_vector).sum(dim=-1)
        similarities.extend(
            float(v)
            for v in cosine.detach().float().cpu().tolist()
        )

    return similarities


def reasoning_pairs_from_steps(
    predicted_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = []
    for step in predicted_steps:
        reasoning_text = str(step.get("text", "")).strip()
        if not reasoning_text:
            continue
        for box in normalize_bbox_list(step.get("bboxes", [])):
            pairs.append({
                "step_index": int(step.get("step_index", len(pairs) + 1)),
                "text": reasoning_text,
                "image_index": box.get("image_index"),
                "bbox": box["bbox"],
            })
    return pairs


def max_reasoning_bbox_overlap(
    pairs: list[dict[str, Any]],
    image_sizes: Any,
    *,
    pred_coordinate_space: CoordinateSpace,
) -> float:
    """
    I = max_{i != j} IoU(B_i, B_j).
    Boxes on different pages are treated as non-overlapping.
    """
    if len(pairs) <= 1:
        return 0.0

    size_map = normalize_image_sizes(image_sizes)
    maximum = 0.0

    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            left, right = pairs[i], pairs[j]

            if not image_indices_compatible(
                left.get("image_index"),
                right.get("image_index"),
                image_sizes=image_sizes,
                allow_missing_image_index=(get_image_count(image_sizes) <= 1),
            ):
                continue

            left_item = {
                "image_index": left.get("image_index"),
                "bbox": left["bbox"],
            }
            right_item = {
                "image_index": right.get("image_index"),
                "bbox": right["bbox"],
            }
            left_size = _resolve_image_size(left_item, size_map)
            right_size = _resolve_image_size(right_item, size_map)

            if left_size is None or right_size is None:
                overlap = bbox_iou(left["bbox"], right["bbox"])
            else:
                left_box = convert_bbox_to_normalized(
                    left["bbox"],
                    left_size[0],
                    left_size[1],
                    pred_coordinate_space,
                )
                right_box = convert_bbox_to_normalized(
                    right["bbox"],
                    right_size[0],
                    right_size[1],
                    pred_coordinate_space,
                )
                overlap = bbox_iou(left_box, right_box)

            maximum = max(maximum, float(overlap))

    return clamp(maximum)



# ============================================================
# 5A. Hierarchical reasoning-order reward
# ============================================================

def normalize_gold_steps_for_order(
    value: Any,
) -> list[dict[str, Any]]:
    """
    Normalize gold reasoning steps for order matching.

    The existing dataset already provides ordered gold_steps. We use this
    ordering as the reference hierarchy and DO NOT require a fragile
    keyword-based page/section/provision classifier.
    """
    if value is None:
        return []

    if isinstance(value, dict):
        value = [value]

    if not isinstance(value, (list, tuple)):
        return []

    normalized = []

    for position, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue

        normalized.append({
            "step_index": int(
                item.get(
                    "step_index",
                    position,
                )
            ),
            "text": str(
                item.get(
                    "text",
                    "",
                )
            ).strip(),
            "bboxes": normalize_bbox_list(
                item.get(
                    "bboxes",
                    [],
                )
            ),
        })

    return normalized


def reasoning_order_alignment_score(
    predicted_steps: list[dict[str, Any]],
    gold_steps: Any,
    image_sizes: Any,
    *,
    pred_coordinate_space: CoordinateSpace = "pixel",
    gold_coordinate_space: CoordinateSpace = "pixel",
    allow_missing_image_index: bool = True,
    semantic_backend: str = "sentence_transformers",
    semantic_model_name_or_path: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
    semantic_device: str = "cpu",
    semantic_local_files_only: bool = True,
    order_text_weight: float = 0.70,
    order_bbox_weight: float = 0.30,
) -> tuple[float, dict[str, Any]]:
    """
    Measure hierarchical reasoning order using reference-step alignment.

    Each predicted reasoning step is aligned to the gold step with the
    highest text + bbox correspondence:

        A(p_t, g_j)
            = w_text * TextSim(p_t, g_j)
            + w_bbox * IoU(p_t, g_j)

    Let m_t denote the best-matching gold-step position. Then:

        S_order =
            1/(K-1) *
            sum_t I(m_{t+1} >= m_t)

    This rewards monotonic progression through the annotated evidence path
    (e.g., page -> section -> provision) while avoiding heuristic keyword
    classification.

    Hierarchy completeness is also checked through coverage of the reference
    positions. For the common 3-step path, a perfect hierarchy should cover
    positions {1, 2, 3}.
    """
    gold_steps = normalize_gold_steps_for_order(
        gold_steps
    )

    if not predicted_steps or not gold_steps:
        return 0.0, {
            "step_order_score":
                0.0,
            "step_order_coverage":
                0.0,
            "step_order_complete":
                0.0,
            "step_order_match_confidence":
                0.0,
            "step_order_transition_count":
                0.0,
            "step_order_correct_transitions":
                0.0,
        }

    if order_text_weight < 0 or order_bbox_weight < 0:
        raise ValueError(
            "Order text/bbox weights must be non-negative."
        )

    if abs(
        order_text_weight
        + order_bbox_weight
        - 1.0
    ) > 1e-6:
        raise ValueError(
            "Order text/bbox weights must sum to 1.0."
        )

    matched_positions = []
    match_confidences = []

    for predicted_step in predicted_steps:
        candidate_scores = []

        for gold_position, gold_step in enumerate(
            gold_steps,
            start=1,
        ):
            text_score, _ = three_aspect_text_score(
                predicted_step.get(
                    "text",
                    "",
                ),
                gold_step.get(
                    "text",
                    "",
                ),
                exact_weight=0.10,
                token_f1_weight=0.10,
                semantic_weight=0.80,
                semantic_backend=semantic_backend,
                semantic_model_name_or_path=(
                    semantic_model_name_or_path
                ),
                semantic_device=semantic_device,
                semantic_local_files_only=(
                    semantic_local_files_only
                ),
            )

            bbox_score, _ = matched_bbox_iou(
                predicted_step.get(
                    "bboxes",
                    [],
                ),
                gold_step.get(
                    "bboxes",
                    [],
                ),
                image_sizes,
                pred_coordinate_space=(
                    pred_coordinate_space
                ),
                gold_coordinate_space=(
                    gold_coordinate_space
                ),
                allow_missing_image_index=(
                    allow_missing_image_index
                ),
            )

            pair_score = clamp(
                order_text_weight
                * text_score
                + order_bbox_weight
                * bbox_score
            )

            candidate_scores.append(
                (
                    pair_score,
                    gold_position,
                )
            )

        best_score, best_position = max(
            candidate_scores,
            key=lambda item: item[0],
        )

        matched_positions.append(
            int(
                best_position
            )
        )
        match_confidences.append(
            float(
                best_score
            )
        )

    if len(matched_positions) <= 1:
        monotonic_order = 1.0
        correct_transitions = 0
        total_transitions = 0
    else:
        transition_flags = [
            float(
                right >= left
            )
            for left, right in zip(
                matched_positions[:-1],
                matched_positions[1:],
            )
        ]

        correct_transitions = int(
            sum(
                transition_flags
            )
        )
        total_transitions = len(
            transition_flags
        )

        monotonic_order = (
            sum(
                transition_flags
            )
            / total_transitions
        )

    # Coverage avoids giving a full order score to paths such as [1, 1, 1].
    unique_matched = len(
        set(
            matched_positions
        )
    )

    required_positions = min(
        len(
            gold_steps
        ),
        len(
            predicted_steps
        ),
    )

    order_coverage = (
        unique_matched
        / required_positions
        if required_positions > 0
        else 0.0
    )

    # Complete hierarchy means all reference levels are visited when the
    # predicted path is long enough to express them.
    required_gold_positions = set(
        range(
            1,
            len(
                gold_steps
            ) + 1,
        )
    )

    hierarchy_complete = float(
        required_gold_positions.issubset(
            set(
                matched_positions
            )
        )
    )

    # Keep the score partially dense:
    # 70% monotonic progression + 30% hierarchical coverage.
    order_score = clamp(
        0.70
        * monotonic_order
        + 0.30
        * order_coverage
    )

    mean_confidence = (
        sum(
            match_confidences
        )
        / len(
            match_confidences
        )
        if match_confidences
        else 0.0
    )

    return order_score, {
        "step_order_score":
            float(
                order_score
            ),
        "step_order_monotonic":
            float(
                monotonic_order
            ),
        "step_order_coverage":
            float(
                order_coverage
            ),
        "step_order_complete":
            float(
                hierarchy_complete
            ),
        "step_order_match_confidence":
            float(
                mean_confidence
            ),
        "step_order_transition_count":
            float(
                total_transitions
            ),
        "step_order_correct_transitions":
            float(
                correct_transitions
            ),
    }


def stepwise_attribution_score(
    predicted_steps: list[dict[str, Any]],
    page_images: dict[int, Image.Image],
    image_sizes: Any,
    answer_accuracy: float,
    gold_is_no_answer: bool,
    gold_steps: Any = None,
    *,
    pred_coordinate_space: CoordinateSpace = "pixel",
    gold_coordinate_space: CoordinateSpace = "pixel",
    allow_missing_image_index: bool = True,
    similarity_threshold: float = 0.45,
    overlap_threshold: float = 0.85,
    accuracy_gate_threshold: float = 0.50,
    order_text_weight: float = 0.70,
    order_bbox_weight: float = 0.30,
    semantic_backend: str = "sentence_transformers",
    semantic_model_name_or_path: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
    semantic_device: str = "cpu",
    semantic_local_files_only: bool = True,
    colqwen2_model_name_or_path: str = "vidore/colqwen2-v1.0-hf",
    colqwen2_device: str = "cuda",
    colqwen2_dtype: str = "bfloat16",
    colqwen2_load_in_4bit: bool = False,
    colqwen2_batch_size: int = 4,
) -> tuple[float, dict[str, Any]]:
    """
    Revised evidence-path reward:

        S = min_k cos(e_img^(k), e_txt^(k))
        I = max_{i != j} IoU(B_i, B_j)

        R_step =
            [ I(S >= tau)
              + I(I <= delta)
              + S_order ] / 3
            * I(R_acc >= epsilon)

    This reward does not compare the intermediate reasoning text with gold
    reasoning text. It instead checks whether the model's own reasoning is
    semantically supported by the visual region it cites.
    """
    pairs = reasoning_pairs_from_steps(predicted_steps)

    if gold_is_no_answer:
        score = float(len(pairs) == 0)
        return score, {
            "step_pair_count": float(len(pairs)),
            "step_min_similarity": score,
            "step_mean_similarity": score,
            "step_max_pairwise_iou": 0.0,
            "step_similarity_pass": score,
            "step_overlap_pass": score,
            "step_accuracy_gate": score,
            "step_crop_success_rate": score,
            "step_image_available": float(bool(page_images)),
            "step_order_score": score,
            "step_order_monotonic": score,
            "step_order_coverage": score,
            "step_order_complete": score,
            "step_order_match_confidence": score,
            "step_order_transition_count": 0.0,
            "step_order_correct_transitions": 0.0,
            "step_semantic_backend": "colqwen2_no_answer_rule",
        }

    if not pairs:
        return 0.0, {
            "step_pair_count": 0.0,
            "step_min_similarity": 0.0,
            "step_mean_similarity": 0.0,
            "step_max_pairwise_iou": 1.0,
            "step_similarity_pass": 0.0,
            "step_overlap_pass": 0.0,
            "step_accuracy_gate": float(
                answer_accuracy >= accuracy_gate_threshold
            ),
            "step_crop_success_rate": 0.0,
            "step_image_available": float(bool(page_images)),
            "step_order_score": 0.0,
            "step_order_monotonic": 0.0,
            "step_order_coverage": 0.0,
            "step_order_complete": 0.0,
            "step_order_match_confidence": 0.0,
            "step_order_transition_count": 0.0,
            "step_order_correct_transitions": 0.0,
            "step_semantic_backend": "colqwen2_not_evaluated",
        }

    image_count = len(page_images)
    crops, texts, valid_pairs = [], [], []

    for pair in pairs:
        image_index = pair.get("image_index")
        if image_count == 1 and image_index in {None, 0, 1}:
            image_index = next(iter(page_images.keys()))
        if image_index is None:
            continue
        try:
            image_index = int(image_index)
        except (TypeError, ValueError):
            continue
        page = page_images.get(image_index)
        if page is None:
            continue

        crop = crop_step_bbox(
            page,
            pair["bbox"],
            pred_coordinate_space,
        )
        if crop is None:
            continue

        crops.append(crop)
        texts.append(pair["text"])
        valid_pairs.append({**pair, "image_index": image_index})

    crop_success_rate = len(valid_pairs) / len(pairs)
    all_pairs_available = (
        len(valid_pairs) == len(pairs)
        and len(valid_pairs) > 0
    )

    similarities = []
    if all_pairs_available:
        similarities = colqwen2_step_similarities(
            crops,
            texts,
            model_name_or_path=colqwen2_model_name_or_path,
            device=colqwen2_device,
            dtype_name=colqwen2_dtype,
            load_in_4bit=colqwen2_load_in_4bit,
            batch_size=colqwen2_batch_size,
        )

    if all_pairs_available and len(similarities) == len(pairs):
        minimum_similarity = min(similarities)
        mean_similarity = sum(similarities) / len(similarities)
        similarity_pass = float(
            minimum_similarity >= float(similarity_threshold)
        )
    else:
        minimum_similarity = 0.0
        mean_similarity = 0.0
        similarity_pass = 0.0

    max_overlap = max_reasoning_bbox_overlap(
        pairs,
        image_sizes,
        pred_coordinate_space=pred_coordinate_space,
    )
    overlap_pass = float(
        max_overlap <= float(overlap_threshold)
    )
    accuracy_gate = float(
        float(answer_accuracy) >= float(accuracy_gate_threshold)
    )

    order_score, order_details = (
        reasoning_order_alignment_score(
            predicted_steps,
            gold_steps,
            image_sizes,
            pred_coordinate_space=(
                pred_coordinate_space
            ),
            gold_coordinate_space=(
                gold_coordinate_space
            ),
            allow_missing_image_index=(
                allow_missing_image_index
            ),
            semantic_backend=(
                semantic_backend
            ),
            semantic_model_name_or_path=(
                semantic_model_name_or_path
            ),
            semantic_device=(
                semantic_device
            ),
            semantic_local_files_only=(
                semantic_local_files_only
            ),
            order_text_weight=(
                order_text_weight
            ),
            order_bbox_weight=(
                order_bbox_weight
            ),
        )
    )

    score = (
        (
            similarity_pass
            + overlap_pass
            + order_score
        )
        / 3.0
        * accuracy_gate
    )

    for crop in crops:
        try:
            crop.close()
        except Exception:
            pass

    return clamp(score), {
        "step_pair_count": float(len(pairs)),
        "step_min_similarity": float(minimum_similarity),
        "step_mean_similarity": float(mean_similarity),
        "step_max_pairwise_iou": float(max_overlap),
        "step_similarity_pass": float(similarity_pass),
        "step_overlap_pass": float(overlap_pass),
        "step_accuracy_gate": float(accuracy_gate),
        "step_crop_success_rate": float(crop_success_rate),
        "step_image_available": float(bool(page_images)),
        **{
            key: float(value)
            for key, value in order_details.items()
            if isinstance(value, (int, float, bool))
        },
        "step_semantic_backend": "colqwen2_pooled_cosine",
    }


# ============================================================
# 6. Binary output-format reward
# ============================================================

def output_format_score(text: str, gold_is_no_answer: bool) -> float:
    """
    Binary format reward:
        correct format -> 1
        incorrect format -> 0

    Required schema:
        <think>...</think>
        <answer>...</answer>
        <evidence>...</evidence>

    For answerable samples, reasoning and evidence must contain parseable
    bbox attributions. For no-answer samples, bbox omission is allowed.
    """
    text = str(text or "")

    if not FULL_FORMAT_PATTERN.fullmatch(text):
        return 0.0

    if not (
        len(THINK_PATTERN.findall(text)) == 1
        and len(ANSWER_PATTERN.findall(text)) == 1
        and len(EVIDENCE_PATTERN.findall(text)) == 1
    ):
        return 0.0

    think_text = extract_tag(text, THINK_PATTERN)
    answer_text = extract_tag(text, ANSWER_PATTERN)
    evidence_text = extract_tag(text, EVIDENCE_PATTERN)

    if not (think_text and answer_text and evidence_text):
        return 0.0

    if not parse_predicted_steps(text):
        return 0.0

    if not gold_is_no_answer:
        if not parse_bboxes(think_text):
            return 0.0
        if not parse_bboxes(evidence_text):
            return 0.0

    return 1.0


# ============================================================
# 7. EasyR1 batch entry point
# ============================================================

def _score_one(
    reward_input: dict[str, Any],
    answer_weight: float,
    grounding_weight: float,
    stepwise_weight: float,
    format_weight: float,
    pred_coordinate_space: CoordinateSpace,
    gold_coordinate_space: CoordinateSpace,
    allow_missing_image_index: bool,
    #grounding_continuous_iou_weight: float,
    #grounding_threshold_iou_weight: float,
    #grounding_validity_weight: float,
    #grounding_iou_threshold: float,
    grounding_continuous_iou_weight: float,
    grounding_gold_coverage_weight: float,
    grounding_merged_gold_iou_weight: float,
    grounding_soft_threshold_weight: float,
    grounding_validity_weight: float,
    grounding_soft_iou_low: float,
    grounding_soft_iou_high: float,
    answer_exact_weight: float,
    answer_token_f1_weight: float,
    answer_semantic_weight: float,
    semantic_backend: str,
    semantic_model_name_or_path: str,
    semantic_device: str,
    semantic_local_files_only: bool,
    step_similarity_threshold: float,
    step_overlap_threshold: float,
    step_accuracy_gate_threshold: float,
    step_order_text_weight: float,
    step_order_bbox_weight: float,
    colqwen2_model_name_or_path: str,
    colqwen2_device: str,
    colqwen2_dtype: str,
    colqwen2_load_in_4bit: bool,
    colqwen2_batch_size: int,
) -> dict[str, Any]:
    """Score one rollout item supplied by EasyR1."""
    if not isinstance(reward_input, dict):
        raise TypeError(
            "Each reward input must be a dictionary, "
            f"but received {type(reward_input).__name__}."
        )

    response = str(reward_input.get("response", "") or "")
    ground_truth = reward_input.get("ground_truth")

    if ground_truth is None:
        raise KeyError("Reward input is missing the 'ground_truth' field.")

    gt = load_ground_truth(ground_truth)

    gold_answer = str(gt.get("gold_answer", "") or "")
    gold_evidence = gt.get("gold_evidence", [])
    gold_steps = gt.get("gold_steps", [])
    image_sizes = gt.get("image_sizes", [])
    gold_is_no_answer = bool(gt.get("is_no_answer", False))

    image_paths = resolve_reward_image_paths(
        reward_input,
        gt,
    )
    page_images = load_reward_page_images(
        image_paths
    )

    if (
        not normalize_image_sizes(image_sizes)
        and page_images
    ):
        image_sizes = derive_image_sizes_from_pages(
            page_images
        )

    predicted_answer = extract_tag(response, ANSWER_PATTERN)
    evidence_text = extract_tag(response, EVIDENCE_PATTERN)
    predicted_evidence = parse_bboxes(evidence_text)
    predicted_steps = parse_predicted_steps(response)

    answer_score, answer_details = answer_accuracy_score(
        predicted_answer,
        gold_answer,
        gold_is_no_answer,
        exact_weight=answer_exact_weight,
        token_f1_weight=answer_token_f1_weight,
        semantic_weight=answer_semantic_weight,
        semantic_backend=semantic_backend,
        semantic_model_name_or_path=semantic_model_name_or_path,
        semantic_device=semantic_device,
        semantic_local_files_only=semantic_local_files_only,
    )
    grounding_score, grounding_details = final_grounding_score(
        predicted_boxes=predicted_evidence,
        gold_boxes=gold_evidence,
        image_sizes=image_sizes,
        gold_is_no_answer=gold_is_no_answer,
        pred_coordinate_space=gt.get(
            "pred_coordinate_space",
            pred_coordinate_space,
        ),
        gold_coordinate_space=gt.get(
            "gold_coordinate_space",
            gold_coordinate_space,
        ),
        allow_missing_image_index=allow_missing_image_index,
        # with strict IoU
        # continuous_iou_weight=grounding_continuous_iou_weight,
        # threshold_iou_weight=grounding_threshold_iou_weight,
        # validity_weight=grounding_validity_weight,
        # iou_threshold=grounding_iou_threshold,
        # Revised grounding components
        continuous_iou_weight=grounding_continuous_iou_weight,
        gold_coverage_weight=grounding_gold_coverage_weight,
        merged_gold_iou_weight=grounding_merged_gold_iou_weight,
        soft_threshold_weight=grounding_soft_threshold_weight,
        validity_weight=grounding_validity_weight,

        # Soft IoU range
        soft_iou_low=grounding_soft_iou_low,
        soft_iou_high=grounding_soft_iou_high,
    )

    stepwise_score, stepwise_details = stepwise_attribution_score(
        predicted_steps=predicted_steps,
        page_images=page_images,
        image_sizes=image_sizes,
        answer_accuracy=answer_score,
        gold_is_no_answer=gold_is_no_answer,
        gold_steps=gold_steps,
        pred_coordinate_space=gt.get(
            "pred_coordinate_space",
            pred_coordinate_space,
        ),
        gold_coordinate_space=gt.get(
            "gold_coordinate_space",
            gold_coordinate_space,
        ),
        allow_missing_image_index=allow_missing_image_index,
        similarity_threshold=step_similarity_threshold,
        overlap_threshold=step_overlap_threshold,
        accuracy_gate_threshold=step_accuracy_gate_threshold,
        order_text_weight=step_order_text_weight,
        order_bbox_weight=step_order_bbox_weight,
        semantic_backend=semantic_backend,
        semantic_model_name_or_path=semantic_model_name_or_path,
        semantic_device=semantic_device,
        semantic_local_files_only=semantic_local_files_only,
        colqwen2_model_name_or_path=colqwen2_model_name_or_path,
        colqwen2_device=colqwen2_device,
        colqwen2_dtype=colqwen2_dtype,
        colqwen2_load_in_4bit=colqwen2_load_in_4bit,
        colqwen2_batch_size=colqwen2_batch_size,
    )

    format_score = output_format_score(response, gold_is_no_answer)

    # Each aspect remains in [0, 1]; unit weights give an overall score in [0, 4].
    overall = (
        answer_weight * answer_score
        + grounding_weight * grounding_score
        + stepwise_weight * stepwise_score
        + format_weight * format_score
    )

    result: dict[str, float] = {
        "overall": float(overall),
        "answer_accuracy": float(answer_score),
        "final_grounding": float(grounding_score),
        "stepwise_attribution": float(stepwise_score),
        "evidence_path": float(stepwise_score),
        "output_format": float(format_score),

        "answer_exact_match": float(
        answer_details["answer_exact_match"]
        ),
        "answer_token_f1": float(
        answer_details["answer_token_f1"]
        ),
        "answer_semantic_similarity": float(
        answer_details["answer_semantic_similarity"]
        ),
        }

    # Add only numeric grounding diagnostics.
    for key, value in grounding_details.items():
        if isinstance(value, (int, float, bool)):
            result[key] = float(value)

    # Add only numeric stepwise diagnostics.
    for key, value in stepwise_details.items():
        if isinstance(value, (int, float, bool)):
            result[key] = float(value)

    # Encode backend labels as numeric indicators rather than strings.
    answer_backend = str(
        answer_details.get(
            "answer_semantic_backend",
            "unknown",
        )
    )

    step_backend = str(
        stepwise_details.get(
            "step_semantic_backend",
            "unknown",
        )
    )

    result["answer_backend_sentence_transformers"] = float(
        answer_backend == "sentence_transformers"
    )
    result["answer_backend_lexical_fallback"] = float(
    answer_backend == "lexical_fallback"
    )
    result["answer_backend_no_answer_rule"] = float(
        answer_backend == "no_answer_rule"
    )

    result["step_backend_colqwen2_pooled_cosine"] = float(
        step_backend == "colqwen2_pooled_cosine"
    )
    result["step_backend_colqwen2_no_answer_rule"] = float(
        step_backend == "colqwen2_no_answer_rule"
    )
    result["step_backend_colqwen2_not_evaluated"] = float(
        step_backend == "colqwen2_not_evaluated"
    )

    for page_image in page_images.values():
        try:
            page_image.close()
        except Exception:
            pass

    return validate_easyr1_metrics(result)

    #return {
    #    "overall": float(overall),
    #    "answer_accuracy": float(answer_score),
    #    "final_grounding": float(grounding_score),
    #    "stepwise_attribution": float(stepwise_score),
    #    "output_format": float(format_score),
    #    "answer_exact_match": float(
    #        answer_details["answer_exact_match"]
    #    ),
    #    "answer_token_f1": float(
    #        answer_details["answer_token_f1"]
    #    ),
    #    "answer_semantic_similarity": float(
    #        answer_details["answer_semantic_similarity"]
    #    ),
    #    "answer_semantic_backend": str(
    #        answer_details["answer_semantic_backend"]
    #    ),
    #    **{
    #        key: float(value)
    #        for key, value in grounding_details.items()
    #    },
    #    **{
    #        key: (
    #            float(value)
    #            if isinstance(value, (int, float))
    #            else str(value)
    #        )
    #        for key, value in stepwise_details.items()
    #    },
    #}


def compute_score(
    reward_inputs: list[dict[str, Any]],
    answer_weight: float = 1.0,
    grounding_weight: float = 1.0,
    stepwise_weight: float = 1.0,
    format_weight: float = 1.0,
    require_valid_weight_sum: bool = False,
    pred_coordinate_space: CoordinateSpace = "pixel",
    gold_coordinate_space: CoordinateSpace = "pixel",
    allow_missing_image_index: bool = True,
    #grounding_continuous_iou_weight: float = 0.80,
    #grounding_threshold_iou_weight: float = 0.10,
    #grounding_validity_weight: float = 0.10,
    #grounding_iou_threshold: float = 0.50,
    # Revised grounding reward weights
    grounding_continuous_iou_weight: float = 0.50,
    grounding_gold_coverage_weight: float = 0.05,
    grounding_merged_gold_iou_weight: float = 0.05,
    grounding_soft_threshold_weight: float = 0.30,
    grounding_validity_weight: float = 0.10,
    grounding_soft_iou_low: float = 0.10,
    grounding_soft_iou_high: float = 0.50,
    answer_exact_weight: float = 0.10,
    answer_token_f1_weight: float = 0.10,
    answer_semantic_weight: float = 0.80,
    semantic_backend: str = "sentence_transformers",
    semantic_model_name_or_path: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
    semantic_device: str = "cpu",
    semantic_local_files_only: bool = True,

    # Revised evidence-path reward:
    # R_step = [I(S >= tau) + I(I <= delta) + S_order] / 3
    #          * I(R_acc >= epsilon)
    step_similarity_threshold: float = 0.45,
    step_overlap_threshold: float = 0.50,
    step_accuracy_gate_threshold: float = 0.50,

    # Order matching:
    # predicted step -> best aligned gold step using text + bbox
    step_order_text_weight: float = 0.70,
    step_order_bbox_weight: float = 0.30,

    # ColQwen2 retriever
    colqwen2_model_name_or_path: str = "vidore/colqwen2-v1.0-hf",
    colqwen2_device: str = "cuda",
    colqwen2_dtype: str = "bfloat16",
    colqwen2_load_in_4bit: bool = False,
    colqwen2_batch_size: int = 4,
) -> list[dict[str, Any]]:
    """Compute four-aspect rewards for an EasyR1 rollout batch.

    The four aspect weights default to 1.0, giving an unnormalized sum in
    [0, 4]. Set require_valid_weight_sum=True only for weights summing to 1.

    EasyR1's batch reward manager calls this function once with a list of
    dictionaries. Each dictionary contains response, response_length, and
    ground_truth. The function must return one score dictionary per item.
    """
    weights = [
        float(answer_weight),
        float(grounding_weight),
        float(stepwise_weight),
        float(format_weight),
    ]

    if any(weight < 0 for weight in weights):
        raise ValueError(f"Reward weights must be non-negative: {weights}")

    valid_spaces = {"auto", "pixel", "normalized", "qwen_1000"}
    if pred_coordinate_space not in valid_spaces:
        raise ValueError(
            f"Unsupported pred_coordinate_space: {pred_coordinate_space}"
        )
    if gold_coordinate_space not in valid_spaces:
        raise ValueError(
            f"Unsupported gold_coordinate_space: {gold_coordinate_space}"
        )

    # Strict grounding
    # grounding_component_sum = (
    #    float(grounding_continuous_iou_weight)
    #    + float(grounding_threshold_iou_weight)
    #    + float(grounding_validity_weight)
    #)
    #if abs(grounding_component_sum - 1.0) > 1e-6:
    #    raise ValueError(
    #        "Grounding component weights must sum to 1.0, "
    #        f"received {grounding_component_sum:.6f}"
    #    )
    grounding_component_weights = [
    float(grounding_continuous_iou_weight),
    float(grounding_gold_coverage_weight),
    float(grounding_merged_gold_iou_weight),
    float(grounding_soft_threshold_weight),
    float(grounding_validity_weight),
    ]

    if any(weight < 0 for weight in grounding_component_weights):
       raise ValueError(
        "Grounding component weights must be non-negative: "
        f"{grounding_component_weights}"
    )

    grounding_component_sum = sum(grounding_component_weights)

    if abs(grounding_component_sum - 1.0) > 1e-6:
       raise ValueError(
        "Grounding component weights must sum to 1.0, "
        f"received {grounding_component_sum:.6f}"
    )

    if not (
      0.0
       <= grounding_soft_iou_low
       < grounding_soft_iou_high
    <= 1.0
    ):
        raise ValueError(
             "Expected "
            "0 <= grounding_soft_iou_low "
        "< grounding_soft_iou_high <= 1."
    )

    validate_three_component_weights(
        answer_exact_weight,
        answer_token_f1_weight,
        answer_semantic_weight,
        name="Answer accuracy",
    )
    if not (-1.0 <= float(step_similarity_threshold) <= 1.0):
        raise ValueError(
            "step_similarity_threshold must be in [-1, 1]."
        )
    if not (0.0 <= float(step_overlap_threshold) <= 1.0):
        raise ValueError(
            "step_overlap_threshold must be in [0, 1]."
        )
    if not (0.0 <= float(step_accuracy_gate_threshold) <= 1.0):
        raise ValueError(
            "step_accuracy_gate_threshold must be in [0, 1]."
        )

    if (
        step_order_text_weight < 0
        or step_order_bbox_weight < 0
    ):
        raise ValueError(
            "step_order_text_weight and step_order_bbox_weight "
            "must be non-negative."
        )

    if abs(
        float(step_order_text_weight)
        + float(step_order_bbox_weight)
        - 1.0
    ) > 1e-6:
        raise ValueError(
            "step_order_text_weight and step_order_bbox_weight "
            "must sum to 1.0."
        )

    weight_sum = sum(weights)
    if require_valid_weight_sum and abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Reward weights must sum to 1.0, received {weight_sum:.6f}"
        )

    if not isinstance(reward_inputs, list):
        raise TypeError(
            "EasyR1 batch reward expects a list of reward-input dictionaries, "
            f"but received {type(reward_inputs).__name__}."
        )

    scores: list[dict[str, Any]] = []

    for index, reward_input in enumerate(reward_inputs):
        try:
            score = _score_one(
                reward_input=reward_input,
                answer_weight=weights[0],
                grounding_weight=weights[1],
                stepwise_weight=weights[2],
                format_weight=weights[3],
                pred_coordinate_space=pred_coordinate_space,
                gold_coordinate_space=gold_coordinate_space,
                allow_missing_image_index=allow_missing_image_index,
                #grounding_continuous_iou_weight=float(
                #    grounding_continuous_iou_weight
                #),
                #grounding_threshold_iou_weight=float(
                #   grounding_threshold_iou_weight
                #),
                #grounding_validity_weight=float(
                #    grounding_validity_weight
                #),
                #grounding_iou_threshold=float(
                #    grounding_iou_threshold
                #),
                grounding_continuous_iou_weight=float(
                grounding_continuous_iou_weight
                ),

                grounding_gold_coverage_weight=float(
                grounding_gold_coverage_weight
                ),

                grounding_merged_gold_iou_weight=float(
                grounding_merged_gold_iou_weight
                ),

                grounding_soft_threshold_weight=float(
                grounding_soft_threshold_weight
                ),

                grounding_validity_weight=float(
                grounding_validity_weight
                ),

                grounding_soft_iou_low=float(
                grounding_soft_iou_low
                ),

                grounding_soft_iou_high=float(
                grounding_soft_iou_high
                ),
                answer_exact_weight=float(answer_exact_weight),
                answer_token_f1_weight=float(answer_token_f1_weight),
                answer_semantic_weight=float(
                    answer_semantic_weight
                ),
                semantic_backend=str(
                    semantic_backend
                ),
                semantic_model_name_or_path=str(
                    semantic_model_name_or_path
                ),
                semantic_device=str(
                    semantic_device
                ),
                semantic_local_files_only=bool(
                    semantic_local_files_only
                ),
                step_similarity_threshold=float(
                    step_similarity_threshold
                ),
                step_overlap_threshold=float(
                    step_overlap_threshold
                ),
                step_accuracy_gate_threshold=float(
                    step_accuracy_gate_threshold
                ),
                step_order_text_weight=float(
                    step_order_text_weight
                ),
                step_order_bbox_weight=float(
                    step_order_bbox_weight
                ),
                colqwen2_model_name_or_path=str(
                    colqwen2_model_name_or_path
                ),
                colqwen2_device=str(
                    colqwen2_device
                ),
                colqwen2_dtype=str(
                    colqwen2_dtype
                ),
                colqwen2_load_in_4bit=bool(
                    colqwen2_load_in_4bit
                ),
                colqwen2_batch_size=int(
                    colqwen2_batch_size
                ),
            )
        except Exception as exc:
            # Keep the Ray worker alive if one sample is malformed.
            print(
                f"[ConReg reward] Failed to score item {index}: "
                f"{type(exc).__name__}: {exc}"
            )
            score = {
                "overall": 0.0,
                "answer_accuracy": 0.0,
                "final_grounding": 0.0,
                "stepwise_attribution": 0.0,
                "evidence_path": 0.0,
                "output_format": 0.0,
                "reward_error": 1.0,
            }
        else:
            score["reward_error"] = 0.0

        scores.append(score)

    return scores

def main(
    reward_inputs: list[dict[str, Any]],
    **kwargs,
):
    return compute_score(
        reward_inputs,
        **kwargs,
    )