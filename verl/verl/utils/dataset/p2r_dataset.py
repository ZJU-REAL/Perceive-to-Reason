"""P2R (Perceive-to-Reason) Dataset with role-based prompt construction.

Extends RLHFDataset to dynamically build perceiver or reasoner prompts
based on the ``role`` parameter in data config. When role is ``reasoner``,
calls a frozen perceiver vLLM service to get bbox predictions, post-processes
the image (highlight + crop), and builds the reasoner prompt with the correct
number of ``<image>`` tags.
"""

import io
import os
import re
import json
import base64
import logging

import requests
import torch
from io import BytesIO
from PIL import Image, ImageDraw

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_IMAGE_EVIDENCE = 3
CROP_PIXELS_DIVISOR = 1


def get_crop_max_pixels(max_pixels: int | None) -> int | None:
    if max_pixels is None:
        return None
    return max_pixels // CROP_PIXELS_DIVISOR

# ─── Perceiver templates ─────────────────────────────────────────────────────────

PERCEIVE_PRE_PROMPT = "Question: {question}\n"
PERCEIVE_IMAGE_POST_PROMPT = (
    "Locate the key visual evidence in the image for answering this question. "
    "Report bbox coordinates in JSON format only: "
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "description"}] '
    "If the question requires the entire image, return an empty list []."
)

# ─── Reasoner templates ──────────────────────────────────────────────────────────

REASON_PRE_PROMPT = (
    "Question: {question}\n"
    "The key visual regions have been highlighted and cropped for you. "
    "Think step by step.\n"
)
REASON_FALLBACK_PRE_PROMPT = (
    "Question: {question}\n"
    "Think step by step.\n"
)
ANSWER_FORM_POST_PROMPTS = {
    "mca": (
        "Provide your detailed reasoning between the <think> </think> tags, "
        "and then answer the question with the option's letter from the given choices "
        "(e.g., A, B, etc.) within the <answer> </answer> tags.\n"
    ),
    "na": (
        "Provide your detailed reasoning between the <think> </think> tags, "
        "and then answer the question using a numerical value (e.g., 42 or 3.1) "
        "within the <answer> </answer> tags.\n"
    ),
    "free_form": (
        "Provide your detailed reasoning between the <think> </think> tags, "
        "and then provide your text answer within the <answer> </answer> tags.\n"
    ),
}

# ─── Bbox utilities (mirrored from evaluation/p2r_utils.py) ──────────────────────


def _is_valid_bbox(bbox):
    coords = bbox.get("bbox_2d", [])
    if len(coords) != 4:
        return False
    x1, y1, x2, y2 = coords
    if not all(isinstance(c, (int, float)) for c in coords):
        return False
    if x1 >= x2 or y1 >= y2:
        return False
    return True


def parse_bboxes(text):
    """Parse bbox JSON from model output. Coords in 0-1000 normalized format."""
    if not text:
        return []
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return [data] if _is_valid_bbox(data) else []
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and _is_valid_bbox(d)]
    except json.JSONDecodeError:
        pass
    pattern = r'\{[^{}]*"bbox_2d"\s*:\s*\[[^\]]+\][^{}]*\}'
    results = []
    for m in re.findall(pattern, text):
        try:
            parsed = json.loads(m)
            if _is_valid_bbox(parsed):
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    return results


def draw_bounding_boxes(image, bboxes):
    """Draw bounding boxes on a copy of the image. Coords in 0-1000 normalized format."""
    img = image.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    colors = ["red", "green", "blue", "yellow", "orange", "purple", "cyan", "magenta"]
    line_width = max(3, int(0.003 * width))
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        coords = bbox["bbox_2d"]
        abs_x1 = int(coords[0] / 1000 * width)
        abs_y1 = int(coords[1] / 1000 * height)
        abs_x2 = int(coords[2] / 1000 * width)
        abs_y2 = int(coords[3] / 1000 * height)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=line_width)
    return img


def crop_region(image, bbox_2d, padding=(0.1, 0.1)):
    """Crop a region from the image with padding. Coords in 0-1000 normalized format."""
    width, height = image.size
    x1, y1, x2, y2 = [c / 1000 for c in bbox_2d]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    pad_x = min(padding[0], 600.0 / width)
    pad_y = min(padding[1], 600.0 / height)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(1, x2 + pad_x)
    y2 = min(1, y2 + pad_y)
    return image.crop((int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)))


# ─── Image helpers ────────────────────────────────────────────────────────────────

_IMAGE_TAG_RE = re.compile(r"<image>\s*")


def _extract_question(content: str) -> str:
    """Strip all <image> tags from prompt content to get pure question text."""
    return _IMAGE_TAG_RE.sub("", content).strip()


def _load_pil_image(raw_image, image_prefix: str = "") -> Image.Image:
    """Convert raw image data (str path / dict / PIL) to PIL Image."""
    if isinstance(raw_image, Image.Image):
        return raw_image.convert("RGB")
    if isinstance(raw_image, dict):
        if "bytes" in raw_image:
            return Image.open(BytesIO(raw_image["bytes"])).convert("RGB")
        if "path" in raw_image:
            path = os.path.join(image_prefix, raw_image["path"]) if image_prefix else raw_image["path"]
            return Image.open(path).convert("RGB")
        if "image" in raw_image and isinstance(raw_image["image"], Image.Image):
            return raw_image["image"].convert("RGB")
    if isinstance(raw_image, str):
        path = os.path.join(image_prefix, raw_image) if image_prefix else raw_image
        return Image.open(path).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(raw_image)}")


def _image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image as a JPEG base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _resize_image(image: Image.Image, max_pixels: int | None) -> Image.Image:
    """Resize image to fit within max_pixels using Qwen VL smart_resize."""
    if max_pixels is None:
        return image
    try:
        from qwen_vl_utils.vision_process import smart_resize
    except ImportError:
        logger.warning("qwen_vl_utils not available, skipping resize")
        return image
    factor = 32
    min_pixels = factor * factor * 4
    w, h = image.size
    new_h, new_w = smart_resize(h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    if (new_h, new_w) != (h, w):
        image = image.resize((new_w, new_h))
    return image


# ─── Prompt builders ──────────────────────────────────────────────────────────────


def _build_perceiver_prompt(question: str) -> list[dict]:
    user_content = (
        "<image>\n"
        + PERCEIVE_PRE_PROMPT.format(question=question)
        + PERCEIVE_IMAGE_POST_PROMPT
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_reasoner_prompt(
    question: str, answer_form: str, n_images: int = 1, has_evidence: bool = True,
) -> list[dict]:
    pre_prompt = REASON_PRE_PROMPT if has_evidence else REASON_FALLBACK_PRE_PROMPT
    post_prompt = ANSWER_FORM_POST_PROMPTS.get(answer_form, ANSWER_FORM_POST_PROMPTS["free_form"])
    image_tags = "<image>\n" * n_images
    user_content = image_tags + pre_prompt.format(question=question) + post_prompt
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_baseline_prompt(question: str, answer_form: str) -> list[dict]:
    """Build a single-turn reasoning prompt (original image + think step by step)."""
    pre_prompt = REASON_FALLBACK_PRE_PROMPT
    post_prompt = ANSWER_FORM_POST_PROMPTS.get(answer_form, ANSWER_FORM_POST_PROMPTS["free_form"])
    user_content = "<image>\n" + pre_prompt.format(question=question) + post_prompt
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ─── Dataset class ────────────────────────────────────────────────────────────────


class P2RDataset(RLHFDataset):
    """RLHFDataset subclass that rebuilds prompts based on training role.

    Config fields:
        role (str): ``"perceiver"`` or ``"reasoner"``. When *null* / absent,
            falls back to the default RLHFDataset behaviour.
        perceiver_host (str): IP of the frozen perceiver vLLM service (reasoner only).
        perceiver_port (int): Port of the frozen perceiver vLLM service (reasoner only).
        perceiver_model (str): Model name for the perceiver vLLM API (reasoner only).
    """

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        self.role = config.get("role", None)
        if self.role is not None and self.role not in ("perceiver", "reasoner", "baseline"):
            raise ValueError(f"Invalid P2R role: {self.role!r}. Must be 'perceiver', 'reasoner', or 'baseline'.")

        self.perceiver_host = config.get("perceiver_host", None)
        self.perceiver_port = config.get("perceiver_port", None)
        self.perceiver_model = config.get("perceiver_model", None)
        self.max_pixels = config.get("max_pixels", None)
        self.image_prefix = config.get("image_prefix", "")
        if self.role == "reasoner":
            if not all([self.perceiver_host, self.perceiver_port, self.perceiver_model]):
                raise ValueError(
                    "perceiver_host, perceiver_port, and perceiver_model "
                    "are required when role='reasoner'."
                )

        super().__init__(data_files, tokenizer, config, processor, max_samples)

    # ── Perceiver service ────────────────────────────────────────────────────

    def _call_perceiver(self, image: Image.Image, question: str) -> str:
        """Call frozen perceiver vLLM service and return raw text output."""
        image = _resize_image(image, self.max_pixels)
        b64 = _image_to_base64(image)
        prompt_text = PERCEIVE_PRE_PROMPT.format(question=question) + PERCEIVE_IMAGE_POST_PROMPT
        image_item = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [image_item, {"type": "text", "text": prompt_text}],
            },
        ]
        url = f"http://{self.perceiver_host}:{self.perceiver_port}/v1/chat/completions"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    url,
                    json={
                        "model": self.perceiver_model,
                        "messages": messages,
                        "max_tokens": 2048,
                        "temperature": 0,
                    },
                    timeout=300,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception:
                logger.warning("Perceiver call failed (attempt %d/%d)", attempt + 1, max_retries, exc_info=True)
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5 * (attempt + 1))
        return ""

    # ── Image post-processing ────────────────────────────────────────────────

    @staticmethod
    def _process_perception(image: Image.Image, perceive_output: str, max_evidence: int = MAX_IMAGE_EVIDENCE) -> tuple[list[Image.Image], bool]:
        """Parse bboxes, draw annotations, crop regions.

        Returns:
            (processed_images, has_evidence)
        """
        all_bboxes = parse_bboxes(perceive_output)
        bboxes = all_bboxes[:max_evidence]
        if not bboxes:
            return [image], False
        annotated = draw_bounding_boxes(image, all_bboxes)
        crops = []
        for b in bboxes:
            try:
                crops.append(crop_region(image, b["bbox_2d"]))
            except Exception:
                logger.warning("Failed to crop bbox %s, skipping", b["bbox_2d"])
        if not crops:
            return [image], False
        return [annotated] + crops, True

    def _ensure_pil_images(self, raw_images: list) -> list[Image.Image]:
        """Convert raw image data to PIL Images, skipping failures."""
        result = []
        for img in raw_images:
            try:
                result.append(_load_pil_image(img, image_prefix=self.image_prefix))
            except Exception:
                logger.warning("Failed to load image, skipping", exc_info=True)
        return result

    # ── Data access ──────────────────────────────────────────────────────────

    def __getitem__(self, item):
        # No role → default RLHFDataset behaviour
        if self.role is None:
            return super().__getitem__(item)

        row_dict: dict = self.dataframe[item]

        # Extract question from raw prompt messages (before _build_messages)
        question = ""
        for msg in row_dict[self.prompt_key]:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, str):
                    question = _extract_question(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            question = _extract_question(part["text"])
                            break
                break

        extra_info = row_dict.get("extra_info") or {}
        answer_form = extra_info.get("answer_form", "free_form")

        # Save raw image reference before any processing (needed by reward fn)
        raw_images = row_dict.get(self.image_key) or []
        raw_image_ref = raw_images[0] if raw_images else None

        if self.role in ("perceiver", "baseline"):
            # Both use original image directly; only prompt differs
            if self.role == "perceiver":
                row_dict[self.prompt_key] = _build_perceiver_prompt(question)
            else:
                row_dict[self.prompt_key] = _build_baseline_prompt(question, answer_form)
            row_dict[self.image_key] = self._ensure_pil_images(
                row_dict.get(self.image_key) or []
            )

        elif self.role == "reasoner":
            raw_images = row_dict.get(self.image_key) or []
            pil_images = self._ensure_pil_images(raw_images)

            perceive_output = ""
            if pil_images:
                original_image = pil_images[0]
                perceive_output = self._call_perceiver(original_image, question)
                processed_images, has_evidence = self._process_perception(
                    original_image, perceive_output, MAX_IMAGE_EVIDENCE,
                )
            else:
                processed_images, has_evidence = [], False

            row_dict[self.prompt_key] = _build_reasoner_prompt(
                question, answer_form, len(processed_images), has_evidence,
            )
            row_dict[self.image_key] = processed_images

        # Convert <image> placeholders → image content items
        row_dict["raw_prompt"] = self._build_messages(row_dict)

        # Inject max_pixels into image content items for rollout
        # First image (annotated/original) gets full max_pixels, crops get reduced
        if self.max_pixels is not None:
            crop_max_pixels = get_crop_max_pixels(self.max_pixels)
            img_idx = 0
            for msg in row_dict["raw_prompt"]:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "image":
                            item["max_pixels"] = self.max_pixels if img_idx == 0 else crop_max_pixels
                            img_idx += 1

        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        # Extra-info handling (same as parent)
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        row_dict["extra_info"]["question"] = question
        if raw_image_ref is not None:
            row_dict["extra_info"]["raw_image"] = raw_image_ref
        if self.role == "reasoner" and perceive_output:
            row_dict["extra_info"]["perceiver_output"] = perceive_output
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get(
            "need_tools_kwargs", self.need_tools_kwargs,
        )
        if need_tools_kwargs and not tools_kwargs:
            logger.warning(
                "tools_kwargs is empty for index %s, data source: %s",
                index, row_dict["data_source"],
            )
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict
