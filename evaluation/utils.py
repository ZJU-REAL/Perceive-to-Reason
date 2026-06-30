import re
import os
import io
import base64
from datetime import datetime, timedelta
import logging
import numpy as np
import json
import requests
from PIL import Image
import torch


def format_time(elapsed_seconds):
    time_delta = timedelta(seconds=int(elapsed_seconds))
    hours = time_delta.seconds // 3600
    minutes = (time_delta.seconds % 3600) // 60
    seconds = time_delta.seconds % 60
    return f"{hours:02}h{minutes:02}m{seconds:02}s"


def setup_logger(log_file, params_dict=None):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting evaluation")
    logger.info("Running parameters:")
    
    if params_dict:
        for key, value in params_dict.items():
            logger.info(f"  {key}: {value}")
    return logger


def json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, np.floating):
        return float(o)
    elif isinstance(o, np.integer):
        return int(o)
    return str(o)


def parse_output(output_text):
    pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(pattern, output_text, re.DOTALL)
    if match:
        try:
            tool_call_arguments = json.loads(match.group(1))
            assert "name" in tool_call_arguments, "'name' must be in tool_call_arguments"
            assert "arguments" in tool_call_arguments, "'arguments' must be in tool_call_arguments"
            assert isinstance(tool_call_arguments["name"], str), "'name' must be a string"
            assert isinstance(tool_call_arguments["arguments"], dict), "'arguments' must be a dictionary"
            return {
                "type": "tool",
                "content": tool_call_arguments
            }
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
            }
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, output_text, re.DOTALL)
    if match:
        return {
            "type": "answer",
            "content": match.group(1),
        }
    return {
        "type": "error",
        "content": "Error: response parse failed",
    }


# ─── BBox utilities ────────────────────────────────────────────────────────────


def compute_iou(box1, box2) -> float:
    """Compute IoU of two boxes in [x1, y1, x2, y2] absolute pixel format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def norm_bbox_to_abs(bbox_norm, width, height):
    """Convert 0-1000 normalized bbox [x1, y1, x2, y2] to absolute pixel coords."""
    x1 = int(bbox_norm[0] / 1000 * width)
    y1 = int(bbox_norm[1] / 1000 * height)
    x2 = int(bbox_norm[2] / 1000 * width)
    y2 = int(bbox_norm[3] / 1000 * height)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def coco_bbox_to_xyxy(bbox):
    """Convert COCO format [x, y, w, h] to [x1, y1, x2, y2]."""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


# ─── Image loading ─────────────────────────────────────────────────────────────


def load_image(image):
    """Load an image from various input types into a PIL RGB image."""

    # PIL image
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    # String (URL / local path / base64)
    if isinstance(image, str):
        # URL
        if image.startswith("http://") or image.startswith("https://"):
            response = requests.get(image, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")

        # Local path
        elif os.path.exists(image):
            return Image.open(image).convert("RGB")

        # Base64 string
        else:
            try:
                image_bytes = base64.b64decode(image)
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                raise ValueError(f"Unrecognized image string: {image[:50]}...")

    # Numpy array
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[0] <= 4 and image.ndim == 3:
            # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))
        return Image.fromarray(np.uint8(image)).convert("RGB")

    # Torch tensor
    if torch.is_tensor(image):
        image = image.detach().cpu()
        if image.ndim == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        if image.shape[0] <= 4 and image.ndim == 3:
            image = image.permute(1, 2, 0)
        image = image.numpy()
        return Image.fromarray(np.uint8(image * 255) if image.max() <= 1 else np.uint8(image)).convert("RGB")

    # Bytes / bytearray
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")

    # Dict (common case for decoded_image)
    if isinstance(image, dict):
        # Case 1: {"bytes": {"0": 137, "1": 80, "2": 78, ...}}
        if "bytes" in image and isinstance(image["bytes"], dict):
            byte_values = bytes(image["bytes"].values())
            return Image.open(io.BytesIO(byte_values)).convert("RGB")

        # Case 2: {"bytes": b"..."} or {"bytes": [137, 80, 78, ...]}
        elif "bytes" in image:
            bytes_field = image["bytes"]
            if isinstance(bytes_field, (bytes, bytearray)):
                return Image.open(io.BytesIO(bytes_field)).convert("RGB")
            elif isinstance(bytes_field, (list, tuple)):
                return Image.open(io.BytesIO(bytes(bytes_field))).convert("RGB")

        # Case 3: {"path": "..."}
        elif "path" in image and os.path.exists(image["path"]):
            return Image.open(image["path"]).convert("RGB")

        # Case 4: {"image_path": "..."}
        elif "image_path" in image and os.path.exists(image["image_path"]):
            return Image.open(image["image_path"]).convert("RGB")

        raise ValueError(f"Unrecognized image dict structure: keys={list(image.keys())}")

    # Unsupported type
    raise TypeError(f"Unsupported image type: {type(image)}")



