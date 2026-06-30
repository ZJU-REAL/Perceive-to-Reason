import re
import json
from PIL import Image, ImageDraw


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
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
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
        # if "label" in bbox:
        #     draw.text((abs_x1 + 8, abs_y1 + 6), bbox["label"], fill=color)
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


def get_video_duration(video_path):
    import decord
    vr = decord.VideoReader(video_path)
    return len(vr) / vr.get_avg_fps()

