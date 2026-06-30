"""P2R reward functions for perceiver and reasoner training.

Reward routing by answer_form:
  - mca:       regex-extract option letter (A–D) and compare with ground truth.
  - na:        numerical grading via ``mathruler.grade_answer``.
  - free_form: call an external verifier model service.

When role='perceiver', the perceiver's bbox output is first post-processed
(highlight + crop), then a frozen reasoner service produces an answer which
is evaluated with the same logic above.

Usage – configure as a custom reward function in the trainer config:

.. code-block:: yaml

    reward:
      custom_reward_function:
        path: pkg://verl.utils.reward_score.p2r_reward
        name: compute_score
        reward_kwargs:
          role: reasoner            # or perceiver
          reasoner_host: null       # required when role=perceiver
          reasoner_port: null
          reasoner_model: null
          verifier_host: null       # required for free_form evaluation
          verifier_port: null
          verifier_model: null
"""

import re
import logging

import requests

from verl.utils.dataset.p2r_dataset import (
    parse_bboxes,
    draw_bounding_boxes,
    crop_region,
    _load_pil_image,
    _image_to_base64,
    _resize_image,
    MAX_IMAGE_EVIDENCE,
    get_crop_max_pixels,
    SYSTEM_PROMPT,
    REASON_PRE_PROMPT,
    REASON_FALLBACK_PRE_PROMPT,
    ANSWER_FORM_POST_PROMPTS,
)

logger = logging.getLogger(__name__)

# ─── Answer extraction ────────────────────────────────────────────────────────────

BBOX_IOU_THRESHOLD = 0.5

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer(text: str) -> str:
    """Extract text inside ``<answer>`` tags; fall back to full text."""
    if not text:
        return ""
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()


# ─── MCA reward (mirrors evaluation/data_utils/hrbench.py) ───────────────────────

_MCA_PREFIXES = [
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
    "Answer:",
    "Option:",
    "The correct answer",
    "The correct option",
]


def _bbox_iou(b1, b2) -> float:
    """Compute IoU of two bboxes. Coords in 0-1000 normalized format."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _mca_reward(answer: str, ground_truth: str) -> float:
    s = answer.strip() if answer else ""
    for prefix in _MCA_PREFIXES:
        s = s.replace(prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return 0.0
    m = re.search(r"[ABCDE]", s)
    pred = m[0] if m else ""
    gt = ground_truth.strip().upper()
    return 1.0 if pred == gt else 0.0


# ─── NA reward ────────────────────────────────────────────────────────────────────


def _na_reward(answer: str, ground_truth: str) -> float:
    try:
        from mathruler.grader import grade_answer
        return float(grade_answer(answer, ground_truth))
    except Exception:
        logger.warning("mathruler.grade_answer failed", exc_info=True)
        return 0.0


# ─── Free-form reward (model verifier) ────────────────────────────────────────────

_VERIFIER_PROMPT = (
    "Given the question and its ground truth answer, "
    "determine if the predicted answer is correct. "
    "Consider semantic equivalence, not just exact string match.\n\n"
    "Question: {question}\n"
    "Ground truth answer: {ground_truth}\n"
    "Predicted answer: {answer}\n\n"
    'Is the predicted answer correct? Reply with only "Yes" or "No".'
)


def _free_form_reward(
    answer: str,
    ground_truth: str,
    question: str,
    verifier_host: str,
    verifier_port,
    verifier_model: str,
) -> float:
    if not all([verifier_host, verifier_port, verifier_model]):
        logger.warning("Verifier service not configured, returning 0.0 for free_form")
        return 0.0
    prompt = _VERIFIER_PROMPT.format(
        question=question, ground_truth=ground_truth, answer=answer,
    )
    messages = [
        {"role": "system", "content": "You are a helpful grading assistant."},
        {"role": "user", "content": prompt},
    ]
    url = f"http://{verifier_host}:{verifier_port}/v1/chat/completions"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                json={
                    "model": verifier_model,
                    "messages": messages,
                    "max_tokens": 64,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return 1.0 if "yes" in text.lower() else 0.0
        except Exception:
            logger.warning("Verifier call failed (attempt %d/%d)", attempt + 1, max_retries, exc_info=True)
            if attempt < max_retries - 1:
                import time
                time.sleep(2 * (attempt + 1))
    return 0.0


# ─── Evaluate answer by answer_form ───────────────────────────────────────────────


def _evaluate_answer(
    answer: str,
    ground_truth: str,
    answer_form: str,
    question: str,
    verifier_host=None,
    verifier_port=None,
    verifier_model=None,
) -> float:
    if answer_form == "mca":
        return _mca_reward(answer, ground_truth)
    if answer_form == "na":
        return _na_reward(answer, ground_truth)
    if answer_form == "free_form":
        return _free_form_reward(
            answer, ground_truth, question,
            verifier_host, verifier_port, verifier_model,
        )
    return 0.0


# ─── Call frozen reasoner service (perceiver training) ────────────────────────────


def _call_reasoner(images, prompt_text, reasoner_host, reasoner_port, reasoner_model, max_pixels=None) -> str:
    content = []
    for i, img in enumerate(images):
        if i == 0:
            img = _resize_image(img, max_pixels)
        b64 = _image_to_base64(img)
        image_item = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        content.append(image_item)
    content.append({"type": "text", "text": prompt_text})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    url = f"http://{reasoner_host}:{reasoner_port}/v1/chat/completions"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                json={
                    "model": reasoner_model,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0,
                },
                timeout=300,
            )
            if resp.status_code != 200:
                logger.warning("Reasoner service returned %d: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            logger.warning("Reasoner call failed (attempt %d/%d)", attempt + 1, max_retries, exc_info=True)
            if attempt < max_retries - 1:
                import time
                time.sleep(5 * (attempt + 1))
    return ""


# ─── Main entry point ─────────────────────────────────────────────────────────────


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    role=None,
    reasoner_host=None,
    reasoner_port=None,
    reasoner_model=None,
    verifier_host=None,
    verifier_port=None,
    verifier_model=None,
    max_pixels=None,
    image_prefix=None,
    **kwargs,
):
    """P2R reward function dispatched by *role* and *answer_form*.

    Configured via ``reward.custom_reward_function.reward_kwargs`` in the
    trainer YAML so that ``role``, service hosts/ports/models are injected
    into every call automatically.
    """
    extra_info = extra_info or {}
    answer_form = extra_info.get("answer_form", "free_form")
    question = extra_info.get("question", "")

    # ── Reasoner / Baseline training: evaluate generated answer directly ──
    if role in ("reasoner", "baseline"):
        answer = _extract_answer(solution_str)
        score = _evaluate_answer(
            answer, ground_truth, answer_form, question,
            verifier_host, verifier_port, verifier_model,
        )
        perceiver_output = extra_info.get("perceiver_output", "")
        return {
            "score": score,
            "perceiver_output": perceiver_output,
            "extracted_answer": answer,
        }

    # ── Perceiver training: post-process → call reasoner → evaluate ──
    if role == "perceiver":
        all_bboxes = parse_bboxes(solution_str)
        bbox_count = len(all_bboxes)
        max_evidence = MAX_IMAGE_EVIDENCE

        # Soft penalty: too many bboxes (0.2 per excess, max 1.0)
        # penalty = 0.0
        # if bbox_count > max_evidence:
        #     excess = bbox_count - max_evidence
        #     penalty = min(excess * 0.2, 1.0)

        # # Penalize: duplicate labels (non-counting questions only)
        # if all_bboxes and not _is_counting_question(question):
        #     labels = [b.get("label", "") for b in all_bboxes]
        #     if len(labels) != len(set(labels)):
        #         return {"score": 0.0, "reasoner_output": f"[PENALIZED] duplicate labels: {labels}", "extracted_answer": ""}

        # Penalize: high IoU between any two bboxes
        if bbox_count >= 2:
            for i in range(bbox_count):
                for j in range(i + 1, bbox_count):
                    iou = _bbox_iou(all_bboxes[i]["bbox_2d"], all_bboxes[j]["bbox_2d"])
                    if iou > BBOX_IOU_THRESHOLD:
                        return {"score": 0.0, "reasoner_output": f"[PENALIZED] bbox IoU={iou:.2f} > {BBOX_IOU_THRESHOLD} between bbox[{i}] and bbox[{j}]", "extracted_answer": "", "bbox_count": bbox_count}

        # Penalize: missing label or label is literally "description"
        for i, b in enumerate(all_bboxes):
            label = b.get("label", "")
            if not label or label.strip().lower() == "description":
                return {"score": 0.0, "reasoner_output": f"[PENALIZED] bbox[{i}] has invalid label: '{label}'", "extracted_answer": "", "bbox_count": bbox_count}

        bboxes = all_bboxes[:max_evidence]

        raw_image = extra_info.get("raw_image")
        if raw_image is None:
            logger.warning("No raw_image in extra_info for perceiver reward")
            return {"score": 0.0, "reasoner_output": "[ERROR] no raw_image in extra_info", "extracted_answer": "", "bbox_count": bbox_count}
        try:
            image = _load_pil_image(raw_image, image_prefix=image_prefix or "")
        except Exception:
            logger.warning("Failed to load image for perceiver reward", exc_info=True)
            return {"score": 0.0, "reasoner_output": "[ERROR] failed to load image", "extracted_answer": "", "bbox_count": bbox_count}

        # Post-process: highlight + crop
        if bboxes:
            annotated = draw_bounding_boxes(image, all_bboxes)
            crop_max_pixels = get_crop_max_pixels(max_pixels)
            crops = []
            for b in bboxes:
                try:
                    crop = _resize_image(crop_region(image, b["bbox_2d"]), crop_max_pixels)
                    crops.append(crop)
                except Exception:
                    logger.warning("Failed to crop bbox %s, skipping", b["bbox_2d"])
            if crops:
                images = [annotated] + crops
                has_evidence = True
            else:
                images = [image]
                has_evidence = False
        else:
            images = [image]
            has_evidence = False

        # Build reasoner prompt
        pre = REASON_PRE_PROMPT if has_evidence else REASON_FALLBACK_PRE_PROMPT
        post = ANSWER_FORM_POST_PROMPTS.get(
            answer_form, ANSWER_FORM_POST_PROMPTS["free_form"],
        )
        prompt_text = pre.format(question=question) + post

        if not all([reasoner_host, reasoner_port, reasoner_model]):
            logger.warning("Reasoner service not configured for perceiver reward")
            return {"score": 0.0, "reasoner_output": "[ERROR] reasoner service not configured", "extracted_answer": "", "bbox_count": bbox_count}
        reasoner_output = _call_reasoner(
            images, prompt_text, reasoner_host, reasoner_port, reasoner_model, max_pixels,
        )

        answer = _extract_answer(reasoner_output)
        score = _evaluate_answer(
            answer, ground_truth, answer_form, question,
            verifier_host, verifier_port, verifier_model,
        )
        # final_score = score * (1.0 - penalty)
        final_score = score
        return {
            "score": final_score,
            "reasoner_output": reasoner_output,
            "extracted_answer": answer,
            "bbox_count": bbox_count,
        }

    raise ValueError(f"Unknown P2R role: {role!r}. Must be 'perceiver', 'reasoner', or 'baseline'.")
