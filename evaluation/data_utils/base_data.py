import logging
import gc
import numpy as np
import json
from dataclasses import dataclass, field
from typing import Optional, Any
from abc import ABC, abstractmethod
from batch_inference_vllm import *
from prompt import *
import pandas as pd
import os
import string
from tqdm import tqdm
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from p2r_utils import parse_bboxes, draw_bounding_boxes, crop_region
from utils import load_image


MAX_NUM_WORKERS_BUILD_MESSAGES = 16


@dataclass
class BaseDatasetSample(ABC):
    row: Any
    prompt_template: dict = field(default_factory=dict)
    prompt_text: str = field(init=False, default="")
    raw_predicted_answer: str = field(init=False, default="")
    processed_predicted_answer: Optional[str] = field(init=False, default=None)
    stats: Optional[dict] = field(init=False, default=None)
    perceive_output: str = field(init=False, default="")
    bboxes: list = field(init=False, default_factory=list)
    bbox_count: int = field(init=False, default=0)
    bbox_avg_size: float = field(init=False, default=0.0)
    image: Any = field(init=False, default=None)

    def read_video_frames(self, video_path: str, nframes: int, max_pixels: int, min_pixels: int, image_patch_size=14):
        raise NotImplementedError("Video evaluation is not supported yet.")

    @abstractmethod
    def build_messages(self, image, system_prompt: str, nframes: int, max_pixels: int, min_pixels: int) -> list:
        ...

    @abstractmethod
    def to_result(self) -> dict:
        ...

    @abstractmethod
    def build_prompt(self, pre_key: str = None) -> str:
        ...

    MAX_IMAGE_EVIDENCE = 3
    CROP_PIXELS_DIVISOR = 1

    @classmethod
    def get_crop_max_pixels(cls, max_pixels: int) -> int:
        return max_pixels // cls.CROP_PIXELS_DIVISOR

    def process_image_perception(self, image, perceive_output, question=""):
        self.perceive_output = perceive_output
        max_ev = self.MAX_IMAGE_EVIDENCE
        all_bboxes = parse_bboxes(perceive_output)
        self.bbox_count = len(all_bboxes)
        # Compute average bbox area in 0-1000 coord system
        if all_bboxes:
            areas = []
            for b in all_bboxes:
                x1, y1, x2, y2 = b["bbox_2d"]
                areas.append(abs(x2 - x1) * abs(y2 - y1))
            self.bbox_avg_size = sum(areas) / len(areas)
        self.bboxes = all_bboxes[:max_ev]
        annotated_image = None
        cropped_images = []
        if self.bboxes:
            annotated_image = draw_bounding_boxes(image, all_bboxes)
            crops = []
            for b in self.bboxes:
                try:
                    crops.append(crop_region(image, b["bbox_2d"]))
                except Exception:
                    logging.warning("Failed to crop bbox %s, skipping", b["bbox_2d"])
            cropped_images = crops
            if not crops:
                annotated_image = None
        return annotated_image, cropped_images

    def rebuild_evidence(self, image):
        """Re-derive annotated image and crops from stored perceive_output."""
        if not self.bboxes:
            return None, []
        all_bboxes = parse_bboxes(self.perceive_output)
        annotated = draw_bounding_boxes(image, all_bboxes)
        crops = []
        for b in self.bboxes:
            try:
                crops.append(crop_region(image, b["bbox_2d"]))
            except Exception:
                logging.warning("Failed to crop bbox %s, skipping", b["bbox_2d"])
        if not crops:
            return None, []
        return annotated, crops


class BaseDatasetEvaluator(ABC):
    sample_cls: type[BaseDatasetSample] = None

    def build_samples(self, data_df: pd.DataFrame, prompt_template: dict, data_dir: str) -> list[BaseDatasetSample]:
        rows = [row for _, row in data_df.iterrows()]
        max_workers = max(1, min(MAX_NUM_WORKERS_BUILD_MESSAGES, len(rows)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            samples = list(executor.map(
                lambda r: self.sample_cls(row=r, prompt_template=prompt_template, data_dir=data_dir), rows
            ))
        return samples

    @staticmethod
    def _build_p2r_image_perceive_messages(image, prompt, system_prompt, max_pixels, min_pixels):
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image,
                     "max_pixels": max_pixels, "min_pixels": min_pixels},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    @staticmethod
    def _build_p2r_image_reason_messages(images, prompt, system_prompt, max_pixels, min_pixels, crop_max_pixels=None):
        content = []
        for i, img in enumerate(images):
            mp = max_pixels if (i == 0 or crop_max_pixels is None) else crop_max_pixels
            content.append({"type": "image", "image": img,
                            "max_pixels": mp, "min_pixels": min_pixels})
        content.append({"type": "text", "text": prompt})
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]


    @abstractmethod
    def eval_results(self, results: list[dict]) -> list[float]:
        ...

    @abstractmethod
    def aggregate_results(self, results: list[dict]) -> dict:
        ...
    

    def get_desc(self) -> str:
        return "Evaluating"

    def evaluate(
        self,
        main_logger,
        data_df, data_dir, model_path,
        eval_mode,
        temperature, top_p, nframes, max_pixels, min_pixels,
        batch_size, debug_size, debug_mode,
    ):
        def extract_answer_text(text_with_tags):
            match = re.search(r"<answer>(.*?)</answer>", text_with_tags, re.DOTALL)
            if match:
                return match.group(1).strip()  
            else:
                return None
            
            
        def extract_final_answer(answer_text, prompt_type="default"):
            if answer_text is None:
                return None
            
            if prompt_type not in ("default", "grounding"):
                extracted_text = extract_answer_text(answer_text)
            else:
                extracted_text = answer_text.strip()
            
            if extracted_text is None or extracted_text.strip() == "":
                return None
                
            return extracted_text

        df_shard = data_df.sample(n=debug_size) if debug_mode else data_df
        if debug_mode:
            main_logger.info(f"Debug mode enabled, randomly processing {debug_size} samples.")

        llm, processor, _ = prepare_llm(model_path, nframes, max_pixels, eval_mode)
        processor.image_processor.size = {"longest_edge": max_pixels, "shortest_edge": min_pixels}

        total_samples = len(df_shard)
        if total_samples == 0:
            main_logger.info("Process has empty shard, skipping processing.")
            return [], {}

        if batch_size is None or batch_size <= 0:
            batch_size = total_samples

        system_prompt = build_system_prompt()
        prompt_template = PROMPT_TEMPLATES.get(eval_mode, PROMPT_TEMPLATES["default"])

        all_samples: list[BaseDatasetSample] = self.build_samples(df_shard, prompt_template, data_dir)

        num_batches = math.ceil(total_samples / batch_size)

        if eval_mode == "p2r":
            main_logger.info("P2R: Perceive + Reason per batch...")
            inference_start = time.time()  # measure pure inference (excludes model loading)
            pbar = tqdm(total=num_batches, desc=f"{self.get_desc()} - P2R", unit="batch")
            for idx in range(0, total_samples, batch_size):
                batch = all_samples[idx : idx + batch_size]
                max_workers = max(1, min(MAX_NUM_WORKERS_BUILD_MESSAGES, len(batch)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    images = list(executor.map(lambda s: load_image(s.image), batch))

                # Stage 1: Perceive
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._build_p2r_image_perceive_messages,
                            img, s.build_perceive_prompt(),
                            system_prompt, max_pixels, min_pixels,
                        )
                        for s, img in zip(batch, images)
                    ]
                    batch_messages = [f.result() for f in futures]
                perceive_outputs = batch_inference_vllm(
                    batch_messages, processor, llm, temperature, top_p, "p2r-perceive",
                )
                perceive_results = []
                for sample, img, output in zip(batch, images, perceive_outputs):
                    question = sample.row.get("question", getattr(sample, '_get_question', lambda: "")())
                    perceive_results.append(sample.process_image_perception(img, output, question))

                # Stage 2: Reason
                max_workers = max(1, min(MAX_NUM_WORKERS_BUILD_MESSAGES, len(batch)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for s, img, (annotated, crops) in zip(batch, images, perceive_results):
                        has_evidence = annotated is not None
                        reason_prompt = s.prompt_text if has_evidence else s.build_prompt(pre_key="pre_prompt")
                        reason_images = [annotated] + crops if has_evidence else [img]
                        crop_max_pixels = self.sample_cls.get_crop_max_pixels(max_pixels) if has_evidence else None
                        futures.append(executor.submit(
                            self._build_p2r_image_reason_messages,
                            reason_images, reason_prompt,
                            system_prompt, max_pixels, min_pixels, crop_max_pixels,
                        ))
                    batch_messages = [f.result() for f in futures]
                reason_outputs = batch_inference_vllm(
                    batch_messages, processor, llm, temperature, top_p, "p2r",
                )
                for sample, output in zip(batch, reason_outputs):
                    sample.raw_predicted_answer = output
                    sample.processed_predicted_answer = extract_final_answer(output, eval_mode)

                del images, perceive_results, batch_messages
                gc.collect()

                pbar.update(1)
            pbar.close()
            self.inference_time = time.time() - inference_start
        else:
            pbar = tqdm(total=num_batches, desc=self.get_desc(), unit="batch")
            inference_start = time.time()
            for idx in range(0, total_samples, batch_size):
                batch = all_samples[idx : idx + batch_size]
                max_workers = max(1, min(MAX_NUM_WORKERS_BUILD_MESSAGES, len(batch)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    images = list(executor.map(lambda s: load_image(s.image), batch))

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(s.build_messages, img, system_prompt, nframes, max_pixels, min_pixels)
                        for s, img in zip(batch, images)
                    ]
                    batch_messages = [f.result() for f in futures]
                raw_answers = batch_inference_vllm(batch_messages, processor, llm, temperature, top_p, eval_mode)
                for sample, raw_answer in zip(batch, raw_answers):
                    sample.raw_predicted_answer = raw_answer
                    sample.processed_predicted_answer = extract_final_answer(raw_answer, eval_mode)

                del images, batch_messages
                gc.collect()

                pbar.update(1)
            pbar.close()
            self.inference_time = time.time() - inference_start

        results = [s.to_result() for s in all_samples]
        results = self.eval_results(results)

        output = self.aggregate_results(results)
        return results, output