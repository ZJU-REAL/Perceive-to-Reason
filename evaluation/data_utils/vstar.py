import re
from data_utils.base_data import *
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class VStarSample(BaseDatasetSample):
    data_dir: str = ""

    def __post_init__(self):
        self.image = self.row["image"]
        self.prompt_text = self.build_prompt()

    def _get_question(self):
        return re.sub(
            r"\s*Answer with the option'?s letter from the given choices directly\.?\s*$",
            "",
            self.row["text"],
        ).strip()

    def build_prompt(self, pre_key=None):
        question = self._get_question()
        if pre_key is None:
            pre_key = "image_pre_prompt" if "image_pre_prompt" in self.prompt_template else "pre_prompt"
        prompt_text = self.prompt_template[pre_key].format(question=question)
        prompt_text += self.prompt_template["mca_post_prompt"]
        return prompt_text

    def build_perceive_prompt(self):
        question = self._get_question()
        prompt = self.prompt_template["perceive_pre_prompt"].format(question=question)
        prompt += self.prompt_template["perceive_image_post_prompt"]
        return prompt

    def build_messages(self, image, system_prompt, nframes, max_pixels, min_pixels):
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image,
                     "max_pixels": max_pixels, "min_pixels": min_pixels},
                    {"type": "text", "text": self.prompt_text},
                ],
            },
        ]

    def to_result(self):
        result = {
            "question_id": self.row["question_id"],
            "question": self.row["text"],
            "question_type": self.row["category"],
            "processed_predicted_answer": self.processed_predicted_answer,
            "ground_truth": self.row["label"],
            "accuracy": 0.0,
            "prompt": self.prompt_text,
            "raw_predicted_answer": self.raw_predicted_answer,
        }
        if self.perceive_output:
            result["perceive_output"] = self.perceive_output
            result["bbox_count"] = self.bbox_count
        if self.stats is not None:
            result.update(self.stats)
        return result


class VStarEvaluator(BaseDatasetEvaluator):
    sample_cls = VStarSample

    def get_desc(self):
        return "Evaluating V-Star"

    def eval_results(self, results):
        def extract_characters_regex(s):
            if s is None:
                return ""
            s = s.strip()
            answer_prefixes = [
                "The best answer is", "The correct answer is", "The answer is", "The answer",
                "The best option is", "The correct option is", "Best answer:", "Best option:",
                "Answer:", "Option:", "The correct answer", "The correct option",
            ]
            for prefix in answer_prefixes:
                s = s.replace(prefix, "")
            if len(s.split()) > 10 and not re.search("[ABCD]", s):
                return ""
            matches = re.search(r"[ABCD]", s)
            return matches[0] if matches else ""

        for result in results:
            pred = extract_characters_regex(result["processed_predicted_answer"])
            gt = result["ground_truth"].strip().upper()
            result["accuracy"] = 1.0 if pred == gt else 0.0

        return results

    def aggregate_results(self, results):
        results_df = pd.DataFrame(results)
        output = {"overall": {"accuracy": 0.0}}

        if results_df.empty:
            return output

        results_df["accuracy"] = pd.to_numeric(results_df["accuracy"], errors="coerce").fillna(0.0)
        has_bbox_count = "bbox_count" in results_df.columns

        for q_type, sub in results_df.groupby("question_type"):
            output[q_type] = {"accuracy": float(sub["accuracy"].mean()) * 100}
            if has_bbox_count:
                output[q_type]["bbox_count"] = float(sub["bbox_count"].mean())

        output["overall"] = {"accuracy": float(results_df["accuracy"].mean()) * 100}
        if has_bbox_count:
            output["overall"]["bbox_count"] = float(results_df["bbox_count"].mean())

        return output
