import json


PROMPT_TEMPLATES = {
    "default":
        {
            "pre_prompt": "Question: {question}\n",
            "mca_post_prompt": "Answer the question with the option's letter from the given choices (e.g., A, B, etc.) directly.\n",
            "na_post_prompt": "Answer the question using a numerical value (e.g., 42 or 3.1) directly.\n",
            "free_form_post_prompt": "Output the answer simply and directly.\n",
            "one_word_post_prompt": "Output the answer using a single word or phrase directly.\n",
            "caption_post_prompt": "Output your text answer directly.\n",
            "yes_no_post_prompt": "Output yes or no directly.\n"
        },
    "thinking":
        {
            "pre_prompt": (
                "Question: {question}\n"
                "Think step by step. \n"
            ),
            "mca_post_prompt": (
                "Provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question with the option's letter from the given choices (e.g., A, B, etc.) within the <answer> </answer> tags.\n"
            ),
            "na_post_prompt": (
                "Provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question using a numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags.\n"
            ),
            "free_form_post_prompt": (
                "Provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question simply within the <answer> </answer> tags.\n"
            ),
            "one_word_post_prompt": (
                "Provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question using a single word or phrase within the <answer> </answer> tags.\n"
            ),
            "caption_post_prompt": (
                "Provide your detailed reasoning between the <think> </think> tags, "
                "and then provide your text answer within the <answer> </answer> tags.\n"
            ),
            "yes_no_post_prompt": (
                "Provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question using yes or no within the <answer> </answer> tags.\n"
            ),
        },
    "p2r":
        {
            "perceive_pre_prompt": "Question: {question}\n",
            "perceive_image_post_prompt": (
                "Locate the key visual evidence in the image for answering this question. "
                "Report bbox coordinates in JSON format only: "
                '[{"bbox_2d": [x1, y1, x2, y2], "label": "<description>"}] '
                "If the question requires the entire image, return an empty list []."
            ),
            "pre_prompt": (
                "Question: {question}\n"
                "Think step by step.\n"
            ),
            "image_pre_prompt": (
                "Question: {question}\n"
                "The key visual regions have been highlighted and cropped for you. "
                "Think step by step.\n"
            ),
            "mca_post_prompt": (
                "Provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question with the option's letter from the given choices "
                "(e.g., A, B, etc.) within the <answer> </answer> tags.\n"
            ),
            "na_post_prompt": (
                "Provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question using a numerical value (e.g., 42 or 3.1) "
                "within the <answer> </answer> tags.\n"
            ),
            "free_form_post_prompt": (
                "Provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question simply within the <answer> </answer> tags.\n"
            ),
            "one_word_post_prompt": (
                "Provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question using a single word or phrase within the <answer> </answer> tags.\n"
            ),
            "caption_post_prompt": (
                "Provide your detailed reasoning between the <think> </think> tags, "
                "and then provide your text answer within the <answer> </answer> tags.\n"
            ),
            "yes_no_post_prompt": (
                "Provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question using yes or no within the <answer> </answer> tags.\n"
            ),
        },
    "grounding":
        {
            "pre_prompt": "Locate the {description} in the image.\n",
            "sentence_pre_prompt": "{description}, locate it in the image.",
            "post_prompt": 'Report the bbox coordinates in JSON format.\n'
        },
}


def build_system_prompt() -> str:
    return "You are a helpful assistant."