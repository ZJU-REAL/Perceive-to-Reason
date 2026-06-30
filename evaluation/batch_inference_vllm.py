import logging
from transformers import AutoProcessor, AutoConfig
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


MAX_NUM_WORKERS_READ_VIDEO = 16
SPATIAL_MERGE_SIZE = 2
MAX_TOKENS: dict[str, int] = {
    "default": 16,
    "thinking": 1024,
    "p2r-perceive": 256,
    "p2r": 1024,
    "grounding": 256,
}


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def prepare_llm(model_config, nframes, max_pixels, eval_mode):
    processor = AutoProcessor.from_pretrained(model_config)
    processor.tokenizer.padding_side = "left"
    
    config = AutoConfig.from_pretrained(model_config)

    image_patch_size=processor.image_processor.patch_size
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    
    llm = LLM(
        model=model_config,
        mm_encoder_tp_mode="data",
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
        dtype="bfloat16",
        max_model_len=16384,
        gpu_memory_utilization=0.75,
        disable_mm_preprocessor_cache=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
            
    return llm, processor, config


def batch_inference_vllm(batch_messages, processor, llm, temperature, top_p, eval_mode):
    max_workers = max(1, min(MAX_NUM_WORKERS_READ_VIDEO, len(batch_messages)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(prepare_inputs_for_vllm, message, processor)
            for message in batch_messages
        ]
        inputs = [f.result() for f in futures]

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens = MAX_TOKENS[eval_mode],
        top_p=top_p,
        repetition_penalty=1.05,
    )
    
    outputs = llm.generate(inputs, sampling_params)
    response_preds = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_preds.append(generated_text)
        
    return response_preds


