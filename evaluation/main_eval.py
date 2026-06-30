import os
import json
from datetime import datetime
import argparse
from utils import setup_logger, format_time, json_default
import pandas as pd
import time
from datasets import load_dataset
from tqdm import tqdm


def prepare_data(task: str, data_dir: str) -> tuple[pd.DataFrame, str]:
    if task == "V-Star":
        data_dir = os.path.join(data_dir, "V-Star-with-BBox")
        return load_dataset(data_dir)["test"].to_pandas(), data_dir

    if task == "HR-Bench":
        data_dir = os.path.join(data_dir, "HR-Bench")
        hrbench_4k_data = load_dataset(data_dir)["hrbench_4k"].to_pandas()
        hrbench_8k_data = load_dataset(data_dir)["hrbench_8k"].to_pandas()
        hrbench_4k_data["benchmark"] = "hrbench_4k"
        hrbench_8k_data["benchmark"] = "hrbench_8k"
        return pd.concat([hrbench_4k_data, hrbench_8k_data], ignore_index=True), data_dir

    elif task == "MME-RealWorld-lite":
        data_dir = os.path.join(data_dir, "MME-RealWorld-lite")
        return load_dataset(data_dir)["train"].to_pandas(), data_dir

    elif task == "MME-RealWorld":
        data_dir = os.path.join(data_dir, "MME-RealWorld")
        return load_dataset(data_dir)["train"].to_pandas(), data_dir

    else:
        raise ValueError(f"Task {task} not recognized for data preparation.")


EVALUATOR_REGISTRY = {
    "V-Star": "data_utils.vstar.VStarEvaluator",
    "HR-Bench": "data_utils.hrbench.HRBenchEvaluator",
    "MME-RealWorld-lite": "data_utils.mme_realworld.MMERealWorldEvaluator",
    "MME-RealWorld": "data_utils.mme_realworld.MMERealWorldEvaluator",
}


def load_evaluator(task: str):
    if task not in EVALUATOR_REGISTRY:
        raise ValueError(f"Task {task} not recognized. Available: {list(EVALUATOR_REGISTRY.keys())}")
    module_path, class_name = EVALUATOR_REGISTRY[task].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--task", default="V-Star")
    parser.add_argument("--eval_mode", default="default")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--nframes", type=int, default=32)
    parser.add_argument("--max_pixels", type=int, default=224 * 224)
    parser.add_argument("--min_pixels", type=int, default=64 * 64)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--debug_size", type=int, default=4)
    parser.add_argument("--debug_mode", action="store_true")
    args = parser.parse_args()

    # === Configuration ===
    start_time = time.time()
    task = args.task
    data_df, data_dir = prepare_data(task, args.data_dir)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.log_dir, task, args.eval_mode, timestamp_str)
    os.makedirs(output_dir, exist_ok=True)

    output_jsonl_file = os.path.join(output_dir, "results.jsonl")
    log_output_file = os.path.join(output_dir, "eval.log")

    params_to_log = vars(args) | {"output_dir": output_dir}

    main_logger = setup_logger(log_output_file, params_to_log)
    main_logger.info("Main script started. Configuration logged.")

    evaluator = load_evaluator(task)
    results, output = evaluator.evaluate(
        main_logger=main_logger,
        data_df=data_df,
        data_dir=data_dir,
        model_path=args.model_path,
        eval_mode=args.eval_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        nframes=args.nframes,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        batch_size=args.batch_size,
        debug_size=args.debug_size,
        debug_mode=args.debug_mode,
    )

    with open(output_jsonl_file, "w") as f:
        for result in results:
            json.dump(result, f, default=json_default, ensure_ascii=False)
            f.write("\n")

    elapsed_time = time.time() - start_time
    inference_time = getattr(evaluator, "inference_time", elapsed_time)
    main_logger.info(f"{task} evaluation completed.")
    main_logger.info(f"Final results saved to: {output_jsonl_file}")
    main_logger.info(f"Total runtime: {format_time(elapsed_time)}")
    main_logger.info(f"Pure inference runtime: {format_time(inference_time)}")

    if output:
        print(f"{task} Evaluation Results:", output)
        main_logger.info(f"{task} Evaluation Results:")
        main_logger.info(json.dumps(output, indent=2))
    else:
        main_logger.info(f"No final evaluation metrics calculated for {task} or evaluation failed.")
