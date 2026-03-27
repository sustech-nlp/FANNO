import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from math_score import compute_score
from vllm import LLM, SamplingParams
from instdiff_tools.analysis import load_jsonlines, write_jsonlines
from typing import List, Dict
from tqdm import tqdm


def build_prompt(item: Dict) -> str:
    return item["instruction"]


def main(
    model_name_or_path: str,
    k: int,
    max_tokens: int,
    data_path: str,
    output_path: str,
):
    # ========== 1. Load data ==========
    data: List[Dict] = load_jsonlines(data_path)

    # ========== 2. Init LLM ==========
    llm = LLM(
        model=model_name_or_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        # tensor_parallel_size=4,
    )

    sampling_params = SamplingParams(
        n=k,
        max_tokens=max_tokens,
        temperature=0.8,
        top_p=0.95,
    )

    # ========== 3. Batch build prompts ==========
    prompts = [build_prompt(item) for item in data]

    # ========== 4. Batch inference ==========
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
    )

    assert len(outputs) == len(data), "Outputs and data size mismatch"

    # ========== 5. Compute PassRate@K ==========
    new_data = []

    for item, out in tqdm(
        zip(data, outputs),
        total=len(data),
        desc=f"Computing PassRate@{k}",
    ):
        generations = [o.text for o in out.outputs]

        reference = item.get("response")
        if reference is None:
            raise ValueError("No reference / answer field found in item.")

        pass_cnt = 0
        scores = []

        for gen in generations:
            score = compute_score(gen, reference)
            scores.append(score)

            if isinstance(score, bool):
                pass_cnt += int(score)
            else:
                pass_cnt += int(score > 0)

        passrate_k = pass_cnt / k

        # 写回
        item[f"passrate@{k}"] = passrate_k
        item[f"passrate@{k}_scores"] = scores
        item["complexity"] = 1 - passrate_k

        new_data.append(item)

    # ========== 6. Save ==========
    write_jsonlines(new_data, output_path)
    print(f"Saved PassRate@{k} results to {output_path}")


if __name__ == "__main__":
    main(
        model_name_or_path="/volume/pt-train/users/wzhang/ghchen/zh/models/Qwen2.5-7B",
        k=8,
        max_tokens=2048,
        data_path="/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/qwen2_5_7b__math_math-1k/compared.jsonl",
        output_path="/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/qwen2_5_7b__math_math-1k/compared_complexity.jsonl",
    )
