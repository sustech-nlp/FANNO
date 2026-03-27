from instdiff_tools.analysis import load_jsonlines, write_jsonlines


data_w_complexity = load_jsonlines("/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__bigcode_bigcode-iter2/compared_complexity.jsonl")
data_wo_complexity = load_jsonlines("/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__bigcode_bigcode-iter1/compared.jsonl")


for item_w, item_wo in zip(data_w_complexity, data_wo_complexity):
    item_wo["complexity"] = item_w["complexity"]
    # item_wo["passrate@8_scores"] = item_w["passrate@8_scores"]
    # item_wo["passrate@8"] = item_w["passrate@8"]

write_jsonlines(data_wo_complexity, "/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/ds-acl/diff_size/llama3_8b__bigcode_bigcode-iter1/compared_complexity.jsonl")