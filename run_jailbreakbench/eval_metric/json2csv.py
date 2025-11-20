import json
import csv

def json_to_csv(json_path: str, csv_path: str):
    # 读取 JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 打开 CSV 文件准备写入
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # 写表头
        writer.writerow(["behavior", "prompt"])

        # 遍历 JSON 里的每个键值对
        for behavior, prompts in data.items():
            # 有的结构是 list，有的可能是单个字符串，这里都兼容一下
            if isinstance(prompts, list):
                for p in prompts:
                    # 跳过空的
                    if not isinstance(p, str):
                        continue
                    writer.writerow([behavior, p])
            elif isinstance(prompts, str):
                writer.writerow([behavior, prompts])

if __name__ == "__main__":
    # 把这里路径改成你的实际文件路径
    json_to_csv("/opt/tiger/sft_entity/DLLM_safety/run_jailbreakbench/eval_metric/test_cases_autodan.json", "/opt/tiger/sft_entity/DLLM_safety/run_jailbreakbench/eval_metric/test_cases_autodan.csv")
