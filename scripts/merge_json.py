import json
import sys
import os

# file_path = "/opt/tiger/llava-video/output/test/full_110_16_16_mf_16000_nqa/2025011218"

if __name__ == "__main__":
    file_path = sys.argv[1]
    thread_num = int(sys.argv[2])
    res = []

    for i in range(thread_num):
        if not os.path.exists(file_path + f"/test_results_rank{i}.json"):
            continue

        with open(file_path + f"/test_results_rank{i}.json", "r") as f:
            res.extend(json.load(f))

    with open(file_path + "/test_results.json", "w") as f:
        json.dump(res, f, indent=2)

    print(file_path + "/test_results.json")