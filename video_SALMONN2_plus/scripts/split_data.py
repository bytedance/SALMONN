import json
import sys
import os
import copy

if __name__ == "__main__":
    data_file = sys.argv[1]
    split_num = sys.argv[2]
    target_dir = sys.argv[3]

    with open(data_file, "r") as f:
        data = json.load(f)

    per_len = len(data) // int(split_num)

    remainder = len(data) % int(split_num)

    if remainder > 0:
        should_add = int(split_num) - remainder
    else:
        should_add=0

    for i in range(should_add):
        data_append = copy.deepcopy(data[-1])
        data_append["should_use"] = False
        data.append(data_append)

    per_len = len(data) // int(split_num)

    group_counts = [per_len] * int(split_num)

    os.makedirs(target_dir, exist_ok=True)

    total_count = 0

    for i in range(int(split_num) - 1):
        # data_file = f"{target_dir}/{i}.json"
        data_file = os.path.join(target_dir, f"{i}.json")
        print(data_file)
        with open(data_file, "w") as f:
            json.dump(data[total_count:total_count+group_counts[i]], f, indent=2)
        print(group_counts[i])
        total_count += group_counts[i]

    data_file = os.path.join(target_dir, f"{int(split_num) - 1}.json")
    with open(data_file, "w") as f:
        json.dump(data[total_count:], f, indent=2)

# data_file = "/mnt/bn/tiktok-mm-3/aiic/users/tangchangli/preprocess_dataset/ytb60CapwAsrAac0-7k10-22kSpeakFiltBadAsr_detailPrompt.json"

# with open (data_file, "r") as f:
#     data = json.load(f)

# # prev_data = random.sample(data, 10)

# per_len = len(data) // 6

# data1 = data[:per_len]
# data2 = data[per_len*1:per_len*2]
# data3 = data[per_len*2:per_len*3]
# data4 = data[per_len*3:per_len*4]
# data5 = data[per_len*4:per_len*5]
# data6 = data[per_len*5:]

# data_file1 = "/mnt/bn/tiktok-mm-3/aiic/users/liyixuan/preprocessed_data/dpo_pipeline_gooddata_detail1.json"
# data_file2 = "/mnt/bn/tiktok-mm-3/aiic/users/liyixuan/preprocessed_data/dpo_pipeline_gooddata_detail2.json"
# data_file3 = "/mnt/bn/tiktok-mm-3/aiic/users/liyixuan/preprocessed_data/dpo_pipeline_gooddata_detail3.json"
# data_file4 = "/mnt/bn/tiktok-mm-3/aiic/users/liyixuan/preprocessed_data/dpo_pipeline_gooddata_detail4.json"
# data_file5 = "/mnt/bn/tiktok-mm-3/aiic/users/liyixuan/preprocessed_data/dpo_pipeline_gooddata_detail5.json"
# data_file6 = "/mnt/bn/tiktok-mm-3/aiic/users/liyixuan/preprocessed_data/dpo_pipeline_gooddata_detail6.json"

# print(len(data1), len(data2), len(data6))

# with open(data_file1, "w") as f:
#     json.dump(data1, f, indent=2)

# with open(data_file2, "w") as f:
#     json.dump(data2, f, indent=2)

# with open(data_file3, "w") as f:
#     json.dump(data3, f, indent=2)

# with open(data_file4, "w") as f:
#     json.dump(data4, f, indent=2)

# with open(data_file5, "w") as f:
#     json.dump(data5, f, indent=2)

# with open(data_file6, "w") as f:
#     json.dump(data6, f, indent=2)