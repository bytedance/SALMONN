import os
import random
import json

subsets = ["train","test"]
score_template = ["The score is {}.","The score of speaker similarity is {}.", "{}."]
dataset_dir = "/path/to/voxceleb_dataset"
result_dir = "/path/to/result_dir" # path to save the results

for subset in subsets:
    if subset == "train":
        data_txt = open(os.path.join(dataset_dir,"voxsim_{}_list_average.txt".format(subset)),"r")
    else:
        data_txt = open(os.path.join(dataset_dir,"voxsim_{}_list.txt".format(subset)),"r")
    data_json = {"annotation":[]}

    while True:
        item = {}
        line = data_txt.readline()[:-1]
        if not line:
            break
        onepiece_data = line.split(',')
        if os.path.exists(os.path.join(dataset_dir,"dev/wav",onepiece_data[0])):
            item.update({"path":os.path.join(dataset_dir,"dev/wav",onepiece_data[0])})
        else:
            item.update({"path":os.path.join(dataset_dir,"test/wav",onepiece_data[0])})
        if os.path.exists(os.path.join(dataset_dir,"dev/wav",onepiece_data[1])):
            item.update({"expand_wav":[os.path.join(dataset_dir,"dev/wav",onepiece_data[1])]})
        else:
            item.update({"expand_wav":[os.path.join(dataset_dir,"test/wav",onepiece_data[1])]})
        item.update({"task":"spk_evaluation_onlyscore"})
        text = random.choice(score_template).format(float(onepiece_data[-1]))
        item.update({"text":text})
        data_json["annotation"].append(item)

    with open(os.path.join(result_dir,"voxsim_{}_onlyscore.json".format(subset)), "w") as f:
        json.dump(data_json, f, indent=4, ensure_ascii=False)