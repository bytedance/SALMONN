import os
import random
import json

subsets = ["train","valid","test"]
score_template = ["The score is {}.","The score of the quality of the speech sample is {}.", "{}."]

AB_test_prefix = ["{}.","It's {}.","{} is better."]
former_options = ["The first","The former"]
latter_options = ["The second","The latter"]

dataset_dir = "/path/to/somos_dataset"
result_dir = "/path/to/result_dir" # path to save the results

# AB test
for subset in subsets:
    data = open(os.path.join(dataset_dir,"training_files/split1/clean/{}_mos_list.txt".format(subset)),"r")
    data_json = {"annotation":[]}
    count = 1
    same_utters = []
    utter = ""
    line = data.readline()[:-1]
    while True:
        line = data.readline()[:-1]
        if not line:
            break
        split = line.find(',')
        wav = line[:split]
        score = float(line[split+1:])
        split2 = wav.rfind('_')
        sign = wav[:split2]
        if sign != utter:
            if len(same_utters) <= 1:
                pass
            else:
                for one_utter in same_utters:
                    repeated = [one_utter]
                    for _ in range(min(count,len(same_utters)-1)):
                        item = {}
                        other_utter = random.choice(same_utters)
                        while other_utter in repeated:
                            other_utter = random.choice(same_utters)
                        repeated.append(other_utter)
                        item.update({"task":"mos_ABtest"})
                        better = 2
                        item.update({"path":os.path.join(dataset_dir,"audios",one_utter["wav"])})
                        item.update({"expand_wav":[os.path.join(dataset_dir,"audios",other_utter["wav"])]})
                        if one_utter["score"] > other_utter["score"]:
                            better = 0
                        elif one_utter["score"] < other_utter["score"]:
                            better = 1
                        else:
                            better = 2
                        prefix_num = random.randint(0,len(AB_test_prefix)-1)
                        if better == 0:
                            answer = random.choice(former_options)
                        elif better == 1:
                            answer = random.choice(latter_options)
                        else:
                            answer = "it's a tie"
                        if prefix_num != 0 and prefix_num != len(AB_test_prefix)-1:
                            answer = answer.lower()
                        text = AB_test_prefix[prefix_num].format(answer)
                        item.update({"text":text})
                        item.update({"abs":abs(one_utter["score"]-other_utter["score"])})
                        if better != 2:
                            data_json["annotation"].append(item)
            same_utters = []
            same_utters.append({"wav":wav,"score":score})
            utter = sign
        else:
            same_utters.append({"wav":wav,"score":score})

    with open(os.path.join(result_dir,"somos_{}_ABtest.json".format(subset)), "w") as f:
        json.dump(data_json, f, indent=4, ensure_ascii=False)
    data.close()

# score
for subset in subsets:
    data = open(os.path.join(dataset_dir,"training_files/split1/clean/{}_mos_list.txt".format(subset)),"r")
    data_json = {"annotation":[]}
    line = data.readline()[:-1]
    while True:
        line = data.readline()[:-1]
        if not line:
            break
        item = {}
        split = line.find(',')
        wav = line[:split]
        score = line[split+1:]
        item.update({"path":os.path.join(dataset_dir,"audios",wav)})
        item.update({"task":"mos_evaluation_onlyscore_somos"})
        text = random.choice(score_template).format(score)
        item.update({"text":text})
        data_json["annotation"].append(item)

    with open(os.path.join(result_dir,"somos_{}_onlyscore.json".format(subset)),"w") as f:
        json.dump(data_json, f, indent=4, ensure_ascii=False)
    data.close()