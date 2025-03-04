import os
import random
import json
import pandas as pd


subsets = ["TRAIN","VAL","TEST"]
score_template = ["The score is {}.","The score of the quality of the speech sample is {}.", "{}."]

noi_template = ["The background noise is {}.", "There is {} background noise in the speech sample."]
noi_des = ["very intrusive","somewhat intrusive","noticeable but not intrusive","slightly noticeable","not noticeable"]
col_dis_template = ["The speech sample is {} distorted and {}."]
col_des = ["very", "fairly", "somewhat", "slightly", "not"]
dis_des = [["very discontinuous","highly discontinuous"],["somewhat discontinuous","slightly discontinuous"],["not smooth","noticeably discontinuous"],["smooth","smoothly continuous"],["fluent","fluently continuous"]]
overall_template = ["The overall quality of the speech sample is {}.","Overall, the quality of the speech sample is {}."]
overall_des = ["bad","poor","fair","good","excellent"]
score_bar = [1,1.8,2.6,3.4,4.2,5.01]

def get_idx(score,bar):
    for i in range(len(bar)):
        if score < bar[i+1]:
            return i

dataset_dir = "/path/to/nisqa_dataset"
result_dir = "/path/to/result_dir" # path to save the results

data = pd.read_csv(os.path.join(dataset_dir,"NISQA_corpus_file.csv"))

# natural language description 
for subset in subsets:
    data_json = {"annotation":[]}
    for i in range(data.shape[0]):
        if subset in data.iloc[i]['db']:
            item = {}
            item.update({"path":os.path.join(dataset_dir,data.iloc[i]['db'],"deg",data.iloc[i]['filename_deg'])})
            item.update({"task":"mos_evaluation_description"})
            text = random.choice(col_dis_template).format(col_des[get_idx(data.iloc[i]['col'],score_bar)],random.choice(dis_des[get_idx(data.iloc[i]['dis'],score_bar)]))
            text += " " + random.choice(noi_template).format(noi_des[get_idx(data.iloc[i]['noi'],score_bar)])
            text += " " + random.choice(overall_template).format(overall_des[get_idx(data.iloc[i]['mos'],score_bar)])
            item.update({"text":text})
            data_json["annotation"].append(item)
    
    with open(os.path.join(result_dir,"nisqa_{}_description.json".format(subset.lower())), "w") as f:
        json.dump(data_json, f, indent=4, ensure_ascii=False)

# score 
for subset in subsets:
    data_json = {"annotation":[]}
    for i in range(data.shape[0]):
        if subset in data.iloc[i]['db']:
            item = {}
            item.update({"path":os.path.join(dataset_dir,data.iloc[i]['db'],"deg",data.iloc[i]['filename_deg'])})
            item.update({"task":"mos_evaluation_onlyscore_nisqa"})
            text = random.choice(score_template).format(str(round(data.iloc[i]['mos'],1)))
            item.update({"text":text})
            data_json["annotation"].append(item)
        
    with open(os.path.join(result_dir,"nisqa_{}_onlyscore.json".format(subset.lower())), "w") as f:
        json.dump(data_json, f, indent=4, ensure_ascii=False)
