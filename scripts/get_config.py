import json
import numpy as np


budget = 8192 * 36
temperature = 0.2
clip_max_visual = 1.5 / 36 * budget
clip_min_visual = 0.5 / 36 * budget
avratio = 2
clip_max_audio = clip_max_visual / avratio
clip_min_audio = clip_min_visual / avratio

with open("/mnt/bn/tiktok-mm-4/aiic/users/guangzhisun/thu_qwenvl3/plots/signals_baseline_audiovisual_norm.json") as fin:
    data = json.load(fin)

entropy_audio = []
entropy_visual = []
cossim_audio = []
cossim_visual = []
entropy_all = []
for datapiece in data:
    for chunk in datapiece[5:]:
        entropy_all.append(chunk["entropy_all"])
        entropy_audio.append(chunk["entropy_audio"])
        entropy_visual.append(chunk["entropy_visual"])
        cossim_visual.append(chunk["cosine_similarity_visual"])
        cossim_audio.append(chunk["cosine_similarity_audio"])
entropy_all = np.array(entropy_all).mean(axis=0)
entropy_audio = np.array(entropy_audio).mean(axis=0)
entropy_visual = np.array(entropy_visual).mean(axis=0)
cossim_visual = (1 - np.array(cossim_visual).astype(np.float32)).mean(axis=0)
cossim_audio = (1 - np.array(cossim_audio).astype(np.float32)).mean(axis=0)
cossim_all = (cossim_visual + cossim_audio) / 2
budgets_audio = (entropy_audio * cossim_audio ** 0.5)
budgets_visual = (entropy_visual * cossim_visual ** 0.5)
# budgets_audio_norm = np.exp(budgets_audio/temperature) / np.exp(budgets_audio/temperature).sum()
# budgets_visual_norm = np.exp(budgets_visual/temperature) / np.exp(budgets_visual/temperature).sum()
budgets_audio_w = budgets_audio / (budgets_audio + budgets_visual * avratio)
budgets_visual_w = budgets_visual * avratio / (budgets_audio + budgets_visual * avratio)

# budgets_all = (budgets_audio_norm + budgets_visual_norm * avratio) / (1 + avratio)
budgets_all = (entropy_all * cossim_all ** 0.5)
budgets_all_norm = np.exp(budgets_all/temperature) / np.exp(budgets_all/temperature).sum()
per_layer_budget_all = budgets_all_norm * budget
per_layer_budget_audio = per_layer_budget_all * budgets_audio_w
per_layer_budget_visual = per_layer_budget_all * budgets_visual_w
per_layer_budget_audio = np.clip(per_layer_budget_audio, clip_min_audio, clip_max_audio)
per_layer_budget_visual = np.clip(per_layer_budget_visual, clip_min_visual, clip_max_visual)

budgets_pair = [[audio, visual] for audio, visual in zip(per_layer_budget_audio.astype(np.int32).tolist(), per_layer_budget_visual.astype(np.int32).tolist())]
per_layer_budgets = per_layer_budget_all.astype(int).tolist()

# layer_id = 1
# budgets_pair = budgets_pair[:layer_id] + per_layer_budgets[layer_id:]

# with open("per_layer_budget_norm_temperature_{}_audiolarge_L1.json".format(temperature), "w") as f:
#     json.dump(per_layer_budgets, f, indent=4)

with open("per_layer_budget_norm_audiovisual_temperature_{}_audiolarge_8k_avratio2.json".format(temperature), "w") as f:
    json.dump(budgets_pair, f, indent=4)
