import json
import random
import sys

in_file = sys.argv[1]
batch_size = int(sys.argv[2])
out_file = in_file.replace(".json", "_batch.json")

with open(in_file, "r") as f:
    data = json.load(f)

random.shuffle(data)

out_data = []

for idx in range(0, len(data), batch_size):
    if idx + batch_size > len(data):
        break
    out_data.append(
        {
            "batch_audio": data[idx:idx+batch_size],
            "conversations": [
                {
                    "from": "human", 
                    "value": ""
                },
                {
                    "from": "gpt", 
                    "value": ""
                }
            ]
        }
    )

print(len(out_data))

print(out_file)

with open(out_file, "w") as f:
    json.dump(out_data, f, indent=4)