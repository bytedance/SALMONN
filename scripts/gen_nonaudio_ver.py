import json
import sys

if __name__ == "__main__":
    data_file = sys.argv[1]
    audio_file = data_file.replace(".json", "_silent.json")
    with open(data_file, "r") as f:
        data = json.load(f)

    for d in data:
        d.pop("use_audio", False)
        d.pop("audio", None)
        d.pop("tos_audio", None)

    with open(audio_file, "w") as f:
        json.dump(data, f, indent=2)

    print(audio_file)