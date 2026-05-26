import json
import sys, os


with open(sys.argv[1]) as fin:
    data = json.load(fin)

print(len(data))
print(sum(data) / len(data))