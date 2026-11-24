import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
data_dir = os.path.join(repo_root, "data", "dataset")

s = {n: {d["formula"] for d in json.load(open(os.path.join(data_dir, f"{n}.json")))}
     for n in ("train", "val", "test", "ood")}
for a in s:
    for b in s:
        if a < b:
            print(a, b, len(s[a] & s[b]))