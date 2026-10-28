import json
s = {n: {d["formula"] for d in json.load(open(f"data/dataset/{n}.json"))}
     for n in ("train", "val", "test", "ood")}
for a in s:
    for b in s:
        if a < b:
            print(a, b, len(s[a] & s[b]))