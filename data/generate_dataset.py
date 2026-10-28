#!/usr/bin/env python3
"""
Dataset generator for LTLfBERT using Spot.

Richiede Spot con Python bindings installati.
Pipeline:
1. Genera formule con randltl
2. Converte LTLf -> automa finito via Spot
3. Campiona tracce positive/negative usando l'automa come oracolo
4. Salva train/val/test/ood in JSON
"""

import json
import os
import random
import subprocess
from typing import List, Dict, Tuple

try:
    import spot
except Exception as e:
    raise RuntimeError(
        "Spot non è installato o i Python bindings non sono disponibili. "
        "Installa Spot prima di usare questo script."
    ) from e

PROPS = ["p0", "p1", "p2", "p3"]


def is_satisfiable(formula: str) -> bool:
    """Controlla se una formula LTLf è soddisfacibile usando Spot."""
    try:
        f = spot.from_ltlf(formula)
        return not spot.is_empty(f.translate('small', 'buchi', 'sbacc'))
    except Exception:
        return False


def random_trace(min_len=3, max_len=8, props=PROPS):
    length = random.randint(min_len, max_len)
    trace = []
    for _ in range(length):
        state = {p for p in props if random.random() > 0.5}
        trace.append(state)
    return trace


def trace_to_str(trace, props=PROPS):
    return " ".join("".join("1" if p in s else "0" for p in props) for s in trace)


def randltl_generate(n=100, tree_size=8, seed=42, props=PROPS) -> List[str]:
    cmd = [
        "randltl",
        "-n", str(n),
        "--tree-size", str(tree_size),
        "--seed", str(seed),
        "-p", str(len(props)),
    ]
    out = subprocess.check_output(cmd, text=True)
    formulas = [line.strip() for line in out.splitlines() if line.strip()]
    return formulas


def build_finite_automaton(formula: str):
    f = spot.from_ltlf(formula)
    aut = f.translate('small', 'buchi', 'sbacc')
    finite = spot.to_finite(aut)
    return finite


def _trace_to_word(trace, props=PROPS) -> List[set]:
    return [set(s) for s in trace]


def accepts_trace(formula: str, trace, props=PROPS) -> bool:
    aut = build_finite_automaton(formula)
    bdict = aut.get_dict()
    current = {aut.get_init_state_number()}
    word = _trace_to_word(trace, props)
    for letter in word:
        nxt = set()
        for s in current:
            for e in aut.out(s):
                cond = spot.bdd_format_formula(bdict, e.cond)
                if cond_holds(cond, letter):
                    nxt.add(e.dst)
        current = nxt
        if not current:
            return False
    for s in current:
        if aut.state_is_accepting(s):
            return True
    return False


def cond_holds(cond_str: str, letter: set) -> bool:
    expr = cond_str.strip()
    if expr in ('1', 'true', 't'): return True
    if expr in ('0', 'false', 'f'): return False
    expr = expr.replace('!', ' not ').replace('&', ' and ').replace('|', ' or ')
    env = {p: (p in letter) for p in PROPS}
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False


def sample_traces_for_formula(formula: str, n_pos=2, n_neg=2, max_attempts=500):
    pos, neg = [], []
    for _ in range(max_attempts):
        if len(pos) >= n_pos and len(neg) >= n_neg: break
        tr = random_trace()
        sat = accepts_trace(formula, tr)
        if sat and len(pos) < n_pos:
            pos.append(tr)
        elif (not sat) and len(neg) < n_neg:
            neg.append(tr)
    return pos, neg


def generate_dataset(
        n_formulas=10000,
        traces_per_formula=4,
        complexity_buckets=(6, 10, 14),
        seed=42,
        output_dir=None,
):
    random.seed(seed)
    if output_dir is None: output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)

    formulas = []
    per_bucket = max(1, n_formulas // len(complexity_buckets))
    for i, tree_size in enumerate(complexity_buckets):
        fs = randltl_generate(n=per_bucket, tree_size=tree_size, seed=seed + i)
        formulas.extend((f, i + 1) for f in fs)

    dataset = []
    skipped = 0
    n_pos = traces_per_formula // 2
    n_neg = traces_per_formula // 2

    for formula, depth in formulas[:n_formulas]:
        try:
            sat_val = is_satisfiable(formula)
            pos, neg = sample_traces_for_formula(formula, n_pos=n_pos, n_neg=n_neg)

            if len(pos) < 1 or len(neg) < 1:
                skipped += 1
                continue

            for tr in pos:
                dataset.append({
                    "formula": formula, "trace": trace_to_str(tr),
                    "label": 1, "sat_label": int(sat_val), "depth": depth,
                })
            for tr in neg:
                dataset.append({
                    "formula": formula, "trace": trace_to_str(tr),
                    "label": 0, "sat_label": int(sat_val), "depth": depth,
                })
        except Exception:
            skipped += 1
            continue

    random.shuffle(dataset)
    ood_depth = max(d for _, d in formulas[:n_formulas]) if formulas else 3
    ood = [x for x in dataset if x["depth"] == ood_depth]
    iid = [x for x in dataset if x["depth"] != ood_depth]

    splits = {"train": iid[:int(0.8 * len(iid))], "val": iid[int(0.8 * len(iid)):int(0.9 * len(iid))],
              "test": iid[int(0.9 * len(iid)):], "ood": ood}

    for name, data in splits.items():
        with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
            json.dump(data, f, indent=2)
        pos = sum(1 for d in data if d["label"] == 1)
        print(f"{name:6s}: {len(data):5d} | pos={pos} neg={len(data) - pos}")

    print(f"Dataset totale: {len(dataset)} esempi | skipped={skipped}")
    return splits


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-formulas", type=int, default=5000)
    args = p.parse_args()
    generate_dataset(n_formulas=args.n_formulas)