#!/usr/bin/env python3
"""
Dataset generator for LTLfBERT using Spot.

Richiede Spot con Python bindings installati.
Pipeline:
1. Genera formule con randltl
2. Deduplica le formule
3. Divide le FORMULE (non le coppie formula-traccia) in train/val/test
4. Separa le formule più complesse come OOD
5. Campiona tracce positive/negative usando l'automa come oracolo
6. Salva train/val/test/ood in JSON
"""

import json
import os
import random
import subprocess
from typing import List, Tuple

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
    if expr in ('1', 'true', 't'):
        return True
    if expr in ('0', 'false', 'f'):
        return False
    expr = expr.replace('!', ' not ').replace('&', ' and ').replace('|', ' or ')
    env = {p: (p in letter) for p in PROPS}
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False


def sample_traces_for_formula(formula: str, n_pos=2, n_neg=2, max_attempts=500):
    pos, neg = [], []
    for _ in range(max_attempts):
        if len(pos) >= n_pos and len(neg) >= n_neg:
            break
        tr = random_trace()
        sat = accepts_trace(formula, tr)
        if sat and len(pos) < n_pos:
            pos.append(tr)
        elif (not sat) and len(neg) < n_neg:
            neg.append(tr)
    return pos, neg


def build_examples(formula: str, depth: int, n_pos: int, n_neg: int):
    """Genera gli esempi (formula, traccia) per UNA formula."""
    sat_val = is_satisfiable(formula)
    pos, neg = sample_traces_for_formula(formula, n_pos=n_pos, n_neg=n_neg)

    if len(pos) < 1 or len(neg) < 1:
        return None

    rows = []
    for label, traces in ((1, pos), (0, neg)):
        for tr in traces:
            rows.append({
                "formula": formula,
                "trace": trace_to_str(tr),
                "label": label,
                "sat_label": int(sat_val),
                "depth": depth,
            })
    return rows


def generate_dataset(
    n_formulas=10000,
    traces_per_formula=4,
    complexity_buckets=(6, 10, 14),
    seed=42,
    output_dir=None,
    train_frac=0.8,
    val_frac=0.1,
    test_frac=0.1,
):
    random.seed(seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if output_dir is None:
        # crea <cartella_dello_script>/dataset/
        output_dir = os.path.join(script_dir, "dataset")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Genera le formule e deduplica (prima occorrenza vince)
    # ------------------------------------------------------------------
    unique_formulas = {}  # formula -> depth
    per_bucket = max(1, n_formulas // len(complexity_buckets))

    for i, tree_size in enumerate(complexity_buckets):
        fs = randltl_generate(n=per_bucket, tree_size=tree_size, seed=seed + i)
        for f in fs:
            if f not in unique_formulas:
                unique_formulas[f] = i + 1

    # Limita al numero richiesto
    unique_formulas = dict(list(unique_formulas.items())[:n_formulas])
    print(f"Formule uniche generate: {len(unique_formulas)}")

    # ------------------------------------------------------------------
    # 2) Separa OOD: tutte le formule del bucket più profondo
    # ------------------------------------------------------------------
    ood_depth = max(unique_formulas.values())
    ood_formulas = sorted(f for f, d in unique_formulas.items() if d == ood_depth)
    iid_formulas = sorted(f for f, d in unique_formulas.items() if d != ood_depth)

    print(f"OOD (depth={ood_depth}): {len(ood_formulas)} formule")
    print(f"IID: {len(iid_formulas)} formule")

    # ------------------------------------------------------------------
    # 3) Split A LIVELLO DI FORMULA (non di esempio)
    # ------------------------------------------------------------------
    random.shuffle(iid_formulas)

    n_iid = len(iid_formulas)
    n_test = int(test_frac * n_iid)
    n_val = int(val_frac * n_iid)
    n_train = n_iid - n_val - n_test

    formula_splits = {
        "train": iid_formulas[:n_train],
        "val": iid_formulas[n_train:n_train + n_val],
        "test": iid_formulas[n_train + n_val:n_train + n_val + n_test],
        "ood": ood_formulas,
    }

    # ------------------------------------------------------------------
    # 4) Sanity check: nessuna formula in più di uno split
    # ------------------------------------------------------------------
    names = list(formula_splits)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            inter = set(formula_splits[names[a]]) & set(formula_splits[names[b]])
            assert not inter, \
                f"LEAKAGE: {len(inter)} formule condivise tra {names[a]} e {names[b]}"
    print("Sanity check leakage: OK (nessuna formula condivisa tra split)")

    # ------------------------------------------------------------------
    # 5) Solo ORA genera le tracce per ogni formula
    # ------------------------------------------------------------------
    n_pos = traces_per_formula // 2
    n_neg = traces_per_formula // 2

    splits = {}
    skipped = 0

    for name, formula_list in formula_splits.items():
        data = []
        for formula in formula_list:
            try:
                depth = unique_formulas[formula]
                rows = build_examples(formula, depth, n_pos, n_neg)
            except Exception:
                rows = None

            if rows is None:
                skipped += 1
                continue
            data.extend(rows)

        # shuffle dei soli esempi DENTRO lo split (sicuro)
        random.shuffle(data)
        splits[name] = data

    # ------------------------------------------------------------------
    # 6) Salva i JSON
    # ------------------------------------------------------------------
    for name, data in splits.items():
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        pos = sum(1 for d in data if d["label"] == 1)
        n_formule = len({d["formula"] for d in data})
        print(f"{name:6s}: {len(data):5d} esempi | {n_formule:4d} formule | "
              f"pos={pos} neg={len(data) - pos}")

    total = sum(len(v) for v in splits.values())
    print(f"Dataset totale: {total} esempi | formule scartate={skipped}")
    return splits


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n-formulas", type=int, default=5000)
    p.add_argument("--traces-per-formula", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    args = p.parse_args()

    generate_dataset(
        n_formulas=args.n_formulas,
        traces_per_formula=args.traces_per_formula,
        seed=args.seed,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )