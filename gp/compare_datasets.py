"""
Confronto tra due dataset LTLfBERT (es. randltl vs GP).

Uso:
  python compare_datasets.py --a data --b data_gp --out report.json
  python compare_datasets.py --run-gp --a data   # lancia prima gp_ltlf.py
"""

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
import pickle

# Assicura che la root del repository sia sempre in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import spot

from gp_ltlf import AP, make_trace_pool, signature, similarity

SPLITS = ["train", "val", "test", "ood"]


# ----------------------------------------------------------------------
# Spot formula -> tupla annidata (stesso formato di gp_ltlf.sem)
# ----------------------------------------------------------------------
def spot_to_nested(s):
    f = spot.formula(s) if isinstance(s, str) else s
    # riduce a {!, &, |, X, F, G, U}; elimina W, R, M, ->, <->, xor
    f = f.unabbreviate("WRMie^")
    return _conv(f)


def _conv(f):
    k = f.kind()
    if f.is_tt():  # true  = p | !p
        return ("OR", ("ap", 0), ("NOT", ("ap", 0)))
    if f.is_ff():
        return ("AND", ("ap", 0), ("NOT", ("ap", 0)))
    if k == spot.op_ap:
        return ("ap", AP.index(f.ap_name()))
    if k == spot.op_Not:
        return ("NOT", _conv(f[0]))
    if k in (spot.op_And, spot.op_Or):
        name = "AND" if k == spot.op_And else "OR"
        node = _conv(f[0])
        for i in range(1, f.size()):  # And/Or di Spot sono n-ari
            node = (name, node, _conv(f[i]))
        return node
    if k in (spot.op_X, spot.op_strong_X):
        return ("X", _conv(f[0]))
    if k == spot.op_F:
        return ("F", _conv(f[0]))
    if k == spot.op_G:
        return ("G", _conv(f[0]))
    if k == spot.op_U:
        return ("U", _conv(f[0]), _conv(f[1]))
    raise ValueError(f"operatore non supportato: {f}")


def size_depth_ops(f):
    """Numero di nodi, profondità e contatore degli operatori."""
    ops = Counter()

    def rec(g):
        ops[g.kind_name() if hasattr(g, "kind_name") else str(g.kind())] += 1
        if g.size() == 0:
            return 1, 1
        parts = [rec(g[i]) for i in range(g.size())]
        return 1 + sum(p[0] for p in parts), 1 + max(p[1] for p in parts)

    n, d = rec(f)
    return n, d, ops


# ----------------------------------------------------------------------
# Caricamento
# ----------------------------------------------------------------------
def load(dirpath):
    data = {}
    for s in SPLITS:
        p = os.path.join(dirpath, f"{s}.json")
        if os.path.exists(p):
            with open(p) as fp:
                data[s] = json.load(fp)
    return data


def entropy(counter):
    tot = sum(counter.values())
    return -sum(c / tot * math.log2(c / tot) for c in counter.values() if c)


# ----------------------------------------------------------------------
# Metriche per singolo dataset
# ----------------------------------------------------------------------
def formula_signatures(formulas, vals, valid):
    """Evaluate the existing held-out pool using the shared Spot oracle."""
    from data.generate_dataset import (
        PROPS,
        build_ltlf_oracle,
        trace_to_spot_word,
    )

    words = []

    for index in range(valid.shape[0]):
        length = int(valid[index].sum())

        trace = [
            {
                proposition
                for column, proposition in enumerate(PROPS)
                if vals[index, position, column]
            }
            for position in range(length)
        ]

        words.append(trace_to_spot_word(trace))

    signatures = []

    for index, formula in enumerate(formulas, start=1):
        try:
            aut = build_ltlf_oracle(formula)
            signatures.append(
                [bool(aut.intersects(word)) for word in words]
            )
        except Exception as exc:
            raise RuntimeError(
                f"Spot evaluation failed for formula: {formula}"
            ) from exc

        if index % 100 == 0:
            print(
                f"[compare/Spot] {index}/{len(formulas)} formulas",
                flush=True,
            )

    return list(formulas), np.asarray(signatures, dtype=bool).reshape(
        len(formulas), valid.shape[0]
    )


def pairwise_stats(S, max_pairs=200_000, rng=None):
    """Statistiche di similarità a coppie (campionate se troppe)."""
    n = S.shape[0]
    rng = rng or np.random.default_rng(0)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)] \
        if n * (n - 1) // 2 <= max_pairs else \
        [tuple(rng.choice(n, 2, replace=False)) for _ in range(max_pairs)]
    sims = np.array([similarity(S[i], S[j][None, :]) for i, j in pairs])
    # distanza al vicino più prossimo (quanto è "isolata" ogni formula)
    nn = []
    for i in range(min(n, 500)):
        others = np.delete(S, i, axis=0)
        nn.append(similarity(S[i], others))
    return {
        "pair_sim_mean": float(sims.mean()),
        "pair_sim_p95": float(np.percentile(sims, 95)),
        "pair_sim_max": float(sims.max()),
        "frac_pairs_sim_gt_0.9": float((sims > 0.9).mean()),
        "nn_sim_mean": float(np.mean(nn)),
        "unique_signatures": int(len({s.tobytes() for s in S})),
        "n_formulas": int(n),
    }


def dataset_metrics(data, vals, valid):
    all_rows = [r for s in data.values() for r in s]
    formulas = sorted({r["formula"] for r in all_rows})
    out = {"n_examples": len(all_rows), "n_unique_formulas": len(formulas)}

    # sintattiche
    sizes, depths, ops = [], [], Counter()
    for f in formulas:
        n, d, o = size_depth_ops(spot.formula(f))
        sizes.append(n)
        depths.append(d)
        ops.update(o)
    out["size_mean"], out["size_std"] = float(np.mean(sizes)), float(np.std(sizes))
    out["depth_mean"] = float(np.mean(depths))
    out["op_entropy_bits"] = entropy(ops)
    out["op_hist"] = dict(ops)

    # tracce
    lens = [len(r["trace"].split()) for r in all_rows]
    labels = [r["label"] for r in all_rows]
    out["trace_len_mean"] = float(np.mean(lens))
    out["label_pos_frac"] = float(np.mean(labels))
    by_trace = {}
    for r in all_rows:
        by_trace.setdefault(r["trace"], []).append((r["formula"], r["label"]))
    out["unique_traces"] = len(by_trace)
    out["trace_dup_frac"] = 1 - len(by_trace) / len(all_rows)
    # tracce condivise da più formule, e con label in conflitto
    shared = [v for v in by_trace.values() if len({f for f, _ in v}) > 1]
    out["traces_shared_by_formulas_frac"] = len(shared) / len(by_trace)
    out["traces_conflicting_labels_frac"] = (
        sum(1 for v in shared if len({l for _, l in v}) > 1) / len(by_trace))

    # leakage tra split (formule uguali in split diversi)
    seen, leak = {}, 0
    for s, rows in data.items():
        for f in {r["formula"] for r in rows}:
            if f in seen:
                leak += 1
            seen[f] = s
    out["formula_leak_across_splits"] = leak

    # semantiche
    ok, S = formula_signatures(formulas, vals, valid)
    out["n_formulas_evaluable"] = len(ok)
    out["sat_rate_mean"] = float(S.mean())
    out["sat_rate_std"] = float(S.mean(1).std())
    out.update(pairwise_stats(S))
    return out, ok, S


# ----------------------------------------------------------------------
# Cross-dataset
# ----------------------------------------------------------------------
def cross_metrics(fa, Sa, fb, Sb):
    common = set(fa) & set(fb)
    # per ogni formula di B, la similarità max con A (e viceversa)
    b_to_a = [similarity(Sb[i], Sa) for i in range(len(fb))]
    a_to_b = [similarity(Sa[i], Sb) for i in range(min(len(fa), 500))]
    return {
        "common_formulas": len(common),
        "B_to_A_max_sim_mean": float(np.mean(b_to_a)),
        "A_to_B_max_sim_mean": float(np.mean(a_to_b)),
        "B_frac_with_equiv_in_A(sim>0.95)": float(np.mean(np.array(b_to_a) > .95)),
    }


def op_kl(opa, opb, eps=1e-9):
    keys = set(opa) | set(opb)
    ta, tb = sum(opa.values()), sum(opb.values())
    pa = np.array([opa.get(k, 0) / ta + eps for k in keys])
    pb = np.array([opb.get(k, 0) / tb + eps for k in keys])
    return float(np.sum(pa * np.log(pa / pb)))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="dir dataset A (randltl)")
    ap.add_argument("--b", default="data_gp", help="dir dataset B (GP)")
    ap.add_argument("--run-gp", action="store_true",
                    help="lancia gp_ltlf.py per generare B")
    ap.add_argument("--gp-args", nargs=argparse.REMAINDER, default=[],
                    help="argomenti extra per gp_ltlf.py")
    ap.add_argument("--eval-pool", type=int, default=1024,
                    help="tracce held-out per le metriche semantiche")
    ap.add_argument("--min-len", type=int, default=3,
                    help="lunghezza minima delle tracce di valutazione")
    ap.add_argument("--max-len", type=int, default=20,
                    help="lunghezza massima delle tracce di valutazione")
    ap.add_argument("--seed", type=int, default=12345,
                    help="diverso dal seed del GP -> pool indipendente")
    ap.add_argument("--out", default="report.json")

    args = ap.parse_args()

    if args.run_gp:
        subprocess.run([sys.executable, "gp_ltlf.py", "--output-dir", args.b,
                        *args.gp_args], check=True)

    rng = np.random.default_rng(args.seed)
    vals, valid = make_trace_pool(args.eval_pool, rng, min_len=args.min_len, max_len=args.max_len)

    A, B = load(args.a), load(args.b)
    ma, fa, Sa = dataset_metrics(A, vals, valid)
    mb, fb, Sb = dataset_metrics(B, vals, valid)
    cross = cross_metrics(fa, Sa, fb, Sb)
    cross["op_KL(A||B)"] = op_kl(ma["op_hist"], mb["op_hist"])

    report = {"A": ma, "B": mb, "cross": cross}
    with open(args.out, "w") as fp:
        json.dump(report, fp, indent=2)

    # tabella a console
    keys = [k for k in ma if k != "op_hist"]
    print(f"{'metrica':42s} {'A':>12s} {'B':>12s}")
    for k in keys:
        va, vb = ma[k], mb[k]
        fmt = (lambda v: f"{v:12.4f}" if isinstance(v, float) else f"{v:12d}")
        print(f"{k:42s} {fmt(va)} {fmt(vb)}")
    print("\nCross:", json.dumps(cross, indent=2))


if __name__ == "__main__":
    main()