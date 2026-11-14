"""
# TODO: usare `#` per don't care symbols
GP (DEAP) per generare formule LTLf il più possibile "ortogonali" tra loro.

Ortogonale = semanticamente diverse, misurato come basso overlap dei
sat-set su un pool di tracce (firma semantica).

Pipeline:
 1. Round di GP: ogni round evolve una popolazione la cui fitness
    penalizza la similarità con le formule già nell'archivio.
 2. Le migliori formule (mutuamente ortogonali) entrano nell'archivio.
 3. Si generano train/val/test/ood riusando le funzioni di
    generate_dataset_randltl.py (stesso formato JSON).
"""

import argparse
import json
import os
import random
import time

import numpy as np
from deap import base, creator, gp, tools

AP = ["p0", "p1", "p2", "p3"]
MIN_LEN = 3  # lunghezza minima delle tracce predefinita
MAX_LEN = 20  # lunghezza massima delle tracce (passi temporali)


# ----------------------------------------------------------------------
# 1. Pool di tracce (stessa distribuzione di random_trace del tuo script)
# ----------------------------------------------------------------------
def make_trace_pool(n, rng, min_len=MIN_LEN, max_len=MAX_LEN, n_ap=len(AP)):
    """Ritorna (vals[n, L, n_ap] bool, valid[n, L] bool).
    valid[t, i] = True se la posizione i esiste nella traccia t (padding)."""
    vals = rng.random((n, max_len, n_ap)) > 0.5
    lens = rng.integers(min_len, max_len + 1, size=n)
    valid = np.arange(max_len)[None, :] < lens[:, None]
    return vals, valid


# ----------------------------------------------------------------------
# 2. Semantica LTLf vettorizzata su tutte le tracce
#    Ogni nodo -> array bool [n_tracce, L]: res[t, i] = (trace_t, i |= f)
# ----------------------------------------------------------------------
def sem(node, vals, valid):
    op = node[0]
    n, L = valid.shape
    if op == "ap":
        return vals[:, :, node[1]] & valid
    if op == "NOT":
        return ~sem(node[1], vals, valid) & valid
    if op == "AND":
        return sem(node[1], vals, valid) & sem(node[2], vals, valid)
    if op == "OR":
        return sem(node[1], vals, valid) | sem(node[2], vals, valid)
    if op == "IMPL":
        a, b = sem(node[1], vals, valid), sem(node[2], vals, valid)
        return (~a | b) & valid

    a = sem(node[1], vals, valid)
    b = sem(node[2], vals, valid) if op == "U" else None
    res = np.zeros((n, L), dtype=bool)
    nxt = np.zeros(n, dtype=bool)  # res[:, i+1]; oltre L è False
    nxt_valid = np.zeros(n, dtype=bool)  # valid[:, i+1]
    for i in range(L - 1, -1, -1):
        v = valid[:, i]
        if op == "X":  # strong next: deve esistere il passo successivo
            cur = nxt_valid & _nxt_val(a, i, L)
        elif op == "F":
            cur = v & (a[:, i] | nxt)
        elif op == "G":  # alla fine della traccia G è vero
            cur = v & a[:, i] & (~nxt_valid | nxt)
        elif op == "U":  # b[i] oppure (a[i] e U al passo dopo)
            cur = v & (b[:, i] | (a[:, i] & nxt))
        else:
            raise ValueError(op)
        res[:, i] = cur
        nxt, nxt_valid = cur, v
    return res


def _nxt_val(a, i, L):
    return a[:, i + 1] if i + 1 < L else np.zeros(a.shape[0], dtype=bool)


def signature(node, vals, valid):
    """Firma semantica: la formula vale sulla traccia (posizione 0)?"""
    return sem(node, vals, valid)[:, 0]


# ----------------------------------------------------------------------
# 3. Conversione albero DEAP -> tupla annidata -> stringa Spot
# ----------------------------------------------------------------------
def to_nested(ind):
    """Walk ricorsivo dell'albero DEAP (notazione prefissa)."""
    def rec(i):
        node = ind[i]
        if node.arity == 0:
            return ("ap", AP.index(node.name)), i + 1
        children, j = [], i + 1
        for _ in range(node.arity):
            c, j = rec(j)
            children.append(c)
        return (node.name, *children), j
    return rec(0)[0]


def to_spot_str(node):
    op = node[0]
    if op == "ap":
        return AP[node[1]]
    if op == "NOT":
        return f"!({to_spot_str(node[1])})"
    if op == "AND":
        return f"({to_spot_str(node[1])} & {to_spot_str(node[2])})"
    if op == "OR":
        return f"({to_spot_str(node[1])} | {to_spot_str(node[2])})"
    if op == "IMPL":
        return f"({to_spot_str(node[1])} -> {to_spot_str(node[2])})"
    if op == "X":
        return f"X[!]({to_spot_str(node[1])})"  # strong next (LTLf)
    if op in ("F", "G"):
        return f"{op}({to_spot_str(node[1])})"
    if op == "U":
        return f"({to_spot_str(node[1])} U {to_spot_str(node[2])})"
    raise ValueError(op)


# ----------------------------------------------------------------------
# 4. Similarità tra firme
# ----------------------------------------------------------------------
def similarity(sig, archive):
    """Calcola la massima correlazione phi assoluta contro l'archivio.

    |phi| = 0: completa indipendenza / ortogonalità statistica.
    |phi| = 1: equivalenza empirica o esatta negazione logica.
    Fornisce una misura bilanciata senza penalizzare le formule dense (sat_rate > 0.5)
    come invece farebbe la metrica di Jaccard.
    """
    sig = np.asarray(sig, dtype=bool)
    archive = np.asarray(archive, dtype=bool)

    if archive.size == 0:
        return 0.0

    if sig.ndim != 1 or archive.ndim != 2:
        raise ValueError("Expected sig[T] and archive[K, T].")

    if sig.size == 0 or archive.shape[1] != sig.size:
        raise ValueError("Signatures must have the same non-zero length.")

    # Controllo rapido identici / opposti esatti
    is_identical = np.all(archive == sig[None, :], axis=1)
    is_opposite = np.all(archive == (~sig)[None, :], axis=1)

    total = float(sig.size)
    inter = np.count_nonzero(archive & sig[None, :], axis=1).astype(float)
    archive_pos = archive.sum(axis=1, dtype=float)
    sig_pos = float(sig.sum())

    numerator = inter * total - archive_pos * sig_pos
    denominator = np.sqrt(
        archive_pos * (total - archive_pos)
        * sig_pos * (total - sig_pos)
    )

    phi = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )

    scores = np.where(is_identical | is_opposite, 1.0, np.abs(phi))
    return float(np.clip(scores.max(), 0.0, 1.0))


# ----------------------------------------------------------------------
# 5. Setup DEAP
# ----------------------------------------------------------------------
def build_pset():
    pset = gp.PrimitiveSet("LTLF", 0)
    # i nomi dei nodi sono usati poi in to_nested
    pset.addPrimitive(lambda a: a, 1, name="NOT")
    pset.addPrimitive(lambda a: a, 1, name="X")
    pset.addPrimitive(lambda a: a, 1, name="F")
    pset.addPrimitive(lambda a: a, 1, name="G")
    pset.addPrimitive(lambda a, b: a, 2, name="AND")
    pset.addPrimitive(lambda a, b: a, 2, name="OR")
    pset.addPrimitive(lambda a, b: a, 2, name="IMPL")
    pset.addPrimitive(lambda a, b: a, 2, name="U")
    for p in AP:
        pset.addTerminal(p, name=p)
    return pset


def setup_toolbox(pset, max_height):
    # un solo obiettivo: massimizzare l'ortogonalità (fitness)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genFull, min_=0, max_=2)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
    # limite di altezza: evita il bloat
    for name in ("mate", "mutate"):
        tb.decorate(name, gp.staticLimit(
            key=lambda ind: ind.height, max_value=max_height))
    return tb


# ----------------------------------------------------------------------
# 6. Fitness
# ----------------------------------------------------------------------
class Evaluator:
    def __init__(self, vals, valid, size_penalty=0.003,
                 sat_lo=0.15, sat_hi=0.85):
        self.vals, self.valid = vals, valid
        self.size_penalty = size_penalty
        self.sat_lo, self.sat_hi = sat_lo, sat_hi
        self.archive = np.zeros((0, valid.shape[0]), dtype=bool)
        self.cache = {}  # stringa formula -> firma (evita ricalcoli)

    def sig_of(self, ind):
        nested = to_nested(ind)
        s = to_spot_str(nested)
        if s not in self.cache:
            self.cache[s] = signature(nested, self.vals, self.valid)
        return s, self.cache[s]

    def __call__(self, ind):
        _, sig = self.sig_of(ind)
        rate = sig.mean()
        # penalità se quasi sempre vera/falsa: non darebbe pos/neg bilanciati
        pen = 0.0
        if rate < self.sat_lo:
            pen += (self.sat_lo - rate) * 3
        if rate > self.sat_hi:
            pen += (rate - self.sat_hi) * 3
        sim = similarity(sig, self.archive)
        return (-sim - pen - self.size_penalty * len(ind),)

    def add(self, sig):
        self.archive = np.vstack([self.archive, sig[None, :]])


def run_round(tb, ev, pop_size, n_gen, cxpb, mutpb, k_add, sim_thr, seen):
    """Un round di GP; ritorna fino a k_add nuove formule mutuamente ortogonali."""
    pop = tb.population(n=pop_size)
    tb.register("evaluate", ev)
    for ind in pop:
        ind.fitness.values = ev(ind)

    for _ in range(n_gen):
        off = list(map(tb.clone, tb.select(pop, len(pop))))
        for a, b in zip(off[::2], off[1::2]):
            if random.random() < cxpb:
                tb.mate(a, b)
                del a.fitness.values, b.fitness.values
        for m in off:
            if random.random() < mutpb:
                tb.mutate(m)
                del m.fitness.values
        for ind in off:
            if not ind.fitness.valid:
                ind.fitness.values = ev(ind)
        # elitismo: i 5 migliori sopravvivono sempre
        pop = tools.selBest(pop, 5) + off[:-5]

    # selezione greedy: prendi i migliori, ma ortogonali anche tra loro
    added = []
    for ind in tools.selBest(pop, len(pop)):
        s, sig = ev.sig_of(ind)
        if s in seen:
            continue
        rate = sig.mean()
        if not (ev.sat_lo <= rate <= ev.sat_hi):
            continue
        if similarity(sig, ev.archive) > sim_thr:
            continue
        ev.add(sig)
        seen.add(s)
        added.append((s, len(ind)))
        if len(added) >= k_add:
            break
    return added


# ----------------------------------------------------------------------
# 7. Dataset (riusa le funzioni del tuo script originale)
# ----------------------------------------------------------------------
def sample_traces_fast(node, n_pos, n_neg, rng, min_len=MIN_LEN, max_len=MAX_LEN, batch_size=256, max_batches=20):
    """Campionamento ultra-veloce di tracce pos/neg usando la semantica vettorizzata sem()."""
    pos, neg = [], []
    for _ in range(max_batches):
        if len(pos) >= n_pos and len(neg) >= n_neg:
            break
        v, val = make_trace_pool(batch_size, rng, min_len=min_len, max_len=max_len)
        s = sem(node, v, val)[:, 0]
        for idx in range(batch_size):
            if s[idx] and len(pos) < n_pos:
                L = int(val[idx].sum())
                pos.append([{AP[j] for j in range(len(AP)) if v[idx, i, j]} for i in range(L)])
            elif (not s[idx]) and len(neg) < n_neg:
                L = int(val[idx].sum())
                neg.append([{AP[j] for j in range(len(AP)) if v[idx, i, j]} for i in range(L)])
            if len(pos) >= n_pos and len(neg) >= n_neg:
                break
    return pos, neg


def build_dataset(formulas, output_dir, traces_per_formula, seed, min_len=MIN_LEN, max_len=MAX_LEN, use_fast_sampler=True):
    from generate_dataset_randltl import (build_dfa, sample_traces_for_formula,
                                          trace_to_str)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    # OOD = formule più grandi (ultimo terzo per dimensione)
    formulas = sorted(formulas, key=lambda x: x[1])
    cut = int(len(formulas) * 0.8)
    iid, ood = formulas[:cut], formulas[cut:]
    random.shuffle(iid)
    n = len(iid)
    splits = {"train": iid[:int(.8 * n)], "val": iid[int(.8 * n):int(.9 * n)],
              "test": iid[int(.9 * n):], "ood": ood}
    half = traces_per_formula // 2
    for name, fl in splits.items():
        data = []
        for f, size in fl:
            try:
                if use_fast_sampler:
                    nested = parse_back(f)
                    pos, neg = sample_traces_fast(nested, half, half, rng, min_len=min_len, max_len=max_len)
                else:
                    pos, neg = sample_traces_for_formula(build_dfa(f), half, half, min_len=min_len, max_len=max_len)
            except Exception as e:  # noqa: BLE001
                print(f"skip {f}: {e}")
                continue
            if not pos or not neg:
                continue
            data += [{"formula": f, "trace": trace_to_str(t), "label": 1,
                      "depth": size} for t in pos]
            data += [{"formula": f, "trace": trace_to_str(t), "label": 0,
                      "depth": size} for t in neg]
        random.shuffle(data)
        with open(os.path.join(output_dir, f"{name}.json"), "w") as fp:
            json.dump(data, fp, indent=2)
        print(f"{name:5s}: {len(data)} esempi")


def verify_against_spot(formulas, vals, valid, n_check=100):
    """Controllo: il mio evaluator concorda con il DFA Spot del tuo script?"""
    from generate_dataset_randltl import accepts_trace, build_dfa
    bad = 0
    total_checks = 0
    for f, _ in formulas[:20]:
        aut = build_dfa(f)
        nested = parse_back(f)
        sig = signature(nested, vals, valid)
        for t in range(min(n_check, valid.shape[0])):
            L = int(valid[t].sum())
            tr = [{AP[j] for j in range(len(AP)) if vals[t, i, j]}
                  for i in range(L)]
            total_checks += 1
            if accepts_trace(aut, tr) != bool(sig[t]):
                bad += 1
    print(f"[verify] disaccordi con Spot: {bad}/{total_checks}")


def parse_back(s):
    """Parser minimale via Spot -> tupla annidata (vedi compare_datasets)."""
    from compare_datasets import spot_to_nested
    return spot_to_nested(s)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-formulas", type=int, default=1000)
    ap.add_argument("--traces-per-formula", type=int, default=4)
    ap.add_argument("--min-len", type=int, default=MIN_LEN,
                    help="lunghezza minima delle tracce")
    ap.add_argument("--max-len", type=int, default=MAX_LEN,
                    help="lunghezza massima delle tracce (passi temporali)")
    ap.add_argument("--pool-size", type=int, default=512,
                    help="tracce usate per la firma semantica")
    ap.add_argument("--pop-size", type=int, default=200)
    ap.add_argument("--n-gen", type=int, default=15)
    ap.add_argument("--k-add", type=int, default=5, help="formule per round")
    ap.add_argument("--sim-thr", type=float, default=0.6,
                    help="soglia massima di similarità per entrare")
    ap.add_argument("--max-height", type=int, default=7,
                    help="altezza massima alberi GP (consigliato >= 7 per orizzonti lunghi)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="data_gp")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--slow-spot-sampler", action="store_true",
                    help="usa l'oracolo Spot invece di sem() per il campionamento finale")
    args = ap.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    vals, valid = make_trace_pool(args.pool_size, rng, min_len=args.min_len, max_len=args.max_len)

    pset = build_pset()
    tb = setup_toolbox(pset, args.max_height)
    ev = Evaluator(vals, valid)

    seen, formulas = set(), []
    t0, stall = time.perf_counter(), 0
    while len(formulas) < args.n_formulas and stall < 50:
        new = run_round(tb, ev, args.pop_size, args.n_gen, 0.7, 0.3,
                        args.k_add, args.sim_thr, seen)
        formulas += new
        stall = 0 if new else stall + 1
        print(f"formule={len(formulas):5d} (+{len(new)}) "
              f"t={time.perf_counter() - t0:.0f}s")

    formulas = formulas[:args.n_formulas]
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "formulas.json"), "w") as fp:
        json.dump(formulas, fp, indent=2)

    if args.verify:
        verify_against_spot(formulas, vals, valid)
    build_dataset(formulas, args.output_dir, args.traces_per_formula, args.seed,
                  min_len=args.min_len, max_len=args.max_len,
                  use_fast_sampler=not args.slow_spot_sampler)


if __name__ == "__main__":
    main()