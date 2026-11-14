"""
Dataset generator for LTLfBERT using Spot.

Pipeline:
1. Generate formulas using randltl
2. Convert LTLf to a finite state machine using Spot
3. Sample positive/negative traces using the finite state machine as an oracle
4. Save the train/val/test/ood datasets in JSON format
"""

import argparse
import json
import os
import random
import time

import spot

DEBUG = True
AP = ["p0", "p1", "p2", "p3"]


# TODO
# ho deciso per ora di togliere i types
# poiché erano sporadici e poco utili per ora
# si può pensare di aggiungerli in futuro
# ma non sarà un lavoro breve


# in ogni istante c'è una probabilità del 50%
# che la proposizione i-esima sia vera.

# IDEA
# magari si puù vedere se con altri algoritmi
# si ottengono risultati diversi?

# IDEA
# ltlfilt --remove-wm
# rewrite operators W and M using U and R
# per ridurre i tipi di simboli?
# https://spot.lre.epita.fr/man/ltlfilt.1.html
def random_trace(min_len=3, max_len=20, ap=AP):
    length = random.randint(min_len, max_len)
    trace = []
    for _ in range(length):
        state = {p for p in ap if random.random() > 0.5}
        trace.append(state)
    return trace


# TODO
# vedere se in fase di tokenizzazione
# effettivamente viene trattata bene
def trace_to_str(trace, ap=AP):
    # 0101 1101 0001 ...
    return " ".join("".join("1" if p in s else "0" for p in ap) for s in trace)


# CHANGE: da subprocess a metodo della libreria
def randltl_generate(n=100, tree_size=8, seed=67, ap=AP):
    f = spot.randltl(
        n=n,
        tree_size=tree_size,
        seed=seed,
        ap=len(ap),
        allow_dup=False
    )
    fs = [str(i) for i in f]
    return fs


# TODO
# vecchio codice (1) per generare automi
# non so se sia corretto perché mi manca la teoria
# funionava con il vecchio accept ma era hadcodato
# e utilizzava eval di python,
# che sicuramente non è la soluzione migliore

# (1)
# buchi_aut = f.translate("small", "buchi", "sbacc")
# aut = spot.to_finite(buchi_aut)

# (2)
# aut = f.translate("finite", "deterministic", "sbacc")

# (2) dà problemi con l'eval ovviamente (alive)
# bisognorebbe farlo usando spot meglio

# UPDATE
# cambiata accepts_trace ma non funziona correttamente
# poiché non ho idea di come fare lol

# UPDATE 2
# nel notebook bisogna vedere bene se funziona e come
# (ho gestito 'alive' come missing key)

# UPDATE 3
# forse ci siamo!
# il vecchio modo (1) crea un automa di buchi
# (che funziona per tracce infinite)
# lo fa "small" e poi lo trasforma in finito,
# quindi senza usare "alive" (poco chiaro)
# ---
# invece la (2) usa la forma più moderna.
# crea già automa per tracce "finite"
# crea direttamente un automa deterministico
# e "sbacc", cioè State-Based Acceptance,
# l'accettazione dipende dallo stato in cui arriva,
# non dall'arco che si attraversa.
# dovrebbe essere meglio per le tracce finite.
# da questo, deduco che è la direzione giusta da prendere,
# lavoro quindi per trovare un accept trace che usa
# solo funzioni spot native, per efficenza e correttezza.
# adesso dovremmo avere un vero DFA.

def build_dfa(formula_str):
    """Costruisce un oracolo Buchi su parole infinite tramite riduzione standard LTLf (alive)."""
    f = spot.formula(formula_str)
    embedded = spot.from_ltlf(f)
    return embedded.translate("small", "buchi", "sbacc")


def trace_to_spot_word(trace, ap=AP):
    """Codifica una traccia finita come parola ultimamente periodica con suffisso !alive e !ap."""
    steps = []
    for state in trace:
        conds = [p if p in state else f"!{p}" for p in ap]
        conds.append("alive")
        steps.append(" & ".join(conds))

    suffix = " & ".join(["!alive", *(f"!{p}" for p in ap)])
    word_str = " ; ".join(steps) + f" ; cycle{{{suffix}}}"
    return spot.parse_word(word_str)


def accepts_trace(aut, trace, ap=AP):
    """Verifica l'accettazione della traccia tramite intersezione nativa in Spot."""
    word = trace_to_spot_word(trace, ap=ap)
    return bool(aut.intersects(word))


def sample_traces_for_formula(aut, n_pos=2, n_neg=2, min_len=3, max_len=20, max_attempts=500):
    pos, neg = [], []
    for _ in range(max_attempts):
        if len(pos) >= n_pos and len(neg) >= n_neg:
            break
        tr = random_trace(min_len=min_len, max_len=max_len)
        sat = accepts_trace(aut, tr)
        if sat and len(pos) < n_pos:
            pos.append(tr)
        elif (not sat) and len(neg) < n_neg:
            neg.append(tr)
    return pos, neg


# TODO rivedi
def generate_dataset(
        n_formulas=1000,
        traces_per_formula=4,
        complexity_buckets=(6, 10, 14),
        seed=67,
        output_dir=None,
        min_len=3,
        max_len=20,
):
    # TODO
    # lo split si deve fare prima della creazione delle tracce!!

    random.seed(seed)
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "data")
    os.makedirs(output_dir, exist_ok=True)

    per_bucket = max(1, n_formulas // len(complexity_buckets))

    iid_formulas = []
    ood_formulas = []
    ood_depth = len(complexity_buckets)

    for i, tree_size in enumerate(complexity_buckets):
        depth = i + 1
        fs = randltl_generate(n=per_bucket, tree_size=tree_size, seed=seed + i)

        if depth == ood_depth:
            ood_formulas.extend((f, depth) for f in fs)
        else:
            iid_formulas.extend((f, depth) for f in fs)

    random.shuffle(iid_formulas)
    n_iid = len(iid_formulas)

    train_f = iid_formulas[: int(0.8 * n_iid)]
    val_f = iid_formulas[int(0.8 * n_iid): int(0.9 * n_iid)]
    test_f = iid_formulas[int(0.9 * n_iid):]

    formula_splits = {
        "train": train_f,
        "val": val_f,
        "test": test_f,
        "ood": ood_formulas,
    }

    n_pos = traces_per_formula // 2
    n_neg = traces_per_formula // 2
    skipped = 0
    total_examples = 0

    for split_name, f_list in formula_splits.items():
        split_data = []

        for formula, depth in f_list:
            try:
                aut = build_dfa(formula)
                pos, neg = sample_traces_for_formula(
                    aut, n_pos=n_pos, n_neg=n_neg, min_len=min_len, max_len=max_len
                )

                if len(pos) < 1 or len(neg) < 1:
                    skipped += 1
                    continue

                for tr in pos:
                    split_data.append({
                        "formula": formula,
                        "trace": trace_to_str(tr),
                        "label": 1,
                        "depth": depth,
                    })
                for tr in neg:
                    split_data.append({
                        "formula": formula,
                        "trace": trace_to_str(tr),
                        "label": 0,
                        "depth": depth,
                    })

            except Exception as e:  # noqa: BLE001
                skipped += 1
                if DEBUG:
                    print(f"Error on formula '{formula}': {e}")
                continue

        random.shuffle(split_data)

        output_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_path, "w") as fp:
            json.dump(split_data, fp, indent=2)

        pos_count = sum(1 for d in split_data if d["label"] == 1)
        total_examples += len(split_data)
        print(f"{split_name:6s}: {len(split_data):5d} esempi | pos={pos_count} neg={len(split_data) - pos_count}")

    print(f"\nDataset totale: {total_examples} esempi | formule saltate={skipped}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-formulas", type=int, default=5000)
    p.add_argument("--traces-per-formula", type=int, default=4)
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--max-len", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    # misuro anche il tempo impiegato
    t0 = time.perf_counter()
    generate_dataset(
        n_formulas=args.n_formulas,
        traces_per_formula=args.traces_per_formula,
        seed=args.seed,
        output_dir=args.output_dir,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    t1 = time.perf_counter() - t0
    print(f"Time: {t1:.2f}s")

