#!/usr/bin/env python3
"""
Evaluate one LTLf dataset without comparing it against another dataset.

Expected files:
    train.json
    val.json
    test.json
    ood.json

Required row fields:
    formula: str
    trace: str, e.g. "0101 1100 0001"
    label: 0 or 1

Dependencies:
    numpy
    Spot Python bindings

Semantic evaluation:
    LTLf -> LTL via spot.from_ltlf()
    finite trace -> infinite word with an alive prefix
    acceptance -> native automaton/word intersection

No dataset files are modified.
"""

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import spot


SPLITS = ("train", "val", "test", "ood")


def summarize(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return None

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def load_dataset(directory, props):
    data = {}
    records = []

    for split in SPLITS:
        path = directory / f"{split}.json"

        if not path.is_file():
            print(f"[load] Missing split: {path}")
            continue

        with path.open(encoding="utf-8") as fp:
            rows = json.load(fp)

        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON list.")

        normalized = []

        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{path}, row {index}: expected an object.")

            if not isinstance(row.get("formula"), str):
                raise ValueError(f"{path}, row {index}: invalid formula.")

            if not isinstance(row.get("trace"), str):
                raise ValueError(f"{path}, row {index}: invalid trace.")

            label = row.get("label")

            if label not in (0, 1):
                raise ValueError(f"{path}, row {index}: label must be 0 or 1.")

            tokens = row["trace"].split()

            if not tokens or any(
                len(token) != len(props) or set(token) - {"0", "1"}
                for token in tokens
            ):
                raise ValueError(
                    f"{path}, row {index}: invalid binary trace "
                    f"for propositions {props}."
                )

            item = {
                "formula": row["formula"],
                "trace": " ".join(tokens),
                "label": int(label),
                "split": split,
            }
            normalized.append(item)
            records.append(item)

        data[split] = normalized
        print(f"[load] {split}: {len(normalized)} examples")

    if not records:
        raise ValueError(f"No examples found in {directory}")

    return data, records


def parse_trace(text, props):
    return [
        {props[column] for column, bit in enumerate(token) if bit == "1"}
        for token in text.split()
    ]


def word_from_trace(trace, props):
    steps = []

    for state in trace:
        literals = [
            proposition if proposition in state else f"!{proposition}"
            for proposition in props
        ]
        steps.append(" & ".join([*literals, "alive"]))

    suffix = " & ".join(["!alive", *(f"!{p}" for p in props)])

    return spot.parse_word(
        " ; ".join(steps) + f" ; cycle{{{suffix}}}"
    )


def build_oracle(formula):
    return spot.from_ltlf(formula).translate("small", "buchi", "sbacc")


def oracle_sanity_checks(props):
    p = props[0]

    cases = [
        (p, [{p}], True),
        (p, [set()], False),
        (f"F {p}", [set(), {p}], True),
        (f"G {p}", [{p}, set()], False),
        (f"G {p}", [{p}], True),
        (f"X[!] {p}", [{p}], False),
        (f"X[!] {p}", [set(), {p}], True),
    ]

    for formula, trace, expected in cases:
        actual = bool(
            build_oracle(formula).intersects(word_from_trace(trace, props))
        )

        if actual != expected:
            raise RuntimeError(
                f"Oracle sanity check failed: {formula}, "
                f"expected={expected}, actual={actual}"
            )

    print(f"[oracle] {len(cases)} sanity checks passed")


def syntax_metrics(formulas, props):
    sizes, depths = [], []
    operators = Counter()
    atoms = Counter()
    allowed = set(props)

    def visit(node):
        if node.kind() == spot.op_ap:
            name = node.ap_name()
            if name not in allowed:
                raise ValueError(f"Unknown atomic proposition: {name}")
            atoms[name] += 1
        else:
            operators[node.kindstr()] += 1

        children = [visit(child) for child in node]

        return (
            1 + sum(size for size, _ in children),
            1 + max((depth for _, depth in children), default=0),
        )

    for formula in formulas:
        size, depth = visit(spot.formula(formula))
        sizes.append(size)
        depths.append(depth)

    total = sum(operators.values())
    entropy = (
        -sum(
            (count / total) * math.log2(count / total)
            for count in operators.values()
        )
        if total
        else 0.0
    )

    return {
        "size": summarize(sizes),
        "depth": summarize(depths),
        "operator_histogram": dict(operators),
        "atomic_proposition_histogram": dict(atoms),
        "operator_entropy_bits": entropy,
    }


def structural_metrics(data, records):
    by_trace = defaultdict(list)
    pair_labels = defaultdict(set)
    triple_counts = Counter()
    examples_per_formula = Counter()

    for row in records:
        formula, trace, label = (
            row["formula"], row["trace"], row["label"]
        )
        by_trace[trace].append(row)
        pair_labels[(formula, trace)].add(label)
        triple_counts[(formula, trace, label)] += 1
        examples_per_formula[formula] += 1

    split_metrics = {}

    for name, rows in data.items():
        split_metrics[name] = {
            "examples": len(rows),
            "unique_formulas": len({row["formula"] for row in rows}),
            "positive_fraction": ratio(
                sum(row["label"] for row in rows), len(rows)
            ),
        }

    leakage = {}

    for left, right in combinations(data, 2):
        a, b = data[left], data[right]
        fa = {row["formula"] for row in a}
        fb = {row["formula"] for row in b}
        ta = {row["trace"] for row in a}
        tb = {row["trace"] for row in b}
        pa = {(row["formula"], row["trace"]) for row in a}
        pb = {(row["formula"], row["trace"]) for row in b}

        leakage[f"{left}__{right}"] = {
            "shared_textual_formulas": len(fa & fb),
            "shared_traces": len(ta & tb),
            "shared_formula_trace_pairs": len(pa & pb),
        }

    contradictory = [
        {"formula": formula, "trace": trace}
        for (formula, trace), labels in pair_labels.items()
        if len(labels) > 1
    ]

    shared_trace_count = sum(
        len({row["formula"] for row in rows}) > 1
        for rows in by_trace.values()
    )

    return {
        "examples": len(records),
        "unique_formulas": len(examples_per_formula),
        "positive_fraction": sum(row["label"] for row in records) / len(records),
        "trace_length": summarize(
            [len(row["trace"].split()) for row in records]
        ),
        "examples_per_formula": summarize(list(examples_per_formula.values())),
        "unique_traces": len(by_trace),
        "trace_duplicate_fraction": 1 - len(by_trace) / len(records),
        "unique_traces_shared_by_multiple_formulas": shared_trace_count,
        "repeated_labeled_rows": sum(
            count - 1 for count in triple_counts.values()
        ),
        "contradictory_formula_trace_pairs": len(contradictory),
        "contradiction_examples": contradictory[:20],
        "splits": split_metrics,
        "cross_split_overlap": leakage,
    }


def random_pool(args, props):
    rng = random.Random(args.seed)
    words = []

    for _ in range(args.pool_size):
        length = rng.randint(args.min_len, args.max_len)
        trace = [
            {p for p in props if rng.random() < 0.5}
            for _ in range(length)
        ]
        words.append(word_from_trace(trace, props))

    return words


def sample_pairs(n, count, rng):
    """Uniform ordered draws, excluding the diagonal; repeats allowed."""
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    left = rng.integers(0, n, size=count)
    right = rng.integers(0, n - 1, size=count)
    right += right >= left

    return left, right


def semantic_metrics(signatures, args):
    n = len(signatures)
    rates = signatures.mean(axis=1)
    rng = np.random.default_rng(args.seed + 1)

    left, right = sample_pairs(n, args.pairs, rng)

    jaccard_values = []
    signed_phi_values = []
    identical_pairs = 0
    complementary_pairs = 0

    for start in range(0, len(left), 2048):
        a = signatures[left[start:start + 2048]]
        b = signatures[right[start:start + 2048]]

        inter = (a & b).sum(axis=1, dtype=float)
        union = (a | b).sum(axis=1, dtype=float)

        jac = np.divide(
            inter, union,
            out=np.ones_like(inter),
            where=union > 0,
        )
        jaccard_values.extend(jac.tolist())

        total = float(signatures.shape[1])
        na = a.sum(axis=1, dtype=float)
        nb = b.sum(axis=1, dtype=float)
        denominator = np.sqrt(na * (total - na) * nb * (total - nb))
        defined = denominator > 0

        phi = (inter[defined] * total - na[defined] * nb[defined]) / (
            denominator[defined]
        )
        signed_phi_values.extend(np.clip(phi, -1, 1).tolist())

        identical_pairs += int(np.all(a == b, axis=1).sum())
        complementary_pairs += int(np.all(a != b, axis=1).sum())

    packed = np.packbits(signatures, axis=1)
    groups = Counter(row.tobytes() for row in packed)
    phi_array = np.asarray(signed_phi_values)

    return {
        "evaluated_formulas": n,
        "pool_size": signatures.shape[1],
        "sat_rate": summarize(rates),
        "all_false_signatures": int((rates == 0).sum()),
        "all_true_signatures": int((rates == 1).sum()),
        "unique_signatures": len(groups),
        "formulas_beyond_one_per_signature": n - len(groups),
        "largest_signature_group": max(groups.values()),
        "pair_sampling": "uniform with replacement, excluding self-pairs",
        "sampled_pairs": len(left),
        "jaccard": summarize(jaccard_values),
        "signed_phi_nonconstant_pairs": summarize(phi_array),
        "absolute_phi_nonconstant_pairs": summarize(np.abs(phi_array)),
        "phi_undefined_pairs": len(left) - len(phi_array),
        "sampled_identical_signature_fraction": ratio(
            identical_pairs, len(left)
        ),
        "sampled_complementary_signature_fraction": ratio(
            complementary_pairs, len(left)
        ),
    }


def run(args):
    started = time.perf_counter()
    props = args.props
    directory = Path(args.dataset).expanduser().resolve()

    if not directory.is_dir():
        raise FileNotFoundError(directory)

    data, records = load_dataset(directory, props)
    formulas = sorted({row["formula"] for row in records})

    structural = structural_metrics(data, records)
    syntax = syntax_metrics(formulas, props)
    oracle_sanity_checks(props)

    rng = random.Random(args.seed + 2)

    # A zero limit means "evaluate every formula".
    selected = (
        sorted(rng.sample(formulas, args.max_formulas))
        if args.max_formulas and len(formulas) > args.max_formulas
        else formulas
    )

    # Label checking samples rows globally, independently of semantic selection.
    check_count = (
        len(records)
        if args.label_checks == 0
        else min(args.label_checks, len(records))
    )
    checked_rows = rng.sample(records, check_count)
    checks_by_formula = defaultdict(list)

    for row in checked_rows:
        checks_by_formula[row["formula"]].append(row)

    # Estimate contamination of negatives sampled from positive rows.
    # Sampling is within each split and excludes the anchor formula.
    risk_by_formula = defaultdict(list)

    for split, rows in data.items():
        positives = [row for row in rows if row["label"] == 1]
        owners = {row["formula"] for row in positives}

        if len(owners) < 2:
            continue

        for _ in range(args.risk_pairs_per_split):
            anchor = rng.choice(positives)
            candidate = rng.choice(positives)

            while candidate["formula"] == anchor["formula"]:
                candidate = rng.choice(positives)

            risk_by_formula[anchor["formula"]].append(
                (split, candidate["trace"])
            )

    pool_words = random_pool(args, props)
    signatures = {}
    mismatches = []
    risk_counts = defaultdict(lambda: [0, 0])

    selected_set = set(selected)
    required = sorted(
        selected_set | set(checks_by_formula) | set(risk_by_formula)
    )

    for index, formula in enumerate(required, start=1):
        try:
            aut = build_oracle(formula)

            if formula in selected_set:
                signatures[formula] = np.array(
                    [bool(aut.intersects(word)) for word in pool_words],
                    dtype=bool,
                )

            # Reuse words and labels locally, then release them with the automaton.
            local_labels = {}

            def evaluate_trace(text):
                if text not in local_labels:
                    word = word_from_trace(parse_trace(text, props), props)
                    local_labels[text] = bool(aut.intersects(word))
                return local_labels[text]

            for row in checks_by_formula.get(formula, []):
                actual = int(evaluate_trace(row["trace"]))

                if actual != row["label"]:
                    mismatches.append({
                        **row,
                        "oracle_label": actual,
                    })

            for split, text in risk_by_formula.get(formula, []):
                satisfies = evaluate_trace(text)
                risk_counts[split][0] += 1
                risk_counts[split][1] += int(satisfies)

        except Exception as exc:
            raise RuntimeError(
                f"Spot evaluation failed for formula: {formula}"
            ) from exc

        if index % 50 == 0 or index == len(required):
            print(
                f"[Spot] {index}/{len(required)} formulas",
                flush=True,
            )

    matrix = np.stack([signatures[formula] for formula in selected])
    semantic = semantic_metrics(matrix, args)

    label_validation = {
        "checked_rows": check_count,
        "all_rows_checked": check_count == len(records),
        "mismatches": len(mismatches),
        "mismatch_fraction": len(mismatches) / check_count,
        "mismatch_examples": mismatches[:20],
    }

    contrastive = {
        "sampling_policy": (
            "Within each split: anchor sampled from positive rows; "
            "candidate sampled from positive rows of a different formula. "
            "This is a proxy, not a measurement of the actual training sampler."
        ),
        "splits": {
            split: {
                "checked_candidate_negatives": counts[0],
                "actually_satisfying_pairs": counts[1],
                "estimated_false_negative_fraction": ratio(
                    counts[1], counts[0]
                ),
            }
            for split, counts in risk_counts.items()
        },
    }

    report = {
        "dataset": str(directory),
        "config": vars(args),
        "structural": structural,
        "syntax": syntax,
        "label_validation": label_validation,
        "semantic": semantic,
        "contrastive_negative_risk": contrastive,
        "elapsed_seconds": time.perf_counter() - started,
    }

    output = (
        Path(args.out).expanduser().resolve()
        if args.out
        else directory / "quality_report.json"
    )

    input_paths = {(directory / f"{split}.json").resolve() for split in SPLITS}
    if output in input_paths:
        raise ValueError("The report must not overwrite an input split.")

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, allow_nan=False)

    print("\n=== Dataset quality ===")
    print(f"Examples:                  {structural['examples']}")
    print(f"Unique formulas:           {structural['unique_formulas']}")
    print(f"Positive fraction:         {structural['positive_fraction']:.4f}")
    print(f"Mean formula size:         {syntax['size']['mean']:.4f}")
    print(f"Mean formula depth:        {syntax['depth']['mean']:.4f}")
    print(f"Unique traces:             {structural['unique_traces']}")
    print(f"Trace duplicate fraction:  {structural['trace_duplicate_fraction']:.4f}")
    print(
        "Contradictory pairs:       "
        f"{structural['contradictory_formula_trace_pairs']}"
    )
    print(f"Label mismatches:          {len(mismatches)}/{check_count}")
    print(f"Semantic formulas checked: {len(selected)}/{len(formulas)}")
    print(f"Unique signatures:         {semantic['unique_signatures']}")
    print(f"Mean sat-rate:             {semantic['sat_rate']['mean']:.4f}")
    print(f"All-false signatures:      {semantic['all_false_signatures']}")
    print(f"All-true signatures:       {semantic['all_true_signatures']}")

    if semantic["jaccard"]:
        print(f"Mean pair Jaccard:         {semantic['jaccard']['mean']:.4f}")
        print(f"Pair Jaccard p95:          {semantic['jaccard']['p95']:.4f}")

    if semantic["absolute_phi_nonconstant_pairs"]:
        value = semantic["absolute_phi_nonconstant_pairs"]["mean"]
        print(f"Mean absolute phi:         {value:.4f}")

    print("\n=== Cross-split overlap ===")
    for pair, values in structural["cross_split_overlap"].items():
        print(f"{pair}: {values}")

    print("\n=== Contrastive negative-risk proxy ===")
    for split, values in contrastive["splits"].items():
        fraction = values["estimated_false_negative_fraction"]
        print(
            f"{split}: {fraction:.2%} satisfying candidate negatives "
            f"over {values['checked_candidate_negatives']} checks"
        )

    print(f"\nReport: {output}")
    print(f"Elapsed: {report['elapsed_seconds']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--props", nargs="+", default=["p0", "p1", "p2", "p3"])
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument(
        "--max-formulas",
        type=int,
        default=500,
        help="Maximum formulas for semantic signatures; 0 means all.",
    )
    parser.add_argument("--pairs", type=int, default=20000)
    parser.add_argument(
        "--label-checks",
        type=int,
        default=2000,
        help="Number of saved rows to validate; 0 means all.",
    )
    parser.add_argument("--risk-pairs-per-split", type=int, default=500)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=8)
    args = parser.parse_args()

    if not args.props or len(set(args.props)) != len(args.props):
        parser.error("--props must contain distinct proposition names.")
    if "alive" in args.props:
        parser.error("'alive' is reserved for the semantic embedding.")
    if args.pool_size < 1 or args.pairs < 1:
        parser.error("--pool-size and --pairs must be positive.")
    if min(
        args.max_formulas, args.label_checks, args.risk_pairs_per_split
    ) < 0:
        parser.error("Sampling limits must be non-negative.")
    if not 1 <= args.min_len <= args.max_len:
        parser.error("Expected 1 <= min-len <= max-len.")

    run(args)


if __name__ == "__main__":
    main()