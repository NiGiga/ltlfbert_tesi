#!/usr/bin/env python3
"""
Generate an LTLf dataset with exactly one example per retained formula.

Pipeline:
1. Run basic Spot oracle sanity checks.
2. Generate formulas with Spot randltl.
3. Deduplicate formulas globally across generation buckets.
4. Assign each formula to exactly one dataset split.
5. Assign one target label to each candidate formula.
6. Find both a positive and a negative witness for each candidate.
7. Save only one witness, matching the assigned target label.
8. Check global formula uniqueness and save the dataset.

Each retained formula produces exactly one row:
    {
        "formula": "...",
        "trace": "0101 1100 ...",
        "label": 0 or 1,
        "depth": generation_bucket_index
    }

Semantic oracle:
- Convert LTLf to LTL with spot.from_ltlf().
- Translate the result into a Buchi automaton.
- Encode a finite trace using an alive prefix and a dead infinite suffix.
- Check acceptance with Spot's native intersection operation.

Important:
- Every textual formula appears at most once across all split JSON files.
- Different formulas may still be semantically equivalent.
- Different formulas may still be associated with the same trace.
- Target labels are approximately balanced within each candidate split.
- Sampling failures can make the final label distribution unbalanced.
- "depth" is a generation bucket index, NOT the actual AST depth.
- OOD contains formulas first encountered in the final generation bucket.
- n_formulas is a candidate budget, not a guaranteed final dataset size.
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

try:
    import spot
except ImportError as exc:
    raise RuntimeError(
        "Spot Python bindings are unavailable in this environment."
    ) from exc


PROPS = ["p0", "p1", "p2", "p3"]
ALIVE = "alive"
SPLIT_NAMES = ("train", "val", "test", "ood")


# ---------------------------------------------------------------------
# Trace generation and serialization
# ---------------------------------------------------------------------
def random_trace(rng, min_len=3, max_len=8, props=PROPS):
    """Sample a non-empty trace with independent Bernoulli(0.5) APs."""
    length = rng.randint(min_len, max_len)

    return [
        {
            proposition
            for proposition in props
            if rng.random() < 0.5
        }
        for _ in range(length)
    ]


def trace_to_str(trace, props=PROPS):
    """Serialize a trace using the fixed proposition order."""
    return " ".join(
        "".join(
            "1" if proposition in state else "0"
            for proposition in props
        )
        for state in trace
    )


def trace_to_spot_word(trace, props=PROPS):
    """Encode a finite trace as a concrete ultimately periodic word."""
    if not trace:
        raise ValueError("Empty traces are not supported.")

    allowed = set(props)
    unknown = set().union(*trace) - allowed

    if unknown:
        raise ValueError(
            f"Unknown propositions in trace: {sorted(unknown)}"
        )

    steps = []

    # The alive proposition is true exactly on the finite trace prefix.
    for state in trace:
        literals = [
            proposition if proposition in state else f"!{proposition}"
            for proposition in props
        ]
        steps.append(" & ".join([*literals, ALIVE]))

    # Fix all propositions in the infinite suffix to obtain a concrete word.
    suffix = " & ".join(
        [f"!{ALIVE}", *(f"!{proposition}" for proposition in props)]
    )
    word_text = " ; ".join(steps) + f" ; cycle{{{suffix}}}"

    return spot.parse_word(word_text)

def temporal_depth(formula):
    """Count temporal operators along the deepest AST path."""
    root = spot.formula(formula)

    temporal_kinds = {
        spot.formula("F p0").kind(),
        spot.formula("G p0").kind(),
        spot.formula("X p0").kind(),
        spot.formula("X[!] p0").kind(),
        spot.formula("p0 U p1").kind(),
        spot.formula("p0 R p1").kind(),
        spot.formula("p0 W p1").kind(),
        spot.formula("p0 M p1").kind(),
    }

    def visit(node):
        child_depth = max(
            (visit(child) for child in node),
            default=0,
        )

        return child_depth + int(node.kind() in temporal_kinds)

    return visit(root)

# ---------------------------------------------------------------------
# LTLf semantic oracle
# ---------------------------------------------------------------------
def build_ltlf_oracle(formula):
    """Construct an infinite-word automaton encoding LTLf semantics."""
    embedded = spot.from_ltlf(formula)
    return embedded.translate("small", "buchi", "sbacc")


def accepts_trace(aut, trace, props=PROPS):
    """Evaluate one finite trace against an already constructed oracle."""
    word = trace_to_spot_word(trace, props=props)
    return bool(aut.intersects(word))


def test_ltlf_oracle():
    """Check basic Boolean, temporal and end-of-trace behavior."""
    cases = [
        ("p0", [{"p0"}], True),
        ("p0", [set()], False),
        ("!p0", [set()], True),
        ("p0 & p1", [{"p0", "p1"}], True),
        ("p0 & p1", [{"p0"}], False),
        ("F p0", [set(), {"p0"}], True),
        ("F p0", [set(), set()], False),
        ("G p0", [{"p0"}, {"p0"}], True),
        ("G p0", [{"p0"}, set()], False),
        ("G p0", [{"p0"}], True),
        ("p0 U p1", [{"p0"}, {"p1"}], True),
        ("p0 U p1", [{"p0"}, {"p0"}], False),
        ("p0 U p1", [{"p1"}], True),
        ("X[!] p0", [{"p0"}], False),
        ("X[!] p0", [set(), {"p0"}], True),
        ("X[!] p0", [{"p0"}, set()], False),
        ("!(X[!] p0)", [{"p0"}], True),
    ]

    automata = {}

    for formula, trace, expected in cases:
        if formula not in automata:
            automata[formula] = build_ltlf_oracle(formula)

        actual = accepts_trace(automata[formula], trace)

        if actual != expected:
            raise AssertionError(
                "LTLf oracle sanity check failed:\n"
                f"  formula: {formula}\n"
                f"  trace: {trace}\n"
                f"  expected: {expected}\n"
                f"  actual: {actual}"
            )

    print(
        f"[oracle] Passed {len(cases)} sanity checks.",
        flush=True,
    )


# ---------------------------------------------------------------------
# Formula generation and global textual deduplication
# ---------------------------------------------------------------------
def randltl_generate(
    n,
    tree_size,
    seed,
    min_temporal_depth=2,
    max_temporal_depth=4,
    max_batches=20,
):
    """Generate formulas satisfying a temporal-depth constraint."""
    if n == 0:
        return []

    accepted = {}
    inspected = 0

    # These are relative generation weights, not final percentages.
    priorities = (
        "ap=3,"
        "false=0,true=0,"
        "not=1,"
        "and=2,or=2,implies=1,equiv=0,xor=0,"
        "F=3,G=3,"
        "X=0,strongX=3,"
        "U=3,R=1,W=0,M=0"
    )

    for batch in range(max_batches):
        remaining = n - len(accepted)

        # Overgenerate because many candidates may fail the depth filter.
        batch_size = max(100, 3 * remaining)

        generator = spot.randltl(
            n=batch_size,
            tree_size=tree_size,
            seed=seed + 10000 * batch,
            ap=len(PROPS),
            allow_dup=False,
            simplify=0,
            ltl_priorities=priorities,
        )

        for candidate in generator:
            inspected += 1
            formula = str(candidate)

            if formula in accepted:
                continue

            depth = temporal_depth(formula)

            if min_temporal_depth <= depth <= max_temporal_depth:
                accepted[formula] = None

            if len(accepted) == n:
                print(
                    f"[temporal filter] accepted={len(accepted)}/{n} "
                    f"inspected={inspected}",
                    flush=True,
                )
                return list(accepted)

    print(
        f"[temporal filter] WARNING: accepted={len(accepted)}/{n} "
        f"after inspecting {inspected} candidates",
        flush=True,
    )

    return list(accepted)


def generate_unique_formulas(n_formulas, complexity_buckets, seed):
    """Deduplicate across all buckets; the first occurrence wins."""
    bucket_count = len(complexity_buckets)
    quotient, remainder = divmod(n_formulas, bucket_count)

    formula_to_bucket = {}
    bucket_stats = []

    for index, tree_size in enumerate(complexity_buckets):
        requested = quotient + int(index < remainder)

        formulas = (
            randltl_generate(
                n=requested,
                tree_size=tree_size,
                seed=seed + index,
            )
            if requested
            else []
        )

        added = 0

        for formula in formulas:
            if formula not in formula_to_bucket:
                formula_to_bucket[formula] = index + 1
                added += 1

        bucket_stats.append({
            "bucket": index + 1,
            "tree_size": tree_size,
            "requested": requested,
            "returned": len(formulas),
            "new_unique": added,
            "discarded_duplicate_occurrences": len(formulas) - added,
        })

        print(
            f"[generation] bucket={index + 1} "
            f"tree_size={tree_size} "
            f"requested={requested} "
            f"returned={len(formulas)} "
            f"new_unique={added}",
            flush=True,
        )

    if not formula_to_bucket:
        raise RuntimeError("No formulas were generated.")

    print(
        f"[generation] Globally unique candidates: "
        f"{len(formula_to_bucket)}",
        flush=True,
    )

    return formula_to_bucket, bucket_stats


# ---------------------------------------------------------------------
# Formula-level splitting
# ---------------------------------------------------------------------
def split_formulas(
    formula_to_bucket,
    ood_bucket,
    rng,
    train_frac,
    val_frac,
    test_frac,
):
    """Assign each candidate formula to exactly one split."""
    iid = sorted(
        formula
        for formula, bucket in formula_to_bucket.items()
        if bucket != ood_bucket
    )
    ood = sorted(
        formula
        for formula, bucket in formula_to_bucket.items()
        if bucket == ood_bucket
    )

    rng.shuffle(iid)

    n_val = int(val_frac * len(iid))
    n_test = int(test_frac * len(iid))

    # Fraction validation guarantees that the remainder matches train_frac
    # up to integer rounding.
    n_train = len(iid) - n_val - n_test

    splits = {
        "train": iid[:n_train],
        "val": iid[n_train:n_train + n_val],
        "test": iid[n_train + n_val:],
        "ood": ood,
    }

    assigned = set()

    for name in SPLIT_NAMES:
        formulas = splits[name]
        formula_set = set(formulas)

        if len(formulas) != len(formula_set):
            raise RuntimeError(f"Repeated candidate inside split {name}.")

        if assigned.intersection(formula_set):
            raise RuntimeError(
                f"Formula leakage detected while assigning split {name}."
            )

        assigned.update(formula_set)

        print(
            f"[split] {name}: {len(formulas)} candidate formulas",
            flush=True,
        )

    if assigned != set(formula_to_bucket):
        raise RuntimeError("Formula assignment is incomplete.")

    print("[split] Global textual formula uniqueness: OK", flush=True)

    return splits


# ---------------------------------------------------------------------
# One target label and one example per formula
# ---------------------------------------------------------------------
def assign_target_labels(n_formulas, rng):
    """Create a shuffled, approximately balanced list of target labels."""
    n_positive = n_formulas // 2

    # Randomize which class receives the extra candidate for odd counts.
    if n_formulas % 2:
        n_positive += rng.randint(0, 1)

    labels = (
        [1] * n_positive
        + [0] * (n_formulas - n_positive)
    )

    rng.shuffle(labels)
    return labels


def build_example(
    formula,
    bucket,
    rng,
    target_label,
    max_attempts,
    min_len,
    max_len,
):
    """
    Require both a positive and a negative witness before keeping a formula.

    Only the witness matching target_label is saved in the dataset.
    The other witness is used internally to validate the candidate.

    Failure to find both witnesses within the sampling budget does not
    prove that the formula is a tautology or a contradiction.
    """
    if target_label not in (0, 1):
        raise ValueError("target_label must be 0 or 1.")

    # Construct one oracle and reuse it for all sampled traces.
    aut = build_ltlf_oracle(formula)

    positive_witness = None
    negative_witness = None

    seen_traces = set()
    oracle_checks = 0

    for attempt in range(1, max_attempts + 1):
        trace = random_trace(
            rng,
            min_len=min_len,
            max_len=max_len,
        )
        serialized = trace_to_str(trace)

        # Do not evaluate the same trace twice for this formula.
        if serialized in seen_traces:
            continue

        seen_traces.add(serialized)

        actual_label = int(accepts_trace(aut, trace))
        oracle_checks += 1

        # Keep the first witness found for each label.
        if actual_label == 1 and positive_witness is None:
            positive_witness = serialized

        elif actual_label == 0 and negative_witness is None:
            negative_witness = serialized

        # Accept the formula only after observing both outcomes.
        if positive_witness is not None and negative_witness is not None:
            selected_trace = (
                positive_witness
                if target_label == 1
                else negative_witness
            )

            row = {
                "formula": formula,
                "trace": selected_trace,
                "label": target_label,
                "depth": bucket,
            }

            return row, {
                "attempts": attempt,
                "oracle_checks": oracle_checks,
                "positive_witness_found": True,
                "negative_witness_found": True,
            }

    # Report which witness was not found without making a logical claim
    # about all possible finite traces.
    if positive_witness is None:
        reason = "positive_witness_not_found"
    elif negative_witness is None:
        reason = "negative_witness_not_found"
    else:
        raise RuntimeError("Unexpected witness sampling state.")

    return None, {
        "reason": reason,
        "target_label": target_label,
        "attempts": max_attempts,
        "oracle_checks": oracle_checks,
        "positive_witness_found": positive_witness is not None,
        "negative_witness_found": negative_witness is not None,
    }

# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def validate_parameters(
    n_formulas,
    traces_per_formula,
    complexity_buckets,
    train_frac,
    val_frac,
    test_frac,
    min_len,
    max_len,
    max_attempts,
):
    if n_formulas < 1:
        raise ValueError("n_formulas must be positive.")

    # Keep this argument only for explicit CLI compatibility.
    if traces_per_formula != 1:
        raise ValueError(
            "Exactly one example per formula is required. "
            "Use --traces-per-formula 1."
        )

    if not complexity_buckets:
        raise ValueError("At least one complexity bucket is required.")

    if any(size < 1 for size in complexity_buckets):
        raise ValueError("Complexity buckets must be positive.")

    if list(complexity_buckets) != sorted(set(complexity_buckets)):
        raise ValueError("Complexity buckets must be strictly increasing.")

    fractions = (train_frac, val_frac, test_frac)

    if any(
        not math.isfinite(value) or value < 0
        for value in fractions
    ):
        raise ValueError(
            "Split fractions must be finite and non-negative."
        )

    if not math.isclose(
        sum(fractions),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Split fractions must sum to 1.")

    if not 1 <= min_len <= max_len:
        raise ValueError("Expected 1 <= min_len <= max_len.")

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")


def check_final_dataset(datasets):
    """Ensure that every saved formula appears exactly once globally."""
    counts = Counter(
        row["formula"]
        for rows in datasets.values()
        for row in rows
    )

    repeated = [
        formula
        for formula, count in counts.items()
        if count != 1
    ]

    if repeated:
        raise RuntimeError(
            f"Global uniqueness check failed for {len(repeated)} formulas."
        )

    total_examples = sum(len(rows) for rows in datasets.values())

    if total_examples != len(counts):
        raise RuntimeError(
            "The number of examples differs from the number of formulas."
        )

    print(
        "[validation] Exactly one example per retained formula: OK",
        flush=True,
    )

    return total_examples


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
def write_json(path, content):
    """Write a JSON file without non-standard NaN/Infinity values."""
    with path.open("w", encoding="utf-8") as fp:
        json.dump(
            content,
            fp,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def save_outputs(output, payloads, overwrite):
    """Stage output files before replacing destination files."""
    output.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        existing = [
            filename
            for filename in payloads
            if (output / filename).exists()
        ]

        if existing:
            raise FileExistsError(
                f"Output files already exist: {existing}. "
                "Use --overwrite to replace them."
            )

    staged = []

    try:
        # Serialize every payload before replacing any destination.
        for filename, content in payloads.items():
            target = output / filename
            temporary = output / f".{filename}.tmp"
            write_json(temporary, content)
            staged.append((temporary, target))

        # Each replacement is atomic, but the whole group is not a
        # transaction across all files.
        for temporary, target in staged:
            os.replace(temporary, target)

    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()


# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------
def generate_dataset(
    n_formulas=5000,
    traces_per_formula=1,
    complexity_buckets=(6, 10, 14),
    seed=42,
    output_dir=None,
    train_frac=0.8,
    val_frac=0.1,
    test_frac=0.1,
    min_len=3,
    max_len=8,
    max_attempts=500,
    overwrite=False,
):
    validate_parameters(
        n_formulas=n_formulas,
        traces_per_formula=traces_per_formula,
        complexity_buckets=complexity_buckets,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        min_len=min_len,
        max_len=max_len,
        max_attempts=max_attempts,
    )

    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(__file__).resolve().parent / "dataset"
    )

    output_names = [
        *(f"{name}.json" for name in SPLIT_NAMES),
        "formula_splits.json",
        "generation_report.json",
    ]

    # Check before running the expensive generation phase.
    if not overwrite:
        existing = [
            filename
            for filename in output_names
            if (output / filename).exists()
        ]

        if existing:
            raise FileExistsError(
                f"Output files already exist in {output}: {existing}. "
                "Use --overwrite to replace them."
            )

    started = time.perf_counter()

    formula_to_bucket, bucket_stats = generate_unique_formulas(
        n_formulas=n_formulas,
        complexity_buckets=complexity_buckets,
        seed=seed,
    )

    # Use independent random generators for separate pipeline stages.
    split_rng = random.Random(seed)
    trace_rng = random.Random(seed + 1)
    label_rng = random.Random(seed + 2)
    shuffle_rng = random.Random(seed + 3)

    formula_splits = split_formulas(
        formula_to_bucket=formula_to_bucket,
        ood_bucket=len(complexity_buckets),
        rng=split_rng,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )

    datasets = {}
    split_stats = {}
    skipped_formulas = []
    sampling_attempts = 0
    total_oracle_checks = 0

    for name in SPLIT_NAMES:
        formula_list = formula_splits[name]
        target_labels = assign_target_labels(
            len(formula_list),
            label_rng,
        )

        rows = []
        failures = Counter()
        skipped_by_label = Counter()

        for index, (formula, target_label) in enumerate(
            zip(formula_list, target_labels),
            start=1,
        ):
            try:
                row, status = build_example(
                    formula=formula,
                    bucket=formula_to_bucket[formula],
                    rng=trace_rng,
                    target_label=target_label,
                    max_attempts=max_attempts,
                    min_len=min_len,
                    max_len=max_len,
                )
            except Exception as exc:
                # Never reinterpret an oracle error as a negative label.
                raise RuntimeError(
                    f"Oracle error in split {name}, formula: {formula}"
                ) from exc

            sampling_attempts += status["attempts"]
            total_oracle_checks += status["oracle_checks"]

            if row is None:
                failures[status["reason"]] += 1
                skipped_by_label[target_label] += 1

                skipped_formulas.append({
                    "formula": formula,
                    "split": name,
                    **status,
                })
            else:
                rows.append(row)

            if index % 100 == 0 or index == len(formula_list):
                print(
                    f"[sampling] {name}: "
                    f"{index}/{len(formula_list)} processed | "
                    f"kept={len(rows)}",
                    flush=True,
                )

        shuffle_rng.shuffle(rows)
        datasets[name] = rows

        positives = sum(row["label"] for row in rows)
        negatives = len(rows) - positives

        split_stats[name] = {
            "candidate_formulas": len(formula_list),
            "candidate_positive_targets": sum(target_labels),
            "candidate_negative_targets": (
                len(target_labels) - sum(target_labels)
            ),
            "kept_formulas": len(rows),
            "examples": len(rows),
            "positive_examples": positives,
            "negative_examples": negatives,
            "positive_fraction": (
                positives / len(rows) if rows else None
            ),
            "skipped_formulas": sum(failures.values()),
            "skipped_positive_targets": skipped_by_label[1],
            "skipped_negative_targets": skipped_by_label[0],
            "failure_reasons": dict(failures),
        }

        print(
            f"[dataset] {name:5s}: "
            f"{len(rows)} formulas | "
            f"{len(rows)} examples | "
            f"pos={positives} neg={negatives} | "
            f"skipped={sum(failures.values())}",
            flush=True,
        )

    total_examples = check_final_dataset(datasets)

    # Do not replace an existing dataset with an entirely empty result.
    if total_examples == 0:
        raise RuntimeError(
            "No examples were generated. Existing files were not replaced."
        )

    total_positives = sum(
        row["label"]
        for rows in datasets.values()
        for row in rows
    )

    report = {
        "config": {
            "n_formulas_requested": n_formulas,
            "traces_per_formula": 1,
            "complexity_buckets": list(complexity_buckets),
            "seed": seed,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
            "min_len": min_len,
            "max_len": max_len,
            "max_attempts": max_attempts,
            "propositions": PROPS,
            "oracle": "from_ltlf + Buchi intersection with alive word",
            "depth_field": "generation bucket index",
            "ood_policy": "first encountered in final generation bucket",
            "label_policy": (
                "approximately balanced targets per candidate split; "
                "no target changes after sampling failure"
            ),
            "formula_acceptance_policy": (
                "require one positive and one negative sampled witness; "
                "save only the witness matching the assigned target label"
            ),
            "global_textual_formula_uniqueness": True,
            "global_trace_uniqueness": False,
            "semantic_formula_deduplication": False,
        },
        "bucket_stats": bucket_stats,
        "unique_candidate_formulas": len(formula_to_bucket),
        "kept_formulas": total_examples,
        "total_examples": total_examples,
        "positive_examples": total_positives,
        "negative_examples": total_examples - total_positives,
        "positive_fraction": total_positives / total_examples,
        "sampling_attempts": sampling_attempts,
        "oracle_checks": total_oracle_checks,
        "splits": split_stats,
        "skipped_formulas": skipped_formulas,
        "elapsed_generation_seconds": time.perf_counter() - started,
    }

    payloads = {
        f"{name}.json": datasets[name]
        for name in SPLIT_NAMES
    }

    # This manifest contains retained formulas only.
    # Skipped candidates are recorded separately in generation_report.json.
    payloads["formula_splits.json"] = {
        name: [
            {
                "formula": row["formula"],
                "depth": row["depth"],
                "label": row["label"],
            }
            for row in rows
        ]
        for name, rows in datasets.items()
    }

    payloads["generation_report.json"] = report

    save_outputs(
        output=output,
        payloads=payloads,
        overwrite=overwrite,
    )

    print(f"\nOutput: {output}")
    print(f"Unique candidate formulas: {len(formula_to_bucket)}")
    print(f"Retained unique formulas: {total_examples}")
    print(f"Examples: {total_examples}")
    print(f"Skipped formulas: {len(skipped_formulas)}")
    print(f"Positive fraction: {total_positives / total_examples:.4f}")
    print(f"Elapsed: {time.perf_counter() - started:.2f}s")

    for name, rows in datasets.items():
        if not rows:
            print(f"WARNING: split '{name}' is empty.")

    if overwrite:
        print(
            "Existing split files and generation metadata were replaced. "
            "Other files in the directory were left unchanged."
        )

    return datasets


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--n-formulas",
        type=int,
        default=5000,
        help="Candidate formula budget before deduplication and sampling.",
    )
    parser.add_argument(
        "--traces-per-formula",
        type=int,
        choices=[1],
        default=1,
        help="Exactly one example per retained formula.",
    )
    parser.add_argument(
        "--complexity-buckets",
        type=int,
        nargs="+",
        default=[6, 10, 14],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=500)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing split files and generation metadata.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run oracle sanity checks without generating a dataset.",
    )

    args = parser.parse_args()

    test_ltlf_oracle()

    if args.test_only:
        return

    generate_dataset(
        n_formulas=args.n_formulas,
        traces_per_formula=args.traces_per_formula,
        complexity_buckets=tuple(args.complexity_buckets),
        seed=args.seed,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        min_len=args.min_len,
        max_len=args.max_len,
        max_attempts=args.max_attempts,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()