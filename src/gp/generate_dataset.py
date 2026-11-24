#!/usr/bin/env python3
"""
Generate an LTLf dataset using Spot with rigorous deduplication and validation.
Adapted from data/generate_dataset.py for use in the gp/ pipeline.

Key Enhancements over the naive randltl generator:
1. Oracle sanity checks (test_ltlf_oracle) covering 17 edge cases.
2. Global deduplication across all generation buckets (zero duplicate formulas).
3. Formula-level splitting with strict validation guaranteeing ZERO formula leakage across splits.
4. Temporal depth filtering (min_temporal_depth <= depth <= max_temporal_depth).
5. Double-witness validation (positive & negative witness required).
6. Support for both single-witness (traces_per_formula=1) and multi-witness (traces_per_formula>1) sampling.
7. Atomic JSON output staging and detailed generation reports.
"""

import argparse
import concurrent.futures
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
AP = PROPS  # Backward-compatibility alias for gp_ltlf
ALIVE = "alive"
SPLIT_NAMES = ("train", "val", "test", "ood")


# ---------------------------------------------------------------------
# Trace generation and serialization
# ---------------------------------------------------------------------
def random_trace(rng=None, min_len=3, max_len=8, props=PROPS):
    """Sample a non-empty trace with independent Bernoulli(0.5) APs."""
    if rng is None:
        rng = random
    length = rng.randint(min_len, max_len)

    return [
        {
            proposition
            for proposition in props
            if rng.random() < 0.5
        }
        for _ in range(length)
    ]


def trace_to_str(trace, props=PROPS, ap=None):
    """Serialize a trace using the fixed proposition order."""
    if ap is not None:
        props = ap
    return " ".join(
        "".join(
            "1" if proposition in state else "0"
            for proposition in props
        )
        for state in trace
    )


def trace_to_spot_word(trace, props=PROPS, ap=None):
    """Encode a finite trace as a concrete ultimately periodic word."""
    if ap is not None:
        props = ap
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
    root = spot.formula(formula) if isinstance(formula, str) else formula

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
    f = spot.formula(formula) if isinstance(formula, str) else formula
    embedded = spot.from_ltlf(f)
    return embedded.translate("small", "buchi", "sbacc")


# Alias for backward compatibility with gp_ltlf.py
build_dfa = build_ltlf_oracle


def accepts_trace(aut, trace, props=PROPS, ap=None):
    """Evaluate one finite trace against an already constructed oracle."""
    if ap is not None:
        props = ap
    word = trace_to_spot_word(trace, props=props)
    return bool(aut.intersects(word))


def sample_traces_for_formula(
    aut,
    n_pos=2,
    n_neg=2,
    min_len=3,
    max_len=20,
    max_attempts=500,
    props=PROPS,
    rng=None
):
    """Sample n_pos positive and n_neg negative traces using the Spot oracle."""
    if rng is None:
        rng = random
    pos, neg = [], []
    for _ in range(max_attempts):
        if len(pos) >= n_pos and len(neg) >= n_neg:
            break
        tr = random_trace(rng, min_len=min_len, max_len=max_len, props=props)
        sat = accepts_trace(aut, tr, props=props)
        if sat and len(pos) < n_pos:
            pos.append(tr)
        elif (not sat) and len(neg) < n_neg:
            neg.append(tr)
    return pos, neg


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
    props=PROPS,
):
    """Generate formulas satisfying a temporal-depth constraint."""
    if n == 0:
        return []

    accepted = {}
    inspected = 0

    # Relative generation weights
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
        batch_size = max(100, 3 * remaining)

        generator = spot.randltl(
            n=batch_size,
            tree_size=tree_size,
            seed=seed + 10000 * batch,
            ap=len(props),
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


def generate_unique_formulas(
    n_formulas,
    complexity_buckets,
    seed,
    min_temporal_depth=2,
    max_temporal_depth=4,
    props=PROPS,
):
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
                min_temporal_depth=min_temporal_depth,
                max_temporal_depth=max_temporal_depth,
                props=props,
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
        f"[generation] Globally unique candidates: {len(formula_to_bucket)}",
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

    print("[split] Global textual formula uniqueness: OK (0 leakage)", flush=True)

    return splits


# ---------------------------------------------------------------------
# Trace sampling & dataset building
# ---------------------------------------------------------------------
def assign_target_labels(n_formulas, rng):
    """Create a shuffled, approximately balanced list of target labels."""
    n_positive = n_formulas // 2
    if n_formulas % 2:
        n_positive += rng.randint(0, 1)

    labels = [1] * n_positive + [0] * (n_formulas - n_positive)
    rng.shuffle(labels)
    return labels


def build_example_single(
    formula,
    bucket,
    rng,
    target_label,
    max_attempts,
    min_len,
    max_len,
    props=PROPS,
):
    """Require both a positive and a negative witness; save witness matching target_label."""
    aut = build_ltlf_oracle(formula)
    positive_witness = None
    negative_witness = None
    seen_traces = set()
    oracle_checks = 0

    for attempt in range(1, max_attempts + 1):
        trace = random_trace(rng, min_len=min_len, max_len=max_len, props=props)
        serialized = trace_to_str(trace, props=props)
        if serialized in seen_traces:
            continue
        seen_traces.add(serialized)

        actual_label = int(accepts_trace(aut, trace, props=props))
        oracle_checks += 1

        if actual_label == 1 and positive_witness is None:
            positive_witness = serialized
        elif actual_label == 0 and negative_witness is None:
            negative_witness = serialized

        if positive_witness is not None and negative_witness is not None:
            selected_trace = positive_witness if target_label == 1 else negative_witness
            row = {
                "formula": formula,
                "trace": selected_trace,
                "label": target_label,
                "depth": bucket,
            }
            return [row], {
                "attempts": attempt,
                "oracle_checks": oracle_checks,
                "positive_witness_found": True,
                "negative_witness_found": True,
            }

    reason = "positive_witness_not_found" if positive_witness is None else "negative_witness_not_found"
    return None, {
        "reason": reason,
        "target_label": target_label,
        "attempts": max_attempts,
        "oracle_checks": oracle_checks,
        "positive_witness_found": positive_witness is not None,
        "negative_witness_found": negative_witness is not None,
    }


def build_example_multi(
    formula,
    bucket,
    rng,
    traces_per_formula,
    max_attempts,
    min_len,
    max_len,
    props=PROPS,
):
    """Sample balanced positive and negative traces per formula."""
    aut = build_ltlf_oracle(formula)
    half = traces_per_formula // 2
    pos, neg = sample_traces_for_formula(
        aut, n_pos=half, n_neg=half,
        min_len=min_len, max_len=max_len,
        max_attempts=max_attempts, props=props, rng=rng
    )

    if len(pos) < half or len(neg) < half:
        reason = "insufficient_positive" if len(pos) < half else "insufficient_negative"
        return None, {
            "reason": reason,
            "attempts": max_attempts,
            "oracle_checks": len(pos) + len(neg),
            "pos_count": len(pos),
            "neg_count": len(neg),
        }

    rows = []
    for tr in pos:
        rows.append({
            "formula": formula,
            "trace": trace_to_str(tr, props=props),
            "label": 1,
            "depth": bucket,
        })
    for tr in neg:
        rows.append({
            "formula": formula,
            "trace": trace_to_str(tr, props=props),
            "label": 0,
            "depth": bucket,
        })

    return rows, {
        "attempts": len(pos) + len(neg),
        "oracle_checks": len(pos) + len(neg),
        "positive_witness_found": True,
        "negative_witness_found": True,
    }


def _worker_multi(args):
    formula, bucket, seed, traces_per_formula, max_attempts, min_len, max_len, props = args
    rng = random.Random(seed)
    return formula, build_example_multi(
        formula=formula,
        bucket=bucket,
        rng=rng,
        traces_per_formula=traces_per_formula,
        max_attempts=max_attempts,
        min_len=min_len,
        max_len=max_len,
        props=props,
    )


def _worker_single(args):
    formula, bucket, seed, target_label, max_attempts, min_len, max_len, props = args
    rng = random.Random(seed)
    return formula, build_example_single(
        formula=formula,
        bucket=bucket,
        rng=rng,
        target_label=target_label,
        max_attempts=max_attempts,
        min_len=min_len,
        max_len=max_len,
        props=props,
    )


def get_executor(n_workers=None):
    if n_workers is None:
        n_workers = os.cpu_count() or 4
    try:
        ex = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
        fut = ex.submit(int, 42)
        fut.result(timeout=2)
        return ex, f"ProcessPoolExecutor ({n_workers} CPU cores)"
    except Exception:
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)
        return ex, f"ThreadPoolExecutor ({n_workers} workers)"


# ---------------------------------------------------------------------
# Validation & Output
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
    if traces_per_formula < 1:
        raise ValueError("traces_per_formula must be >= 1.")
    if traces_per_formula > 1 and traces_per_formula % 2 != 0:
        raise ValueError("traces_per_formula > 1 must be an even number for 50/50 balance.")
    if not complexity_buckets:
        raise ValueError("At least one complexity bucket is required.")
    if any(size < 1 for size in complexity_buckets):
        raise ValueError("Complexity buckets must be positive.")
    if list(complexity_buckets) != sorted(set(complexity_buckets)):
        raise ValueError("Complexity buckets must be strictly increasing.")

    fractions = (train_frac, val_frac, test_frac)
    if any(not math.isfinite(value) or value < 0 for value in fractions):
        raise ValueError("Split fractions must be finite and non-negative.")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1.")
    if not 1 <= min_len <= max_len:
        raise ValueError("Expected 1 <= min_len <= max_len.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")


def check_final_dataset(datasets, traces_per_formula=1):
    """Ensure consistency across generated split files."""
    formula_counts = Counter(
        row["formula"]
        for rows in datasets.values()
        for row in rows
    )

    if traces_per_formula == 1:
        repeated = [f for f, c in formula_counts.items() if c != 1]
        if repeated:
            raise RuntimeError(f"Global uniqueness failed for {len(repeated)} formulas.")
    else:
        # Check no formula appears across multiple splits
        split_sets = {name: {row["formula"] for row in rows} for name, rows in datasets.items()}
        for s1 in SPLIT_NAMES:
            for s2 in SPLIT_NAMES:
                if s1 != s2 and split_sets[s1].intersection(split_sets[s2]):
                    raise RuntimeError(f"Formula leakage detected between {s1} and {s2}!")

    total_examples = sum(len(rows) for rows in datasets.values())
    print(f"[validation] Dataset verified: {total_examples} examples across {len(formula_counts)} unique formulas. OK", flush=True)
    return total_examples


def write_json(path, content):
    with path.open("w", encoding="utf-8") as fp:
        json.dump(content, fp, indent=2, ensure_ascii=False, allow_nan=False)


def find_repo_root():
    curr = Path(__file__).resolve().parent
    while curr != curr.parent:
        if (curr / ".git").exists() or (curr / "src").exists():
            return curr
        curr = curr.parent
    return Path(__file__).resolve().parent.parent.parent


def save_outputs(output, payloads, overwrite):
    output.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = [fn for fn in payloads if (output / fn).exists()]
        if existing:
            raise FileExistsError(
                f"Output files already exist in {output}: {existing}. Use --overwrite to replace them."
            )

    staged = []
    try:
        for filename, content in payloads.items():
            target = output / filename
            temporary = output / f".{filename}.tmp"
            write_json(temporary, content)
            staged.append((temporary, target))
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()


# ---------------------------------------------------------------------
# Main Generation Function
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
    min_temporal_depth=2,
    max_temporal_depth=4,
    max_attempts=500,
    overwrite=False,
    n_workers=None,
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

    if output_dir is not None:
        p = Path(output_dir).expanduser()
        if not p.is_absolute():
            base_dir = find_repo_root()
            output = (base_dir / p).resolve()
        else:
            output = p.resolve()
    else:
        output = find_repo_root() / "data" / "dataset"

    output_names = [*(f"{name}.json" for name in SPLIT_NAMES), "formula_splits.json", "generation_report.json"]
    if not overwrite:
        existing = [fn for fn in output_names if (output / fn).exists()]
        if existing:
            raise FileExistsError(f"Output files already exist in {output}: {existing}. Use --overwrite.")

    started = time.perf_counter()
    formula_to_bucket, bucket_stats = generate_unique_formulas(
        n_formulas=n_formulas,
        complexity_buckets=complexity_buckets,
        seed=seed,
        min_temporal_depth=min_temporal_depth,
        max_temporal_depth=max_temporal_depth,
    )

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

    executor, exec_mode = get_executor(n_workers=n_workers)
    print(f"[sampling] Parallel trace sampling active: {exec_mode}", flush=True)

    with executor:
        for name in SPLIT_NAMES:
            formula_list = formula_splits[name]
            target_labels = assign_target_labels(len(formula_list), label_rng) if traces_per_formula == 1 else None

            tasks = []
            for index, formula in enumerate(formula_list, start=1):
                task_seed = trace_rng.randint(0, 10**9)
                if traces_per_formula == 1:
                    target_label = target_labels[index - 1] if target_labels else 1
                    tasks.append((formula, formula_to_bucket[formula], task_seed, target_label, max_attempts, min_len, max_len, PROPS))
                else:
                    tasks.append((formula, formula_to_bucket[formula], task_seed, traces_per_formula, max_attempts, min_len, max_len, PROPS))

            worker_fn = _worker_single if traces_per_formula == 1 else _worker_multi

            rows = []
            failures = Counter()

            for index, (formula, (batch_rows, status)) in enumerate(
                executor.map(worker_fn, tasks, chunksize=50), start=1
            ):
                sampling_attempts += status["attempts"]
                total_oracle_checks += status["oracle_checks"]

                if batch_rows is None:
                    failures[status["reason"]] += 1
                    skipped_formulas.append({"formula": formula, "split": name, **status})
                else:
                    rows.extend(batch_rows)

                if index % 500 == 0 or index == len(formula_list):
                    print(f"[sampling] {name}: {index}/{len(formula_list)} processed | rows={len(rows)}", flush=True)

            shuffle_rng.shuffle(rows)
            datasets[name] = rows
            positives = sum(row["label"] for row in rows)
            negatives = len(rows) - positives

        split_stats[name] = {
            "candidate_formulas": len(formula_list),
            "kept_examples": len(rows),
            "positive_examples": positives,
            "negative_examples": negatives,
            "positive_fraction": positives / len(rows) if rows else None,
            "skipped_formulas": sum(failures.values()),
            "failure_reasons": dict(failures),
        }

        print(f"[dataset] {name:5s}: {len(rows):5d} examples (pos={positives}, neg={negatives}) | skipped={sum(failures.values())}", flush=True)

    total_examples = check_final_dataset(datasets, traces_per_formula=traces_per_formula)
    total_positives = sum(row["label"] for rows in datasets.values() for row in rows)

    report = {
        "config": {
            "n_formulas_requested": n_formulas,
            "traces_per_formula": traces_per_formula,
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
            "global_textual_formula_uniqueness": True,
            "zero_leakage_guaranteed": True,
        },
        "bucket_stats": bucket_stats,
        "unique_candidate_formulas": len(formula_to_bucket),
        "kept_examples": total_examples,
        "positive_examples": total_positives,
        "negative_examples": total_examples - total_positives,
        "positive_fraction": total_positives / total_examples if total_examples else 0,
        "sampling_attempts": sampling_attempts,
        "oracle_checks": total_oracle_checks,
        "splits": split_stats,
        "skipped_formulas": skipped_formulas,
        "elapsed_generation_seconds": time.perf_counter() - started,
    }

    payloads = {f"{name}.json": datasets[name] for name in SPLIT_NAMES}
    payloads["formula_splits.json"] = {
        name: [
            {"formula": row["formula"], "depth": row["depth"], "label": row["label"]}
            for row in rows
        ]
        for name, rows in datasets.items()
    }
    payloads["generation_report.json"] = report

    save_outputs(output=output, payloads=payloads, overwrite=overwrite)
    print(f"\n[Done] Output saved to: {output} (elapsed: {time.perf_counter() - started:.2f}s)")
    return datasets


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-formulas", type=int, default=5000, help="Candidate formula budget.")
    parser.add_argument("--traces-per-formula", type=int, default=1, help="Traces per formula (1 or even number).")
    parser.add_argument("--complexity-buckets", type=int, nargs="+", default=[6, 10, 14])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/dataset")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--min-len", type=int, default=3, help="Minimum trace length (timesteps).")
    parser.add_argument("--max-len", type=int, default=8, help="Maximum trace length (timesteps / temporal horizon).")
    parser.add_argument("--min-temporal-depth", type=int, default=2, help="Minimum temporal operator depth.")
    parser.add_argument("--max-temporal-depth", type=int, default=4, help="Maximum temporal operator depth.")
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--n-workers", type=int, default=None, help="Number of worker processes (default: all CPUs).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--test-only", action="store_true")

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
        min_temporal_depth=args.min_temporal_depth,
        max_temporal_depth=args.max_temporal_depth,
        max_attempts=args.max_attempts,
        overwrite=args.overwrite,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
