"""
Backward-compatibility wrapper for generate_dataset.py.

This module re-exports all functions and classes from gp/generate_dataset.py,
which replaces the legacy randltl generator with a robust, verified generator
featuring:
  - 17 Spot oracle sanity checks
  - Global deduplication across complexity buckets
  - Verified 0 formula leakage across dataset splits
  - Temporal depth filtering
  - Double-witness validation
  - Atomic JSON writes and generation reports
"""

from generate_dataset import (  # noqa: F401
    AP,
    PROPS,
    accepts_trace,
    build_dfa,
    build_ltlf_oracle,
    generate_dataset,
    randltl_generate,
    random_trace,
    sample_traces_for_formula,
    test_ltlf_oracle,
    trace_to_spot_word,
    trace_to_str,
    main,
)

if __name__ == "__main__":
    main()
