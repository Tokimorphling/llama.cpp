#!/usr/bin/env python3
"""Compare two llama-debug float32 logits dumps."""

from __future__ import annotations

import argparse
import array
import heapq
import json
import math
from pathlib import Path


def read_f32(path: Path) -> array.array:
    values = array.array("f")
    with path.open("rb") as f:
        values.fromfile(f, path.stat().st_size // values.itemsize)
    return values


def topk(values: array.array, k: int) -> list[int]:
    k = min(k, len(values))
    return heapq.nlargest(k, range(len(values)), key=values.__getitem__)


def logsumexp(values: array.array) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def kl_divergence(base: array.array, candidate: array.array) -> float:
    log_z_base = logsumexp(base)
    log_z_candidate = logsumexp(candidate)
    total = 0.0
    for b, c in zip(base, candidate):
        log_p = b - log_z_base
        total += math.exp(log_p) * (log_p - (c - log_z_candidate))
    return total


def compare(base: array.array, candidate: array.array, ks: list[int]) -> dict[str, object]:
    if len(base) != len(candidate):
        raise ValueError(f"logit length mismatch: base={len(base)} candidate={len(candidate)}")

    n = len(base)
    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    dot = 0.0
    norm_base = 0.0
    norm_candidate = 0.0
    max_abs_idx = 0

    for i, (b, c) in enumerate(zip(base, candidate)):
        diff = c - b
        adiff = abs(diff)
        if adiff > max_abs:
            max_abs = adiff
            max_abs_idx = i
        sum_abs += adiff
        sum_sq += diff * diff
        dot += b * c
        norm_base += b * b
        norm_candidate += c * c

    base_top = {k: topk(base, k) for k in ks}
    candidate_top = {k: topk(candidate, k) for k in ks}
    overlaps = {
        str(k): len(set(base_top[k]).intersection(candidate_top[k])) / float(k)
        for k in ks
    }

    return {
        "n": n,
        "max_abs": max_abs,
        "max_abs_idx": max_abs_idx,
        "mean_abs": sum_abs / n,
        "rmse": math.sqrt(sum_sq / n),
        "cosine": dot / math.sqrt(norm_base * norm_candidate) if norm_base > 0.0 and norm_candidate > 0.0 else 0.0,
        "top1_match": base_top[1][0] == candidate_top[1][0],
        "top1_base": base_top[1][0],
        "top1_candidate": candidate_top[1][0],
        "topk_overlap": overlaps,
        "kl_base_to_candidate": kl_divergence(base, candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="baseline float32 logits .bin")
    parser.add_argument("--candidate", required=True, type=Path, help="candidate float32 logits .bin")
    parser.add_argument("--top-k", default="1,5,10", help="comma-separated top-k overlap values")
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    ks = sorted({int(x) for x in args.top_k.split(",") if x})
    if 1 not in ks:
        ks.insert(0, 1)

    result = compare(read_f32(args.base), read_f32(args.candidate), ks)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"n: {result['n']}")
    print(f"max_abs: {result['max_abs']:.9g} @ token {result['max_abs_idx']}")
    print(f"mean_abs: {result['mean_abs']:.9g}")
    print(f"rmse: {result['rmse']:.9g}")
    print(f"cosine: {result['cosine']:.9g}")
    print(f"top1_match: {str(result['top1_match']).lower()} base={result['top1_base']} candidate={result['top1_candidate']}")
    for k, overlap in result["topk_overlap"].items():
        print(f"top{k}_overlap: {overlap:.9g}")
    print(f"kl_base_to_candidate: {result['kl_base_to_candidate']:.9g}")


if __name__ == "__main__":
    main()
