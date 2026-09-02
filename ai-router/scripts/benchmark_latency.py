"""
scripts/benchmark_latency.py
──────────────────────────────
ONNX inference latency benchmark.

Target: p95 < 300 ms per image pair (change detection or fusion).
Reports p50 and p95 latency over 50 runs after 5 warm-up runs.

Usage (after exporting ONNX models):
    python satquery/inference/onnx_export.py --model all
    python scripts/benchmark_latency.py

Exits with code 1 if p95 exceeds 300 ms threshold.
"""
from __future__ import annotations

import platform
import statistics
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

WARMUP_RUNS  = 5
MEASURE_RUNS = 50
LATENCY_P95_TARGET_MS = 300.0   # SIH requirement


def _time_fn(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - t0) * 1000.0   # ms


def benchmark_change_detector() -> dict:
    from satquery.inference.onnx_runner import OnnxChangeDetector

    try:
        detector = OnnxChangeDetector()
    except FileNotFoundError as e:
        return {"error": str(e)}

    t1 = np.random.rand(1, 3, 256, 256).astype(np.float32)
    t2 = np.random.rand(1, 3, 256, 256).astype(np.float32)

    # Warm-up
    for _ in range(WARMUP_RUNS):
        detector.run(t1, t2)

    # Measure
    latencies = [_time_fn(detector.run, t1, t2) for _ in range(MEASURE_RUNS)]
    return {
        "model":  "SiameseChangeDetector (ONNX)",
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(sorted(latencies)[int(0.95 * MEASURE_RUNS)], 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
    }


def benchmark_fusion_model() -> dict:
    from satquery.inference.onnx_runner import OnnxFusionModel

    try:
        fusion = OnnxFusionModel()
    except FileNotFoundError as e:
        return {"error": str(e)}

    opt = np.random.rand(1, 3, 224, 224).astype(np.float32)
    sar = np.random.rand(1, 2, 224, 224).astype(np.float32)

    for _ in range(WARMUP_RUNS):
        fusion.run(opt, sar)

    latencies = [_time_fn(fusion.run, opt, sar) for _ in range(MEASURE_RUNS)]
    return {
        "model":  "OpticalSARFusion (ONNX)",
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(sorted(latencies)[int(0.95 * MEASURE_RUNS)], 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
    }


def main() -> None:
    print(f"\n{'='*56}")
    print(f"SatQuery AI — ONNX Latency Benchmark")
    print(f"Warm-up: {WARMUP_RUNS} runs | Measure: {MEASURE_RUNS} runs")
    print(f"Target: p95 < {LATENCY_P95_TARGET_MS} ms")
    print(f"{'='*56}\n")

    results = [
        benchmark_change_detector(),
        benchmark_fusion_model(),
    ]

    failed = False
    for r in results:
        if "error" in r:
            print(f"  [SKIP] {r['error']}", flush=True)
            continue
        status = "✓ PASS" if r["p95_ms"] < LATENCY_P95_TARGET_MS else "✗ FAIL"
        print(f"  {status}  {r['model']}", flush=True)
        print(f"           p50={r['p50_ms']} ms | p95={r['p95_ms']} ms "
              f"| min={r['min_ms']} ms | max={r['max_ms']} ms", flush=True)
        if r["p95_ms"] >= LATENCY_P95_TARGET_MS:
            failed = True

    print(f"\n{'='*56}", flush=True)
    if failed:
        print("BENCHMARK FAILED: p95 latency exceeds 300 ms target.", flush=True)
        print("Consider: quantisation (INT8), smaller input resolution,", flush=True)
        print("or enabling GPU/CoreML execution provider.", flush=True)
        sys.exit(1)
    else:
        print("BENCHMARK PASSED: All models meet 300 ms p95 target.", flush=True)


if __name__ == "__main__":
    main()
