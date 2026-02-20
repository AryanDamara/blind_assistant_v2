"""
Performance Benchmark
---------------------
Measures latency of SafetyManager, ObjectDetector, and the
complete detection → safety → alert pipeline.

Runs against synthetic data (no camera or model file needed
for SafetyManager benchmarks; YOLO benchmarks require the model).

Run:
    python tests/benchmark.py
"""

import sys
import os
import time
import json
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# --------------- Helpers ---------------

class MockDetection:
    """Synthetic detection for benchmarking."""
    def __init__(self, class_name='person', confidence=0.9,
                 bbox=(200, 50, 440, 470), bbox_center=(320, 260), bbox_height=420):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.bbox_center = bbox_center
        self.bbox_height = bbox_height


def percentile(data, p):
    """Calculate p-th percentile."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def benchmark(func, iterations=1000, label=""):
    """Run a function N times and report latency statistics."""
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)
    
    result = {
        'label': label,
        'iterations': iterations,
        'avg_ms': round(statistics.mean(times_ms), 4),
        'min_ms': round(min(times_ms), 4),
        'max_ms': round(max(times_ms), 4),
        'median_ms': round(statistics.median(times_ms), 4),
        'stdev_ms': round(statistics.stdev(times_ms), 4) if len(times_ms) > 1 else 0,
        'p95_ms': round(percentile(times_ms, 95), 4),
        'p99_ms': round(percentile(times_ms, 99), 4),
    }
    return result


def print_result(result):
    """Pretty-print a benchmark result."""
    print(f"\n{'=' * 50}")
    print(f"  {result['label']}")
    print(f"{'=' * 50}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Avg:    {result['avg_ms']:>8.4f} ms")
    print(f"  Median: {result['median_ms']:>8.4f} ms")
    print(f"  Min:    {result['min_ms']:>8.4f} ms")
    print(f"  Max:    {result['max_ms']:>8.4f} ms")
    print(f"  StdDev: {result['stdev_ms']:>8.4f} ms")
    print(f"  P95:    {result['p95_ms']:>8.4f} ms")
    print(f"  P99:    {result['p99_ms']:>8.4f} ms")
    fps = 1000.0 / result['avg_ms'] if result['avg_ms'] > 0 else float('inf')
    print(f"  ~FPS:   {fps:>8.1f}")


# --------------- Benchmarks ---------------

def bench_safety_manager():
    """Benchmark SafetyManager.analyze() with synthetic detections."""
    from safety_manager import SafetyManager

    sm = SafetyManager(config_path="nonexistent.yaml")

    # Prepare synthetic detections
    detections_1 = [MockDetection()]
    detections_5 = [
        MockDetection('person', 0.9, (200, 50, 440, 470), (320, 260), 420),
        MockDetection('chair', 0.8, (50, 200, 120, 350), (85, 275), 150),
        MockDetection('car', 0.7, (500, 100, 620, 400), (560, 250), 300),
        MockDetection('bicycle', 0.6, (350, 300, 400, 450), (375, 375), 150),
        MockDetection('bottle', 0.5, (100, 400, 130, 450), (115, 425), 50),
    ]
    detections_10 = detections_5 * 2

    results = []

    # 1 detection
    r = benchmark(lambda: sm.analyze(detections_1, 640, 0), iterations=5000,
                  label="SafetyManager.analyze (1 detection)")
    print_result(r)
    results.append(r)

    # 5 detections
    r = benchmark(lambda: sm.analyze(detections_5, 640, 0), iterations=5000,
                  label="SafetyManager.analyze (5 detections)")
    print_result(r)
    results.append(r)

    # 10 detections
    r = benchmark(lambda: sm.analyze(detections_10, 640, 0), iterations=2000,
                  label="SafetyManager.analyze (10 detections)")
    print_result(r)
    results.append(r)

    return results


def bench_stair_detector():
    """Benchmark StairDetector with synthetic depth maps."""
    import numpy as np

    try:
        from stair_detector import StairDetector
    except ImportError:
        print("[SKIP] StairDetector not available")
        return []

    sd = StairDetector(config_path="nonexistent.yaml")

    # Synthetic stair depth map
    h, w = 240, 320
    stair_map = np.zeros((h, w), dtype=np.float32)
    for i in range(6):
        y_start = 70 + i * 25
        y_end = y_start + 25
        if y_end <= h:
            stair_map[y_start:y_end, :] = 2.0 + i * 0.3

    # Flat depth map
    flat_map = np.ones((h, w), dtype=np.float32) * 3.0

    results = []

    # Stair map
    r = benchmark(lambda: sd.detect(stair_map), iterations=2000,
                  label="StairDetector.detect (stair pattern)")
    print_result(r)
    results.append(r)
    sd._last_alert_time = 0  # Reset cooldown between iterations

    # Flat map
    r = benchmark(lambda: sd.detect(flat_map), iterations=2000,
                  label="StairDetector.detect (flat surface)")
    print_result(r)
    results.append(r)

    return results


def bench_distance_estimation():
    """Benchmark distance estimation in isolation."""
    from safety_manager import SafetyManager

    sm = SafetyManager(config_path="nonexistent.yaml")

    results = []

    # Calibrated object
    r = benchmark(lambda: sm._estimate_distance('person', 400, 0.9), iterations=10000,
                  label="Distance estimation (calibrated)")
    print_result(r)
    results.append(r)

    # Fallback object
    r = benchmark(lambda: sm._estimate_distance('bottle', 200, 0.8), iterations=10000,
                  label="Distance estimation (fallback)")
    print_result(r)
    results.append(r)

    return results


# --------------- Main ---------------

def main():
    print("=" * 60)
    print("  VOICE NAVIGATION SYSTEM — PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")
    print()

    all_results = []

    print("\n>>> Safety Manager Benchmarks")
    all_results.extend(bench_safety_manager())

    print("\n>>> Distance Estimation Benchmarks")
    all_results.extend(bench_distance_estimation())

    print("\n>>> Stair Detector Benchmarks")
    all_results.extend(bench_stair_detector())

    # Save results to JSON
    output_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'python_version': sys.version.split()[0],
            'results': all_results
        }, f, indent=2)
    print(f"\n[Benchmark] Results saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for r in all_results:
        fps = 1000 / r['avg_ms'] if r['avg_ms'] > 0 else float('inf')
        status = "✅" if r['avg_ms'] < 1.0 else "⚠️"
        print(f"  {status} {r['label']}: {r['avg_ms']:.4f}ms avg ({fps:.0f} FPS)")

    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()
