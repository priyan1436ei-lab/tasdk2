#!/usr/bin/env python3
"""
sliding_window_max_average.py

Standalone script to:
- Solve maximum average subarray of fixed size k (naive and optimized)
- Compare performance on random data
- Visualize sliding-window movement with matplotlib animation

Dependencies:
    python 3.8+
    numpy
    matplotlib

Usage:
    python sliding_window_max_average.py --demo
    python sliding_window_max_average.py --benchmark
    python sliding_window_max_average.py --visualize --n 200 --k 20

Note: Saving animations to GIF/MP4 requires ffmpeg or imagemagick installed and on PATH.
"""

import argparse
import time
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation


def max_average_subarray_naive(nums: List[float], k: int) -> Tuple[float, int]:
    """
    Naive O(n*k) approach: compute every k-length window sum directly.
    Returns (max_average, start_index_of_window).
    """
    n = len(nums)
    if k <= 0 or k > n:
        raise ValueError("k must be between 1 and len(nums)")
    max_sum = None
    max_idx = 0
    for i in range(n - k + 1):
        s = 0.0
        for j in range(i, i + k):
            s += nums[j]
        if (max_sum is None) or (s > max_sum):
            max_sum = s
            max_idx = i
    return max_sum / k, max_idx


def max_average_subarray_sliding(nums: List[float], k: int) -> Tuple[float, int]:
    """
    Optimized O(n) sliding window: maintain running sum for window size k.
    Returns (max_average, start_index_of_window).
    """
    n = len(nums)
    if k <= 0 or k > n:
        raise ValueError("k must be between 1 and len(nums)")
    window_sum = sum(nums[:k])
    max_sum = window_sum
    max_idx = 0
    for i in range(k, n):
        window_sum += nums[i] - nums[i - k]
        # window starting at i - k + 1
        if window_sum > max_sum:
            max_sum = window_sum
            max_idx = i - k + 1
    return max_sum / k, max_idx


def benchmark_compare(n: int = 100000, k: int = 100, trials: int = 3, seed: Optional[int] = 0):
    """
    Compare naive and optimized methods on a random array of size n with window k.
    For large n, naive will be slow; the function will skip naive if it appears too expensive.
    """
    if seed is not None:
        np.random.seed(seed)
    nums = np.random.randn(n).astype(float).tolist()

    print(f"Benchmark: n={n}, k={k}, trials={trials}")
    # Time optimized method
    opt_times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        avg_opt, idx_opt = max_average_subarray_sliding(nums, k)
        t1 = time.perf_counter()
        opt_times.append(t1 - t0)
    print(f"Optimized sliding-window: avg={avg_opt:.6f} start={idx_opt} time_mean={np.mean(opt_times):.6f}s")

    # For naive, guard against huge runs
    naive_times = []
    max_naive_n = 50000  # heuristic cap where naive might still be feasible for small k
    if n * k > 5e7:
        print("Skipping naive approach due to large n*k (would be too slow). Try smaller n or k to run naive.")
        return
    for _ in range(trials):
        t0 = time.perf_counter()
        avg_naive, idx_naive = max_average_subarray_naive(nums, k)
        t1 = time.perf_counter()
        naive_times.append(t1 - t0)
    print(f"Naive approach: avg={avg_naive:.6f} start={idx_naive} time_mean={np.mean(naive_times):.6f}s")
    # correctness check
    assert abs(avg_opt - avg_naive) < 1e-9 and idx_opt == idx_naive, "Results mismatch between methods!"
    print("Results match. Optimized approach is faster by factor:", np.mean(naive_times) / np.mean(opt_times))


def visualize_window(nums: List[float], k: int, interval: int = 100, save: Optional[str] = None):
    """
    Visualize the sliding window moving across nums.
    - nums: list or 1D numpy array
    - k: window size
    - interval: ms between animation frames
    - save: optional filename to save animation (gif/mp4). Requires ffmpeg/Magick.
    """
    n = len(nums)
    if k <= 0 or k > n:
        raise ValueError("k must be between 1 and len(nums)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios':[2, 1]})
    x = np.arange(n)
    ax1.plot(x, nums, color="tab:blue")
    ax1.set_title("Data with sliding window")
    ax1.set_xlim(0, n-1)
    ymin, ymax = min(nums), max(nums)
    margin = (ymax - ymin) * 0.15 if (ymax - ymin) != 0 else 1.0
    ax1.set_ylim(ymin - margin, ymax + margin)

    # Highlighted rectangle for window
    rect = patches.Rectangle((0 - 0.5, ymin - margin), k, ymax - ymin + 2 * margin, linewidth=1,
                             edgecolor='tab:orange', facecolor='tab:orange', alpha=0.2)
    ax1.add_patch(rect)

    # Small subplot showing average over window index as it moves
    ax2.set_xlim(0, n - k)
    ax2.set_ylim(min(nums) - margin, max(nums) + margin)
    ax2.set_title("Window average as it slides")
    avg_vals = []
    avg_line, = ax2.plot([], [], color='tab:green')
    current_point, = ax2.plot([], [], marker='o', color='tab:red')

    # Precompute averages for speed
    window_sum = sum(nums[:k])
    avg_vals.append(window_sum / k)
    for i in range(k, n):
        window_sum += nums[i] - nums[i - k]
        avg_vals.append(window_sum / k)
    avg_vals = np.array(avg_vals)

    def init():
        avg_line.set_data([], [])
        current_point.set_data([], [])
        rect.set_x(-0.5)
        return rect, avg_line, current_point

    def update(frame):
        # frame is window start index: 0 .. n-k
        rect.set_x(frame - 0.5)
        avg_line.set_data(np.arange(len(avg_vals[:frame+1])), avg_vals[:frame+1])
        current_point.set_data(frame, avg_vals[frame])
        ax2.set_ylim(avg_vals.min() - margin*0.2, avg_vals.max() + margin*0.2)
        return rect, avg_line, current_point

    frames = n - k + 1
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=interval)

    if save:
        print(f"Saving animation to {save} (this may take a moment)...")
        anim.save(save, dpi=150)
        print("Saved animation.")
    else:
        plt.tight_layout()
        plt.show()


def demo_small_example():
    nums = [1, 12, -5, -6, 50, 3]
    k = 4
    print("Example nums:", nums)
    print("k =", k)
    avg_naive, idx_naive = max_average_subarray_naive(nums, k)
    avg_opt, idx_opt = max_average_subarray_sliding(nums, k)
    print("Naive -> max_avg:", avg_naive, "start:", idx_naive)
    print("Optimized -> max_avg:", avg_opt, "start:", idx_opt)


def main():
    parser = argparse.ArgumentParser(description="Max Average Subarray (fixed-size sliding window) demo")
    parser.add_argument("--demo", action="store_true", help="Run a small demo example")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--visualize", action="store_true", help="Run visualization")
    parser.add_argument("--n", type=int, default=2000, help="Array size for benchmark/visualization")
    parser.add_argument("--k", type=int, default=50, help="Window size")
    parser.add_argument("--save", type=str, default=None, help="Save visualization to file (e.g., out.mp4 or out.gif)")
    args = parser.parse_args()

    if args.demo:
        demo_small_example()
    if args.benchmark:
        # choose n and k sensibly
        benchmark_compare(n=args.n, k=args.k, trials=3, seed=1)
    if args.visualize:
        # For very large n we show only first 500 points for clarity
        n_show = min(args.n, 800)
        nums = np.random.randn(n_show).cumsum()  # nicer-looking signal
        visualize_window(nums.tolist(), args.k, interval=50, save=args.save)
    if not (args.demo or args.benchmark or args.visualize):
        parser.print_help()

if __name__ == "__main__":
    main()
