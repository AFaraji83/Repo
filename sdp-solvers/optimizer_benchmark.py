"""
Change #1: Optimizer Benchmarking.

Runs every optimizer in vqe_optimizers.OPTIMIZER_REGISTRY (COBYLA, Powell, SPSA, QNSPSA,
L-BFGS-B, SLSQP) on identical separation-oracle problems -- either saved snapshot matrices
(sdp-solvers/snapshots/<instance>/manifest.csv, produced by generate_snapshots.py, if present)
or a handful of fresh blocks pulled directly from an instance's initial MOSEK solve, so this
script has no hard dependency on having already run Task 4 -- and compares:

    - convergence rate: energy trajectory (vqe_trajectory['energy']), and iterations-to-threshold
    - runtime: wall-clock solve_time per call
    - number of objective evaluations: nfev
    - oracle accuracy: |VQE estimate - exact classical eigenvalue| (and relative error)

adaptive_stop is turned OFF for the benchmark runs specifically (unlike normal cutting-plane
operation, where it's on by default) -- comparing "how many iterations does optimizer X take to
converge" is the whole point here, so every optimizer runs its full maxiter budget rather than
exiting the moment it clears the PSD threshold.

Each optimizer gets NUM_REPEATS independent seeded attempts per problem; failures are caught
per-attempt (one bad combination shouldn't kill the whole sweep) and logged separately.

Usage:
    python optimizer_benchmark.py [--instance control2] [--maxiter 150] [--repeats 3]
"""

import argparse
import csv
import os
import time

import configurable_VQE_based_CP as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from VQESubroutine import VQESubroutine
from vqe_optimizers import OPTIMIZER_REGISTRY

RESULTS_FIELDNAMES = [
    "timestamp", "instance", "block_id", "n_original", "n_padded", "num_qubits", "optimizer_name",
    "repeat", "seed", "maxiter", "true_min_eigenvalue", "vqe_estimate", "abs_error", "relative_error",
    "solve_time", "nfev", "iterations_to_threshold", "num_pauli_terms", "num_cuts_found",
]
ERRORS_FIELDNAMES = ["timestamp", "instance", "block_id", "optimizer_name", "repeat", "seed", "error"]


def get_benchmark_blocks(instance):
    """
    Returns a list of (block_id, matrix, n_original) test problems for the benchmark.

    Prefers Task 4's saved snapshots (sdp-solvers/snapshots/<instance>/manifest.csv) if they
    exist, since those are real candidate matrices pulled from an actual cutting-plane
    trajectory. Falls back to the blocks straight out of the instance's very first (uncut)
    MOSEK solve otherwise, so this script runs standalone without requiring Task 4 first.
    """
    manifest_path = f"snapshots/{instance}/manifest.csv"
    if os.path.exists(manifest_path):
        blocks = []
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                matrix = np.load(row["matrix_path"])
                blocks.append((int(row["block_id"]), matrix, int(row["n_original"])))
        print(f"Loaded {len(blocks)} benchmark matrices from {manifest_path}")
        return blocks

    print(f"No snapshot manifest found at {manifest_path}; falling back to {instance}'s first MOSEK solve.")
    block_list, A_list, b_list = cp.read_instance(instance)
    model, X = cp.create_master_problem(block_list, A_list, b_list, "linear")
    _status, X_values, _objVal, _t = cp.solve_master_problem(model, X, block_list, "MOSEK")
    model.dispose()
    blocks = []
    for b, n_original in enumerate(block_list):
        noiseless_matrix = np.where(np.abs(X_values[b]) < cp.TOL, 0, X_values[b])
        matrix = cp.matrix_prep_for_vqe_psd_check(noiseless_matrix)
        blocks.append((b, matrix, n_original))
    return blocks


def run_one(vqe, optimizer_cls, matrix, n_original, maxiter, seed):
    optimizer = optimizer_cls(maxiter=maxiter, seed=seed) if "seed" in optimizer_cls.__init__.__code__.co_varnames \
        else optimizer_cls(maxiter=maxiter)
    np.random.seed(seed)  # VQESubroutine's cold-start initial_point draw (see solve()) uses the legacy global RNG
    trajectory, params, cuts, solve_time, leaked_amplitude, num_pauli_terms = vqe.solve(
        matrix, optimizer, initial_point=None, energy_threshold=-cp.TOL, overlap_threshold=0.9,
        global_pool=None, collect_multiple_vectors=True, original_dim=n_original,
        adaptive_stop=False,  # full trajectory, for a fair convergence-rate comparison
    )
    exact_eigenvalue = np.linalg.eigvalsh(matrix[:n_original, :n_original]).min()
    vqe_estimate = trajectory["energy"][-1]
    abs_error = abs(vqe_estimate - exact_eigenvalue)
    denom = abs(exact_eigenvalue)
    relative_error = (abs_error / denom) if denom > cp.TOL else (0.0 if abs_error < cp.TOL else float("inf"))

    # first index where the running-best energy cleared the PSD threshold, i.e. how many
    # evaluations *would* have been needed under normal (adaptive_stop=True) operation
    running_best = np.minimum.accumulate(trajectory["energy"])
    below = np.where(running_best < -cp.TOL)[0]
    iterations_to_threshold = int(below[0] + 1) if len(below) else None

    return {
        "true_min_eigenvalue": exact_eigenvalue, "vqe_estimate": vqe_estimate, "abs_error": abs_error,
        "relative_error": relative_error, "solve_time": solve_time, "nfev": trajectory["nfev"],
        "iterations_to_threshold": iterations_to_threshold, "num_pauli_terms": num_pauli_terms,
        "num_cuts_found": len(cuts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default="control2")
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ansatz-type", default="sparsity_aware")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--errors-path", default=None)
    args = parser.parse_args()

    results_path = args.results_path or f"optimizer_benchmark_{args.instance}_results.csv"
    errors_path = args.errors_path or f"optimizer_benchmark_{args.instance}_errors.csv"

    blocks = get_benchmark_blocks(args.instance)
    vqe = VQESubroutine(ansatz_type=args.ansatz_type, execution_mode="noiseless", reps=args.reps)

    total_runs = len(blocks) * len(OPTIMIZER_REGISTRY) * args.repeats
    run_count = 0

    for block_id, matrix, n_original in blocks:
        num_qubits = int(np.log2(matrix.shape[0]))
        n_padded = matrix.shape[0]

        for optimizer_name, optimizer_cls in OPTIMIZER_REGISTRY.items():
            for repeat in range(1, args.repeats + 1):
                run_count += 1
                seed = hash((block_id, optimizer_name, repeat)) % (2 ** 31)
                print(f"[{run_count}/{total_runs}] block={block_id} optimizer={optimizer_name} repeat={repeat}")
                try:
                    metrics = run_one(vqe, optimizer_cls, matrix, n_original, args.maxiter, seed)
                    row = {
                        "timestamp": time.time(), "instance": args.instance, "block_id": block_id,
                        "n_original": n_original, "n_padded": n_padded, "num_qubits": num_qubits,
                        "optimizer_name": optimizer_name, "repeat": repeat, "seed": seed,
                        "maxiter": args.maxiter, **metrics,
                    }
                    cp.log_row(results_path, row, RESULTS_FIELDNAMES)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    cp.log_row(errors_path, {
                        "timestamp": time.time(), "instance": args.instance, "block_id": block_id,
                        "optimizer_name": optimizer_name, "repeat": repeat, "seed": seed, "error": str(e),
                    }, ERRORS_FIELDNAMES)

    print(f"Done. Results in {results_path}, errors (if any) in {errors_path}")
    summarize(results_path)


def summarize(results_path):
    """Prints a per-optimizer summary table and writes a convergence-rate comparison plot."""
    if not os.path.exists(results_path):
        return
    import pandas as pd  # local import: only needed for this convenience summary
    df = pd.read_csv(results_path)
    if df.empty:
        return

    summary = df.groupby("optimizer_name").agg(
        mean_solve_time=("solve_time", "mean"), mean_nfev=("nfev", "mean"),
        mean_abs_error=("abs_error", "mean"), median_abs_error=("abs_error", "median"),
        mean_iterations_to_threshold=("iterations_to_threshold", "mean"),
        success_rate=("num_cuts_found", lambda s: (s > 0).mean()),
    ).round(4)
    print("\n=== Optimizer benchmark summary ===")
    print(summary.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, title) in zip(axes, [
        ("mean_solve_time", "Mean solve time (s)"),
        ("mean_nfev", "Mean # objective evaluations"),
        ("mean_abs_error", "Mean |VQE estimate - exact eigenvalue|"),
    ]):
        summary[col].plot(kind="bar", ax=ax, color="#2E6F9E")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    plot_path = results_path.replace(".csv", "_summary.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Summary plot written to {plot_path}")


if __name__ == "__main__":
    main()
