"""
Task 4 driver: run the VQE-based cutting-plane loop (configurable_VQE_based_CP) on control2
with the locked configuration, saving candidate-matrix + VQE-optimal-parameter snapshots at
a configurable set of iterations. Produces sdp-solvers/snapshots/control2/manifest.csv plus
the usual iteration/results CSVs, all consumed downstream by tier_a_oracle_benchmark.py and
(for the locked-config reference) tier_b_noisy_run.py.
"""

import logging
import socket
import time

import configurable_VQE_based_CP as cp
from qiskit_algorithms.optimizers import COBYLA

logging.basicConfig(filename='generate_snapshots_log.txt', level=logging.INFO,
                     format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
cp.logger.setLevel(logging.INFO)

INSTANCE = "control2"
ITR_LIMIT = 250
SNAPSHOT_ITERATIONS = [1, 25, 50, 100, 250]
SNAPSHOT_DIR = f"snapshots/{INSTANCE}"
RANDOM_SEED = 42
VM_ID = socket.gethostname()

ITERATION_LOG_PATH = f"{INSTANCE}_snapshot_gen_iterations.csv"
CANDIDATE_LOG_PATH = f"{INSTANCE}_snapshot_gen_vqe_candidates.csv"
RESULTS_LOG_PATH = f"{INSTANCE}_snapshot_gen_results.csv"

if __name__ == "__main__":
    block_list, A_list, b_list = cp.read_instance(INSTANCE)
    best_solution, best_value, solve_time = cp.solve_SDP_with_MOSEK(block_list, A_list, b_list)

    # Locked configuration (per the task spec): MOSEK master, SOC valid inequalities + SOC
    # cuts, sparsity-aware ansatz reps=3, COBYLA(maxiter=200), noiseless, cut purging at the
    # module's default patience (5) since no specific value was pinned, diversity filtering on
    # (VQESubroutine's default overlap_threshold=0.9, already applied unconditionally).
    # Deliberately excludes cut_purge_patience: it only feeds compute_config_hash below,
    # since ITERATION_LOG_FIELDNAMES/CANDIDATE_LOG_FIELDNAMES are fixed column lists that
    # don't include it (configurable_VQE_based_CP.__main__'s own config_dict follows the
    # same pattern), and csv.DictWriter raises on unlisted keys.
    config_dict = {
        "instance": INSTANCE, "solver": "MOSEK", "valid_cut_type": "soc", "add_soc_cuts": True,
        "collect_multiple_cuts": True, "ansatz_type": "sparsity_aware", "ansatz_layers": 3,
        "execution_mode": "noiseless", "shots": None, "optimizer_name": "COBYLA", "optimizer_maxiter": 200,
    }
    config_hash = cp.compute_config_hash({**config_dict, "cut_purge_patience": 5})
    log_context = {**config_dict, "config_hash": config_hash, "random_seed": RANDOM_SEED, "repeat": 1}

    rss_at_run_start_mb = cp.current_rss_mb()
    (termination_reason, itr, linear_cuts_added, linear_cuts_purged, soc_cuts_added,
     master_total_time, sep_total_time, X_values, objVal) = cp.solve_SDP_by_cutting_plane(
        block_list, A_list, b_list, "MOSEK", valid="soc", add_soc_cuts=True,
        optimizer=COBYLA(maxiter=200), collect_multiple_cuts=True, itr_limit=ITR_LIMIT,
        ansatz_type="sparsity_aware", execution_mode="noiseless", reps=3,
        shots=None, thermal_relaxation=False, seed=RANDOM_SEED,
        log_filepath=ITERATION_LOG_PATH, log_context=log_context, cut_purge_patience=5,
        snapshot_iterations=SNAPSHOT_ITERATIONS, snapshot_dir=SNAPSHOT_DIR,
        candidate_log_filepath=CANDIDATE_LOG_PATH,
    )

    result_row = {
        "timestamp": time.time(), "vm_id": VM_ID, "config_hash": config_hash,
        "random_seed": RANDOM_SEED, "instance": INSTANCE, "termination_reason": termination_reason,
        "iterations": itr, "linear_cuts_added": linear_cuts_added, "linear_cuts_purged": linear_cuts_purged,
        "soc_cuts_added": soc_cuts_added, "master_total_time": master_total_time, "sep_total_time": sep_total_time,
        "objVal": objVal, "best_value_MOSEK": best_value, "rss_at_run_start_mb": rss_at_run_start_mb,
        "peak_rss_mb": cp.peak_rss_mb(),
    }
    cp.log_row(RESULTS_LOG_PATH, result_row, list(result_row.keys()))
    print(f"Result: termination_reason={termination_reason}, iterations={itr}, objVal={objVal}, best_value(MOSEK)={best_value}")
    print(f"Snapshots written to {SNAPSHOT_DIR}/ (manifest.csv lists {len(SNAPSHOT_ITERATIONS)} iterations x {len(block_list)} blocks)")
