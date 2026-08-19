"""
Task 6 (Tier B): one full noisy cutting-plane run. Same locked configuration as
generate_snapshots.py's Tier A reference run (MOSEK, valid='soc', add_soc_cuts=True,
sparsity_aware, COBYLA(maxiter=200), reps=3, cut purging on) but execution_mode='noisy',
shots=4096, itr_limit=100, 3 repeats -- to see how the loop behaves (false-PSD certificates
included) when the separation oracle runs under simulated device noise + finite sampling.

No noiseless comparison run here: generate_snapshots.py's 250-iteration noiseless run already
has per-iteration logs to truncate at iteration 100 for comparison.
"""

import logging
import socket
import time

import configurable_VQE_based_CP as cp
from qiskit_algorithms.optimizers import COBYLA

logging.basicConfig(filename='tier_b_noisy_run_log.txt', level=logging.INFO,
                     format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
cp.logger.setLevel(logging.INFO)

INSTANCE = "control2"
ITR_LIMIT = 100
SHOTS = 4096
NUM_REPEATS = 3
BASE_SEED = 42
VM_ID = socket.gethostname()

ITERATION_LOG_PATH = f"{INSTANCE}_tier_b_noisy_iterations.csv"
CANDIDATE_LOG_PATH = f"{INSTANCE}_tier_b_noisy_vqe_candidates.csv"
RESULTS_LOG_PATH = f"{INSTANCE}_tier_b_noisy_results.csv"
ERROR_LOG_PATH = f"{INSTANCE}_tier_b_noisy_errors.csv"
ERROR_LOG_FIELDNAMES = [
    "timestamp", "vm_id", "instance", "config_hash", "execution_mode", "shots", "random_seed", "repeat", "error",
]

if __name__ == "__main__":
    block_list, A_list, b_list = cp.read_instance(INSTANCE)
    best_solution, best_value, solve_time = cp.solve_SDP_with_MOSEK(block_list, A_list, b_list)

    # Deliberately excludes cut_purge_patience: it only feeds compute_config_hash below,
    # since ITERATION_LOG_FIELDNAMES/CANDIDATE_LOG_FIELDNAMES are fixed column lists that
    # don't include it (configurable_VQE_based_CP.__main__'s own config_dict follows the
    # same pattern), and csv.DictWriter raises on unlisted keys.
    config_dict = {
        "instance": INSTANCE, "solver": "MOSEK", "valid_cut_type": "soc", "add_soc_cuts": True,
        "collect_multiple_cuts": True, "ansatz_type": "sparsity_aware", "ansatz_layers": 3,
        "execution_mode": "noisy", "shots": SHOTS, "optimizer_name": "COBYLA", "optimizer_maxiter": 200,
    }
    config_hash = cp.compute_config_hash({**config_dict, "cut_purge_patience": 5})

    results = []
    for repeat in range(1, NUM_REPEATS + 1):
        seed = BASE_SEED + repeat  # distinct, still-deterministic seed per repeat
        log_context = {**config_dict, "config_hash": config_hash, "random_seed": seed, "repeat": repeat}
        print(f"=== Tier B noisy run: repeat {repeat}/{NUM_REPEATS}, seed={seed} ===")

        rss_at_run_start_mb = cp.current_rss_mb()
        try:
            (termination_reason, itr, linear_cuts_added, linear_cuts_purged, soc_cuts_added,
             master_total_time, sep_total_time, X_values, objVal) = cp.solve_SDP_by_cutting_plane(
                block_list, A_list, b_list, "MOSEK", valid="soc", add_soc_cuts=True,
                optimizer=COBYLA(maxiter=200), collect_multiple_cuts=True, itr_limit=ITR_LIMIT,
                ansatz_type="sparsity_aware", execution_mode="noisy", reps=3,
                shots=SHOTS, thermal_relaxation=False, seed=seed,
                log_filepath=ITERATION_LOG_PATH, log_context=log_context, cut_purge_patience=5,
                candidate_log_filepath=CANDIDATE_LOG_PATH,
            )
            # Reset the per-block diversity/history globals before the next repeat, as the
            # module's own __main__ does between configs (they're module-level dicts, not
            # reset by solve_SDP_by_cutting_plane itself).
            cp.violating_vectors.clear()
            cp.exact_min_eigenvalues.clear()
            cp.vqe_min_eigenvalue_estimates.clear()
            cp.master_objective_values.clear()
            cp.leaked_amplitudes.clear()
        except Exception as e:
            print(f"Run failed (repeat {repeat}): {e}")
            cp.logger.exception("Tier B run failed (repeat %d)", repeat)
            cp.log_row(ERROR_LOG_PATH, {
                "timestamp": time.time(), "vm_id": VM_ID, "instance": INSTANCE, "config_hash": config_hash,
                "execution_mode": "noisy", "shots": SHOTS, "random_seed": seed, "repeat": repeat, "error": str(e),
            }, ERROR_LOG_FIELDNAMES)
            termination_reason, itr = str(e), None
            linear_cuts_added = linear_cuts_purged = soc_cuts_added = None
            master_total_time = sep_total_time = objVal = None

        result_row = {
            "timestamp": time.time(), "vm_id": VM_ID, "config_hash": config_hash, "instance": INSTANCE,
            "execution_mode": "noisy", "shots": SHOTS, "random_seed": seed, "repeat": repeat,
            "termination_reason": termination_reason, "iterations": itr,
            "linear_cuts_added": linear_cuts_added, "linear_cuts_purged": linear_cuts_purged,
            "soc_cuts_added": soc_cuts_added, "master_total_time": master_total_time,
            "sep_total_time": sep_total_time, "objVal": objVal, "best_value_MOSEK": best_value,
            "rss_at_run_start_mb": rss_at_run_start_mb, "peak_rss_mb": cp.peak_rss_mb(),
        }
        results.append(result_row)
        cp.log_row(RESULTS_LOG_PATH, result_row, list(result_row.keys()))
        print(f"Result: termination_reason={termination_reason}, iterations={itr}, objVal={objVal}")

    print(f"Done. {len(results)} repeats written to {RESULTS_LOG_PATH}; per-iteration log in {ITERATION_LOG_PATH}")
