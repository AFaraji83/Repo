"""
Task 5 (Tier A): standalone separation-oracle benchmark. For every saved snapshot matrix
(sdp-solvers/snapshots/<instance>/manifest.csv, produced by generate_snapshots.py), calls
VQESubroutine.solve() directly -- no master problem, no cutting-plane loop -- across a grid
of execution modes/shot counts, and compares the estimate against the known exact minimum
eigenvalue (already computed via eigvalsh when the snapshot was taken).

Each grid point is a single, independent separation call evaluated cold (no warm start),
wrapped in its own try/except so one failure doesn't kill the grid, with rows flushed+fsynced
as they're produced (see log_row).
"""

import csv
import hashlib
import time

import configurable_VQE_based_CP as cp
import numpy as np
import qiskit
import qiskit_aer
import qiskit_ibm_runtime
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from VQESubroutine import VQESubroutine

QISKIT_VERSION = qiskit.__version__
QISKIT_AER_VERSION = qiskit_aer.__version__
QISKIT_IBM_RUNTIME_VERSION = qiskit_ibm_runtime.__version__

MANIFEST_PATH = "snapshots/control2/manifest.csv"
RESULTS_PATH = "tier_a_control2_results.csv"
ERRORS_PATH = "tier_a_control2_errors.csv"

ANSATZ_TYPE = "sparsity_aware"
REPS = 3
OPTIMIZER_MAXITER = 200
NUM_REPEATS = 3
FAKE_BACKEND = "fake_fez"

# mode -> list of shot settings to sweep (None = exact, used only for 'noiseless')
MODE_SHOTS_GRID = [
    ("noiseless", None),
    ("shots", 1024), ("shots", 4096), ("shots", 8192),
    ("noisy", 1024), ("noisy", 4096), ("noisy", 8192),
]

RESULTS_FIELDNAMES = [
    "timestamp", "snapshot_iteration", "block_id", "n_original", "n_padded", "num_qubits",
    "mode", "shots", "backend_name", "repeat", "seed", "true_min_eigenvalue", "vqe_estimate",
    "abs_error", "relative_error", "solve_time", "num_pauli_terms", "leaked_amplitude",
    "optimizer_nfev", "entanglement_edges", "config_hash",
    "qiskit_version", "qiskit_aer_version", "qiskit_ibm_runtime_version",
]
ERRORS_FIELDNAMES = ["timestamp", "snapshot_iteration", "block_id", "mode", "shots", "repeat", "seed", "error"]


def read_manifest(manifest_path):
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def deterministic_seed(*parts) -> int:
    """Hash-derived seed so every grid point is reproducible and individually re-runnable."""
    digest = hashlib.sha256("_".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def prep_matrix(raw_matrix):
    """Mirror the preprocessing the cutting-plane loop applies before calling VQE (see
    configurable_VQE_based_CP.psd_check): scrub sub-tolerance noise, then zero-pad to the
    next power of two."""
    scrubbed = np.where(np.abs(raw_matrix) < cp.TOL, 0, raw_matrix)
    return cp.matrix_prep_for_vqe_psd_check(scrubbed)


def entanglement_edge_count(vqe: VQESubroutine, padded_matrix) -> int:
    op = SparsePauliOp.from_operator(padded_matrix, atol=1e-12)
    ent_map = vqe._get_entanglement(op)
    return len(ent_map) if isinstance(ent_map, list) else 0


def get_or_build_vqe(cache: dict, mode: str, shots) -> VQESubroutine:
    key = (mode, shots)
    if key not in cache:
        cache[key] = VQESubroutine(
            ansatz_type=ANSATZ_TYPE, execution_mode=mode, reps=REPS, shots=shots,
            backend_name=FAKE_BACKEND if mode == "noisy" else None,
        )
    return cache[key]


def run_one(vqe, padded_matrix, n_original, seed, config_hash):
    # Seeds the global legacy RNG that VQESubroutine.solve's cold-start initial point draws
    # from (np.random.uniform) -- see solve_SDP_by_cutting_plane for why this mechanism was
    # chosen over a per-call np.random.Generator (bit-for-bit reproducible reruns).
    np.random.seed(seed)
    trajectory, _params, _collected, solve_time, leaked_amplitude, num_pauli_terms = vqe.solve(
        padded_matrix, COBYLA(maxiter=OPTIMIZER_MAXITER), initial_point=None,
        energy_threshold=-cp.TOL, overlap_threshold=0.9, global_pool=None,
        collect_multiple_vectors=False, original_dim=n_original,
    )
    vqe_estimate = trajectory["energy"][-1]
    nfev = trajectory.get("nfev")
    return vqe_estimate, nfev, solve_time, leaked_amplitude, num_pauli_terms


if __name__ == "__main__":
    manifest_rows = read_manifest(MANIFEST_PATH)
    print(f"Loaded {len(manifest_rows)} snapshots from {MANIFEST_PATH}")

    vqe_cache = {}
    total_runs = len(manifest_rows) * len(MODE_SHOTS_GRID) * NUM_REPEATS
    run_count = 0

    for snap in manifest_rows:
        iteration = int(snap["iteration"])
        block_id = int(snap["block_id"])
        n_original = int(snap["n_original"])
        n_padded = int(snap["n_padded"])
        true_min_eigenvalue = float(snap["true_min_eigenvalue"])
        raw_matrix = np.load(snap["matrix_path"])
        padded_matrix = prep_matrix(raw_matrix)
        num_qubits = int(np.log2(padded_matrix.shape[0]))

        for mode, shots in MODE_SHOTS_GRID:
            vqe = get_or_build_vqe(vqe_cache, mode, shots)
            entanglement_edges = entanglement_edge_count(vqe, padded_matrix)
            config_hash = cp.compute_config_hash({
                "ansatz_type": ANSATZ_TYPE, "reps": REPS, "execution_mode": mode, "shots": shots,
                "optimizer_maxiter": OPTIMIZER_MAXITER,
            })

            for repeat in range(1, NUM_REPEATS + 1):
                run_count += 1
                seed = deterministic_seed(snap["snapshot_id"], mode, shots, repeat)
                print(f"[{run_count}/{total_runs}] snapshot={snap['snapshot_id']} mode={mode} shots={shots} repeat={repeat}")
                try:
                    vqe_estimate, nfev, solve_time, leaked_amplitude, num_pauli_terms = run_one(
                        vqe, padded_matrix, n_original, seed, config_hash
                    )
                    abs_error = abs(vqe_estimate - true_min_eigenvalue)
                    denom = abs(true_min_eigenvalue)
                    relative_error = (abs_error / denom) if denom > cp.TOL else (0.0 if abs_error < cp.TOL else float("inf"))

                    row = {
                        "timestamp": time.time(), "snapshot_iteration": iteration, "block_id": block_id,
                        "n_original": n_original, "n_padded": n_padded, "num_qubits": num_qubits,
                        "mode": mode, "shots": shots, "backend_name": vqe.backend_name, "repeat": repeat,
                        "seed": seed, "true_min_eigenvalue": true_min_eigenvalue, "vqe_estimate": vqe_estimate,
                        "abs_error": abs_error, "relative_error": relative_error, "solve_time": solve_time,
                        "num_pauli_terms": num_pauli_terms, "leaked_amplitude": leaked_amplitude,
                        "optimizer_nfev": nfev, "entanglement_edges": entanglement_edges,
                        "config_hash": config_hash, "qiskit_version": QISKIT_VERSION,
                        "qiskit_aer_version": QISKIT_AER_VERSION, "qiskit_ibm_runtime_version": QISKIT_IBM_RUNTIME_VERSION,
                    }
                    cp.log_row(RESULTS_PATH, row, RESULTS_FIELDNAMES)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    cp.log_row(ERRORS_PATH, {
                        "timestamp": time.time(), "snapshot_iteration": iteration, "block_id": block_id,
                        "mode": mode, "shots": shots, "repeat": repeat, "seed": seed, "error": str(e),
                    }, ERRORS_FIELDNAMES)

    print(f"Done. Results in {RESULTS_PATH}, errors (if any) in {ERRORS_PATH}")
