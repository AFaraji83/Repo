"""
Standalone, budget-conscious hardware benchmark for the VQE separation oracle. Compares a
VQE minimum-eigenvalue estimate against exact/noiseless/noisy/hardware baselines on the SAME
control2 candidate matrices, without running the full cutting-plane loop on a QPU (only
10 minutes of IBM open-plan QPU time total).

Stages (see module docstring sections below and the task spec this implements):
  A. Select snapshots -- read snapshots/control2/manifest.csv (produced offline by
     generate_snapshots.py) and pick the early/middle/late iterations available.
  B. Classical reference -- VQESubroutine.solve() in 'noiseless' and 'noisy' (1024/4096
     shots) mode, 3 repeats each, no QPU. Persists the noiseless run's optimal parameters
     theta_hat (one per repeat) to THETA_HAT_PATH -- Stage C's only input besides the
     manifest itself.
  C. Hardware evaluation -- binds each persisted theta_hat to the ansatz and evaluates a
     single expectation value per (snapshot, repeat) on a real backend, all pubs submitted
     in one Batch/job. Defaults to --meter-only (exactly one evaluation) so QPU cost per
     evaluation can be inspected before committing to --full-grid.
  D. Output -- both stages append to the same results CSV (RESULTS_FIELDNAMES), written
     incrementally with flush+fsync; failures are logged to ERRORS_PATH without aborting
     the run.

Stages A/B never touch the network/QPU and can be run (`--stage prepare`) entirely offline;
Stage C (`--stage hardware`) reads only the files A/B produced, so the QPU is touched only
when explicitly requested.
"""

import argparse
import csv
import hashlib
import json
import os
import time

import configurable_VQE_based_CP as cp
import numpy as np
import qiskit
import qiskit_aer
import qiskit_ibm_runtime
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import Batch, QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
from VQESubroutine import VQESubroutine

QISKIT_VERSION = qiskit.__version__
QISKIT_AER_VERSION = qiskit_aer.__version__
QISKIT_IBM_RUNTIME_VERSION = qiskit_ibm_runtime.__version__

# Manifest is instance-specific by directory; hardcoding control2 here (rather than taking
# --instance) is what makes "control2 only" structural instead of a filter that could drift.
MANIFEST_PATH = "snapshots/control2/manifest.csv"
THETA_HAT_PATH = "hardware_benchmark_theta_hat.csv"
RESULTS_PATH = "hardware_benchmark_results.csv"
ERRORS_PATH = "hardware_benchmark_errors.csv"

ANSATZ_TYPE = "sparsity_aware"
REPS = 3
OPTIMIZER_MAXITER = 200
NUM_REPEATS = 3
NOISY_SHOTS_GRID = [1024, 4096]
FAKE_BACKEND = "fake_fez"
RUNTIME_CHANNEL = "ibm_quantum_platform"  # matches VQESubroutine._setup_engine's 'hardware' branch
DEFAULT_HARDWARE_SHOTS = 4096
RESILIENCE_LEVEL = 0  # unmitigated baseline -- mitigation multiplies QPU cost several-fold

RESULTS_FIELDNAMES = [
    "iteration", "block_id", "n_original", "n_padded", "num_qubits",
    "mode", "shots", "backend_name", "repeat", "seed",
    "true_min_eigenvalue", "estimate", "abs_error", "relative_error",
    "solve_time", "qpu_seconds", "num_pauli_terms", "entanglement_edges",
    "transpiled_depth", "two_qubit_gate_count",
    "qiskit_version", "qiskit_aer_version", "qiskit_ibm_runtime_version",
]
ERRORS_FIELDNAMES = ["timestamp", "stage", "snapshot_id", "iteration", "block_id", "mode", "shots", "repeat", "seed", "error"]
THETA_HAT_FIELDNAMES = [
    "snapshot_id", "iteration", "block_id", "repeat", "seed",
    "theta_hat_json", "num_qubits", "n_original", "n_padded", "true_min_eigenvalue",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def read_manifest(manifest_path):
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def deterministic_seed(*parts) -> int:
    """Hash-derived seed so every (snapshot, mode, shots, repeat) combo is reproducible and
    individually re-runnable -- same mechanism as tier_a_oracle_benchmark.py."""
    digest = hashlib.sha256("_".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def prep_matrix(raw_matrix):
    """Mirror the preprocessing the cutting-plane loop applies before calling VQE (see
    configurable_VQE_based_CP.psd_check): scrub sub-tolerance noise, then zero-pad to the
    next power of two. save_snapshot() persists the raw (unpadded, unscrubbed) candidate
    matrix, so this must be re-applied here exactly as psd_check does it."""
    scrubbed = np.where(np.abs(raw_matrix) < cp.TOL, 0, raw_matrix)
    return cp.matrix_prep_for_vqe_psd_check(scrubbed)


def log_error(stage, snapshot_id, iteration, block_id, mode, shots, repeat, seed, error):
    print(f"  FAILED [{stage}] snapshot={snapshot_id} mode={mode} shots={shots} repeat={repeat}: {error}")
    cp.log_row(ERRORS_PATH, {
        "timestamp": time.time(), "stage": stage, "snapshot_id": snapshot_id, "iteration": iteration,
        "block_id": block_id, "mode": mode, "shots": shots, "repeat": repeat, "seed": seed, "error": str(error),
    }, ERRORS_FIELDNAMES)


def relative_error_of(abs_error, true_value):
    denom = abs(true_value)
    return (abs_error / denom) if denom > cp.TOL else (0.0 if abs_error < cp.TOL else float("inf"))


# ---------------------------------------------------------------------------
# Stage A -- select snapshots
# ---------------------------------------------------------------------------

def select_snapshots(manifest_rows):
    """Picks the earliest, middle (by sorted-index), and latest iterations present in the
    control2 manifest, and returns every block row saved at those iterations."""
    iterations = sorted({int(r["iteration"]) for r in manifest_rows})
    if not iterations:
        raise ValueError(f"No snapshots found in {MANIFEST_PATH}.")

    chosen = sorted({iterations[0], iterations[len(iterations) // 2], iterations[-1]})
    labels = {}
    if len(chosen) == 1:
        labels[chosen[0]] = "early/middle/late"
    elif len(chosen) == 2:
        labels[chosen[0]] = "early"
        labels[chosen[-1]] = "late"
    else:
        labels[chosen[0]] = "early"
        labels[chosen[1]] = "middle"
        labels[chosen[-1]] = "late"

    selected = [r for r in manifest_rows if int(r["iteration"]) in chosen]
    selected.sort(key=lambda r: (int(r["iteration"]), int(r["block_id"])))
    return selected, labels


def print_selection(selected_rows, labels):
    print(f"Selected {len(selected_rows)} control2 snapshot(s) spanning {len(labels)} iteration(s):")
    for r in selected_rows:
        itr = int(r["iteration"])
        n_padded = int(r["n_padded"])
        num_qubits = int(np.log2(n_padded))
        print(
            f"  [{labels.get(itr, '?')}] iter={itr} block={r['block_id']} snapshot_id={r['snapshot_id']} "
            f"n_original={r['n_original']} n_padded={n_padded} num_qubits={num_qubits} "
            f"true_min_eigenvalue={float(r['true_min_eigenvalue']):.6f}"
        )


# ---------------------------------------------------------------------------
# Stage B -- classical reference (no QPU)
# ---------------------------------------------------------------------------

def get_or_build_vqe(cache: dict, mode: str, shots) -> VQESubroutine:
    key = (mode, shots)
    if key not in cache:
        cache[key] = VQESubroutine(
            ansatz_type=ANSATZ_TYPE, execution_mode=mode, reps=REPS, shots=shots,
            backend_name=FAKE_BACKEND if mode == "noisy" else None,
        )
    return cache[key]


def circuit_stats(vqe: VQESubroutine, padded_matrix):
    """Builds (and caches, via VQESubroutine's own ansatz_cache) the op/ansatz for this
    matrix and returns entanglement-edge count plus post-transpile depth/2Q-gate-count.
    Depth/2Q-count are only meaningful once a PassManager has actually transpiled the
    circuit against a backend ISA (noisy/hardware); for noiseless (vqe.pm is None) they're
    reported as None rather than a pre-transpile number that would misleadingly look like
    a hardware-relevant figure."""
    num_qubits = int(np.log2(padded_matrix.shape[0]))
    op = SparsePauliOp.from_operator(padded_matrix, atol=1e-12)
    ent_map = vqe._get_entanglement(op)
    entanglement_edges = len(ent_map) if isinstance(ent_map, list) else 0
    execution_ansatz, _abstract_ansatz = vqe._get_ansatz(num_qubits, ent_map)

    if vqe.pm is None:
        return entanglement_edges, None, None
    transpiled_depth = execution_ansatz.depth()
    two_qubit_gate_count = sum(1 for instr in execution_ansatz.data if instr.operation.num_qubits == 2)
    return entanglement_edges, transpiled_depth, two_qubit_gate_count


def run_classical_reference(selected_rows):
    """Stage B. For each selected snapshot, runs noiseless + noisy(1024/4096) VQE, 3 repeats
    each, writing RESULTS_FIELDNAMES rows. Every noiseless repeat's optimal parameters are
    additionally persisted to THETA_HAT_PATH -- Stage C's sole offline input."""
    vqe_cache = {}
    mode_shots_grid = [("noiseless", None)] + [("noisy", s) for s in NOISY_SHOTS_GRID]
    total_runs = len(selected_rows) * len(mode_shots_grid) * NUM_REPEATS
    run_count = 0

    for snap in selected_rows:
        iteration = int(snap["iteration"])
        block_id = int(snap["block_id"])
        n_original = int(snap["n_original"])
        n_padded = int(snap["n_padded"])
        true_min_eigenvalue = float(snap["true_min_eigenvalue"])
        raw_matrix = np.load(snap["matrix_path"])
        padded_matrix = prep_matrix(raw_matrix)
        num_qubits = int(np.log2(padded_matrix.shape[0]))

        for mode, shots in mode_shots_grid:
            vqe = get_or_build_vqe(vqe_cache, mode, shots)
            entanglement_edges, transpiled_depth, two_qubit_gate_count = circuit_stats(vqe, padded_matrix)

            for repeat in range(1, NUM_REPEATS + 1):
                run_count += 1
                seed = deterministic_seed(snap["snapshot_id"], mode, shots, repeat)
                print(f"[{run_count}/{total_runs}] snapshot={snap['snapshot_id']} mode={mode} shots={shots} repeat={repeat}")
                try:
                    np.random.seed(seed)
                    trajectory, theta_hat, _collected, solve_time, _leaked, num_pauli_terms = vqe.solve(
                        padded_matrix, COBYLA(maxiter=OPTIMIZER_MAXITER), initial_point=None,
                        energy_threshold=-cp.TOL, overlap_threshold=0.9, global_pool=None,
                        collect_multiple_vectors=False, original_dim=n_original,
                    )
                    estimate = trajectory["energy"][-1]
                    abs_error = abs(estimate - true_min_eigenvalue)

                    row = {
                        "iteration": iteration, "block_id": block_id, "n_original": n_original,
                        "n_padded": n_padded, "num_qubits": num_qubits, "mode": mode, "shots": shots,
                        "backend_name": vqe.backend_name, "repeat": repeat, "seed": seed,
                        "true_min_eigenvalue": true_min_eigenvalue, "estimate": estimate,
                        "abs_error": abs_error, "relative_error": relative_error_of(abs_error, true_min_eigenvalue),
                        "solve_time": solve_time, "qpu_seconds": 0.0, "num_pauli_terms": num_pauli_terms,
                        "entanglement_edges": entanglement_edges, "transpiled_depth": transpiled_depth,
                        "two_qubit_gate_count": two_qubit_gate_count, "qiskit_version": QISKIT_VERSION,
                        "qiskit_aer_version": QISKIT_AER_VERSION, "qiskit_ibm_runtime_version": QISKIT_IBM_RUNTIME_VERSION,
                    }
                    cp.log_row(RESULTS_PATH, row, RESULTS_FIELDNAMES)

                    if mode == "noiseless":
                        cp.log_row(THETA_HAT_PATH, {
                            "snapshot_id": snap["snapshot_id"], "iteration": iteration, "block_id": block_id,
                            "repeat": repeat, "seed": seed, "theta_hat_json": json.dumps(list(np.asarray(theta_hat).tolist())),
                            "num_qubits": num_qubits, "n_original": n_original, "n_padded": n_padded,
                            "true_min_eigenvalue": true_min_eigenvalue,
                        }, THETA_HAT_FIELDNAMES)
                except Exception as e:
                    log_error("prepare", snap["snapshot_id"], iteration, block_id, mode, shots, repeat, seed, e)


# ---------------------------------------------------------------------------
# Stage C -- hardware evaluation (uses QPU)
# ---------------------------------------------------------------------------

def get_qpu_seconds(job, result):
    """Best-effort extraction of QPU time actually billed for this job. job.usage() is the
    primary, documented source; the fallbacks exist only because metadata shape has shifted
    across qiskit-ibm-runtime releases and this script should degrade to 'unknown' rather
    than crash on a version mismatch."""
    try:
        usage = job.usage()
        if usage is not None:
            return float(usage)
    except Exception:
        pass
    try:
        usage = job.metrics().get("usage", {}).get("quantum_seconds")
        if usage is not None:
            return float(usage)
    except Exception:
        pass
    try:
        return float(result.metadata["execution"]["execution_spans"].duration)
    except Exception:
        pass
    return None


def load_theta_hat_rows():
    if not os.path.isfile(THETA_HAT_PATH):
        raise FileNotFoundError(
            f"{THETA_HAT_PATH} not found -- run `--stage prepare` first (Stage B persists the "
            f"noiseless theta_hat that Stage C binds; Stage C never runs the optimizer itself)."
        )
    with open(THETA_HAT_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (int(r["iteration"]), int(r["block_id"]), int(r["repeat"])))
    return rows


def build_pub(vqe, manifest_by_id, theta_row):
    snap = manifest_by_id[theta_row["snapshot_id"]]
    raw_matrix = np.load(snap["matrix_path"])
    padded_matrix = prep_matrix(raw_matrix)
    num_qubits = int(np.log2(padded_matrix.shape[0]))

    op = SparsePauliOp.from_operator(padded_matrix, atol=1e-12)
    ent_map = vqe._get_entanglement(op)
    entanglement_edges = len(ent_map) if isinstance(ent_map, list) else 0
    execution_ansatz, _abstract_ansatz = vqe._get_ansatz(num_qubits, ent_map)
    isa_op = op.apply_layout(execution_ansatz.layout)
    theta_hat = np.array(json.loads(theta_row["theta_hat_json"]))

    pub = (execution_ansatz, isa_op, theta_hat)
    meta = {
        "snapshot_id": theta_row["snapshot_id"], "iteration": int(theta_row["iteration"]),
        "block_id": int(theta_row["block_id"]), "repeat": int(theta_row["repeat"]), "seed": int(theta_row["seed"]),
        "n_original": int(snap["n_original"]), "n_padded": int(snap["n_padded"]), "num_qubits": num_qubits,
        "true_min_eigenvalue": float(snap["true_min_eigenvalue"]), "num_pauli_terms": len(op),
        "entanglement_edges": entanglement_edges, "transpiled_depth": execution_ansatz.depth(),
        "two_qubit_gate_count": sum(1 for instr in execution_ansatz.data if instr.operation.num_qubits == 2),
    }
    return pub, meta


def run_hardware_stage(manifest_rows, backend_name, shots, meter_only):
    manifest_by_id = {r["snapshot_id"]: r for r in manifest_rows}
    theta_rows = load_theta_hat_rows()
    if meter_only:
        theta_rows = theta_rows[:1]

    service = QiskitRuntimeService(channel=RUNTIME_CHANNEL)
    if backend_name is not None:
        real_backend = service.backend(backend_name)
    else:
        real_backend = service.least_busy(operational=True, simulator=False)
    print(f"Hardware backend: {real_backend.name}")

    # Reuses VQESubroutine purely for its ansatz-building machinery (pm / _get_ansatz /
    # _get_entanglement) -- consistent with how the caching layer is meant to be shared, and
    # with tier_a_oracle_benchmark.py already reaching into these same "private" helpers.
    # Its own self.estimator (a bare, non-batched RuntimeEstimator) is deliberately unused --
    # this script manages its own Batch + EstimatorV2 below so every pub goes in one job.
    vqe = VQESubroutine(ansatz_type=ANSATZ_TYPE, execution_mode="hardware", backend_name=real_backend.name, reps=REPS)

    pubs, metas = [], []
    for theta_row in theta_rows:
        try:
            pub, meta = build_pub(vqe, manifest_by_id, theta_row)
            pubs.append(pub)
            metas.append(meta)
        except Exception as e:
            log_error("hardware-build", theta_row["snapshot_id"], theta_row["iteration"], theta_row["block_id"],
                       "hardware", shots, theta_row["repeat"], theta_row["seed"], e)

    if not pubs:
        print("No pubs to submit (all builds failed or theta_hat file was empty); nothing sent to hardware.")
        return

    print(f"Submitting {len(pubs)} circuit(s) to {real_backend.name}, {shots} shots/circuit, resilience_level={RESILIENCE_LEVEL}"
          f"{' [METER-ONLY]' if meter_only else ' [FULL GRID]'}")
    for meta in metas:
        print(f"  snapshot={meta['snapshot_id']} repeat={meta['repeat']} num_qubits={meta['num_qubits']} "
              f"transpiled_depth={meta['transpiled_depth']} two_qubit_gates={meta['two_qubit_gate_count']}")

    start = time.time()
    with Batch(backend=real_backend) as batch:
        estimator = RuntimeEstimator(mode=batch)
        estimator.options.resilience_level = RESILIENCE_LEVEL
        estimator.options.default_shots = shots
        job = estimator.run(pubs)
        print(f"Job submitted: {job.job_id()}. Waiting for result...")
        result = job.result()
    solve_time = time.time() - start

    qpu_seconds_total = get_qpu_seconds(job, result)
    if qpu_seconds_total is None:
        print("WARNING: could not determine QPU seconds from job.usage()/job.metrics()/result.metadata.")
        qpu_seconds_per_pub = None
    else:
        # IBM Runtime reports usage at job granularity, not per-pub; averaging is the best
        # available per-row figure when >1 pub shares a job (meter-only always has exactly 1,
        # so this is exact in that case).
        qpu_seconds_per_pub = qpu_seconds_total / len(pubs)

    print(f"Reported QPU time for this job: {qpu_seconds_total} seconds "
          f"({'exact -- single evaluation' if meter_only else f'total across {len(pubs)} evaluations, {qpu_seconds_per_pub:.4f}s/eval avg'})")

    for i, meta in enumerate(metas):
        try:
            estimate = float(result[i].data.evs)
            abs_error = abs(estimate - meta["true_min_eigenvalue"])
            row = {
                "iteration": meta["iteration"], "block_id": meta["block_id"], "n_original": meta["n_original"],
                "n_padded": meta["n_padded"], "num_qubits": meta["num_qubits"], "mode": "hardware", "shots": shots,
                "backend_name": real_backend.name, "repeat": meta["repeat"], "seed": meta["seed"],
                "true_min_eigenvalue": meta["true_min_eigenvalue"], "estimate": estimate, "abs_error": abs_error,
                "relative_error": relative_error_of(abs_error, meta["true_min_eigenvalue"]),
                "solve_time": solve_time / len(pubs), "qpu_seconds": qpu_seconds_per_pub,
                "num_pauli_terms": meta["num_pauli_terms"], "entanglement_edges": meta["entanglement_edges"],
                "transpiled_depth": meta["transpiled_depth"], "two_qubit_gate_count": meta["two_qubit_gate_count"],
                "qiskit_version": QISKIT_VERSION, "qiskit_aer_version": QISKIT_AER_VERSION,
                "qiskit_ibm_runtime_version": QISKIT_IBM_RUNTIME_VERSION,
            }
            cp.log_row(RESULTS_PATH, row, RESULTS_FIELDNAMES)
        except Exception as e:
            log_error("hardware-extract", meta["snapshot_id"], meta["iteration"], meta["block_id"],
                       "hardware", shots, meta["repeat"], meta["seed"], e)

    if meter_only:
        print("Meter-only run complete. Inspect the QPU-seconds figure above, then re-launch with --full-grid.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "hardware"], default="prepare",
                         help="'prepare' runs Stages A+B (no QPU). 'hardware' runs Stage C (uses QPU).")
    parser.add_argument("--manifest", default=MANIFEST_PATH, help="Path to the control2 snapshot manifest.")
    parser.add_argument("--backend", default=None, help="Hardware stage only: explicit backend name (default: least busy).")
    parser.add_argument("--shots", type=int, default=DEFAULT_HARDWARE_SHOTS, help="Hardware stage only: shots per circuit.")
    parser.add_argument("--meter-only", dest="meter_only", action="store_true", default=True,
                         help="Hardware stage only (default): submit exactly one evaluation and report QPU seconds.")
    parser.add_argument("--full-grid", dest="full_grid", action="store_true",
                         help="Hardware stage only: submit the complete (snapshot x repeat) grid instead of one meter-only evaluation.")
    args = parser.parse_args()

    if args.stage == "prepare":
        manifest_rows = read_manifest(args.manifest)
        print(f"Loaded {len(manifest_rows)} snapshot(s) from {args.manifest}")
        selected_rows, labels = select_snapshots(manifest_rows)
        print_selection(selected_rows, labels)
        run_classical_reference(selected_rows)
        print(f"Done. Results in {RESULTS_PATH}, theta_hat in {THETA_HAT_PATH}, errors (if any) in {ERRORS_PATH}")
    else:
        manifest_rows = read_manifest(args.manifest)
        run_hardware_stage(manifest_rows, args.backend, args.shots, meter_only=not args.full_grid)


if __name__ == "__main__":
    main()