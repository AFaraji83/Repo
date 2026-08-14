"""
Cutting-plane solver for semidefinite programs (SDPs).
Implements a cutting-plane method that iteratively refines a linear relaxation
of a block-diagonal SDP by adding valid inequalities derived from violated
positive-semidefiniteness constraints. The separation problem identifies
violated constraints using a variational quantum eigensolver (VQE).
"""

import csv
import hashlib
import logging
import os
import resource
import socket
import time
from collections import defaultdict

import mosek.fusion as msk
import numpy as np
import qiskit
import qiskit_aer
import qiskit_ibm_runtime
from gurobipy import GRB, Model, quicksum
from matplotlib import pyplot as plt
from qiskit_algorithms.optimizers import COBYLA
from VQESubroutine import VQESubroutine

# Numerical tolerance for PSD checks
TOL = 1.0e-6

# Pinned per output row: fake-backend calibrations (and other primitive behavior) change
# across releases, so every CSV row records exactly which versions produced it.
QISKIT_VERSION = qiskit.__version__
QISKIT_AER_VERSION = qiskit_aer.__version__
QISKIT_IBM_RUNTIME_VERSION = qiskit_ibm_runtime.__version__

# module-level logger used across functions (configured in __main__)
logger = logging.getLogger(__name__)

# Dict container for VQE histories
vqe_min_eigenvalue_estimates = defaultdict(list)
# Dict container for exact min eigenvalues computed by classical solvers (for analysis/comparison with VQE results)
exact_min_eigenvalues = defaultdict(list)
# Dict container for leaked amplitude (probability mass in the zero-padded junk subspace) per block, per VQE call
leaked_amplitudes = defaultdict(list)
# Global history of all violating vectors collected across iterations for each block (used for diversity checks to avoid adding cuts from similar vectors)
violating_vectors = defaultdict(list)
# Global history of objective values from the master problem across iterations (for analysis/plotting after runs)
master_objective_values = []

# Fixed schema for per-iteration (per-block) result rows, written incrementally to CSV as each iteration completes.
ITERATION_LOG_FIELDNAMES = [
    "timestamp", "instance", "config_hash", "solver", "valid_cut_type", "add_soc_cuts",
    "collect_multiple_cuts", "ansatz_type", "ansatz_layers", "execution_mode", "optimizer_name",
    "optimizer_maxiter", "random_seed", "repeat", "iteration", "block_id", "n_original", "n_padded", "padding_overhead",
    "true_min_eigenvalue", "vqe_min_eigenvalue_estimate", "leaked_amplitude", "optimizer_nfev",
    "num_pauli_terms", "master_objective", "master_solve_time", "separation_time", "num_cuts_added_this_iter", "is_psd",
    "rss_before_sep", "rss_after_sep", "rss_before_master", "rss_after_master",
    "shots", "vqe_seed", "backend_name", "thermal_relaxation",
    "num_candidates_seen", "num_candidates_accepted", "max_energy_true_quad_gap",
    "qiskit_version", "qiskit_aer_version", "qiskit_ibm_runtime_version",
    # Added for adaptive/improved runs
    "ansatz_mode", "max_reps", "adapt_grad_tol", "leakage_penalty",
]

# Fixed schema for the snapshot manifest (Task 4): maps each saved snapshot file pair
# (matrix + VQE optimal parameters) to its metadata, so downstream scripts (Tier A/B) don't
# have to parse filenames.
SNAPSHOT_MANIFEST_FIELDNAMES = [
    "snapshot_id", "matrix_path", "params_path", "iteration", "block_id", "n_original", "n_padded",
    "true_min_eigenvalue", "num_pauli_terms", "ansatz_type", "ansatz_layers", "execution_mode", "config_hash",
]

# Fixed schema for the per-candidate diagnostics log (Task 2): one row per cost-function
# evaluation, within a given cutting-plane (iteration, block), where the estimator reported
# energy < energy_threshold -- i.e. one row per entry in VQESubroutine.solve's
# vqe_trajectory["collected_candidate_*"] lists, whether or not the classical guard accepted
# it as a cut. Deliberately a separate, long-format file rather than list-valued columns on
# ITERATION_LOG_FIELDNAMES: the two logs are at different grains (one row per (iteration,
# block) vs. one row per COBYLA-internal candidate), and keeping them separate means neither
# file needs a json.loads()-and-explode step before analysis.
CANDIDATE_LOG_FIELDNAMES = [
    "timestamp", "instance", "config_hash", "solver", "valid_cut_type", "add_soc_cuts",
    "collect_multiple_cuts", "ansatz_type", "ansatz_layers", "execution_mode", "optimizer_name",
    "optimizer_maxiter", "random_seed", "repeat", "iteration", "block_id", "eval_index",
    "estimator_energy", "true_quad", "accepted", "shots", "vqe_seed", "backend_name",
    "thermal_relaxation", "qiskit_version", "qiskit_aer_version", "qiskit_ibm_runtime_version",
    "ansatz_mode", "max_reps", "adapt_grad_tol", "leakage_penalty",
]

def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def current_rss_mb() -> float | None:
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        return None

def log_row(filepath, row: dict, fieldnames: list):
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())

def csv_to_excel(csv_path, xlsx_path):
    import pandas as pd
    pd.read_csv(csv_path).to_excel(xlsx_path, index=False)

def compute_config_hash(config: dict) -> str:
    normalized = str(sorted((k, str(v)) for k, v in config.items()))
    return hashlib.sha1(normalized.encode()).hexdigest()[:10]

def next_power_of_two(n: int) -> int:
    return n if (n & (n - 1)) == 0 else 1 << (n - 1).bit_length()

def read_instance(instance: str):
    print("Reading instance " + instance + " ...")
    logger.info("Reading instance %s ...", instance)
    file_path = f"./instances/sdplib/{instance}.dat-s"
    block_list = []
    A_list = []
    b_list = []
    with open(file_path) as file:
        m = 0
        nblocks = 0
        lineNumber = 0
        for line in file:
            if line[0] == "{":
                line = line[1:-2]
                splitted = line.split(",")
            else:
                splitted = line.split()
            if splitted[0] != "\"" and splitted[0] != "*":
                if lineNumber == 0:
                    m = int(splitted[0])
                if lineNumber == 1:
                    nblocks = int(splitted[0])
                if lineNumber == 2:
                    for bl in range(nblocks):
                        block_list.append(abs(int(splitted[bl])))
                    for k in range(m+1):
                        A_list.append([])
                        for bl in block_list:
                            A_list[k].append(np.zeros((bl, bl)))
                if lineNumber == 3:
                    b_list = [0] + [float(splitted[j]) for j in range(m)]
                if lineNumber > 3:
                    matno = int(splitted[0])
                    blkno = int(splitted[1])
                    i = int(splitted[2])
                    j = int(splitted[3])
                    entry = float(splitted[4])
                    A_list[matno][blkno-1][i-1,j-1] = A_list[matno][blkno-1][j-1,i-1] = entry
                lineNumber += 1
        return block_list, A_list, b_list

def psd_check(A: np.ndarray, block_id, vqe_instance, optimizer, initial_point, collect_multiple_cuts=False):
    if A.shape[0] != A.shape[1] or A.shape[0] <= 0:
        raise ValueError("Input matrix must be square and of positive dimension.")
    if A.shape[0] == 1:
        print("Trivial 1x1 block detected in PSD check.")
        if A[0, 0] < -TOL:
            return False, [np.array([1.0])], 0.0, initial_point, 0.0, None, None, {"energy": [], "true_quad": [], "accepted": []}
        else:
            return True, [], 0.0, initial_point, 0.0, None, None, {"energy": [], "true_quad": [], "accepted": []}
    elif np.max(np.abs(A)) < TOL:
        print("Matrix is effectively zero. Smallest eigenvalue is 0.0")
        return True, [], 0.0, initial_point, 0.0, None, None, {"energy": [], "true_quad": [], "accepted": []}
    else:
        noiseless_matrix = np.where(np.abs(A) < TOL, 0, A)
        original_dim = A.shape[0]
        matrix_for_vqe_psd_check = matrix_prep_for_vqe_psd_check(noiseless_matrix)
        return psd_check_vqe(matrix_for_vqe_psd_check, block_id, vqe_instance, optimizer, initial_point, collect_multiple_cuts, original_dim)

def matrix_prep_for_vqe_psd_check(A: np.ndarray):
    matrix = np.asarray(A, dtype=complex)
    dim = matrix.shape[0]
    if (dim & (dim - 1)) != 0:
        next_power_of_two = 1 << (dim - 1).bit_length()
        matrix = np.pad(matrix, ((0, next_power_of_two - dim), (0, next_power_of_two - dim)), mode='constant', constant_values=0)
    return matrix

def psd_check_vqe(A: np.ndarray, block_id, vqe_instance, optimizer, initial_point, collect_multiple_cuts=False, original_dim=None):
    time_start = time.time()
    exact_eigenvalue = np.linalg.eigvalsh(A[:original_dim, :original_dim]).min()
    exact_eigenvalue_time = time.time() - time_start
    print(f"Exact eigenvalue computed in {exact_eigenvalue_time:.4f} seconds.")
    exact_min_eigenvalues[block_id].append(exact_eigenvalue)
    print(f"Exact minimum eigenvalue (classical solver): {exact_eigenvalue:.8f}")

    is_psd = True
    try:
        # Pass energy_threshold = -1e-4 (loosened)
        trajectory, optimal_parameters, collected_state_vectors, vqe_solve_time, leaked_amplitude, num_pauli_terms = vqe_instance.solve(
            A, optimizer, initial_point,
            energy_threshold=-1e-4,   # loosened
            overlap_threshold=0.9,
            global_pool=violating_vectors[block_id],
            collect_multiple_vectors=collect_multiple_cuts,
            original_dim=original_dim,
        )
        print(f"VQE optimization completed in {vqe_solve_time:.4f} seconds, with {len(collected_state_vectors)} violating vectors.")
        print(f"Minimum eigenvalue estimate from VQE: {trajectory['energy'][-1]:.8f}")
        print(f"Leaked amplitude (padding subspace): {leaked_amplitude:.8f}")
        print(f"Number of Pauli terms: {num_pauli_terms}")
        nfev = trajectory.get("nfev")
        candidate_diagnostics = {
            "energy": trajectory.get("collected_candidate_energy", []),
            "true_quad": trajectory.get("collected_candidate_true_quad", []),
            "accepted": trajectory.get("collected_candidate_accepted", []),
        }
        vqe_min_eigenvalue_estimates[block_id].append(trajectory['energy'][-1])
        leaked_amplitudes[block_id].append(leaked_amplitude)
        if collected_state_vectors:
            is_psd = False
        return is_psd, collected_state_vectors, vqe_solve_time, optimal_parameters, leaked_amplitude, nfev, num_pauli_terms, candidate_diagnostics
    except Exception as e:
        raise RuntimeError(f"VQE failed during PSD check: {e}") from e

def _as_mosek_matrix(A: np.ndarray):
    rows, cols = np.nonzero(A)
    if rows.size < 0.3 * A.size:
        return msk.Matrix.sparse(A.shape[0], A.shape[1], rows.astype(np.int32), cols.astype(np.int32), A[rows, cols])
    return msk.Matrix.dense(A)

def add_objective_constraints_with_MOSEK(model, X, nblocks, m, A_list: list, b_list: list):
    model.objective(msk.ObjectiveSense.Maximize, msk.Expr.add([msk.Expr.dot(_as_mosek_matrix(A_list[0][b]), X[b]) for b in range(nblocks)]))
    for k in range(1,m+1):
        model.constraint(msk.Expr.add([msk.Expr.dot(_as_mosek_matrix(A_list[k][b]), X[b]) for b in range(nblocks)]), msk.Domain.equalsTo(b_list[k]))

def solve_SDP_with_MOSEK(block_list: list, A_list: list, b_list: list):
    print("*** Solving with MOSEK ***")
    logger.info("*** Solving with MOSEK ***")
    m = len(A_list) - 1
    nblocks = len(block_list)
    with msk.Model("one-shot SDP") as model:
        X = [model.variable(msk.Domain.inPSDCone(block_list[b])) for b in range(nblocks)]
        add_objective_constraints_with_MOSEK(model, X, nblocks, m, A_list, b_list)
        start_time = time.time()
        model.solve()
        end_time = time.time()
        print("*** SDP solve with MOSEK complete ***")
        print("Time= " + str(end_time-start_time))
        print("Optimal Value= " + str(model.primalObjValue()))
        logger.info("*** SDP solve with MOSEK complete ***")
        logger.info("Time= %f", end_time - start_time)
        logger.info("Optimal Solution: %s", [X[b].level() for b in range(nblocks)])
        return X, model.primalObjValue(), end_time - start_time

def add_valid_inequalities(model, X, n: int, valid: str, solver: str):
    if solver == "Gurobi":
        model.addConstrs(X[i,i] >= 0 for i in range(n))
    if solver == "MOSEK" and valid != "soc":
        model.constraint(X.diag(), msk.Domain.greaterThan(0.0))
    s2 = np.sqrt(2)
    if valid == "linear":
        coeff = [1, -1, 1+s2, 1-s2, -1+s2, -1-s2]
        if solver == "Gurobi":
            for a in coeff:
                model.addConstrs(X[i,i] + 2*a*X[i,j] + a*a*X[j,j] >= 0 for i in range(n) for j in range(i))
        if solver == "MOSEK":
            for a in coeff:
                for i in range(n):
                    for j in range(i):
                        model.constraint(msk.Expr.add([X.index(i, i), msk.Expr.mul(2.0 * a, X.index(i, j)), msk.Expr.mul(a * a, X.index(j, j))]), msk.Domain.greaterThan(0.0))
    elif valid == "soc":
        if solver == "Gurobi":
            model.addConstrs(X[i,i]*X[j,j] >= X[i,j]*X[i,j] for i in range(n) for j in range(i))
        if solver == "MOSEK":
            for i in range(n):
                for j in range(i):
                    model.constraint(msk.Expr.vstack(X.index(i, i), X.index(j, j), msk.Expr.mul(s2, X.index(i, j))), msk.Domain.inRotatedQCone())
    else:
        raise ValueError("Invalid option for 'valid'. Accepted values are 'linear' or 'soc'.")

def create_master_problem(block_list: list, A_list: list, b_list: list, solver: str, valid="linear"):
    m = len(A_list) - 1
    nblocks = len(block_list)
    if any(len(A_list[k]) != nblocks for k in range(0, m + 1)):
        raise ValueError("Each A_list[k] must have the same number of blocks as block_list.")
    if len(b_list) != m + 1:
        raise ValueError("Length of b_list must be m + 1, where m is the number of constraints.")
    if solver == "Gurobi":
        model = Model()
        model.setParam("OutputFlag", False)
        model.setParam("DualReductions", 0)
        model.setParam("InfUnbdInfo", 1)
        model.setParam("ConcurrentMethod", 3)
        X = []
        for b in range(nblocks):
            block_size = block_list[b]
            X_block = model.addVars(block_size, block_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"X_{b}")
            X.append(X_block)
        model.addConstrs(X[b][i,j] == X[b][j,i] for b in range(nblocks) for i in range(block_list[b]) for j in range(i))
        model.addConstrs(quicksum(A_list[k][b][i,j]*X[b][i,j] for b in range(nblocks) for i in range(block_list[b]) for j in range(block_list[b])) == b_list[k] for k in range(1,m+1))
        model.setObjective(quicksum(A_list[0][b][i,j]*X[b][i,j] for b in range(nblocks) for i in range(block_list[b]) for j in range(block_list[b])), GRB.MAXIMIZE)
    if solver == "MOSEK":
        model = msk.Model("master LP")
        X = [model.variable([block_list[b],block_list[b]], msk.Domain.unbounded()) for b in range(nblocks)]
        add_objective_constraints_with_MOSEK(model, X, nblocks, m, A_list, b_list)
        model.setSolverParam("intpntCoTolPfeas", 1e-6)
        model.setSolverParam("intpntCoTolDfeas", 1e-6)
        model.setSolverParam("intpntCoTolMuRed", 1e-6)
        for b in range(nblocks):
            for i in range(block_list[b]):
                for j in range(i):
                    model.constraint(msk.Expr.sub(X[b].index(i, j), X[b].index(j, i)), msk.Domain.equalsTo(0.0))
    for b in range(nblocks):
        add_valid_inequalities(model, X[b], block_list[b], valid, solver)
    return model, X

def solve_master_problem(model, X: list, block_list: list, solver: str):
    if solver == "Gurobi":
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        master_solve_time = end_time - start_time
        status = model.Status
        if status == GRB.OPTIMAL or status == GRB.SUBOPTIMAL:
            X_values = get_solution_Gurobi(model, X, block_list, attr="X")
            objVal = model.ObjVal
            return status, X_values, objVal, master_solve_time
        elif status == GRB.UNBOUNDED:
            X_values = get_solution_Gurobi(model, X, block_list, attr="UnbdRay")
            return status, X_values, None, master_solve_time
        else:
            return status, None, None, master_solve_time
    if solver == "MOSEK":
        start_time = time.time()
        model.solve()
        end_time = time.time()
        master_solve_time = end_time - start_time
        if model.getPrimalSolutionStatus() == msk.SolutionStatus.Optimal:
            X_values = get_solution_MOSEK(model, X, block_list)
            objVal = model.primalObjValue()
            return "Optimal", X_values, objVal, master_solve_time
        elif model.getPrimalSolutionStatus() == msk.SolutionStatus.Certificate:
            X_values = get_solution_MOSEK(model, X, block_list)
            return "Unbounded", X_values, None, master_solve_time
        else:
            status = (f"primal={model.getPrimalSolutionStatus()}, dual={model.getDualSolutionStatus()}, problem={model.getProblemStatus()}")
            return status, None, None, master_solve_time

def get_solution_Gurobi(model: Model, X: list, block_list: list, attr: str):
    nblocks = len(X)
    X_values = []
    for b in range(nblocks):
        block_size = block_list[b]
        var_list = [X[b][i, j] for i in range(block_size) for j in range(block_size)]
        vals = model.getAttr(attr, var_list)
        X_block = np.array(vals, dtype=float).reshape((block_size, block_size))
        X_values.append(X_block)
    return X_values

def get_solution_MOSEK(model: msk.Model, X: list, block_list: list):
    model.acceptedSolutionStatus(msk.AccSolutionStatus.Anything)
    nblocks = len(X)
    X_values = [X[b].level().reshape(block_list[b], block_list[b]) for b in range(nblocks)]
    return X_values

def add_linear_cut(model, X_block, u_vector: np.ndarray, block_size: int, solver: str):
    con = None
    if solver == "Gurobi":
        con = model.addConstr(quicksum(X_block[i,j]*u_vector[i]*u_vector[j] for i in range(block_size) for j in range(block_size)) >= 0)
    if solver == "MOSEK":
        trimmed_u_vector = u_vector[:block_size]
        con = model.constraint(msk.Expr.dot(trimmed_u_vector, msk.Expr.mul(X_block, trimmed_u_vector)), msk.Domain.greaterThan(0.0))
    return con

def add_soc_cut(model, X_block, u_vectors: list, block_size: int, solver: str):
    U = np.column_stack([np.asarray(u).ravel()[:block_size] for u in u_vectors[:2]])
    if solver == "Gurobi":
        Y = model.addVars(2, 2, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        model.addConstr(Y[0, 1] == Y[1, 0])
        model.addConstrs(Y[l, l] >= 0 for l in range(2))
        model.addConstr(Y[0, 0] * Y[1, 1] >= Y[0, 1] * Y[0, 1])
        model.addConstrs(
            (Y[k, l] == quicksum(U[i, k] * X_block[i, j] * U[j, l] for i in range(block_size) for j in range(block_size)))
            for k in range(2) for l in range(k + 1)
        )
    s2 = np.sqrt(2)
    if solver == "MOSEK":
        Y = model.variable([2, 2], msk.Domain.unbounded())
        model.constraint(msk.Expr.sub(Y.index(0, 1), Y.index(1, 0)), msk.Domain.equalsTo(0.0))
        model.constraint(msk.Expr.vstack(Y.index(0, 0), Y.index(1, 1), msk.Expr.mul(s2, Y.index(0, 1))), msk.Domain.inRotatedQCone())
        for k in range(2):
            for l in range(k + 1):
                u_k = U[:, k]
                u_l = U[:, l]
                model.constraint(msk.Expr.sub(Y.index(k, l), msk.Expr.dot(u_k, msk.Expr.mul(X_block, u_l))), msk.Domain.equalsTo(0.0))

def purge_inactive_linear_cuts(cut_registry: list, solver: str, patience: int, dual_tol: float, itr: int, logger=None):
    if solver != "MOSEK" or patience is None:
        return 0
    num_removed = 0
    for b, entries in enumerate(cut_registry):
        kept = []
        for entry in entries:
            try:
                dual_val = float(np.max(np.abs(np.asarray(entry["con"].dual()))))
            except Exception:
                kept.append(entry)
                continue
            if dual_val > dual_tol:
                entry["inactive_streak"] = 0
                kept.append(entry)
                continue
            entry["inactive_streak"] += 1
            if entry["inactive_streak"] >= patience:
                entry["con"].remove()
                num_removed += 1
            else:
                kept.append(entry)
        cut_registry[b] = kept
    if num_removed and logger is not None:
        logger.info("Iteration %d: purged %d inactive linear cuts", itr, num_removed)
    return num_removed

def save_snapshot(snapshot_dir, itr, block_id, matrix, params, n_original, n_padded, true_min_eigenvalue,
                   num_pauli_terms, ansatz_type, reps, execution_mode, config_hash):
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_id = f"iter{itr:04d}_block{block_id}"
    matrix_path = os.path.join(snapshot_dir, f"matrix_{snapshot_id}.npy")
    params_path = os.path.join(snapshot_dir, f"params_{snapshot_id}.npy")
    np.save(matrix_path, matrix)
    np.save(params_path, params)
    manifest_row = {
        "snapshot_id": snapshot_id, "matrix_path": matrix_path, "params_path": params_path,
        "iteration": itr, "block_id": block_id, "n_original": n_original, "n_padded": n_padded,
        "true_min_eigenvalue": true_min_eigenvalue, "num_pauli_terms": num_pauli_terms,
        "ansatz_type": ansatz_type, "ansatz_layers": reps, "execution_mode": execution_mode,
        "config_hash": config_hash,
    }
    log_row(os.path.join(snapshot_dir, "manifest.csv"), manifest_row, SNAPSHOT_MANIFEST_FIELDNAMES)
    logger.info("Saved snapshot %s to %s", snapshot_id, snapshot_dir)

def solve_SDP_by_cutting_plane(block_list: list, A_list: list, b_list: list, solver: str, valid = "linear", add_soc_cuts=False, optimizer=None, collect_multiple_cuts=False, itr_limit = 10, ansatz_type='hardware_efficient', execution_mode='noiseless', backend_name=None, reps=2, shots=None, thermal_relaxation=False, seed=None, log_filepath=None, log_context=None, cut_purge_patience=5, cut_dual_tol=1e-8, snapshot_iterations=None, snapshot_dir=None, candidate_log_filepath=None, ansatz_mode='fixed', max_reps=6, adapt_grad_tol=1e-4, adapt_pool_size=30, leakage_penalty=10.0, parallel_separation=False):
    print("*** Starting cutting plane ***")
    logger.info("*** Starting cutting plane ***")
    model, X = create_master_problem(block_list, A_list, b_list, solver, valid)
    nblocks = len(block_list)
    status = None
    X_values = None
    objVal = None
    linear_cuts_added = 0
    soc_cuts_added = 0
    linear_cuts_purged = 0
    master_total_time = 0
    sep_total_time = 0
    itr = 0
    is_optimal = False
    termination_reason = ""
    cut_registry = [[] for _ in range(nblocks)]
    optimizer = optimizer if optimizer is not None else COBYLA(maxiter=100)
    last_optimal_points = [None] * nblocks
    vqe_instance = VQESubroutine(
        ansatz_type=ansatz_type,
        execution_mode=execution_mode,
        backend_name=backend_name,
        reps=reps,
        shots=shots,
        thermal_relaxation=thermal_relaxation,
        ansatz_mode=ansatz_mode,
        max_reps=max_reps,
        adapt_grad_tol=adapt_grad_tol,
        adapt_pool_size=adapt_pool_size,
        leakage_penalty=leakage_penalty,
    )
    if seed is not None:
        np.random.seed(seed)
    snapshot_iterations = set(snapshot_iterations) if snapshot_iterations else set()
    for b in range(nblocks):
        vqe_min_eigenvalue_estimates[b] = []
        exact_min_eigenvalues[b] = []
        violating_vectors[b] = []
        leaked_amplitudes[b] = []
    try:
        while itr < itr_limit and not is_optimal:
            itr += 1
            logger.info("Iteration %d", itr)
            rss_before_master = current_rss_mb()
            status, X_values, objVal, master_solve_time = solve_master_problem(model, X, block_list, solver)
            rss_after_master = current_rss_mb()
            master_total_time += master_solve_time
            if objVal is not None:
                master_objective_values.append(objVal)
            logger.info("Time(master)= %f", master_solve_time)
            logger.info("ObjVal %s", str(objVal))
            logger.info("Peak RSS(MB)= %.1f", peak_rss_mb())
            if solver == "Gurobi" and status == GRB.INFEASIBLE:
                termination_reason = "The original problem is proven to be infeasible since the master problem is infeasible."
                break
            elif solver == "Gurobi" and status == GRB.NUMERIC:
                termination_reason = "Numeric difficulties encountered!"
                break
            elif X_values is None:
                termination_reason = f"Master problem returned no solution (solver={solver}, status={status})."
                break
            else:
                linear_cuts_purged += purge_inactive_linear_cuts(cut_registry, solver, cut_purge_patience, cut_dual_tol, itr, logger)
                num_violating_vectors = 0
                rss_before_sep = current_rss_mb()
                # Parallel or sequential separation
                if parallel_separation and nblocks > 1:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=nblocks) as pool:
                        block_results = list(pool.map(lambda b: psd_check(X_values[b], b, vqe_instance, optimizer, last_optimal_points[b], collect_multiple_cuts), range(nblocks)))
                else:
                    block_results = [psd_check(X_values[b], b, vqe_instance, optimizer, last_optimal_points[b], collect_multiple_cuts) for b in range(nblocks)]
                rss_after_sep = current_rss_mb()
                for b in range(nblocks):
                    is_psd, u_vectors, sep_time, last_optimal_points[b], leaked_amplitude, optimizer_nfev, num_pauli_terms, candidate_diagnostics = block_results[b]
                    num_violating_vectors += len(u_vectors)
                    sep_total_time += sep_time
                    logger.info("Time(sep)= %f", sep_time)
                    logger.info("Block %d is PSD?: %s", b, str(is_psd))
                    n_original = block_list[b]
                    n_padded = next_power_of_two(n_original)
                    if snapshot_dir is not None and itr in snapshot_iterations:
                        save_snapshot(
                            snapshot_dir, itr, b, X_values[b], last_optimal_points[b],
                            n_original, n_padded,
                            exact_min_eigenvalues[b][-1] if exact_min_eigenvalues[b] else None,
                            num_pauli_terms, ansatz_type, reps, execution_mode,
                            (log_context or {}).get("config_hash"),
                        )
                    if not is_psd:
                        logger.info("Adding cutting planes for block %d", b)
                        for u in u_vectors:
                            con = add_linear_cut(model, X[b], u, block_list[b], solver)
                            linear_cuts_added += 1
                            if solver == "MOSEK" and cut_purge_patience is not None:
                                cut_registry[b].append({"con": con, "inactive_streak": 0})
                        if add_soc_cuts and len(u_vectors) >= 2:
                            add_soc_cut(model, X[b], u_vectors[-2:], block_list[b], solver)
                            soc_cuts_added += 1
                    else:
                        logger.info("No cutting planes added for block %d", b)
                    candidate_energies = candidate_diagnostics["energy"]
                    candidate_true_quads = candidate_diagnostics["true_quad"]
                    candidate_accepted = candidate_diagnostics["accepted"]
                    num_candidates_seen = len(candidate_energies)
                    num_candidates_accepted = sum(candidate_accepted)
                    max_energy_true_quad_gap = (
                        max(abs(e - q) for e, q in zip(candidate_energies, candidate_true_quads, strict=False))
                        if candidate_energies else None
                    )
                    if log_filepath is not None:
                        row = dict(log_context or {})
                        row.update({
                            "timestamp": time.time(),
                            "iteration": itr,
                            "block_id": b,
                            "n_original": n_original,
                            "n_padded": n_padded,
                            "padding_overhead": n_padded - n_original,
                            "true_min_eigenvalue": exact_min_eigenvalues[b][-1] if exact_min_eigenvalues[b] else None,
                            "vqe_min_eigenvalue_estimate": vqe_min_eigenvalue_estimates[b][-1] if vqe_min_eigenvalue_estimates[b] else None,
                            "leaked_amplitude": leaked_amplitude,
                            "optimizer_nfev": optimizer_nfev,
                            "num_pauli_terms": num_pauli_terms,
                            "master_objective": objVal,
                            "master_solve_time": master_solve_time,
                            "separation_time": sep_time,
                            "num_cuts_added_this_iter": len(u_vectors),
                            "is_psd": is_psd,
                            "rss_before_sep": rss_before_sep,
                            "rss_after_sep": rss_after_sep,
                            "rss_before_master": rss_before_master,
                            "rss_after_master": rss_after_master,
                            "shots": shots,
                            "vqe_seed": seed,
                            "backend_name": vqe_instance.backend_name,
                            "thermal_relaxation": thermal_relaxation,
                            "num_candidates_seen": num_candidates_seen,
                            "num_candidates_accepted": num_candidates_accepted,
                            "max_energy_true_quad_gap": max_energy_true_quad_gap,
                            "qiskit_version": QISKIT_VERSION,
                            "qiskit_aer_version": QISKIT_AER_VERSION,
                            "qiskit_ibm_runtime_version": QISKIT_IBM_RUNTIME_VERSION,
                            # The following are already in log_context, but we explicitly add them for safety
                            "ansatz_mode": ansatz_mode,
                            "max_reps": max_reps,
                            "adapt_grad_tol": adapt_grad_tol,
                            "leakage_penalty": leakage_penalty,
                        })
                        log_row(log_filepath, row, ITERATION_LOG_FIELDNAMES)
                    if candidate_log_filepath is not None:
                        for eval_index, (est_energy, true_quad, accepted) in enumerate(
                            zip(candidate_energies, candidate_true_quads, candidate_accepted, strict=False)
                        ):
                            candidate_row = dict(log_context or {})
                            candidate_row.update({
                                "timestamp": time.time(),
                                "iteration": itr,
                                "block_id": b,
                                "eval_index": eval_index,
                                "estimator_energy": est_energy,
                                "true_quad": true_quad,
                                "accepted": accepted,
                                "shots": shots,
                                "vqe_seed": seed,
                                "backend_name": vqe_instance.backend_name,
                                "thermal_relaxation": thermal_relaxation,
                                "qiskit_version": QISKIT_VERSION,
                                "qiskit_aer_version": QISKIT_AER_VERSION,
                                "qiskit_ibm_runtime_version": QISKIT_IBM_RUNTIME_VERSION,
                                "ansatz_mode": ansatz_mode,
                                "max_reps": max_reps,
                                "adapt_grad_tol": adapt_grad_tol,
                                "leakage_penalty": leakage_penalty,
                            })
                            log_row(candidate_log_filepath, candidate_row, CANDIDATE_LOG_FIELDNAMES)
                if num_violating_vectors == 0:
                    is_optimal = True
                    false_psd_certificate = any(exact_min_eigenvalues[b] and exact_min_eigenvalues[b][-1] < -TOL for b in range(nblocks))
                    if false_psd_certificate:
                        termination_reason = "Terminating because all blocks are PSD (per VQE oracle). WARNING: false_psd_certificate=True -- exact EVD found a block with a negative eigenvalue that VQE missed."
                        logger.warning("False PSD certificate detected at termination! Exact min eigenvalues per block: %s", [exact_min_eigenvalues[b][-1] if exact_min_eigenvalues[b] else None for b in range(nblocks)])
                    else:
                        termination_reason = "Terminating because all blocks are PSD. Optimal solution found."
                elif itr == itr_limit:
                    termination_reason = "Terminating because iteration limit was reached."
    finally:
        model.dispose()
    print("*** Cutting plane terminated ***")
    print(termination_reason)
    print("Total Time(master)= " + str(master_total_time))
    print("Total Time(sep)   = " + str(sep_total_time))
    print("Total iterations = " + str(itr))
    print("Total linear cuts added = " + str(linear_cuts_added))
    print("Total linear cuts purged= " + str(linear_cuts_purged))
    print("Total SOC cuts added    = " + str(soc_cuts_added))
    print("Final objective value " + str(objVal))
    print("***  ***")
    logger.info("*** Cutting plane terminated ***")
    logger.info(termination_reason)
    logger.info("Total Time(master)= %f", master_total_time)
    logger.info("Total Time(sep)   = %f", sep_total_time)
    logger.info("Total iterations = %d", itr)
    logger.info("Total linear cuts added = %d", linear_cuts_added)
    logger.info("Total linear cuts purged= %d", linear_cuts_purged)
    logger.info("Total SOC cuts added    = %d", soc_cuts_added)
    logger.info("Final objective value %s", str(objVal))
    return termination_reason, itr, linear_cuts_added, linear_cuts_purged, soc_cuts_added, master_total_time, sep_total_time, X_values, objVal

def generate_plots_for_given_configuration(instance, experiment_config, avg_eigenvalue_trajectories, avg_master_objective_values, best_value):
    nblocks = len(avg_eigenvalue_trajectories["exact"])
    fig1 = plt.figure(figsize=(15, 5 * nblocks))
    for block_id in range(nblocks):
        plt.subplot(nblocks, 1, block_id + 1)
        plt.plot(avg_eigenvalue_trajectories["exact"][block_id], label='Exact Minimum Eigenvalue (Classical Solver)', marker='o', linewidth=2, markersize=4)
        plt.plot(avg_eigenvalue_trajectories["vqe"][block_id], label='Average VQE Minimum Eigenvalue Estimate', marker='x', linewidth=2, markersize=5)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='PSD Boundary (Eigenvalue=0)')
        plt.xlabel('Iteration', fontsize=11)
        plt.ylabel('Minimum Eigenvalue', fontsize=11)
        plt.title(f'Block {block_id} - Exact vs VQE Eigenvalue Trajectory', fontsize=12, fontweight='bold')
        plt.legend(fontsize=9, loc='best')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{instance}_blockwise_eigenvalue_trajectories_config_{experiment_config}.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    fig2 = plt.figure(figsize=(12, 7))
    plt.plot(avg_master_objective_values, label='Average Master Objective Value', marker='o', linewidth=2, markersize=4)
    plt.axhline(y=best_value, color='g', linestyle='--', linewidth=2, label='Best Value from CVXPY+MOSEK')
    plt.xlabel('Iteration', fontsize=11)
    plt.ylabel('Master Objective Value', fontsize=11)
    plt.title('Master Objective Value Trajectory', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{instance}_master_objective_trajectory_config_{experiment_config}.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

def generate_VQE_comparison_plots(instance, all_vqe_min_eigenvalue_avg_estimates):
    if not all_vqe_min_eigenvalue_avg_estimates:
        print("No VQE trajectory data available for plotting")
        return
    first_config_trajectories = next(iter(all_vqe_min_eigenvalue_avg_estimates.values()))
    num_blocks = len(first_config_trajectories)
    fig = plt.figure(figsize=(15, 5 * num_blocks))
    for block_id in range(num_blocks):
        plt.subplot(num_blocks, 1, block_id + 1)
        for config, vqe_trajectories in all_vqe_min_eigenvalue_avg_estimates.items():
            solver, valid, add_soc_cuts, collect_multiple_cuts, ansatz_type, execution_mode, p, optimizer_name, optimizer_maxiter = config
            if block_id in vqe_trajectories:
                trajectory = vqe_trajectories[block_id]
                label = f'Solver: {solver}, Valid: {valid}, Add SOC Cuts: {add_soc_cuts}, Multi-Cuts: {collect_multiple_cuts}, Ansatz: {ansatz_type}, Layers: {p}, Mode: {execution_mode}, Opt: {optimizer_name}(maxiter={optimizer_maxiter})'
                plt.plot(trajectory, label=label, marker='o', linewidth=2, markersize=4)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='PSD Boundary (Eigenvalue=0)')
        plt.xlabel('Iteration', fontsize=11)
        plt.ylabel('Average Minimum Eigenvalue Estimate', fontsize=11)
        plt.title(f'Block {block_id} - VQE Performance Across Configurations', fontsize=12, fontweight='bold')
        plt.legend(fontsize=9, loc='best')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{instance}_vqe_eigenvalue_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def write_results_to_file(results, filename="results.xlsx"):
    if results:
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_excel(filename, index=False)
            print(f"Wrote {filename}")
        except Exception:
            csv_path = "results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
            print(f"Pandas not available or write failed; wrote {csv_path} instead")
    else:
        print("No results collected; nothing to write.")

if __name__ == "__main__":
    # Restrict to hinf1-hinf5 for this test
    instances = [f'hinf{i}' for i in range(1,6)]
    print(f"Running instances: {instances}")
    solvers = ["MOSEK"]
    initial_valid_cut_type = ["soc"]
    soc_cuts_added_options = [True]
    collect_multiple_cuts_options = [True]
    # Use LBFGSB optimizer (gradient-based, supports batched evaluations)
    from vqe_optimizers import LBFGSB
    optimizers = [LBFGSB(maxiter=150, tol=1e-6)]
    ansatz_types = ['sparsity_aware']  # can also use hardware_efficient
    execution_modes = ['noiseless']
    # Use adaptive depth, max 6 layers
    ansatz_mode = 'adaptive_depth'
    max_reps = 6
    adapt_grad_tol = 1e-4
    leakage_penalty = 10.0
    ansatz_layers = [3]  # initial depth, but adaptive will override
    num_repeats = 1

    logging.basicConfig(filename='vqe_improved_hinf1-5_log.txt', level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
    logger.setLevel(logging.INFO)

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    VM_ID = socket.gethostname()

    CUT_PURGE_PATIENCE = 3  # enable purging
    ITERATION_LOG_PATH = "vqe_improved_iterations.csv"
    RESULTS_LOG_PATH = "vqe_improved_results.csv"
    ERROR_LOG_PATH = "vqe_improved_errors.csv"
    ERROR_LOG_FIELDNAMES = [
        "timestamp", "vm_id", "instance", "config_hash", "solver", "valid_cut_type", "add_soc_cuts",
        "collect_multiple_cuts", "ansatz_type", "ansatz_layers", "execution_mode", "optimizer_name",
        "optimizer_maxiter", "random_seed", "repeat", "error",
        "ansatz_mode", "max_reps", "adapt_grad_tol", "leakage_penalty",
    ]

    results = []
    for instance in instances:
        try:
            block_list, A_list, b_list = read_instance(instance)
            best_solution, best_value, solve_time = solve_SDP_with_MOSEK(block_list, A_list, b_list)
        except Exception as e:
            print(f"Failed to read/solve instance {instance}, skipping it: {e}")
            logger.exception("Failed to read/solve instance %s", instance)
            log_row(ERROR_LOG_PATH, {"timestamp": time.time(), "vm_id": VM_ID, "instance": instance, "error": str(e)}, ERROR_LOG_FIELDNAMES)
            continue

        all_vqe_min_eigenvalue_avg_estimates = dict()
        for solver in solvers:
            for valid in initial_valid_cut_type:
                for add_soc_cuts in soc_cuts_added_options:
                    for collect_multiple_cuts in collect_multiple_cuts_options:
                        for ansatz_type in ansatz_types:
                            for execution_mode in execution_modes:
                                for optimizer in optimizers:
                                    for p in ansatz_layers:
                                        print(f"=== Instance: {instance}, Solver: {solver}, Valid Cut Type: {valid}, Add SOC Cuts: {add_soc_cuts}, Collect Multiple Cuts: {collect_multiple_cuts}, Optimizer: {optimizer.__class__.__name__}, Optimizer Maxiter: {optimizer.settings['maxiter']}, Ansatz Type: {ansatz_type}, Ansatz Layers: {p}, Execution Mode: {execution_mode}, Ansatz Mode: {ansatz_mode} ===")
                                        logger.info(f"=== Instance: {instance}, Solver: {solver}, Valid Cut Type: {valid}, Add SOC Cuts: {add_soc_cuts}, Collect Multiple Cuts: {collect_multiple_cuts}, Optimizer: {optimizer.__class__.__name__}, Optimizer Maxiter: {optimizer.settings['maxiter']}, Ansatz Type: {ansatz_type}, Ansatz Layers: {p}, Execution Mode: {execution_mode}, Ansatz Mode: {ansatz_mode} ===")
                                        optimizer_maxiter = optimizer.settings["maxiter"]
                                        experiment_config = (solver, valid, add_soc_cuts, collect_multiple_cuts, ansatz_type, execution_mode, p, optimizer.__class__.__name__, optimizer_maxiter)
                                        all_eigenvalue_trajectories = {repeat: {"exact": dict(), "vqe": dict()} for repeat in range(1, num_repeats + 1)}
                                        all_master_objective_values = {repeat: [] for repeat in range(1, num_repeats + 1)}
                                        config_dict = {
                                            "instance": instance, "solver": solver, "valid_cut_type": valid,
                                            "add_soc_cuts": add_soc_cuts, "collect_multiple_cuts": collect_multiple_cuts,
                                            "ansatz_type": ansatz_type, "ansatz_layers": p,
                                            "execution_mode": execution_mode, "optimizer_name": optimizer.__class__.__name__,
                                            "optimizer_maxiter": optimizer_maxiter,
                                            "ansatz_mode": ansatz_mode, "max_reps": max_reps,
                                            "adapt_grad_tol": adapt_grad_tol, "leakage_penalty": leakage_penalty,
                                        }
                                        config_hash = compute_config_hash(config_dict)

                                        for repeat in range(1, num_repeats + 1):
                                            logger.info(f"--- Run {repeat}/{num_repeats} ---")
                                            log_context = {**config_dict, "config_hash": config_hash, "random_seed": RANDOM_SEED, "repeat": repeat}
                                            rss_at_run_start_mb = current_rss_mb()
                                            try:
                                                termination_reason, itr, linear_cuts_added, linear_cuts_purged, soc_cuts_added, master_total_time, sep_total_time, X_values, objVal = solve_SDP_by_cutting_plane(
                                                    block_list, A_list, b_list, solver,
                                                    valid=valid,
                                                    add_soc_cuts=add_soc_cuts,
                                                    collect_multiple_cuts=collect_multiple_cuts,
                                                    optimizer=optimizer,
                                                    itr_limit=250,
                                                    ansatz_type=ansatz_type,
                                                    execution_mode=execution_mode,
                                                    backend_name="ibm_fez",
                                                    reps=p,
                                                    log_filepath=ITERATION_LOG_PATH,
                                                    log_context=log_context,
                                                    cut_purge_patience=CUT_PURGE_PATIENCE,
                                                    ansatz_mode=ansatz_mode,
                                                    max_reps=max_reps,
                                                    adapt_grad_tol=adapt_grad_tol,
                                                    leakage_penalty=leakage_penalty,
                                                    parallel_separation=False,
                                                )
                                                all_eigenvalue_trajectories[repeat]["exact"] = exact_min_eigenvalues.copy()
                                                all_eigenvalue_trajectories[repeat]["vqe"] = vqe_min_eigenvalue_estimates.copy()
                                                all_master_objective_values[repeat].extend(master_objective_values.copy())
                                                violating_vectors.clear()
                                                exact_min_eigenvalues.clear()
                                                vqe_min_eigenvalue_estimates.clear()
                                                master_objective_values.clear()
                                                leaked_amplitudes.clear()
                                            except Exception as e:
                                                print(f"Run failed with solver {solver} for {instance} with initial valid cuts {valid}, with SOC cuts added {add_soc_cuts}, collect multiple cuts option {collect_multiple_cuts}, optimizer {optimizer.__class__.__name__} with maxiter {optimizer_maxiter}, ansatz type {ansatz_type}, ansatz layers {p}, execution mode {execution_mode}, in (repeat {repeat}): {e}")
                                                logger.exception("Run failed (repeat %d)", repeat)
                                                # Ensure log_context includes the extra fields for error log
                                                log_row(ERROR_LOG_PATH, {**log_context, "timestamp": time.time(), "vm_id": VM_ID, "error": str(e)}, ERROR_LOG_FIELDNAMES)
                                                termination_reason = str(e)
                                                itr = None
                                                linear_cuts_added = None
                                                linear_cuts_purged = None
                                                soc_cuts_added = None
                                                master_total_time = None
                                                sep_total_time = None
                                                objVal = None

                                            result_row = {
                                                "timestamp": time.time(),
                                                "vm_id": VM_ID,
                                                "config_hash": config_hash,
                                                "random_seed": RANDOM_SEED,
                                                "instance": instance,
                                                "solver": solver,
                                                "valid": valid,
                                                "add_soc_cuts": add_soc_cuts,
                                                "collect_multiple_cuts": collect_multiple_cuts,
                                                "optimizer": optimizer.__class__.__name__,
                                                "optimizer_maxiter": optimizer_maxiter,
                                                "ansatz_type": ansatz_type,
                                                "ansatz_layers": p,
                                                "execution_mode": execution_mode,
                                                "repeat": repeat,
                                                "termination_reason": termination_reason,
                                                "iterations": itr,
                                                "linear_cuts_added": linear_cuts_added,
                                                "linear_cuts_purged": linear_cuts_purged,
                                                "soc_cuts_added": soc_cuts_added,
                                                "master_total_time": master_total_time,
                                                "sep_total_time": sep_total_time,
                                                "objVal": objVal,
                                                "rss_at_run_start_mb": rss_at_run_start_mb,
                                                "peak_rss_mb": peak_rss_mb(),
                                            }
                                            results.append(result_row)
                                            log_row(RESULTS_LOG_PATH, result_row, list(result_row.keys()))
                                            print(f"Result: termination_reason={termination_reason}, iterations={itr}, linear_cuts={linear_cuts_added}, linear_cuts_purged={linear_cuts_purged}, soc_cuts={soc_cuts_added}, master_time={master_total_time}, sep_time={sep_total_time}, objVal={objVal}")
                                            print("***  ***\n")
                                            logger.info(f"Result: termination_reason={termination_reason}, iterations={itr}, linear_cuts={linear_cuts_added}, linear_cuts_purged={linear_cuts_purged}, soc_cuts={soc_cuts_added}, master_time={master_total_time}, sep_time={sep_total_time}, objVal={objVal}")

                                        successful_repeats = [repeat for repeat in all_eigenvalue_trajectories if all_eigenvalue_trajectories[repeat]["exact"]]
                                        if not successful_repeats:
                                            print(f"All repeats failed for experiment config {experiment_config}; skipping plot generation.")
                                            logger.warning("All repeats failed for experiment config %s; skipping plot generation.", experiment_config)
                                            continue
                                        avg_eigenvalue_trajectories = {"exact": dict(), "vqe": dict()}
                                        for block_id in range(len(block_list)):
                                            max_iters_for_block = max(len(all_eigenvalue_trajectories[repeat]["exact"][block_id]) for repeat in successful_repeats)
                                            for repeat in successful_repeats:
                                                if len(all_eigenvalue_trajectories[repeat]["exact"][block_id]) < max_iters_for_block:
                                                    all_eigenvalue_trajectories[repeat]["exact"][block_id].extend([all_eigenvalue_trajectories[repeat]["exact"][block_id][-1]] * (max_iters_for_block - len(all_eigenvalue_trajectories[repeat]["exact"][block_id])))
                                                if len(all_eigenvalue_trajectories[repeat]["vqe"][block_id]) < max_iters_for_block:
                                                    all_eigenvalue_trajectories[repeat]["vqe"][block_id].extend([all_eigenvalue_trajectories[repeat]["vqe"][block_id][-1]] * (max_iters_for_block - len(all_eigenvalue_trajectories[repeat]["vqe"][block_id])))
                                            avg_eigenvalue_trajectories["exact"][block_id] = [np.mean([all_eigenvalue_trajectories[repeat]["exact"][block_id][iter_id] for repeat in successful_repeats]) for iter_id in range(max_iters_for_block)]
                                            avg_eigenvalue_trajectories["vqe"][block_id] = [np.mean([all_eigenvalue_trajectories[repeat]["vqe"][block_id][iter_id] for repeat in successful_repeats]) for iter_id in range(max_iters_for_block)]
                                        all_vqe_min_eigenvalue_avg_estimates[experiment_config] = avg_eigenvalue_trajectories["vqe"].copy()
                                        avg_master_objective_values = []
                                        objective_repeats = [repeat for repeat in successful_repeats if all_master_objective_values[repeat]]
                                        if not objective_repeats:
                                            print(f"No repeats recorded a master objective value for experiment config {experiment_config}; skipping objective-trajectory plot.")
                                            logger.warning("No repeats recorded a master objective value for experiment config %s; skipping objective-trajectory plot.", experiment_config)
                                            generate_plots_for_given_configuration(instance, experiment_config, avg_eigenvalue_trajectories, avg_master_objective_values, best_value)
                                            continue
                                        max_length = max(len(all_master_objective_values[repeat]) for repeat in objective_repeats)
                                        print(f"Maximum length of master objective value trajectories across repeats for this experiment configuration: {max_length}")
                                        for repeat in objective_repeats:
                                            if len(all_master_objective_values[repeat]) < max_length:
                                                print(f"Run {repeat} has fewer master objective values ({len(all_master_objective_values[repeat])}) than the maximum length ({max_length}). Padding with the last recorded value for averaging.")
                                                all_master_objective_values[repeat].extend([all_master_objective_values[repeat][-1]] * (max_length - len(all_master_objective_values[repeat])))
                                        avg_master_objective_values = [np.mean([all_master_objective_values[repeat][iter_id] for repeat in objective_repeats]) for iter_id in range(max_length)]
                                        generate_plots_for_given_configuration(instance, experiment_config, avg_eigenvalue_trajectories, avg_master_objective_values, best_value)
        generate_VQE_comparison_plots(instance, all_vqe_min_eigenvalue_avg_estimates)

    write_results_to_file(results, filename="vqe_improved_hinf1-5_results.xlsx")