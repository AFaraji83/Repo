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
from concurrent.futures import ThreadPoolExecutor

import mosek.fusion as msk
import numpy as np
from gurobipy import GRB, Model, quicksum
from matplotlib import pyplot as plt
# Change #1 (Optimizer Benchmarking): all six interchangeable optimizers are available here;
# solve_SDP_by_cutting_plane's `optimizer=` argument accepts an instance of any of them.
from vqe_optimizers import COBYLA, Powell, SPSA, QNSPSA, LBFGSB, SLSQP, OPTIMIZER_REGISTRY
from VQESubroutine import VQESubroutine

# Numerical tolerance for PSD checks
TOL = 1.0e-6

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
]

def peak_rss_mb() -> float:
    """
    Peak resident-set size (high-water mark, not current usage) of this process in MiB,
    for diagnosing memory growth across cutting-plane iterations. ru_maxrss is reported
    in KiB on Linux.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def current_rss_mb() -> float:
    """
    Current (not peak) resident-set size of this process in MiB, read directly from
    /proc/self/status (VmRSS). Used to sample memory at specific points within a single
    cutting-plane iteration (before/after separation, before/after the master solve) so
    memory growth can be attributed to a specific step rather than just the run as a whole.
    Linux-only (procfs).
    """
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024

def log_row(filepath, row: dict, fieldnames: list):
    """
    Append a single row to a CSV file, creating it (with header) if it doesn't exist yet.
    Flushes and fsyncs after every write so a crash mid-run loses at most the in-flight row,
    not the whole file (unlike writing Excel incrementally, which can corrupt the entire workbook).
    """
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def csv_to_excel(csv_path, xlsx_path):
    """Convert a results CSV to Excel. Read-only w.r.t. the CSV, so safe to run at any time, even mid-experiment."""
    import pandas as pd
    pd.read_csv(csv_path).to_excel(xlsx_path, index=False)


def compute_config_hash(config: dict) -> str:
    """Short, stable hash of a config dict for easy grouping/filtering of results later (e.g. pandas groupby)."""
    normalized = str(sorted((k, str(v)) for k, v in config.items()))
    return hashlib.sha1(normalized.encode()).hexdigest()[:10]


def next_power_of_two(n: int) -> int:
    """Smallest power of two >= n (the padded dimension a block of size n is zero-padded up to for VQE)."""
    return n if (n & (n - 1)) == 0 else 1 << (n - 1).bit_length()

def read_instance(instance: str):
    """
    ReadInstance(instance)
    Read and parse an SDP instance file from the local sdplib directory and build
    dense block matrices for the objective / constraint data.
    Parameters
    ----------
    instance : str
        Base name of the instance file (without the ".dat-s" suffix). The function
        opens the file at "./instances/sdplib/{instance}.dat-s".
    Returns
    -------
    block_list : list[int]
        List of block sizes (one integer per block). Note: the current parser does
        not account for special-case negative block sizes (which some formats use 
        to denote diagonal blocks); it simply appends the integer values as given.
    A_list : list[list[numpy.ndarray]]
        A_list is a list of length m+1 (where m is the number of constraints read
        from the file). Each element A_list[k] is itself a list of numpy arrays,
        one array per block listed in block_list. Each block array is a dense
        square numpy.ndarray of shape (block_size, block_size). The function 
        allocates full dense blocks (it ignores the sparse structure).
    b_list : list[float]
        List of objective function coefficients of length m+1. The returned list is
        prefixed with a 0 at index 0 (i.e., b_list[0] == 0) and b_list[1..m] are
        the floats parsed from the file.
    """
    # this code ignores the sparse structure !!!!
    # we may want to incorporate that in the future
    
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
                if lineNumber == 2: # this needs to be generalized to ignore {},() 
                    for bl in range(nblocks):
                        block_list.append(abs(int(splitted[bl]))) # block size can be negative => the block is diagonal
                    for k in range(m+1):
                        A_list.append([])
                        for bl in block_list:
                            A_list[k].append(np.zeros((bl, bl)))
                if lineNumber == 3: # this needs to be generalized to ignore {},()
                    # print(splitted)
                    b_list = [0] + [float(splitted[j]) for j in range(m)]
                if lineNumber > 3:
                    # print(splitted)
                    matno = int(splitted[0])
                    blkno = int(splitted[1])
                    i = int(splitted[2])
                    j = int(splitted[3])
                    entry = float(splitted[4])
                    # print(matno, ii, jj, entry)
                    A_list[matno][blkno-1][i-1,j-1] = A_list[matno][blkno-1][j-1,i-1] = entry
                    # it would suffice to store the lower triangular part of each matrix!!!                    
                lineNumber += 1
                # print(line.strip())
        return block_list, A_list, b_list

def psd_check(A: np.ndarray, block_id, vqe_instance, optimizer, initial_point, collect_multiple_cuts=False,
              opt_state=None):
    """
    Check whether the matrix A is positive semidefinite (PSD) using a VQE-based method. The function first performs 
    input validation to ensure that A is square and of positive dimension. It then handles trivial cases: if A is 1x1, 
    it checks the single entry; if A is effectively zero, it treats it as PSD. For non-trivial cases, it scrubs small 
    entries below a certain tolerance to reduce noise and keep the Pauli string count low for the VQE check. Finally, 
    it calls the psd_check_vqe function to perform the actual VQE-based PSD check and returns the results.

    opt_state: optimizer-internal state (e.g. QNSPSA's running metric-tensor estimate) carried in from
        this same block's previous cutting-plane iteration, and returned again for the caller to pass
        into the *next* iteration -- Change #4's optimizer-state reuse.
    """
    # INPUT VALIDATION
    # Check that the matrix is square and its dimension is greater than 0
    if A.shape[0] != A.shape[1] or A.shape[0] <= 0:
        raise ValueError("Input matrix must be square and of positive dimension.")
    
    # HANDLE TRIVIAL CASES AND REDUCE NOISE FOR NON-TRIVIAL CASES
    # (trivial cases never touch the quantum circuit, so there is no zero-padded subspace to leak into)
    if A.shape[0] == 1: # 1x1 (scalar) blocks
        print("Trivial 1x1 block detected in PSD check.")
        if A[0, 0] < -TOL: # Exact classical check for a scalar: eigenvalue is the single entry.
            return False, [np.array([1.0])], 0.0, initial_point, 0.0, None, None, opt_state # Violating vector is simply [1.0]; no SparsePauliOp is built for a trivial block, so num_pauli_terms is None
        else:
            return True, [], 0.0, initial_point, 0.0, None, None, opt_state
    elif np.max(np.abs(A)) < TOL: # effectively zero matrix
        print("Matrix is effectively zero. Smallest eigenvalue is 0.0")
        return True, [], 0.0, initial_point, 0.0, None, None, opt_state
    else: # Scrub the noise out to keep the Pauli string count low
        noiseless_matrix = np.where(np.abs(A) < TOL, 0, A)
        original_dim = A.shape[0] # dimension before zero-padding to the next power of two, needed for leaked-amplitude tracking
        matrix_for_vqe_psd_check = matrix_prep_for_vqe_psd_check(noiseless_matrix)
        return psd_check_vqe(matrix_for_vqe_psd_check, block_id, vqe_instance, optimizer, initial_point,
                              collect_multiple_cuts, original_dim, opt_state)

def matrix_prep_for_vqe_psd_check(A: np.ndarray):
    """
    Prepare the input matrix A for the VQE-based PSD check by ensuring it is a square matrix of dimension that is a power of
    two. The function first converts the input to a complex numpy array, then checks if the dimension is already a power of 
    two. If not, it pads the matrix with zeros to the next power of two in both dimensions. This ensures compatibility with
    quantum algorithms that typically require input sizes that are powers of two.
    Parameters
    ----------
    A : np.ndarray
        The input matrix to be prepared for the VQE-based PSD check. Must be square and of positive dimension.
    Returns
    -------
    matrix : np.ndarray
        The prepared matrix, which is a square complex numpy array of dimension that is a power of two
    """
    matrix = np.asarray(A, dtype=complex)
    dim = matrix.shape[0]

    # Pad the matrix A to the next power of two if necessary
    if (dim & (dim - 1)) != 0:
        next_power_of_two = 1 << (dim - 1).bit_length()
        # Use complex padding to match the matrix dtype
        matrix = np.pad(matrix, ((0, next_power_of_two - dim), (0, next_power_of_two - dim)), mode='constant', constant_values=0)

    return matrix

def psd_check_vqe(A: np.ndarray, block_id, vqe_instance, optimizer, initial_point, collect_multiple_cuts=False,
                   original_dim=None, opt_state=None):
    """
    Perform a PSD check using VQE to estimate the minimum eigenvalue of the matrix A. If the estimated minimum eigenvalue
    is negative (up to a given tolerance), the function collects the corresponding eigenvector as a violating vector for
    use in generating a cut for the master problem. The function also computes the exact minimum eigenvalue using a classical
    solver for reference and analysis purposes, but this is not used in the separation step itself. The VQE optimization
    is warm-started with the provided initial parameters to speed up convergence. The function returns whether A is PSD,
    the list of collected violating vectors (already ranked most-violated-first by VQESubroutine.solve -- Change #5),
    the time taken to run the VQE, the optimal parameters found by VQE, the leaked amplitude (fraction of probability
    mass in the zero-padded junk subspace, given original_dim), the number of cost-function evaluations (nfev) the
    optimizer used, and the optimizer's internal state to reuse next iteration (Change #4).
    """
        
    # Compute minimum eigenvalue using exact classical solver for reference and store it for analysis
    # Note: This is not used in the separation step; it's only for analysis and comparison with VQE results.
    time_start = time.time()
    exact_eigenvalue = np.linalg.eigvalsh(A[:original_dim, :original_dim]).min() 
    exact_eigenvalue_time = time.time() - time_start
    print(f"Exact eigenvalue computed in {exact_eigenvalue_time:.4f} seconds.")
    exact_min_eigenvalues[block_id].append(exact_eigenvalue)
    print(f"Exact minimum eigenvalue (classical solver): {exact_eigenvalue:.8f}")

    # Initialize a boolean flag to track whether the input matrix is PSD
    is_psd = True

    try:
        # Run VQE to estimate the minimum eigenvalue and optimal parameters
        trajectory, optimal_parameters, collected_state_vectors, vqe_solve_time, leaked_amplitude, num_pauli_terms = vqe_instance.solve(A, optimizer, initial_point, energy_threshold=-TOL, overlap_threshold=0.9, global_pool=violating_vectors[block_id], collect_multiple_vectors=collect_multiple_cuts, original_dim=original_dim, opt_state=opt_state)
        print(f"VQE optimization completed in {vqe_solve_time:.4f} seconds, with {len(collected_state_vectors)} violating vectors.")
        print(f"Minimum eigenvalue estimate from VQE: {trajectory['energy'][-1]:.8f}")
        print(f"Leaked amplitude (padding subspace): {leaked_amplitude:.8f}")
        print(f"Number of Pauli terms: {num_pauli_terms}")
        nfev = trajectory.get("nfev")
        new_opt_state = trajectory.get("opt_state")

        # Record the VQE estimate, the leaked amplitude, and update the PSD flag if any violating vectors were collected
        vqe_min_eigenvalue_estimates[block_id].append(trajectory['energy'][-1])
        leaked_amplitudes[block_id].append(leaked_amplitude)
        if collected_state_vectors: # If any violating vectors were collected, mark the matrix as not PSD
            is_psd = False

        return is_psd, collected_state_vectors, vqe_solve_time, optimal_parameters, leaked_amplitude, nfev, num_pauli_terms, new_opt_state
    except Exception as e:
        raise RuntimeError(f"VQE failed during PSD check: {e}") from e

def _as_mosek_matrix(A: np.ndarray):
    """
    Wrap a numpy array as a MOSEK Fusion matrix, using a sparse triplet representation
    when most entries are zero. SDP-LIB constraint matrices are parsed from a sparse
    triplet file format (read_instance) into dense numpy buffers, but each constraint
    typically only touches a handful of entries; passing them to Fusion as msk.Matrix.dense
    pays O(n^2) memory per constraint (times m constraints) for data that is almost always
    mostly zero.
    """
    rows, cols = np.nonzero(A)
    if rows.size < 0.3 * A.size:
        return msk.Matrix.sparse(A.shape[0], A.shape[1], rows.astype(np.int32), cols.astype(np.int32), A[rows, cols])
    return msk.Matrix.dense(A)


def add_objective_constraints_with_MOSEK(model, X, nblocks, m, A_list: list, b_list: list):
    # Objective
    model.objective(msk.ObjectiveSense.Maximize, msk.Expr.add([msk.Expr.dot(_as_mosek_matrix(A_list[0][b]), X[b]) for b in range(nblocks)]))

    # Each constraint is a sum of inner products
    for k in range(1,m+1):
        model.constraint(msk.Expr.add([msk.Expr.dot(_as_mosek_matrix(A_list[k][b]), X[b]) for b in range(nblocks)]), msk.Domain.equalsTo(b_list[k]))

def solve_SDP_with_MOSEK(block_list: list, A_list: list, b_list: list): 
    """Solve a block-diagonal SDP with MOSEK.

    Maximize sum_b trace(A_list[0][b] @ X_b)
    subject to sum_b trace(A_list[k][b] @ X_b) == b_list[k]  for k=1..m
              X_b is PSD for each block b.
    Parameters
    ----------
    block_list : list[int]
        List of block sizes (one integer per block).
    A_list : list[list[numpy.ndarray]]
        A_list is a list of length m+1 (where m is the number of constraints). Each element A_list[k] is itself a list of numpy arrays,
        one array per block listed in block_list. Each block array is a dense square numpy.ndarray of shape (block_size, block_size).
    b_list : list[float]
        List of objective function coefficients of length m+1. The list is prefixed with a 0 at index 0 (i.e., b_list[0] == 0) and
        b_list[1..m] are the floats parsed from the file.
    Returns
    -------
        X_values: list of optimal block matrices X_b.
        optimal_value: optimal objective value.
    """
    #
    # \max { A_0 \cdot X + b_0 : A_k \cdot X = b_k, k=1,\dots,m, X \succeq 0}
    #

    print("*** Solving with MOSEK ***")
    logger.info("*** Solving with MOSEK ***")

    m = len(A_list) - 1 # number of constraints
    nblocks = len(block_list)
    
    with msk.Model("one-shot SDP") as model:    
        # Setting up the variables
        X = [model.variable(msk.Domain.inPSDCone(block_list[b])) for b in range(nblocks)]
            
        add_objective_constraints_with_MOSEK(model, X, nblocks, m, A_list, b_list)
           
        # Solve
        # model.setLogHandler(sys.stdout)            # Add logging
        # model.writeTask("sdp.ptf")                # Save problem in readable format
        start_time = time.time()
        model.solve()
        end_time = time.time()
    
        # end of solution process, print summary and update logger
        print("*** SDP solve with MOSEK complete ***") # what to do when the status is UNKOWNN!!!
        print("Time= " + str(end_time-start_time))
        print("Optimal Value= " + str(model.primalObjValue()))
        # print("Optimal Solution:", [X[b].value for b in range(nblocks)])
        print("***  ***")
        logger.info("*** SDP solve with MOSEK complete ***")
        logger.info("Time= %f", end_time - start_time)
        logger.info("Optimal Solution: %s", [X[b].level() for b in range(nblocks)])
        
        # # Retrieve result
        # print("X1:\n{0}".format(np.reshape(X1.level(), (3,3))))
        # print("X2:\n{0}".format(np.reshape(X2.level(), (4,4))))

        return X, model.primalObjValue(), end_time - start_time

def add_valid_inequalities(model, X, n: int, valid: str, solver: str):
    """
    Add valid inequalities to a model for a symmetric matrix variable X.
    This function augments the given optimization model by adding
    constraints that are valid for positive semidefinite (PSD) matrices. 
    It always enforces nonnegativity of diagonal entries and can add either
    a set of linear inequalities or pairwise second-order-cone (SOC)-type 
    inequalities depending on the choice of the `valid` option.
    Parameters
    ----------
    model : gurobipy.Model or mosek.Model
        Optimization model to which constraints will be added. 
    X : gurobipy.VarDict or mosek.Variable
        Gurobi / MOSEK variable object representing a symmetric (block) matrix variable.
    n : int
        Dimension of the matrix X (number of rows/columns). 
    valid : str
        Choice of additional valid inequalities to add (in addition to the
        nonnegative diagonal). Options:
          - "linear" : Add a family of linear inequalities of the form
            X[i,i] + 2*a*X[i,j] + a*a*X[j,j] >= 0 for each pair (i, j), with
            a taken from the set {1, -1, 1+sqrt(2), 1-sqrt(2), -1+sqrt(2), -1-sqrt(2)}.
            These linear cuts are derived from nonnegativity of quadratic
            forms (e.g., (e_i + a e_j)^T X (e_i + a e_j) >= 0) and are
            used to strengthen relaxations of PSD constraints.
          - "soc"    : Add pairwise SOC-type inequalities
            X[i,i]*X[j,j] >= X[i,j]^2 for each pair (i, j). These are
            implied by positive semidefiniteness and are commonly used as
            2x2 principal-minor (determinant) inequalities.
    solver: Gurobi or MOSEK
    Raises
    ------
    ValueError
        If `valid` is not one of the accepted options ("linear" or "soc").
    """
    
    # nonnegative diagonal
    if solver == "Gurobi":
        model.addConstrs(X[i,i] >= 0 for i in range(n))
    if solver == "MOSEK" and valid != "soc":
        model.constraint(X.diag(), msk.Domain.greaterThan(0.0))
    
    s2 = np.sqrt(2)
    if valid == "linear": 
        coeff = [1, -1, 1+s2, 1-s2, -1+s2, -1-s2] # https://doi.org/10.1007/s10589-020-00255-2
        # this can further be improved by adding more linear inequalities: https://doi.org/10.1016/j.disopt.2021.100643, https://doi.org/10.1287/moor.26.2.193.10561
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
    """
    Create the master problem (relaxation for the SDP) for a cutting-plane solver.
    The function builds and returns a Gurobi Model and a list of block variable objects X each
    representing a block-diagonal symmetric matrix. The model objective is to maximize the linear
    form given by A_list[0] applied to X, subject to a set of linear equality constraints 
    (A_list[1..m] · X == b_list[1..m]) and a collection of block-wise valid inequalities
    (as produced by the helper add_valid_inequalities). Symmetry constraints are enforced for
    each block.
    Parameters
    ----------
    block_list : list[int]
        List of block sizes.
    A_list : list[list[numpy.ndarray]]
        List/sequence of coefficient collections for the objective and constraints.
    b_list : list[float]
        Right-hand side values for the linear equality constraints. 
    valid : str, optional
        The type of valid inequalities to be added. Default is "linear". Options:
          - "linear" : Add a family of linear inequalities derived from nonnegativity
            of quadratic forms.
          - "soc"    : Add pairwise second-order-cone (SOC)-type inequalities
    Returns
    -------
    model : gurobipy.Model
        A Gurobi model object with:
        - Objective: maximize A_list[0] · X
        - Constraints: A_list[k] · X == b_list[k] for k=1..m
        - Valid inequalities added block-wise as per the `valid` option.
    X : list
        A list of Gurobi VarDict objects (one per block) returned in the same order as block_list.
    Raises
    ------
    ValueError
        If `block_list`, `A_list`, and `b_list` are not consistent in shape.
    """
    #
    # \max { A_0 \cdot X : A_k \cdot X = b_k, k=1,\dots,m + valid ineq } 
    #
    m = len(A_list) - 1 # number of constraints
    nblocks = len(block_list) # number of blocks

    # Input validation
    if any(len(A_list[k]) != nblocks for k in range(0, m + 1)):
        raise ValueError("Each A_list[k] must have the same number of blocks as block_list.")
    if len(b_list) != m + 1:
        raise ValueError("Length of b_list must be m + 1, where m is the number of constraints.")
        
    if solver == "Gurobi":
        model = Model()
        model.setParam("OutputFlag", False)
        model.setParam("DualReductions", 0) # by setting this 0, Gurobi does not give status INF_OR_UNBD (status=4)
        model.setParam("InfUnbdInfo", 1) # by setting this 1, Gurobi provides the unbounded ray when the problem is unbounded
        # model.setParam("Method", 3) # by setting this 3, Gurobi uses concurrent solve - only for LPs
        model.setParam("ConcurrentMethod", 3) # by setting this 3, Gurobi concurrently solve dual and primal simplex - only for LPs
        # when the master is solved by barrier and it is unbounded, can we still find the unbounded ray? No, we cannot!!!
    
        X = []
        for b in range(nblocks):
            block_size = block_list[b]
            X_block = model.addVars(block_size, block_size, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"X_{b}")
            # print(f"Block {b} size = {block_size}")
            # print(X_block)
            # print("***********************")
            X.append(X_block)
        
        model.addConstrs(X[b][i,j] == X[b][j,i] for b in range(nblocks) for i in range(block_list[b]) for j in range(i)) # symmetric
        model.addConstrs(quicksum(A_list[k][b][i,j]*X[b][i,j] for b in range(nblocks) for i in range(block_list[b]) for j in range(block_list[b])) == b_list[k] for k in range(1,m+1))
        
        model.setObjective(quicksum(A_list[0][b][i,j]*X[b][i,j] for b in range(nblocks) for i in range(block_list[b]) for j in range(block_list[b])), GRB.MAXIMIZE)
            
    if solver == "MOSEK":
        model = msk.Model("master LP")  
        # Setting up the variables
        X = [model.variable([block_list[b],block_list[b]], msk.Domain.unbounded()) for b in range(nblocks)]
            
        add_objective_constraints_with_MOSEK(model, X, nblocks, m, A_list, b_list)

        # Loosened conic tolerances (100x default 1e-8) to improve robustness on ill-conditioned instances
        model.setSolverParam("intpntCoTolPfeas", 1e-6)
        model.setSolverParam("intpntCoTolDfeas", 1e-6)
        model.setSolverParam("intpntCoTolMuRed", 1e-6)

        # symmetric
        # for b in range(nblocks):
        #     model.constraint(msk.Expr.sub(X[b], X[b].T), msk.Domain.equalsTo(0.0))
        for b in range(nblocks):
            for i in range(block_list[b]):
                for j in range(i):
                    model.constraint(msk.Expr.sub(X[b].index(i, j), X[b].index(j, i)), msk.Domain.equalsTo(0.0)) 
                    # model.constraint(X[b].index(i, j) == X[b].index(j, i)) 
    
    for b in range(nblocks):
        add_valid_inequalities(model, X[b], block_list[b], valid, solver)

    return model, X
    
def solve_master_problem(model, X: list, block_list: list, solver: str):
    """
    Solve the given master problem model using Gurobi or MOSEK.
    Parameters
    ----------
    model : gurobipy.Model or mosek.Model
        The Gurobi model representing the master problem to be solved.
    X : list
        List of Gurobi or MOSEK VarDict objects representing the block variables in the model.
    solver : Gurobi or MOSEK
    Returns
    -------
    status : int
        The status code of the optimization result (model.Status for Gurobi).
    X_values : list of numpy.ndarray
        List of optimal block matrices X_b if the problem is bounded; otherwise, None.
    objVal : float
        The optimal objective value if the problem is bounded; otherwise, None.
    master_solve_time : float
        Time taken to solve the master problem in seconds.
    """
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

    if solver == "MOSEK": ############################
        start_time = time.time()
        model.solve()
        end_time = time.time()
        master_solve_time = end_time - start_time
        
        # status = model.ProblemStatus()
        
        # we may have to check more solutionstatus or problemstatus cases!!!!!
        # in particular, what if primal is infeasible!!!!!!
        if model.getPrimalSolutionStatus() == msk.SolutionStatus.Optimal:
            X_values = get_solution_MOSEK(model, X, block_list)
            objVal = model.primalObjValue()
            return "Optimal", X_values, objVal, master_solve_time
        elif model.getPrimalSolutionStatus() == msk.SolutionStatus.Certificate:
            X_values = get_solution_MOSEK(model, X, block_list)
            return "Unbounded", X_values, None, master_solve_time
        else:
            # Neither an optimal solution nor an unboundedness certificate: surface the actual
            # primal/dual/problem status so the caller's "no solution" message is diagnosable
            # (previously this returned a bare " ", which rendered as an empty status).
            status = (
                f"primal={model.getPrimalSolutionStatus()}, "
                f"dual={model.getDualSolutionStatus()}, "
                f"problem={model.getProblemStatus()}"
            )
            return status, None, None, master_solve_time
        
def get_solution_Gurobi(model: Model, X: list, block_list: list, attr: str):
    """
    Extract the solution/the unbounded rays from the Gurobi model and the variable dictionary.
    Parameters
    ----------
    model : gurobipy.Model
        The Gurobi model from which to extract the solution.
    X : list
        List of Gurobi VarDict objects representing the block variables in the model.
    attr : str
        The attribute to extract from the variables (e.g., "X" for solution values,
        "UnbdRay" for unbounded rays).
    Returns
    -------
    X_values : list of numpy.ndarray
        List of block matrices X_b obtained from the specified attribute of the variables.
    """
    nblocks = len(X)
    X_values = []
    for b in range(nblocks):
        block_size = block_list[b]
        
        # build varlist in row-major order and fetch attribute values in one call
        var_list = [X[b][i, j] for i in range(block_size) for j in range(block_size)]
        vals = model.getAttr(attr, var_list)  # returned in same order as var_list
        X_block = np.array(vals, dtype=float).reshape((block_size, block_size))
        X_values.append(X_block)

    return X_values

def get_solution_MOSEK(model: msk.Model, X: list, block_list: list):
    """
    Extract the solution/the unbounded rays from the MOSEK model and the variable dictionary.
    Parameters
    ----------
    model : mosek.Model
        The MOSEK model from which to extract the solution.
    X : list
        List of MOSEK VarDict objects representing the block variables in the model.
    Returns
    -------
    X_values : list of numpy.ndarray
        List of block matrices X_b obtained from the specified attribute of the variables.
    """
    model.acceptedSolutionStatus(msk.AccSolutionStatus.Anything)
    
    nblocks = len(X)
    X_values = [X[b].level().reshape(block_list[b], block_list[b]) for b in range(nblocks)]
    return X_values

def add_linear_cut(model, X_block, u_vector: np.ndarray, block_size: int, solver: str):
    """
    Add a linear cut for the given block variable using the provided direction vector.
    Parameters
    ----------
    model: gurobipy or MOSEK Model object.
    X_block: gurobipy VarDict for the block (indexed by (i,j)).
    u_vector: 1-D array-like vector (length block_size).
    block_size: int, size of the block.
    solver: Gurobi or MOSEK
    Returns
    -------
    con : the added constraint handle (gurobipy.Constr or mosek.fusion.Constraint), so
        callers can track it (e.g. for later dual-based purging of inactive cuts).
    """
    con = None
    if solver == "Gurobi":
        con = model.addConstr(quicksum(X_block[i,j]*u_vector[i]*u_vector[j] for i in range(block_size) for j in range(block_size)) >= 0)

    if solver == "MOSEK":
        # MOSEK's Fusion Expr.mul only has an array(double,ndim=1) overload -- no float32/complex
        # variant exists, and it raises a ValueError rather than silently upcasting. u_vector can
        # arrive as float32 (cudaq's GPU target uses single-precision statevectors by default;
        # see VQESubroutine._bind_circuit_and_extract_state) or, in principle, from any other
        # future separation-oracle source, so this cast happens here too, not just at the source.
        trimmed_u_vector = u_vector[:block_size].astype(np.float64)  # Ensure the vector is of the correct length and dtype
        con = model.constraint(msk.Expr.dot(trimmed_u_vector, msk.Expr.mul(X_block, trimmed_u_vector)), msk.Domain.greaterThan(0.0))
    return con

def add_soc_cut(model, X_block, u_vectors: list, block_size: int, solver: str):
    """
    Add a 2x2 SOC / determinant cut using two direction vectors.
    Parameters
    ----------
    model: gurobipy Model object.
    X_block: gurobipy VarDict for the block (indexed by (i,j)).
    u_vectors: list of two 1-D array-like vectors (each of length block_size).
    block_size: int, size of the block.
    solver: Gurobi or MOSEK
    """
    
    # Build n x 2 matrix with the two vectors as columns, trimming any VQE padding
    # (u_vectors may be padded to the next power of two) down to block_size. Cast to float64
    # regardless of the input vectors' dtype -- see add_linear_cut for why (MOSEK's Fusion
    # Expr.mul below has no float32 overload).
    U = np.column_stack([np.asarray(u).ravel()[:block_size].astype(np.float64) for u in u_vectors[:2]])  # shape (block_size, 2)
    
    if solver == "Gurobi":
        # Create Y variables (2x2)
        Y = model.addVars(2, 2, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        model.addConstr(Y[0, 1] == Y[1, 0])
        model.addConstrs(Y[l, l] >= 0 for l in range(2))
        model.addConstr(Y[0, 0] * Y[1, 1] >= Y[0, 1] * Y[0, 1])
    
        # y_kl = u_k^T X u_l  for k,l in {0,1}  (only k<=l added; symmetry enforced)
        model.addConstrs(
            (Y[k, l] == quicksum(U[i, k] * X_block[i, j] * U[j, l]
                                 for i in range(block_size) for j in range(block_size)))
            for k in range(2) for l in range(k + 1)
        )
    s2 = np.sqrt(2)
    if solver == "MOSEK":
        Y = model.variable([2, 2], msk.Domain.unbounded())
        model.constraint(msk.Expr.sub(Y.index(0, 1), Y.index(1, 0)), msk.Domain.equalsTo(0.0))
        model.constraint(msk.Expr.vstack(Y.index(0, 0), Y.index(1, 1), msk.Expr.mul(s2, Y.index(0, 1))), msk.Domain.inRotatedQCone())
        for k in range(2):
            for l in range(k + 1):  # Only map lower/upper triangle due to symmetry
                # Extract columns k and l from matrix U
                u_k = U[:, k]
                u_l = U[:, l]
                
                # Add constraint: Y[k, l] == rhs_expr, i.e. u_k^T X u_l, computed via
                # Expr.mul/Expr.dot rather than Expr.dot(np.outer(u_k, u_l), X_block) --
                # see add_linear_cut for why the outer-product form is avoided.
                model.constraint(msk.Expr.sub(Y.index(k, l), msk.Expr.dot(u_k, msk.Expr.mul(X_block, u_l))), msk.Domain.equalsTo(0.0))

def purge_inactive_linear_cuts(cut_registry: list, solver: str, patience: int, dual_tol: float, itr: int, logger=None):
    """
    Drop linear cuts that have been non-binding (dual ~ 0) for `patience` consecutive
    checks, so the master problem doesn't accumulate an unbounded number of cuts over
    a long cutting-plane run. SOC cuts are left alone: unlike a linear cut, an SOC cut's
    auxiliary Y variable and its four linking constraints all have to be removed together
    or the remaining pieces reference a deleted variable, so partial removal isn't safe.
    MOSEK-only: Gurobi's Pi/QCPi duals require re-solving in a mode this module doesn't
    otherwise use, so Gurobi runs simply keep every cut (unchanged prior behavior).
    Parameters
    ----------
    cut_registry : list[list[dict]]
        Per-block list of {"con": constraint handle, "inactive_streak": int}, mutated in place.
    solver : str
        "Gurobi" or "MOSEK"; purging only runs for "MOSEK".
    patience : int or None
        Consecutive non-binding checks required before a cut is removed. None disables purging.
    dual_tol : float
        A cut's dual magnitude below this is treated as "not binding this iteration".
    itr : int
        Current iteration number, for logging only.
    Returns
    -------
    num_removed : int
        Number of cuts removed this call (0 if purging is disabled or nothing qualified).
    """
    if solver != "MOSEK" or patience is None:
        return 0
    num_removed = 0
    for b, entries in enumerate(cut_registry):
        kept = []
        for entry in entries:
            try:
                dual_val = float(np.max(np.abs(np.asarray(entry["con"].dual()))))
            except Exception:
                # No dual available this round (e.g. the master returned an unboundedness
                # certificate rather than a primal-dual optimal solution) -- keep the cut and
                # re-check next time rather than risk dropping one we can't actually evaluate.
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

def solve_SDP_by_cutting_plane(block_list: list, A_list: list, b_list: list, solver: str, valid = "linear", add_soc_cuts=False, optimizer=None, collect_multiple_cuts=False, itr_limit = 10, ansatz_type='hardware_efficient', execution_mode='noiseless', backend_name=None, reps=2, log_filepath=None, log_context=None, cut_purge_patience=5, cut_dual_tol=1e-8, ansatz_mode='fixed', max_reps=6, adapt_grad_tol=1e-3, adapt_pool_size=30, enable_pruning=False, pruning_threshold=0.05, parallel_separation=False, adaptive_stop=True, max_cuts_per_block=None):
    """
    Solve a block-structured semidefinite program (SDP) using a cutting-plane master/separation approach.
    This routine constructs an initial (relaxed) master problem for a block-diagonal SDP, then iteratively:
        1. Solves the master problem,
        2. Checks each block of the returned matrix variables for positive semidefiniteness (PSD),
        3. If any block is not PSD, generates separating cuts (linear or SOC) from violating eigenvectors and
             adds them to the master problem,
        4. Repeats until all blocks are PSD or the iteration limit is reached.
    The algorithm is intended to be used with the following helper functions available in the same module or scope:
        - create_master_problem(block_list, A_list, b_list, valid)
                -> returns (model, X) where model is a solver model and X is a list/structure of block matrix variables.
        - solve_master_problem(model, X)
                -> returns (status, X_values, objVal, master_solve_time).
        - psd_check(X_value, block_id, estimator, ansatz, optimizer, initial_point, collect_multiple_cuts)
                -> returns (is_psd, u_vectors, vqe_solve_time, optimal_parameters).
        - add_linear_cut(model, X_block_var, u, block_spec)
                -> adds a linear (hyperplane) cut corresponding to vector u for the given block variable.
        - add_soc_cut(model, X_block_var, [u1, u2], block_spec)
                -> adds a second-order-cone type cut constructed from two violating vectors.
    Parameters
    ----------
    block_list : sequence
            Description of the block structure of the SDP variable X. Each element should describe one block
            (for example an integer block size or a block specification object expected by create_master_problem).
    A_list : sequence
            List (or other sequence) of linear constraint matrices/operators that define the equality constraints
            A_k · X = b_k for the SDP. Structure must match expectations of create_master_problem.
    b_list : sequence
            Right-hand-side values corresponding to A_list, i.e. b_k scalars (or appropriate objects) for the constraints.
    solver : Gurobi or MOSEK
            The solver to be used for the master problem. Must be one of the supported solvers (Gurobi or MOSEK).
    valid : str, optional (default="linear")
            Type of valid inequalities used to initialize the master problem. Passed through to create_master_problem;
            accepted values depend on that function (commonly "linear" or other relaxations).
    add_soc_cuts : bool, optional (default=False)
            If True, second-order-cone (SOC)-type cuts will be added if possible in addition to linear cuts when a block 
            is found to be non-PSD.
    optimizer : callable, optional (default=COBYLA(maxiter=50))
            The optimizer to be used for VQE-based separation.
    collect_multiple_cuts : bool, optional (default=False)
            If True, all identified violating vectors will be collected for each non-PSD block.
    itr_limit : int, optional (default=10)
            Maximum number of cutting-plane iterations to perform.
    ansatz_type : str, optional (default='hardware_efficient')
            The type of ansatz to be used in the VQE subroutine for separation. Passed to the VQESubroutine constructor.
    execution_mode : str, optional (default='noiseless')
            The execution mode for the VQE subroutine (e.g., 'noiseless', 'noisy', 'simulator'). Passed to the VQESubroutine constructor.
    backend_name : str, optional (default=None)
            The CUDA-Q hardware target name for the VQE subroutine (e.g., 'quantinuum', 'ionq', 'iqm'), only
            used when execution_mode='hardware' (CUDA-Q selects the simulator target automatically for
            'noiseless'/'noisy'). Passed to the VQESubroutine constructor.
    reps : int, optional (default=2)
            The number of repetitions (layers) in the ansatz for the VQE subroutine.
    log_filepath : str, optional (default=None)
            If provided, one row per block per iteration is appended (crash-safe, flushed+fsynced) to this CSV
            file as the run progresses, using log_context for the run-level fields (see ITERATION_LOG_FIELDNAMES).
    log_context : dict, optional (default=None)
            Run-level fields (instance, config_hash, repeat, random_seed, etc.) merged into every logged row.
    cut_purge_patience : int or None, optional (default=5)
            Number of consecutive iterations a linear cut must be non-binding (dual ~0) before it is
            removed from the master problem. None disables purging. MOSEK-only; see
            purge_inactive_linear_cuts for why SOC cuts aren't purged and why Gurobi is unaffected.
    cut_dual_tol : float, optional (default=1e-8)
            Dual magnitude below which a linear cut is considered non-binding for purge purposes.
    ansatz_mode : str, optional (default='fixed')
            'fixed' (original fixed-depth ansatz), 'adaptive_depth' (grows depth up to max_reps
            only as far as needed), or 'adapt_pool' (ADAPT-VQE-lite, grown from the Hamiltonian's
            own Pauli terms). See VQESubroutine's docstring for details of each.
    max_reps : int, optional (default=6)
            Depth/operator-count ceiling for 'adaptive_depth' and 'adapt_pool'.
    adapt_grad_tol : float, optional (default=1e-3)
            Gradient-magnitude stopping threshold for 'adapt_pool' growth.
    adapt_pool_size : int, optional (default=30)
            Cap on candidate pool operators considered per 'adapt_pool' growth step.
    enable_pruning : bool, optional (default=False)
            If True, a 'fixed'-mode ansatz whose final rotation layer converges to ~0 has that
            layer dropped from the cached ansatz for that block shape on the next iteration.
    pruning_threshold : float, optional (default=0.05)
            Angular tolerance (radians) for treating a rotation as "prunable".
    parallel_separation : bool, optional (default=False)
            If True, each iteration's independent per-block PSD/separation calls are dispatched
            via a thread pool instead of strictly one after another. Defaults to False: real runs
            showed that concurrent threads sharing cudaq's single global GPU context eventually
            produces "No QPUs are available for this target" errors and, under sustained load, a
            segfault. VQESubroutine now serializes every actual GPU call behind an internal lock
            (self._gpu_lock) specifically so this option can no longer crash if you do enable it --
            but with that lock in place, enabling it buys little to no real concurrency for the
            GPU work itself (only the surrounding Python bookkeeping between blocks can overlap).
            Genuine concurrent GPU dispatch across blocks would need cudaq's 'nvidia-mqpu' target
            with each thread pinned to its own QPU id, which hasn't been implemented or tested here.
    adaptive_stop : bool, optional (default=True)
            If True, the VQE optimizer stops as soon as it clears energy_threshold rather than
            always running to its full maxiter. Set False for full, comparable-length runs (e.g.
            when benchmarking optimizers against each other).
    max_cuts_per_block : int or None, optional (default=None)
            If set, caps how many of a block's collected (already most-violated-first-ranked)
            cuts are actually added to the master problem in a given iteration.
    """
    #
    # \max { A_0 \cdot X : A_k \cdot X = b_k, k=1,\dots,m + valid ineq + cutting planes } 
    #

    print("*** Starting cutting plane ***")
    logger.info("*** Starting cutting plane ***")
    
    ####################################################################
    # create the initial master problem
    model, X = create_master_problem(block_list, A_list, b_list, solver, valid)
    nblocks = len(block_list) # number of blocks in the problem   
    ####################################################################
    # cutting-plane loop variables
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
    # per-block registry of {"con": constraint handle, "inactive_streak": int} for linear cuts,
    # used by purge_inactive_linear_cuts to drop cuts that stop being binding (see its docstring)
    cut_registry = [[] for _ in range(nblocks)]
    ####################################################################
    # VQE setup for separation
    optimizer = optimizer if optimizer is not None else COBYLA(maxiter=100) # default optimizer for VQE, can be overridden by user input; COBYLA is a good general-purpose choice for small to medium-sized problems, but users may want to experiment with others (e.g., SLSQP, SPSA) depending on the problem structure and size. Note that the choice of optimizer can significantly impact the convergence and solve times of the VQE-based separation step, so it may be beneficial to allow for flexibility here.
    last_optimal_points = [None] * nblocks # we keep track of the last optimal parameters (angles) found for each block to warm-start the VQE in the next iteration, which can speed up convergence and reduce solve times.
    last_optimizer_states = [None] * nblocks # Change #4: same idea, but for optimizer-internal state (e.g. QNSPSA's running metric-tensor estimate) rather than the ansatz parameters themselves.
    vqe_instance = VQESubroutine(ansatz_type=ansatz_type, execution_mode=execution_mode, backend_name=backend_name,
                                  reps=reps, ansatz_mode=ansatz_mode, max_reps=max_reps, adapt_grad_tol=adapt_grad_tol,
                                  adapt_pool_size=adapt_pool_size, enable_pruning=enable_pruning,
                                  pruning_threshold=pruning_threshold) # create an instance of the VQESubroutine class to be used for the PSD checks in the separation step. One instance is shared across all blocks and iterations, which is what makes the ansatz/parameter/optimizer-state caching (Change #3/#4) and its Change #5 thread-safety lock meaningful.
    ####################################################################
    # Dicts to store histories of VQE estimates and exact eigenvalues for analysis and plotting after runs
    for b in range(nblocks):
        vqe_min_eigenvalue_estimates[b] = [] # initialize empty list for each block to store the history of minimum eigenvalue estimates from VQE across VQE-based cutting plane iterations
        exact_min_eigenvalues[b] = [] # initialize empty list for each block to store the history of exact minimum eigenvalues computed by classical solvers across VQE-based cutting plane iterations (for comparison/analysis with VQE estimates)
        violating_vectors[b] = [] # initialize empty list for each block to store the history of all violating vectors collected across iterations for that block (used for diversity checks to avoid adding cuts from similar vectors)
        leaked_amplitudes[b] = [] # initialize empty list for each block to store the history of leaked amplitudes (padding-subspace probability mass) across iterations
    ####################################################################
    # main cutting-plane loop
    try:
        while itr < itr_limit and not is_optimal:
            itr += 1
            logger.info("Iteration %d", itr)
            ####################################################################
            # solve master problem
            rss_before_master = current_rss_mb()
            status, X_values, objVal, master_solve_time = solve_master_problem(model, X, block_list, solver)
            rss_after_master = current_rss_mb()
            master_total_time += master_solve_time
            if objVal is not None:
                master_objective_values.append(objVal) # record this iteration's master objective for the objective-trajectory plots/analysis
            logger.info("Time(master)= %f", master_solve_time)
            logger.info("ObjVal %s", str(objVal))
            logger.info("Peak RSS(MB)= %.1f", peak_rss_mb())
            # model.write("lp "+sepMethod + " " + str(itr) + ".lp")
            ####################################################################
            # separation step
            #
            # MOSEK - incorporate infesible or unknown status etc. !!!!!!!!!!!!
            #
            if solver == "Gurobi" and status == GRB.INFEASIBLE:
                termination_reason = "The original problem is proven to be infeasible since the master problem is infeasible."
                break
            elif solver == "Gurobi" and status == GRB.NUMERIC:
                termination_reason = "Numeric difficulties encountered!" # Should we do something in this case?
                break
            elif X_values is None:
                # MOSEK returned neither an optimal solution nor a certificate (e.g. infeasible/unknown status)
                termination_reason = f"Master problem returned no solution (solver={solver}, status={status})."
                break
            else:
                # drop linear cuts that have gone non-binding for `cut_purge_patience` consecutive
                # iterations, using duals from the solve just above, before adding this iteration's cuts
                linear_cuts_purged += purge_inactive_linear_cuts(cut_registry, solver, cut_purge_patience, cut_dual_tol, itr, logger)
                # check positive-semidefiniteness of each block matrix
                #
                # Change #5 ("parallel separation of independent PSD blocks"): each block's VQE
                # separation call is independent of every other block's -- they read different
                # slices of X_values and don't touch the master model -- so when
                # parallel_separation=True, all nblocks calls are dispatched via a thread pool
                # instead of running strictly one after another. IMPORTANT CORRECTION: an earlier
                # version of this comment claimed cudaq's calls release the GIL cleanly enough for
                # this to safely overlap GPU work across threads -- that was untested and turned
                # out to be wrong; real runs hit "No QPUs are available for this target" and then a
                # segfault under concurrent load. VQESubroutine now serializes every actual GPU
                # call behind its own internal lock, so this no longer crashes, but also no longer
                # gives real GPU-level concurrency -- which is why parallel_separation now defaults
                # to False. The shared vqe_instance's caches (ansatz_cache, param_memory,
                # opt_state_memory) are still protected by their own separate lock regardless.
                # Model mutation (add_linear_cut/add_soc_cut) always happens afterward, in block
                # order, on this (the only) thread -- Mosek/Gurobi model objects are not meant to
                # be touched concurrently, independent of the GPU-locking question above.
                num_violating_vectors = 0
                rss_before_sep = current_rss_mb()

                def _separate_block(b):
                    return psd_check(X_values[b], b, vqe_instance, optimizer, last_optimal_points[b],
                                      collect_multiple_cuts, opt_state=last_optimizer_states[b])

                if parallel_separation and nblocks > 1:
                    with ThreadPoolExecutor(max_workers=nblocks) as pool:
                        block_results = list(pool.map(_separate_block, range(nblocks)))
                else:
                    block_results = [_separate_block(b) for b in range(nblocks)]

                rss_after_sep = current_rss_mb()

                for b in range(nblocks):
                    (is_psd, u_vectors, sep_time, last_optimal_points[b], leaked_amplitude,
                     optimizer_nfev, num_pauli_terms, last_optimizer_states[b]) = block_results[b]
                    num_violating_vectors += len(u_vectors)
                    sep_total_time += sep_time
                    logger.info("Time(sep)= %f", sep_time)
                    logger.info("Block %d is PSD?: %s", b, str(is_psd))
                    # add cutting planes if block is not PSD
                    cuts_this_block = []
                    if not is_psd:
                        logger.info("Adding cutting planes for block %d", b)
                        # add linear (and SOC) cuts based on the violating vectors found.
                        # u_vectors is already ranked most-violated-first by VQESubroutine.solve
                        # (Change #5's "ranking cuts according to violation magnitude"); optionally
                        # cap how many of them get added this iteration via max_cuts_per_block.
                        cuts_this_block = u_vectors if max_cuts_per_block is None else u_vectors[:max_cuts_per_block]
                        for u in cuts_this_block:
                            con = add_linear_cut(model, X[b], u, block_list[b], solver) # add linear cut
                            linear_cuts_added += 1
                            if solver == "MOSEK" and cut_purge_patience is not None:
                                cut_registry[b].append({"con": con, "inactive_streak": 0})
                        if add_soc_cuts:
                            if len(cuts_this_block) >= 2: # need at least two negative eigenvectors for SOC cut
                                add_soc_cut(model, X[b], cuts_this_block[:2], block_list[b], solver) # the two *most* violated vectors, now that u_vectors is ranked most-violated-first
                                soc_cuts_added += 1
                    else: # no cutting planes added
                        logger.info("No cutting planes added for block %d", b)

                    # incremental, crash-safe per-block/per-iteration logging
                    if log_filepath is not None:
                        n_original = block_list[b]
                        n_padded = next_power_of_two(n_original)
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
                            "num_cuts_added_this_iter": len(cuts_this_block),
                            "is_psd": is_psd,
                            "rss_before_sep": rss_before_sep,
                            "rss_after_sep": rss_after_sep,
                            "rss_before_master": rss_before_master,
                            "rss_after_master": rss_after_master,
                        })
                        log_row(log_filepath, row, ITERATION_LOG_FIELDNAMES)
                # check termination condition
                if num_violating_vectors == 0:
                    is_optimal = True
                    # Diagnostic only: cross-check the VQE-based PSD claim against the exact eigenvalues
                    # already computed by psd_check this iteration (see exact_min_eigenvalues). VQE is a
                    # heuristic oracle and can miss a real negative eigenvalue (false PSD certificate);
                    # this never changes is_optimal or the returned solution, only the reported reason.
                    false_psd_certificate = any(exact_min_eigenvalues[b] and exact_min_eigenvalues[b][-1] < -TOL for b in range(nblocks))
                    if false_psd_certificate:
                        termination_reason = "Terminating because all blocks are PSD (per VQE oracle). WARNING: false_psd_certificate=True -- exact EVD found a block with a negative eigenvalue that VQE missed."
                        logger.warning("False PSD certificate detected at termination! Exact min eigenvalues per block: %s", [exact_min_eigenvalues[b][-1] if exact_min_eigenvalues[b] else None for b in range(nblocks)])
                    else:
                        termination_reason = "Terminating because all blocks are PSD. Optimal solution found."
                elif itr == itr_limit:
                    termination_reason = "Terminating because iteration limit was reached."
    finally:
        # Release the master model's native (Gurobi/MOSEK) resources as soon as this run ends,
        # regardless of how it ended (normal termination, early break, or an exception propagating
        # out of the loop, e.g. from psd_check_vqe's RuntimeError on VQE failure) -- otherwise
        # repeated runs in the __main__ sweep leak one model's worth of native memory per run.
        # Model.dispose() is the same call a `with msk.Model(...)` block makes on __exit__ (used
        # for the one-shot MOSEK solve above); this model is created without a `with` block since
        # it needs to stay open across iterations.
        model.dispose()

    # end of cutting-plane loop, print summary and update logger
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
    """
    Generate plots for a given experiment configuration showing exact vs VQE eigenvalues
    per block and master objective value trajectory.
    """
    nblocks = len(avg_eigenvalue_trajectories["exact"]) # number of blocks is inferred from the length of the eigenvalue trajectory dicts
    
    # Figure 1: Block-wise eigenvalue trajectories
    fig1 = plt.figure(figsize=(15, 5 * nblocks))
    for block_id in range(nblocks):
        plt.subplot(nblocks, 1, block_id + 1)
        plt.plot(avg_eigenvalue_trajectories["exact"][block_id], label='Exact Minimum Eigenvalue (Classical Solver)', 
                marker='o', linewidth=2, markersize=4)
        plt.plot(avg_eigenvalue_trajectories["vqe"][block_id], label='Average VQE Minimum Eigenvalue Estimate', 
                marker='x', linewidth=2, markersize=5)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='PSD Boundary (Eigenvalue=0)')
        plt.xlabel('Iteration', fontsize=11)
        plt.ylabel('Minimum Eigenvalue', fontsize=11)
        plt.title(f'Block {block_id} - Exact vs VQE Eigenvalue Trajectory', fontsize=12, fontweight='bold')
        plt.legend(fontsize=9, loc='best')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{instance}_blockwise_eigenvalue_trajectories_config_{experiment_config}.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Figure 2: Master objective value trajectory
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
    """
    Generate a figure with separate subplots for each block, showing VQE performance
    across different configurations within each subplot.
    """
    # Validate input data
    if not all_vqe_min_eigenvalue_avg_estimates:
        print("No VQE trajectory data available for plotting")
        return
    
    # Determine the number of blocks from the first configuration
    first_config_trajectories = next(iter(all_vqe_min_eigenvalue_avg_estimates.values()))
    num_blocks = len(first_config_trajectories)
    
    # Create figure with subplots for each block
    fig = plt.figure(figsize=(15, 5 * num_blocks))
    
    # For each block, create a subplot showing all configurations
    for block_id in range(num_blocks):
        plt.subplot(num_blocks, 1, block_id + 1)
        
        # Plot trajectories for all configurations for this block
        for config, vqe_trajectories in all_vqe_min_eigenvalue_avg_estimates.items():
            solver, valid, add_soc_cuts, collect_multiple_cuts, ansatz_type, execution_mode, p, optimizer_name, optimizer_maxiter = config

            if block_id in vqe_trajectories:
                trajectory = vqe_trajectories[block_id]
                label = f'Solver: {solver}, Valid: {valid}, Add SOC Cuts: {add_soc_cuts}, Multi-Cuts: {collect_multiple_cuts}, Ansatz: {ansatz_type}, Layers: {p}, Mode: {execution_mode}, Opt: {optimizer_name}(maxiter={optimizer_maxiter})'
                plt.plot(trajectory, label=label, marker='o', linewidth=2, markersize=4)
        
        # Add PSD boundary line
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
    # Write results to Excel if possible, otherwise CSV
    if results:
        try:
            import pandas as pd

            df = pd.DataFrame(results)
            df.to_excel(filename, index=False)
            print(f"Wrote {filename}")
        except Exception:
            import csv

            csv_path = "results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
            print(f"Pandas not available or write failed; wrote {csv_path} instead")
    else:
        print("No results collected; nothing to write.")

if __name__ == "__main__":
    # Define the instances to run the experiments on
    instances = [f'control{i}' for i in range(1,6)]
    # Define experiment parameters
    solvers = ["MOSEK"] # solvers to use for the master problem (Gurobi or MOSEK)
    initial_valid_cut_type = ["soc"] # initial valid inequalities to add to the master problem (linear or soc)
    soc_cuts_added_options = [True] # whether to add SOC cuts in addition to linear cuts (True or False)
    collect_multiple_cuts_options = [True] # whether to collect multiple violating vectors per block (True or False)
    optimizers = [COBYLA(maxiter=200)] # optimizers for VQE (COBYLA(maxiter=200), SPSA(maxiter=200) etc.)
    ansatz_types = ['sparsity_aware'] # ansatz types for VQE (hardware_efficient or sparsity_aware)
    execution_modes = ['noiseless'] # execution modes for VQE (noiseless, noisy, or hardware)
    ansatz_layers = [3] # number of ansatz layers (any integer >= 1)
    num_repeats = 1 # number of repeats for each (instance, configuration) to average over VQE stochasticity (any integer >= 1)

    # configure logging; functions use module-level `logger`
    logging.basicConfig(filename='vqe_control1-5_log.txt', level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
    # logging.basicConfig(filename='control_prelim_stage2-vm2_log_cp_vqe_wo_thermal_noise.txt', level=logging.INFO,
    #                     format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
    logger.setLevel(logging.INFO)

    # Fixed seed so runs are reproducible; logged per-row below in case anomalies need to be replayed.
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    VM_ID = socket.gethostname()

    # Cut-management knob (see solve_SDP_by_cutting_plane docstring): purge cuts that go non-binding for several iterations,
    # to keep the master problem from growing without bound.
    CUT_PURGE_PATIENCE = None

    # Crash-safe, append-only CSV logs written incrementally as the experiment progresses (see log_row).
    # Convert to Excel afterwards (or at any time, mid-run, since it's a read-only operation) with csv_to_excel().
    ITERATION_LOG_PATH = "vqe_control1-5_iterations.csv"
    RESULTS_LOG_PATH = "vqe_control1-5_results.csv"
    ERROR_LOG_PATH = "vqe_control1-5_errors.csv"
    ERROR_LOG_FIELDNAMES = [
        "timestamp", "vm_id", "instance", "config_hash", "solver", "valid_cut_type", "add_soc_cuts",
        "collect_multiple_cuts", "ansatz_type", "ansatz_layers", "execution_mode", "optimizer_name",
        "optimizer_maxiter", "random_seed", "repeat", "error",
    ]

    # Collect rows for each (instance, repeat)
    results = []

    for instance in instances:
        try:
            # Read instance
            block_list, A_list, b_list = read_instance(instance)
            # initial solve with MOSEK for reference
            best_solution, best_value, solve_time = solve_SDP_with_MOSEK(block_list, A_list, b_list)
        except Exception as e:
            # a broken instance file or a MOSEK failure shouldn't take down the rest of the batch
            print(f"Failed to read/solve instance {instance}, skipping it: {e}")
            logger.exception("Failed to read/solve instance %s", instance)
            log_row(ERROR_LOG_PATH, {"timestamp": time.time(), "vm_id": VM_ID, "instance": instance, "error": str(e)}, ERROR_LOG_FIELDNAMES)
            continue

        all_vqe_min_eigenvalue_avg_estimates = dict() # store the VQE lowest energy estimates for all runs with different experiment configurations (for analysing the effect of different settings on the VQE convergence behavior)

        # run experiments with different settings
        for solver in solvers:
            for valid in initial_valid_cut_type:
                for add_soc_cuts in soc_cuts_added_options:
                    for collect_multiple_cuts in collect_multiple_cuts_options:
                        for ansatz_type in ansatz_types:
                            #for execution_mode, optimizer in zip(execution_modes, optimizers, strict=False):
                            for execution_mode in execution_modes:
                                #for optimizer, p in zip(optimizers, ansatz_layers, strict=False):
                                for optimizer in optimizers:
                                    for p in ansatz_layers:
                                        print(f"=== Instance: {instance}, Solver: {solver}, Valid Cut Type: {valid}, Add SOC Cuts: {add_soc_cuts}, Collect Multiple Cuts: {collect_multiple_cuts}, Optimizer: {optimizer.__class__.__name__}, Optimizer Maxiter: {optimizer.settings["maxiter"]}, Ansatz Type: {ansatz_type}, Ansatz Layers: {p}, Execution Mode: {execution_mode} ===")
                                        logger.info(f"=== Instance: {instance}, Solver: {solver}, Valid Cut Type: {valid}, Add SOC Cuts: {add_soc_cuts}, Collect Multiple Cuts: {collect_multiple_cuts}, Optimizer: {optimizer.__class__.__name__}, Optimizer Maxiter: {optimizer.settings["maxiter"]}, Ansatz Type: {ansatz_type}, Ansatz Layers: {p}, Execution Mode: {execution_mode} ===")
                                        
                                        optimizer_maxiter = optimizer.settings["maxiter"] # varies across optimizer instances of the same class (e.g. COBYLA(maxiter=200) vs COBYLA(maxiter=500)), so it must be tracked alongside optimizer_name to distinguish configs
                                        experiment_config = (solver, valid, add_soc_cuts, collect_multiple_cuts, ansatz_type, execution_mode, p, optimizer.__class__.__name__, optimizer_maxiter)

                                        # Eigenvalue histories for all repeats with this experiment configuration
                                        all_eigenvalue_trajectories = {repeat: {"exact": dict(), "vqe": dict()} for repeat in range(1, num_repeats + 1)} # store exact and VQE eigenvalues for each block and iteration
                                        all_master_objective_values = {repeat: [] for repeat in range(1, num_repeats + 1)} # store master objective values for each iteration to analyze convergence behavior across different runs with the same experiment configuration
                                        # Config identity shared by every row (iteration/result/error) produced by this experiment configuration.
                                        config_dict = {
                                            "instance": instance, "solver": solver, "valid_cut_type": valid,
                                            "add_soc_cuts": add_soc_cuts, "collect_multiple_cuts": collect_multiple_cuts,
                                            "ansatz_type": ansatz_type, "ansatz_layers": p,
                                            "execution_mode": execution_mode, "optimizer_name": optimizer.__class__.__name__,
                                            "optimizer_maxiter": optimizer_maxiter,
                                        }
                                        config_hash = compute_config_hash(config_dict)

                                        for repeat in range(1, num_repeats + 1):
                                            logger.info(f"--- Run {repeat}/{num_repeats} ---")
                                            log_context = {**config_dict, "config_hash": config_hash, "random_seed": RANDOM_SEED, "repeat": repeat}
                                            rss_at_run_start_mb = current_rss_mb()
                                            try:
                                                termination_reason, itr, linear_cuts_added, linear_cuts_purged, soc_cuts_added, master_total_time, sep_total_time, X_values, objVal = solve_SDP_by_cutting_plane(block_list, A_list, b_list, solver, valid=valid, add_soc_cuts=add_soc_cuts, collect_multiple_cuts=collect_multiple_cuts, optimizer=optimizer, itr_limit=250, ansatz_type=ansatz_type, execution_mode=execution_mode, backend_name="quantinuum", reps=p, log_filepath=ITERATION_LOG_PATH, log_context=log_context, cut_purge_patience=CUT_PURGE_PATIENCE)
                                                all_eigenvalue_trajectories[repeat]["exact"] = exact_min_eigenvalues.copy() # store the trajectory of exact min eigenvalues for this run
                                                all_eigenvalue_trajectories[repeat]["vqe"] = vqe_min_eigenvalue_estimates.copy() # store the trajectory of VQE eigenvalue estimates for this run
                                                all_master_objective_values[repeat].extend(master_objective_values.copy()) # store the master objective values history for this run

                                                # Clear the global history of collected violating vectors before each repeat within the same experiment configuration
                                                violating_vectors.clear()
                                                exact_min_eigenvalues.clear() # clear the global history of exact eigenvalues for the next experiment configuration
                                                vqe_min_eigenvalue_estimates.clear() # clear the global history of VQE eigenvalue estimates for the next experiment configuration
                                                master_objective_values.clear() # clear the global history of master objective values for the next experiment configuration
                                                leaked_amplitudes.clear() # clear the global history of leaked amplitudes for the next experiment configuration
                                            except Exception as e:
                                                # record failure and continue, so one bad instance/config doesn't lose the rest of an overnight batch
                                                print(f"Run failed with solver {solver} for {instance} with initial valid cuts {valid}, with SOC cuts added {add_soc_cuts}, collect multiple cuts option {collect_multiple_cuts}, optimizer {optimizer.__class__.__name__} with maxiter {optimizer_maxiter}, ansatz type {ansatz_type}, ansatz layers {p}, execution mode {execution_mode}, in (repeat {repeat}): {e}")
                                                logger.exception("Run failed (repeat %d)", repeat)
                                                log_row(ERROR_LOG_PATH, {**log_context, "timestamp": time.time(), "vm_id": VM_ID, "error": str(e)}, ERROR_LOG_FIELDNAMES)
                                                termination_reason = str(e)
                                                itr = None
                                                linear_cuts_added = None
                                                linear_cuts_purged = None
                                                soc_cuts_added = None
                                                master_total_time = None
                                                sep_total_time = None
                                                objVal = None

                                            # Append result row (kept in memory for the final Excel export, and written immediately to CSV so an
                                            # overnight crash doesn't lose everything but the last in-flight run).
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

                                            # Compact feedback
                                            print(f"Result: termination_reason={termination_reason}, iterations={itr}, linear_cuts={linear_cuts_added}, linear_cuts_purged={linear_cuts_purged}, soc_cuts={soc_cuts_added}, master_time={master_total_time}, sep_time={sep_total_time}, objVal={objVal}")
                                            print("***  ***\n")
                                            logger.info(f"Result: termination_reason={termination_reason}, iterations={itr}, linear_cuts={linear_cuts_added}, linear_cuts_purged={linear_cuts_purged}, soc_cuts={soc_cuts_added}, master_time={master_total_time}, sep_time={sep_total_time}, objVal={objVal}")

                                        # Compute average trajectories across repeats for this experiment configuration.
                                        # Repeats that raised an exception never populated "exact"/"vqe" (see the
                                        # success-only assignment above), so exclude them here rather than KeyError.
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

                                            # Compute average trajectories for the current block across repeats
                                            avg_eigenvalue_trajectories["exact"][block_id] = [np.mean([all_eigenvalue_trajectories[repeat]["exact"][block_id][iter_id] for repeat in successful_repeats]) for iter_id in range(max_iters_for_block)]
                                            avg_eigenvalue_trajectories["vqe"][block_id] = [np.mean([all_eigenvalue_trajectories[repeat]["vqe"][block_id][iter_id] for repeat in successful_repeats]) for iter_id in range(max_iters_for_block)]

                                        all_vqe_min_eigenvalue_avg_estimates[experiment_config] = avg_eigenvalue_trajectories["vqe"].copy() # store the average VQE eigenvalue trajectories for this experiment configuration for later analysis of the effect of different settings on the VQE convergence behavior

                                        # Compute progress of the average master objective value across repeats for this experiment configuration.
                                        # Restrict to repeats that recorded at least one objective value: a repeat can be in
                                        # successful_repeats (no exception raised) yet still have an empty history if the master
                                        # problem never returned a solution, even on iteration 1 (see the `objVal is not None` guard
                                        # around master_objective_values.append above) - padding with [-1] on an empty list would crash.
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
                                            # The number of master objective values recorded should be equal to the number of iterations + 1 (including the initial solve before any cuts are added). If some runs have fewer recorded values due to early termination, we can pad the history with the last recorded value to ensure consistent averaging across repeats.
                                            if len(all_master_objective_values[repeat]) < max_length:
                                                print(f"Run {repeat} has fewer master objective values ({len(all_master_objective_values[repeat])}) than the maximum length ({max_length}). Padding with the last recorded value for averaging.")
                                                all_master_objective_values[repeat].extend([all_master_objective_values[repeat][-1]] * (max_length - len(all_master_objective_values[repeat])))
                                        avg_master_objective_values = [np.mean([all_master_objective_values[repeat][iter_id] for repeat in objective_repeats]) for iter_id in range(max_length)]

                                        generate_plots_for_given_configuration(instance, experiment_config, avg_eigenvalue_trajectories, avg_master_objective_values, best_value) # generate plots for this experiment configuration using the collected global history of VQE trajectories, exact eigenvalues and master objective values.   

        generate_VQE_comparison_plots(instance, all_vqe_min_eigenvalue_avg_estimates) # generate comparison plots of the average VQE minimum eigenvalue trajectories across iterations for different experiment configurations to analyze the effect of different settings on the VQE convergence behavior.                    

    # After all experiments are done, write the collected results to a file for analysis.
    write_results_to_file(results, filename="vqe_control1-5_results.xlsx")