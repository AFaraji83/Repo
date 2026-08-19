"""
Cutting-plane solver for semidefinite programs (SDPs).
Implements a cutting-plane method that iteratively refines a linear relaxation
of a block-diagonal SDP by adding valid inequalities derived from violated
positive-semidefiniteness constraints. The separation problem identifies
violated constraints using eigenvalue decomposition.
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
from gurobipy import GRB, Model, quicksum
from scipy.linalg import eigh

# Numerical tolerance for PSD checks
TOL = 1.0e-6

# module-level logger used across functions (configured in __main__)
logger = logging.getLogger(__name__)

# Global history of all violating vectors collected across iterations for each block (used for
# diversity checks to avoid adding cuts from vectors that overlap ones already found)
violating_vectors = defaultdict(list)

# Fixed schema for per-iteration (per-block) result rows, written incrementally to CSV as each iteration completes.
ITERATION_LOG_FIELDNAMES = [
    "timestamp", "instance", "config_hash", "solver", "valid_cut_type", "add_soc_cuts",
    "random_seed", "iteration", "block_id", "n_original",
    "true_min_eigenvalue", "master_objective", "master_solve_time", "separation_time",
    "num_cuts_added_this_iter", "is_psd",
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
        
def check_diversity_and_save_eigenvector(vector, collected_vectors, block_id, overlap_threshold=0.9):
    """
    Checks if a (unit-norm) eigenvector is diverse against violating_vectors[block_id] via cosine
    similarity -- since eigh returns unit-norm eigenvectors, a plain dot product already is the
    cosine similarity. A diverse vector is appended to both collected_vectors (this call's cuts)
    and violating_vectors[block_id] (seeding future diversity comparisons, both later in this call
    and in later cutting-plane iterations); a non-diverse vector is dropped and touches neither
    list. Mirrors the intent of VQESubroutine._check_diversity_and_save_state, though PSD_check_EVD
    handles that routine's force-collect case itself (see its docstring) rather than this helper,
    since EVD -- unlike VQE's online optimization -- already has every candidate in hand up front.
    """
    is_diverse = True
    if len(violating_vectors[block_id]) > 0:
        # Compute inner product against ALL vectors in pool
        overlaps = np.abs(np.dot(violating_vectors[block_id], vector))
        # If all overlaps are strictly less than the threshold, the vector passes the diversity check
        is_diverse = bool(np.all(overlaps < overlap_threshold))

    if is_diverse:
        collected_vectors.append(vector)
        violating_vectors[block_id].append(vector)

def PSD_check_EVD(A: np.ndarray, block_id):
    """
    Check whether a matrix is positive semidefinite using eigenvalue decomposition.
    Parameters
    ----------
    A : array_like (n x n)
        Square matrix to test for positive semidefiniteness.
    block_id : hashable
        Key into the module-level violating_vectors pool, which persists previously collected
        violating vectors for this block across cutting-plane iterations. Candidate eigenvectors
        that overlap (cosine similarity >= 0.9, see check_diversity_and_save_eigenvector) any
        vector already in violating_vectors[block_id] are dropped, unless dropping every candidate
        would leave a genuinely non-PSD matrix with no reported cut, in which case the
        most-violating candidate is force-collected as a cut without being added to the pool.
    Returns
    -------
    is_psd : bool
        - True if all eigenvalues of A are nonnegative within the numerical tolerance -TOL,
          False otherwise.
    u_vectors : list[numpy.ndarray]
        List of eigenvectors (as numpy ndarrays) corresponding to negative eigenvalues
        of A (i.e., violating vectors u with u^T A u < 0), filtered for diversity against
        violating_vectors[block_id]. If A is PSD, the list is empty.
    sep_time : float
        Time taken (in seconds) to perform the eigenvalue decomposition and identify
        violating vectors.
    min_eigenvalue : float
        The smallest eigenvalue of A, for analysis/logging (e.g. tracking convergence of the
        cutting-plane method towards PSD-ness across iterations).
    Raises
    ------
    TypeError
        If the input cannot be interpreted as a numpy ndarray.
    ValueError
        If the input matrix A is not square.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a 2-D square array.")

    is_psd = True
    u_vectors = []
    start_time = time.time()
    # compute all eigenvalues and check nonnegativity up to the specified tolerance
    vals = np.linalg.eigvalsh(A) # for symmetric/Hermitian matrices, returned in ascending order
    min_eigenvalue = float(vals[0])
    # identify indices of negative eigenvalues beyond tolerance
    neg_idx = np.where(vals < -TOL)[0]
    # if any negative eigenvalues found, compute corresponding eigenvectors
    if neg_idx.size > 0:
        is_psd = False
        # negative eigenvalues are at the low end (vals sorted ascending),
        # compute only that contiguous range of eigenpairs
        w, v = eigh(A, subset_by_index=(neg_idx[0], neg_idx[-1]))
        candidate_vectors = [v[:, i] for i, val in enumerate(w) if val < -TOL]

        # Apply the diversity filter to each candidate (most-violating first, since w is ascending),
        # dropping ones that overlap a vector already in violating_vectors[block_id].
        for vec in candidate_vectors:
            check_diversity_and_save_eigenvector(vec, u_vectors, block_id)

        # Guarantee a genuine PSD violation is always reported: if every candidate duplicated a
        # vector already in the pool (u_vectors ended up empty) the matrix is still provably
        # non-PSD, so force-collect the most-violating candidate (smallest eigenvalue) rather than
        # let the diversity filter turn a real violation into a false PSD certificate.
        if not u_vectors:
            u_vectors.append(candidate_vectors[0])
    return is_psd, u_vectors, time.time() - start_time, min_eigenvalue

def PSD_check(A: np.ndarray, block_id):
    """
    Check whether a matrix is positive semidefinite using eigenvalue decomposition.
    Parameters
    ----------
    A : array-like (n x n)
        Square matrix to test for positive semidefiniteness.
    block_id : hashable
        Key into the module-level violating_vectors pool, forwarded to PSD_check_EVD for
        diversity filtering (see its docstring). Trivial cases (1x1 blocks, effectively-zero
        matrices) bypass the diversity filter entirely, same as the VQE-based psd_check.
    Returns
    -------
    is_psd : bool
        - True if the matrix A is determined to be positive semidefinite.
        - False if a violating vector was found (indicating A is not PSD).
    u_vectors : list[list[float]]
        - List of violating vectors u such that u^T A u < 0, if any were found.
        - Empty list if no violating vector was found (indicating A may be PSD).
    sep_time : float
        Time taken (in seconds) to perform the PSD check.
    min_eigenvalue : float
        The smallest eigenvalue of A.
    Raises
    ------
    ValueError
        If the specified method is not recognized or if required parameters are missing/invalid.
    """
    # Input validation
    if A.shape[0] != A.shape[1] or A.shape[0] <= 0:
        raise ValueError("Input matrix must be square and of positive dimension.")

    # Preprocess the input matrix to handle trivial cases and reduce noise for non-trivial cases
    matrix_for_psd_check = A.copy() # create a copy to avoid modifying the original matrix
    # Handle trivial corner cases directly
    if A.shape[0] == 1: # 1x1 (scalar) blocks
        print("Trivial 1x1 block detected in PSD check.")
        if A[0, 0] < -TOL: # Exact classical check for a scalar: eigenvalue is the single entry.
            return False, [np.array([1.0])], 0.0, float(A[0, 0]) # Violating vector is simply [1.0]
        else:
            return True, [], 0.0, float(A[0, 0])
    elif np.max(np.abs(A)) < TOL: # effectively zero matrix
        print("Matrix is effectively zero. Smallest eigenvalue is 0.0")
        return True, [], 0.0, 0.0
    else: # Scrub the noise out to reduce numerical noise
        matrix_for_psd_check = np.where(np.abs(A) < TOL, 0, A)

    return PSD_check_EVD(matrix_for_psd_check, block_id)
    
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
    model : gurobipy.Model or mosek.fusion.Model
        Optimization model to which constraints will be added. 
    X : gurobipy.VarDict or Mosek
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
    
    s2 = np.sqrt(2)
    # nonnegative diagonal (always enforced, independent of the `valid` choice below)
    if solver == "Gurobi":
        model.addConstrs(X[i,i] >= 0 for i in range(n))
    if solver == "MOSEK" and valid != "soc":
        model.constraint(X.diag(), msk.Domain.greaterThan(0.0))

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
    model : gurobipy.Model or MOSEK.model
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

    if solver == "MOSEK": 
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
        # u^T X u expressed as a matrix-vector product (Expr.mul) followed by a vector dot,
        # instead of Expr.dot(np.outer(u, u), X_block). The outer product form materializes
        # a dense block_size x block_size numpy array *per violating eigenvector, per iteration*
        # with no pruning of old cuts across the (up to itr_limit) iterations, which is what
        # drove the OOM on control10's 100x100 block. This form is O(block_size) instead.
        con = model.constraint(msk.Expr.dot(u_vector, msk.Expr.mul(X_block, u_vector)), msk.Domain.greaterThan(0.0))
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

    # Build n x 2 matrix with the two vectors as columns
    U = np.column_stack([np.asarray(u).ravel() for u in u_vectors[:2]])  # shape (block_size, 2)

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

def solve_SDP_by_cutting_plane(block_list: list, A_list: list, b_list: list, solver: str, valid="linear", add_soc_cuts=False, itr_limit = 10, log_filepath=None, log_context=None, cut_purge_patience=5, cut_dual_tol=1e-8):
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
        - PSD_check(X_value, block_id)
                -> returns (is_psd: bool, u_vectors: list of violating vectors, sep_time: float, min_eigenvalue: float).
                   u_vectors is diversity-filtered against violating_vectors[block_id] (see
                   PSD_check_EVD / check_diversity_and_save_eigenvector) so cuts that overlap a
                   vector already found in an earlier iteration are dropped, except when doing so
                   would leave a genuinely non-PSD block with no reported cut.
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
    valid : str, optional (default="linear")
            Type of validity/relaxation constraints used to initialize the master problem. Passed through to
            create_master_problem; accepted values depend on that function (commonly "linear" or other relaxations).
    add_soc_cuts : bool, optional (default=False)
            If True, second-order-cone (SOC)-type cuts will be added if possible in addition to linear cuts when a block 
            is found to be non-PSD.
    itr_limit : int, optional (default=10)
            Maximum number of cutting-plane iterations to perform.
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
    # Reset the per-block diversity pool for this run: it accumulates violating vectors across all
    # iterations below, seeding future diversity comparisons (see PSD_check_EVD).
    for b in range(nblocks):
        violating_vectors[b] = []
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
                # Covers MOSEK's non-Optimal/non-Certificate primal solution statuses (e.g. infeasible,
                # unknown, near-optimal), which solve_master_problem signals by returning X_values=None.
                termination_reason = f"Master problem returned no solution (solver={solver}, status={status})."
                break
            else:
                # drop linear cuts that have gone non-binding for `cut_purge_patience` consecutive
                # iterations, using duals from the solve just above, before adding this iteration's cuts
                linear_cuts_purged += purge_inactive_linear_cuts(cut_registry, solver, cut_purge_patience, cut_dual_tol, itr, logger)
                # check positive-semidefiniteness of each block matrix
                all_u_vectors = []
                for b in range(nblocks): # this could be parallelized !!!
                    rss_before_sep = current_rss_mb()
                    is_psd, u_vectors, sep_time, min_eigenvalue = PSD_check(X_values[b], b)
                    rss_after_sep = current_rss_mb()
                    all_u_vectors.append(u_vectors)
                    sep_total_time += sep_time
                    logger.info("Time(sep)= %f", sep_time)
                    logger.info("Block %d is PSD?: %s", b, str(is_psd))
                    # add cutting planes if block is not PSD
                    if not is_psd:
                        logger.info("Adding cutting planes for block %d", b)
                        # add linear (and SOC) cuts based on the violating vectors found
                        for u in u_vectors:
                            con = add_linear_cut(model, X[b], u, block_list[b], solver) # add linear cut
                            linear_cuts_added += 1
                            if solver == "MOSEK" and cut_purge_patience is not None:
                                cut_registry[b].append({"con": con, "inactive_streak": 0})
                        if add_soc_cuts:
                            if len(u_vectors) >= 2: # need at least two negative eigenvectors for SOC cut
                                add_soc_cut(model, X[b], u_vectors[:2], block_list[b], solver) # only use first two vectors (they have the smallest eigenvalues if from EVD)
                                soc_cuts_added += 1
                    else: # no cutting planes added
                        logger.info("No cutting planes added for block %d", b)

                    # incremental, crash-safe per-block/per-iteration logging
                    if log_filepath is not None:
                        row = dict(log_context or {})
                        row.update({
                            "timestamp": time.time(),
                            "iteration": itr,
                            "block_id": b,
                            "n_original": block_list[b],
                            "true_min_eigenvalue": min_eigenvalue,
                            "master_objective": objVal,
                            "master_solve_time": master_solve_time,
                            "separation_time": sep_time,
                            "num_cuts_added_this_iter": len(u_vectors),
                            "is_psd": is_psd,
                            "rss_before_sep": rss_before_sep,
                            "rss_after_sep": rss_after_sep,
                            "rss_before_master": rss_before_master,
                            "rss_after_master": rss_after_master,
                        })
                        log_row(log_filepath, row, ITERATION_LOG_FIELDNAMES)
                # check termination condition
                if all(len(u_vectors) == 0 for u_vectors in all_u_vectors):
                    is_optimal = True
                    termination_reason = "Terminating because all blocks are PSD. Optimal solution found."
                elif itr == itr_limit:
                    termination_reason = "Terminating because iteration limit was reached."
    finally:
        # Release the master model's native (Gurobi/MOSEK) resources as soon as this run ends,
        # regardless of how it ended (normal termination, early break, or an exception propagating
        # out of the loop) -- otherwise repeated runs in a batch (see __main__) leak one model's
        # worth of native memory per run. Model.dispose() is the same call a `with msk.Model(...)`
        # block would make on __exit__ (used elsewhere for the one-shot MOSEK solve); this model
        # is created without a `with` block since it needs to stay open across iterations.
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

if __name__ == "__main__":
    # Define the instances to run the experiments on
    instances = [f'control{i}' for i in range(1,6)]
    # Define experiment parameters
    solvers = ["MOSEK"] # solvers to use for the master problem (Gurobi or MOSEK)
    initial_valid_cut_type = ["soc"] # initial valid inequalities to add to the master problem (linear or soc)
    soc_cuts_added_options = [True] # whether to add SOC cuts in addition to linear cuts (True or False)
    

    # configure logging to write to log.txt; functions use module-level `logger`
    logging.basicConfig(filename='evd_control1-5_log.txt', level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
    logger.setLevel(logging.INFO)

    # Fixed seed for reproducibility; logged per-row below in case anomalies need to be replayed.
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    VM_ID = socket.gethostname()

    # Cut-management knob (see solve_SDP_by_cutting_plane docstring): purge cuts that go non-binding for several iterations,
    # to keep the master problem from growing without bound.
    CUT_PURGE_PATIENCE = None

    # Crash-safe, append-only CSV logs written incrementally as the experiment progresses (see log_row).
    # Convert to Excel afterwards (or at any time, mid-run, since it's a read-only operation) with csv_to_excel().
    ITERATION_LOG_PATH = "evd_control1-5_cutting_plane_iterations.csv"
    RESULTS_LOG_PATH = "evd_control1-5_cutting_plane_results.csv"
    ERROR_LOG_PATH = "evd_control1-5_cutting_plane_errors.csv"
    ERROR_LOG_FIELDNAMES = [
        "timestamp", "vm_id", "instance", "config_hash", "solver", "valid_cut_type",
        "add_soc_cuts", "random_seed", "error",
    ]

    # Collect results in memory for a final Excel export.
    results = []

    for instance in instances:
        try:
            # read instance
            block_list, A_list, b_list = read_instance(instance)

            # initial solve with MOSEK for reference
            best_solution, best_value, solve_time = solve_SDP_with_MOSEK(block_list, A_list, b_list)
        except Exception as e:
            # a broken instance file or a MOSEK failure shouldn't take down the rest of the batch
            print(f"Failed to read/solve instance {instance}, skipping it: {e}")
            logger.exception("Failed to read/solve instance %s", instance)
            log_row(ERROR_LOG_PATH, {"timestamp": time.time(), "vm_id": VM_ID, "instance": instance, "error": str(e)}, ERROR_LOG_FIELDNAMES)
            continue

        # run experiments for all combinations -- some combinations may not be possible with Gurobi
        for solver in solvers:
            for valid in initial_valid_cut_type:
                for add_soc_cuts in soc_cuts_added_options:
                    print(f"=== Instance: {instance}, Solver: {solver}, Valid: {valid}, Add SOC Cuts: {add_soc_cuts} ===")
                    logger.info(f"=== Instance: {instance}, Solver: {solver}, Valid: {valid}, Add SOC Cuts: {add_soc_cuts} ===")

                    # Config identity shared by every row (iteration/result/error) produced by this experiment configuration.
                    config_dict = {
                        "instance": instance, "solver": solver, "valid_cut_type": valid, 
                        "add_soc_cuts": add_soc_cuts, "cut_purge_patience": CUT_PURGE_PATIENCE,
                    }
                    config_hash = compute_config_hash(config_dict)

                    # Deliberately not the full config_dict: ITERATION_LOG_FIELDNAMES / ERROR_LOG_FIELDNAMES
                    # are fixed column lists that don't include the cut-management knob (it only
                    # feeds compute_config_hash above), and csv.DictWriter raises on unlisted keys.
                    log_context = {
                        "instance": instance, "solver": solver, "valid_cut_type": valid, "add_soc_cuts": add_soc_cuts,
                        "config_hash": config_hash, "random_seed": RANDOM_SEED
                    }
                    rss_at_run_start_mb = current_rss_mb()
                    try:
                        termination_reason, itr, linear_cuts_added, linear_cuts_purged, soc_cuts_added, master_total_time, sep_total_time, X_values, objVal = solve_SDP_by_cutting_plane(block_list, A_list, b_list, solver, valid, add_soc_cuts, itr_limit=250, log_filepath=ITERATION_LOG_PATH, log_context=log_context, cut_purge_patience=CUT_PURGE_PATIENCE)
                    except Exception as e:
                        # record failure and continue, so one bad config doesn't lose the rest of an overnight batch
                        print(f"Run failed with solver {solver} for {instance} with initial valid cuts {valid} and with SOC cuts added {add_soc_cuts}: {e}")
                        logger.exception("Run failed")
                        log_row(ERROR_LOG_PATH, {**log_context, "timestamp": time.time(), "vm_id": VM_ID, "error": str(e)}, ERROR_LOG_FIELDNAMES)
                        termination_reason = str(e)
                        itr = None
                        linear_cuts_added = None
                        linear_cuts_purged = None
                        soc_cuts_added = None
                        master_total_time = None
                        sep_total_time = None
                        objVal = None

                    # Append result row (kept in memory for the final Excel export, and written immediately to CSV
                    # so an overnight crash doesn't lose everything but the last in-flight run).
                    result_row = {
                        "timestamp": time.time(),
                        "vm_id": VM_ID,
                        "config_hash": config_hash,
                        "random_seed": RANDOM_SEED,
                        "instance": instance,
                        "solver": solver,
                        "valid": valid,
                        "add_soc_cuts": add_soc_cuts,
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

                    # compact feedback
                    print(f"Result: iterations={itr}, linear_cuts={linear_cuts_added}, linear_cuts_purged={linear_cuts_purged}, soc_cuts={soc_cuts_added}, master_time={master_total_time}, sep_time={sep_total_time}, objVal={objVal}")
                    print("***  ***\n")

    # Write results to Excel if possible, otherwise CSV (results.csv/xlsx here are a convenience export of the
    # in-memory list; the incremental CSVs above are the crash-safe source of truth).
    if results:
        try:
            import pandas as pd

            df = pd.DataFrame(results)
            df.to_excel("evd_control1-5_results.xlsx", index=False)
            print("Wrote evd_control1-5_results.xlsx")
        except Exception:
            csv_path = "evd_control1-5_results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
            print(f"Pandas not available or write failed; wrote {csv_path} instead")
    else:
        print("No results collected; nothing to write.")