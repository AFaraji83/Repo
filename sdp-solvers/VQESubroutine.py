import hashlib
import itertools
import threading
import time
from collections import deque
from functools import reduce

import cudaq
import numpy as np
from cudaq import spin


def _matrix_to_pauli_terms(matrix, atol=1e-12):
    """
    Decompose a 2^n x 2^n Hermitian matrix into a sum of n-qubit Pauli strings.

    Uses the standard Pauli basis expansion: coeff_P = Tr(P @ M) / 2^n for every P in
    {I,X,Y,Z}^n, keeping only terms whose coefficient magnitude exceeds `atol`.

    Returns a list of (pauli_string, coefficient) tuples, where pauli_string is an
    n-character string over {I,X,Y,Z} written most-significant-qubit-first, e.g. "IXZ"
    for qubit order (q2, q1, q0).
    """
    dim = matrix.shape[0]
    num_qubits = int(np.log2(dim))
    if 2 ** num_qubits != dim:
        raise ValueError(f"Matrix dimension {dim} is not a power of two.")

    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    basis = {"I": I, "X": X, "Y": Y, "Z": Z}

    terms = []
    for combo in itertools.product("IXYZ", repeat=num_qubits):
        pauli_matrix = reduce(np.kron, (basis[c] for c in combo))
        coeff = np.trace(pauli_matrix @ matrix) / dim
        if abs(coeff) > atol:
            terms.append(("".join(combo), coeff.real))
    return terms


def _pauli_terms_to_spin_op(terms, num_qubits):
    """Build a cudaq SpinOperator from (pauli_string, coeff) terms (see _matrix_to_pauli_terms)."""
    gate_for_char = {"X": spin.x, "Y": spin.y, "Z": spin.z}
    op = 0.0 * spin.i(0)
    for pauli_string, coeff in terms:
        factors = []
        for q in range(num_qubits):
            char = pauli_string[num_qubits - 1 - q]
            if char != "I":
                factors.append(gate_for_char[char](q))
        term = coeff * reduce(lambda a, b: a * b, factors, spin.i(0)) if factors else coeff * spin.i(0)
        op = op + term
    return op


def _reorder_pauli_word(pauli_string, num_qubits):
    """cudaq's kernel.exp_pauli expects the word in qubit-0-first order; our internal
    pauli_string convention (from _matrix_to_pauli_terms) is most-significant-qubit-first,
    matching Qiskit's SparsePauliOp -- this just reverses the string between the two."""
    return pauli_string[::-1]


class VQESubroutine:
    def __init__(self, ansatz_type='hardware_efficient', execution_mode='noiseless', backend_name=None,
                 reps=2, ansatz_mode='fixed', max_reps=6, adapt_grad_tol=1e-3, adapt_pool_size=30,
                 enable_pruning=False, pruning_threshold=0.05, param_memory_size=5):
        """
        ansatz_type: 'hardware_efficient' or 'sparsity_aware' -- only affects ansatz_mode='fixed'
            and 'adaptive_depth' (both use the RY/CX real-amplitudes-style layer structure).
            ansatz_mode='adapt_pool' ignores this; its structure comes entirely from the
            Hamiltonian's own Pauli terms (Change #3's "Hamiltonian-aware circuit generation").
        execution_mode: 'noiseless', 'noisy', or 'hardware'
        backend_name: a CUDA-Q hardware target name, required when execution_mode='hardware'.
        reps: number of RY/CX layers for ansatz_mode='fixed'. Ignored by 'adaptive_depth'
            (which searches depth itself, up to max_reps) and 'adapt_pool' (which grows
            operator-by-operator, up to max_reps operators).

        --- Change #3: Adaptive Variational Circuits ---
        ansatz_mode: 'fixed' (original real_amplitudes-style circuit, depth fixed at `reps`),
            'adaptive_depth' (starts at 1 layer and grows, up to max_reps, stopping as soon as
            a cheap trial optimization clears the PSD threshold -- see _select_adaptive_depth),
            or 'adapt_pool' (lightweight ADAPT-VQE: greedily builds the circuit one
            Hamiltonian-derived Pauli-rotation operator at a time, by gradient magnitude, up to
            max_reps operators or until the largest available gradient drops below
            adapt_grad_tol -- see _build_adapt_pool_ansatz).
        max_reps: depth/operator-count ceiling for 'adaptive_depth' and 'adapt_pool'.
        adapt_grad_tol: gradient-magnitude stopping threshold for 'adapt_pool' growth.
        adapt_pool_size: cap on how many of the Hamiltonian's Pauli terms (by |coefficient|,
            largest first) are considered as pool operators for 'adapt_pool' -- gradient
            screening is one extra circuit evaluation per pool operator per growth step, so an
            unbounded pool (some blocks here have 500+ Pauli terms) would make growth itself the
            bottleneck; this keeps that bounded and predictable.

        --- Change #5's "circuit pruning" support ---
        enable_pruning: if True, after any 'fixed'-mode solve, checks whether the *last*
            rotation layer's angles all sit within pruning_threshold of 0 (mod 2*pi); if so, the
            next ansatz built for that same (qubit-count, entanglement) shape drops one layer.
            Only ever removes whole layers (not individual gates) -- see _get_ansatz.
        pruning_threshold: angular tolerance (radians) for "close enough to zero to prune".

        --- Change #4: Transfer Learning ---
        param_memory_size: how many recent optimal parameter vectors to keep, per ansatz shape,
            as a fallback informed initial guess when solve() is called with initial_point=None
            (e.g. the very first time a given block shape is seen). The primary transfer-learning
            path is still the caller passing the *previous iteration's* optimal parameters back
            in as initial_point (already supported); this is a secondary net for cold starts.
        """
        self.ansatz_type = ansatz_type
        self.execution_mode = execution_mode
        self.reps = reps
        self.ansatz_mode = ansatz_mode
        self.max_reps = max_reps
        self.adapt_grad_tol = adapt_grad_tol
        self.adapt_pool_size = adapt_pool_size
        self.enable_pruning = enable_pruning
        self.pruning_threshold = pruning_threshold

        self.target_name = None
        self.noise_model = None
        self.num_trajectories = None
        self.ansatz_cache = {}                                   # cache_key -> (kernel, num_params)
        self.prunable_cache_keys = set()                         # cache_key -> eligible to shrink by one layer next build
        self.param_memory = {}                                   # cache_key -> deque[np.ndarray] (Change #4)
        self.opt_state_memory = {}                                # cache_key -> optimizer state dict (Change #4)
        self._param_memory_size = param_memory_size
        self._lock = threading.Lock()  # ansatz_cache/param_memory/opt_state_memory are shared across
                                        # blocks if the caller parallelizes separation (Change #5's
                                        # "parallel separation of independent PSD blocks")

        self._setup_engine(backend_name)

    def _setup_engine(self, backend_name):
        """
        Selects the CUDA-Q target used for energy evaluation. Both the 'noiseless' and 'noisy'
        paths use the 'nvidia' target, which is cuStateVec/cuQuantum-accelerated GPU statevector
        simulation (Change #2's "NVIDIA cuStateVec and cuQuantum acceleration" -- this is CUDA-Q's
        built-in route to that, rather than something to configure separately). If you don't have
        a GPU available, swap 'nvidia' for 'qpp-cpu' here and in _bind_circuit_and_extract_state.
        """
        if self.execution_mode == 'noiseless':
            self.target_name = 'nvidia'

        elif self.execution_mode == 'noisy':
            self.target_name = 'nvidia'
            self.num_trajectories = 2000
            self.noise_model = self._build_default_noise_model()

        elif self.execution_mode == 'hardware':
            if backend_name is None:
                raise ValueError(
                    "execution_mode='hardware' requires an explicit backend_name "
                    "(CUDA-Q has no automatic 'least busy backend' lookup across providers)."
                )
            self.target_name = backend_name

        else:
            raise ValueError(f"Unknown execution_mode: {self.execution_mode}")

        if backend_name is not None and self.execution_mode != 'hardware':
            print(f"Note: backend_name={backend_name!r} is ignored outside of execution_mode='hardware'.")

    @staticmethod
    def _build_default_noise_model():
        noise = cudaq.NoiseModel()
        depol_1q = cudaq.DepolarizationChannel(0.001)
        depol_2q = cudaq.Depolarization2(0.01)
        readout = cudaq.BitFlipChannel(0.01)
        for gate in ("rx", "ry", "rz", "h", "x"):
            noise.add_all_qubit_channel(gate, depol_1q)
        for gate in ("cx", "cz"):
            noise.add_all_qubit_channel(gate, depol_2q, num_controls=1)
        noise.add_all_qubit_channel("mz", readout)
        return noise

    def _get_entanglement(self, terms, num_qubits):
        if self.ansatz_type == 'hardware_efficient':
            return 'linear'
        edges = set()
        for pauli_string, _coeff in terms:
            active = [i for i, char in enumerate(pauli_string[::-1]) if char != 'I']
            if len(active) >= 2:
                for idx, q1 in enumerate(active):
                    for q2 in active[idx + 1:]:
                        edges.add(tuple(sorted((q1, q2))))
        return sorted(edges) if edges else 'linear'

    # ------------------------------------------------------------------
    # Ansatz construction: 'fixed' / 'adaptive_depth' share this builder;
    # 'adapt_pool' has its own (see _build_adapt_pool_ansatz below).
    # ------------------------------------------------------------------
    def _build_fixed_kernel(self, num_qubits, pairs, reps):
        kernel, thetas = cudaq.make_kernel(list)
        qubits = kernel.qalloc(num_qubits)
        param_idx = 0
        for _layer in range(reps):
            for q in range(num_qubits):
                kernel.ry(thetas[param_idx], qubits[q])
                param_idx += 1
            for (control, target) in pairs:
                kernel.cx(qubits[control], qubits[target])
        for q in range(num_qubits):
            kernel.ry(thetas[param_idx], qubits[q])
            param_idx += 1
        return kernel, param_idx

    def _get_ansatz(self, num_qubits, entanglement_map, forced_reps=None):
        """
        Builds and caches a real_amplitudes-style ansatz (ansatz_mode='fixed' or the depth chosen
        by 'adaptive_depth'). If this shape was flagged as prunable by a previous solve() (Change
        #5's circuit pruning -- see solve()'s post-processing), the cached depth is one layer
        shallower than it was before, and the stale (now wrong-length) param/state memory for
        this shape is cleared.
        """
        pairs = (list(zip(range(num_qubits - 1), range(1, num_qubits)))
                 if entanglement_map == 'linear' else entanglement_map)
        map_key = 'linear' if entanglement_map == 'linear' else tuple(entanglement_map)
        cache_key = (num_qubits, map_key, self.ansatz_mode)

        with self._lock:
            if cache_key in self.prunable_cache_keys:
                # Shrink by exactly one layer and forget stale-shaped memory for this key.
                _old_kernel, old_num_params = self.ansatz_cache.get(cache_key, (None, num_qubits * (self.reps + 1)))
                old_reps = max(1, old_num_params // num_qubits - 1)
                new_reps = max(1, old_reps - 1)
                kernel, num_params = self._build_fixed_kernel(num_qubits, pairs, new_reps)
                self.ansatz_cache[cache_key] = (kernel, num_params)
                self.param_memory.pop(cache_key, None)
                self.opt_state_memory.pop(cache_key, None)
                self.prunable_cache_keys.discard(cache_key)
            elif cache_key not in self.ansatz_cache:
                reps = forced_reps if forced_reps is not None else self.reps
                kernel, num_params = self._build_fixed_kernel(num_qubits, pairs, reps)
                self.ansatz_cache[cache_key] = (kernel, num_params)

            return self.ansatz_cache[cache_key]

    def _select_adaptive_depth(self, hamiltonian, num_qubits, entanglement_map, energy_threshold):
        """
        Change #3, 'adaptive_depth' mode: tries reps=1,2,...,max_reps, running a cheap trial
        optimization (COBYLA, a small maxiter) at each depth, and stops growing as soon as a
        trial's best energy clears energy_threshold -- i.e. this depth is already enough to
        certify the block isn't PSD, so there's no need to pay for a deeper (more expensive)
        circuit. Falls back to max_reps if nothing clears the threshold. Trial optimizations are
        cheap and thrown away; only the *depth* they find is kept, and the real solve() below
        does the full optimization at that depth from scratch (or from a warm start, if given).
        """
        from vqe_optimizers import COBYLA  # local import: keeps VQESubroutine importable standalone
        pairs = (list(zip(range(num_qubits - 1), range(1, num_qubits)))
                 if entanglement_map == 'linear' else entanglement_map)
        best_depth = self.max_reps
        for trial_reps in range(1, self.max_reps + 1):
            kernel, num_params = self._build_fixed_kernel(num_qubits, pairs, trial_reps)
            x0 = np.random.uniform(0, 2 * np.pi, num_params)
            trial_opt = COBYLA(maxiter=25)
            result = trial_opt.minimize(lambda p: self._observe_one(kernel, hamiltonian, p), x0)
            if result.fun < energy_threshold:
                best_depth = trial_reps
                break
        return best_depth

    # ------------------------------------------------------------------
    # 'adapt_pool' ansatz: Change #3's ADAPT-VQE-lite, Hamiltonian-aware construction.
    # ------------------------------------------------------------------
    def _build_adapt_pool_ansatz(self, hamiltonian, terms, num_qubits, energy_threshold):
        """
        Lightweight ADAPT-VQE. Pool operators are exp(i*theta*P) for the Hamiltonian's own
        (highest-|coefficient|, up to adapt_pool_size) Pauli terms -- literally Hamiltonian-aware,
        as opposed to a fixed hardware-efficient layer pattern. Growth is greedy: at each step,
        every remaining pool operator is screened by a central-difference energy gradient at
        theta=0 (all screening evaluations for a step are dispatched together via
        cudaq.observe_async and gathered afterward, so the pool scan itself uses concurrent GPU
        dispatch -- Change #2), the largest-|gradient| operator is appended with a fresh
        parameter, and growth stops once max_reps operators have been added or the best
        remaining gradient is below adapt_grad_tol.

        This does not use the ansatz_cache: pool selection is inherently specific to the
        Hamiltonian being separated (different candidate matrices produce different Pauli terms
        even for the same block shape), so there's little to gain from caching structure across
        calls the way 'fixed'/'adaptive_depth' do across iterations.

        Pool construction note: these SDP blocks are always real symmetric matrices, so their
        Pauli decomposition only ever contains terms with an *even* number of Y factors (Y is
        the only imaginary Pauli matrix; Tr(P@M) is real for Hermitian P,M in general, but for
        real M specifically it's nonzero only when P itself works out real, which requires an
        even Y-count). Using one of those terms directly as an exp(i*theta*P) generator would be
        a dead end: for real symmetric H, real symmetric P (even-Y), and the real reference
        state |++...+>, the energy gradient at theta=0 is <psi|[H,P]|psi>, and [H,P] of two real
        symmetric matrices is real *antisymmetric* -- for which v^T A v = 0 for any real v. That
        gradient is exactly zero, not just small, for every even-Y pool candidate (confirmed
        empirically before writing this). The fix is to grow the pool from *odd*-Y variants of
        the Hamiltonian's real terms instead (flip one non-identity factor on each term to Y):
        odd-Y Paulis are purely imaginary, so i*P is real *antisymmetric* -- a genuine real
        rotation generator, giving a nonzero gradient. This keeps the pool Hamiltonian-derived
        (still "Hamiltonian-aware", per the task description) while actually being usable.
        """
        real_terms = sorted(((s, c) for s, c in terms if set(s) != {"I"}), key=lambda t: -abs(t[1]))
        pool_words = []
        seen = set()
        for pauli_string, _coeff in real_terms:
            word = _reorder_pauli_word(pauli_string, num_qubits)  # qubit-0-first, for exp_pauli
            for i, ch in enumerate(word):
                if ch in ("I", "Y"):
                    continue
                variant = word[:i] + "Y" + word[i + 1:]
                if variant not in seen:
                    seen.add(variant)
                    pool_words.append(variant)
                if len(pool_words) >= self.adapt_pool_size:
                    break
            if len(pool_words) >= self.adapt_pool_size:
                break

        kernel, thetas = cudaq.make_kernel(list)
        qubits = kernel.qalloc(num_qubits)
        for q in range(num_qubits):  # break |0...0>'s trivial symmetry before the first operator
            kernel.h(qubits[q])

        selected_words = []
        num_params = 0
        eps = 1e-3
        available = list(range(len(pool_words)))

        while available and num_params < self.max_reps:
            eps_grad = eps
            # Screen every remaining candidate operator's gradient at theta=0, all in one
            # async-dispatched batch (Change #2's concurrent evaluation across the pool scan).
            plus_futures, minus_futures = [], []
            cudaq.set_target(self.target_name)
            for idx in available:
                cand_kernel, cand_thetas = cudaq.make_kernel(list)
                cand_qubits = cand_kernel.qalloc(num_qubits)
                for q in range(num_qubits):
                    cand_kernel.h(cand_qubits[q])
                for w, val in zip(selected_words, self._fitted_angles):
                    cand_kernel.exp_pauli(val, cand_qubits, w)
                cand_kernel.exp_pauli(cand_thetas[0], cand_qubits, pool_words[idx])
                plus_futures.append((idx, cudaq.observe_async(cand_kernel, hamiltonian, [eps_grad])))
                minus_futures.append((idx, cudaq.observe_async(cand_kernel, hamiltonian, [-eps_grad])))

            grads = {}
            plus_vals = {idx: fut.get().expectation() for idx, fut in plus_futures}
            minus_vals = {idx: fut.get().expectation() for idx, fut in minus_futures}
            for idx in available:
                grads[idx] = abs((plus_vals[idx] - minus_vals[idx]) / (2 * eps_grad))

            best_idx = max(available, key=lambda i: grads[i])
            if grads[best_idx] < self.adapt_grad_tol:
                break

            selected_words.append(pool_words[best_idx])
            available.remove(best_idx)
            num_params += 1

            # Re-fit all angles selected so far with a short COBYLA run, so the next round's
            # gradient screening (and the eventual full solve()) starts from a sensible point.
            from vqe_optimizers import COBYLA
            fit_kernel, fit_thetas = cudaq.make_kernel(list)
            fit_qubits = fit_kernel.qalloc(num_qubits)
            for q in range(num_qubits):
                fit_kernel.h(fit_qubits[q])
            for i, w in enumerate(selected_words):
                fit_kernel.exp_pauli(fit_thetas[i], fit_qubits, w)
            x0 = np.zeros(num_params) if not hasattr(self, "_fitted_angles") or not self._fitted_angles \
                else np.concatenate([self._fitted_angles, [0.0]])
            fit_result = COBYLA(maxiter=50).minimize(
                lambda p: self._observe_one(fit_kernel, hamiltonian, p), x0)
            self._fitted_angles = list(fit_result.x)

        # Build the final kernel with exactly the selected operators, in order.
        final_kernel, final_thetas = cudaq.make_kernel(list)
        final_qubits = final_kernel.qalloc(num_qubits)
        for q in range(num_qubits):
            final_kernel.h(final_qubits[q])
        for i, w in enumerate(selected_words):
            final_kernel.exp_pauli(final_thetas[i], final_qubits, w)

        return final_kernel, len(selected_words)

    def _observe_one(self, kernel, hamiltonian, params):
        cudaq.set_target(self.target_name)
        observe_kwargs = {}
        if self.noise_model is not None:
            observe_kwargs["noise_model"] = self.noise_model
            observe_kwargs["num_trajectories"] = self.num_trajectories
        return cudaq.observe(kernel, hamiltonian, list(params), **observe_kwargs).expectation()

    def _batch_observe(self, kernel, hamiltonian, param_list):
        """
        Change #2 (Improved GPU Utilization): dispatches every parameter vector in param_list as
        a separate cudaq.observe_async call *before* blocking on any of their results, so all of
        them are queued on the GPU together instead of the classical optimizer waiting on one
        full round-trip per evaluation. This is what SPSA/QNSPSA's paired perturbation
        evaluations and L-BFGS-B/SLSQP's finite-difference gradients are routed through (see
        vqe_optimizers.py's batch_fun usage) -- the GPU stops sitting idle between sequential
        single-point calls, which was the exact gap the task description flagged.
        """
        cudaq.set_target(self.target_name)
        observe_kwargs = {}
        if self.noise_model is not None:
            observe_kwargs["noise_model"] = self.noise_model
            observe_kwargs["num_trajectories"] = self.num_trajectories
        futures = [cudaq.observe_async(kernel, hamiltonian, list(p), **observe_kwargs) for p in param_list]
        return [f.get().expectation() for f in futures]

    def _bind_circuit_and_extract_state(self, kernel, params):
        ideal_target = 'nvidia'
        current_target = cudaq.get_target().name
        try:
            if current_target != ideal_target:
                cudaq.set_target(ideal_target)
            state = cudaq.get_state(kernel, list(params))
            return np.array(state, copy=True).real
        finally:
            if current_target != ideal_target:
                cudaq.set_target(current_target)

    def _check_diversity_and_save_state(self, state, magnitude, collected_state_vectors,
                                         collected_magnitudes, global_pool, overlap_threshold=0.9,
                                         force_collect=False):
        """
        Same diversity-filter role as before, now also tracking each collected vector's
        violation magnitude (Change #5's "ranking cuts according to violation magnitude" --
        the caller sorts collected_state_vectors/collected_magnitudes together afterward).
        """
        is_diverse = True
        if global_pool is not None and len(global_pool) > 0:
            overlaps = np.abs(np.dot(global_pool, state))
            is_diverse = bool(np.all(overlaps < overlap_threshold))

        if is_diverse or force_collect:
            collected_state_vectors.append(state)
            collected_magnitudes.append(magnitude)
        if is_diverse and global_pool is not None:
            global_pool.append(state)

    def solve(self, matrix, optimizer, initial_point=None, energy_threshold=-1.0e-6, overlap_threshold=0.9,
              global_pool=None, collect_multiple_vectors=False, original_dim=None, opt_state=None,
              adaptive_stop=True):
        """
        global_pool: list maintained by the outer algorithm, seeding future diversity comparisons.
        original_dim: dimension before zero-padding, for the leaked-amplitude check.
        opt_state: optimizer-internal state (currently QNSPSA's metric-tensor average) carried in
            from a previous call for the *same* SDP block, if the caller is managing that
            explicitly (Change #4). If None, falls back to this instance's own memory for the
            ansatz shape involved.
        adaptive_stop: Change #5's "adaptive stopping criteria for the VQE" -- when True (and the
            optimizer honors it, i.e. all six in vqe_optimizers.py do), the optimizer stops as
            soon as its running best energy clears energy_threshold, rather than always running
            to maxiter. Set False to force a full run (e.g. for the optimizer benchmark, where
            comparing full trajectories is the point).
        """
        num_qubits = int(np.log2(matrix.shape[0]))
        try:
            terms = _matrix_to_pauli_terms(matrix, atol=1e-12)
        except Exception as e:
            raise TypeError(f"Input cannot be converted to a Pauli-decomposed operator: {e}") from e
        num_pauli_terms = len(terms)
        if num_pauli_terms == 0:
            print("Operator is empty after conversion. Skipping.")
            print("Matrix = ", matrix)
            return None, initial_point, [], 0.0, 0.0, num_pauli_terms

        ent_map = self._get_entanglement(terms, num_qubits)
        hamiltonian = _pauli_terms_to_spin_op(terms, num_qubits)

        # --- Change #3: pick/build the ansatz for this call ---
        if self.ansatz_mode == 'adapt_pool':
            self._fitted_angles = []
            kernel, num_params = self._build_adapt_pool_ansatz(hamiltonian, terms, num_qubits, energy_threshold)
            cache_key = None  # not cached; see _build_adapt_pool_ansatz docstring
        else:
            if self.ansatz_mode == 'adaptive_depth':
                chosen_depth = self._select_adaptive_depth(hamiltonian, num_qubits, ent_map, energy_threshold)
                kernel, num_params = self._get_ansatz(num_qubits, ent_map, forced_reps=chosen_depth)
            else:
                kernel, num_params = self._get_ansatz(num_qubits, ent_map)
            map_key = 'linear' if ent_map == 'linear' else tuple(ent_map)
            cache_key = (num_qubits, map_key, self.ansatz_mode)

        vqe_trajectory = {"energy": [], "eval_counts": []}
        collected_state_vectors = []
        collected_magnitudes = []
        evaluation_count = 0

        def cost_func(ansatz_params):
            nonlocal evaluation_count
            evaluation_count += 1
            energy = self._observe_one(kernel, hamiltonian, ansatz_params)
            vqe_trajectory["energy"].append(float(energy))
            vqe_trajectory["eval_counts"].append(evaluation_count)
            if collect_multiple_vectors and energy < energy_threshold:
                ideal_state = self._bind_circuit_and_extract_state(kernel, ansatz_params)
                self._check_diversity_and_save_state(ideal_state, float(energy), collected_state_vectors,
                                                      collected_magnitudes, global_pool, overlap_threshold)
            return energy

        def cost_func_batch(param_list):
            """Change #2: routes SPSA/QNSPSA's perturbation pairs and L-BFGS-B/SLSQP's
            finite-difference gradient perturbations through one concurrent GPU dispatch."""
            nonlocal evaluation_count
            energies = self._batch_observe(kernel, hamiltonian, param_list)
            for params, energy in zip(param_list, energies):
                evaluation_count += 1
                vqe_trajectory["energy"].append(float(energy))
                vqe_trajectory["eval_counts"].append(evaluation_count)
                if collect_multiple_vectors and energy < energy_threshold:
                    ideal_state = self._bind_circuit_and_extract_state(kernel, params)
                    self._check_diversity_and_save_state(ideal_state, float(energy), collected_state_vectors,
                                                          collected_magnitudes, global_pool, overlap_threshold)
            return energies

        def state_fn(params):
            return self._bind_circuit_and_extract_state(kernel, params)

        # --- Change #4: informed cold-start fallback + optimizer-state reuse ---
        with self._lock:
            memory = self.param_memory.get(cache_key) if cache_key is not None else None
            if opt_state is None and cache_key is not None:
                opt_state = self.opt_state_memory.get(cache_key)

        if initial_point is None:
            if memory:
                initial_point = np.mean(np.stack(memory), axis=0)
            else:
                initial_point = np.random.uniform(0, 2 * np.pi, num_params)
        elif len(initial_point) != num_params:
            # Shape changed since this initial_point was captured (e.g. pruning shrank the
            # ansatz, or adaptive_depth/adapt_pool picked a different depth this call) --
            # a stale warm start can't be reused, so fall back to memory/cold start instead of
            # raising, since the caller (the cutting-plane loop) doesn't know the ansatz shape.
            initial_point = (np.mean(np.stack(memory), axis=0) if memory
                              else np.random.uniform(0, 2 * np.pi, num_params))

        optimizer.settings["adaptive_stop_below"] = energy_threshold if adaptive_stop else None

        minimize_kwargs = {}
        if getattr(optimizer, "supports_batch", False):
            minimize_kwargs["batch_fun"] = cost_func_batch
        if getattr(optimizer, "needs_state_fn", False):
            minimize_kwargs["state_fn"] = state_fn
        minimize_kwargs["state"] = opt_state

        start_time = time.time()
        result = optimizer.minimize(cost_func, initial_point, **minimize_kwargs)
        solve_time = time.time() - start_time
        # Trust our own count of cost_func/cost_func_batch invocations over the optimizer's
        # self-reported nfev: scipy-backed optimizers report their *internal* nfev, which can
        # undercount once the adaptive-stop callback exits early (the callback's own extra
        # fun(xk) probe doesn't line up 1:1 with scipy's internal per-iteration evaluation
        # count), and this evaluation_count is exact by construction -- it's incremented exactly
        # once per real circuit evaluation, inside cost_func/cost_func_batch themselves.
        vqe_trajectory["nfev"] = evaluation_count
        vqe_trajectory["opt_state"] = result.state
        vqe_trajectory["stopped_early"] = result.stopped_early
        vqe_trajectory["convergence_history"] = result.history

        optimal_state = self._bind_circuit_and_extract_state(kernel, result.x)

        if not collected_state_vectors and result.fun < energy_threshold:
            self._check_diversity_and_save_state(optimal_state, float(result.fun), collected_state_vectors,
                                                  collected_magnitudes, global_pool, overlap_threshold,
                                                  force_collect=True)

        # Change #5: rank collected cuts by violation magnitude (most negative first), so the
        # caller's SOC-cut selection (which wants the *most* violated vectors) and any cut-count
        # limiting downstream both see the strongest witnesses first, regardless of the order VQE
        # happened to discover them in during the optimization trajectory.
        if collected_state_vectors:
            order = np.argsort(collected_magnitudes)  # ascending: most negative (most violated) first
            collected_state_vectors = [collected_state_vectors[i] for i in order]

        # --- Change #4: remember this shape's result for next time ---
        with self._lock:
            if cache_key is not None:
                self.param_memory.setdefault(cache_key, deque(maxlen=self._param_memory_size)).append(result.x)
                if result.state is not None:
                    self.opt_state_memory[cache_key] = result.state

        # --- Change #5 support / circuit pruning (Change #3/#5 overlap): flag a near-zero final
        # rotation layer so the *next* build for this shape drops it.
        if self.enable_pruning and self.ansatz_mode == 'fixed' and cache_key is not None:
            final_layer = np.asarray(result.x[-num_qubits:])
            wrapped = np.abs(np.mod(final_layer + np.pi, 2 * np.pi) - np.pi)
            if np.all(wrapped < self.pruning_threshold) and num_params > num_qubits:
                with self._lock:
                    self.prunable_cache_keys.add(cache_key)

        leaked_amplitude = float(np.sum(optimal_state[original_dim:] ** 2)) if original_dim is not None else 0.0

        return vqe_trajectory, result.x, collected_state_vectors, solve_time, leaked_amplitude, num_pauli_terms
