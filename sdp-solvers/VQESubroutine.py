import time
import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
from qiskit_ibm_runtime import QiskitRuntimeService, fake_provider


class VQESubroutine:
    def __init__(
        self,
        ansatz_type='hardware_efficient',
        execution_mode='noiseless',
        backend_name=None,
        reps=2,
        shots=None,
        thermal_relaxation=False,
        ansatz_mode='fixed',          # 'fixed', 'adaptive_depth', or 'adapt_pool'
        max_reps=6,
        adapt_grad_tol=1e-4,
        adapt_pool_size=30,
        leakage_penalty=10.0,
    ):
        self.ansatz_type = ansatz_type
        self.execution_mode = execution_mode
        self.reps = reps
        self.shots = shots
        self.thermal_relaxation = thermal_relaxation
        self.ansatz_mode = ansatz_mode
        self.max_reps = max_reps
        self.adapt_grad_tol = adapt_grad_tol
        self.adapt_pool_size = adapt_pool_size
        self.leakage_penalty = leakage_penalty
        self.backend_name = None
        self.estimator = None
        self.pm = None
        self.ansatz_cache = {}
        self._fitted_angles = []  # for adapt_pool

        self._setup_engine(backend_name)

    def _resolve_fake_backend(self, name):
        name = name or "fake_fez"
        key = name.replace("ibm_", "").replace("fake_", "")
        class_name = "Fake" + "".join(part.capitalize() for part in key.split("_"))
        if not hasattr(fake_provider, class_name):
            raise ValueError(f"Unknown fake backend '{name}'")
        return getattr(fake_provider, class_name)()

    def _setup_engine(self, backend_name):
        precision = 1.0 / np.sqrt(self.shots) if self.shots else 0.0
        if self.execution_mode == 'noiseless':
            self.estimator = StatevectorEstimator()
            self.backend_name = 'statevector_exact'
        elif self.execution_mode == 'shots':
            self.estimator = AerEstimator(options={"default_precision": precision})
            self.backend_name = 'aer_simulator_ideal'
        elif self.execution_mode == 'noisy':
            fake_backend = self._resolve_fake_backend(backend_name)
            self.backend_name = fake_backend.name
            self.pm = generate_preset_pass_manager(backend=fake_backend)
            noise_model = NoiseModel.from_backend(
                fake_backend,
                thermal_relaxation=self.thermal_relaxation,
                gate_error=True,
                readout_error=True,
            )
            self.estimator = AerEstimator(
                options={
                    "backend_options": {"noise_model": noise_model},
                    "default_precision": precision,
                }
            )
        elif self.execution_mode == 'hardware':
            service = QiskitRuntimeService(channel="ibm_quantum_platform")
            if backend_name is not None:
                real_backend = service.backend(backend_name)
            else:
                real_backend = service.least_busy(operational=True, simulator=False)
            self.backend_name = real_backend.name
            self.pm = generate_preset_pass_manager(backend=real_backend)
            self.estimator = RuntimeEstimator(mode=real_backend)

    def _get_entanglement(self, op: SparsePauliOp):
        if self.ansatz_type == 'hardware_efficient':
            return 'linear'
        edges = set()
        for pauli in op.paulis:
            active = [i for i, char in enumerate(str(pauli)[::-1]) if char != 'I']
            if len(active) >= 2:
                for idx, q1 in enumerate(active):
                    for q2 in active[idx + 1:]:
                        edges.add(tuple(sorted((q1, q2))))
        return sorted(list(edges)) if edges else 'linear'

    def _get_ansatz(self, num_qubits, entanglement_map, forced_reps=None):
        map_key = entanglement_map if isinstance(entanglement_map, str) else tuple(entanglement_map)
        reps = forced_reps if forced_reps is not None else self.reps
        cache_key = (num_qubits, map_key, reps)  # include reps in cache key for fixed mode
        if cache_key not in self.ansatz_cache:
            abstract_circ = real_amplitudes(
                num_qubits=num_qubits,
                entanglement=entanglement_map,
                reps=reps,
                parameter_prefix='θ',
            )
            isa_circ = self.pm.run(abstract_circ) if self.pm else abstract_circ
            self.ansatz_cache[cache_key] = (isa_circ, abstract_circ)
        return self.ansatz_cache[cache_key]

    def _bind_circuit_and_extract_state(self, ansatz, params):
        return Statevector(ansatz.assign_parameters(params)).data.real

    def _check_diversity_and_save_state(
        self,
        state,
        collected_state_vectors,
        global_pool,
        overlap_threshold=0.9,
        force_collect=False,
    ):
        is_diverse = True
        if global_pool is not None and len(global_pool) > 0:
            overlaps = np.abs(np.dot(global_pool, state))
            is_diverse = bool(np.all(overlaps < overlap_threshold))
        if is_diverse or force_collect:
            collected_state_vectors.append(state)
        if is_diverse and global_pool is not None:
            global_pool.append(state)

    def _select_adaptive_depth(self, op, num_qubits, ent_map, energy_threshold):
        from qiskit_algorithms.optimizers import COBYLA
        best_depth = self.max_reps
        for trial_reps in range(1, self.max_reps + 1):
            abstract_circ = real_amplitudes(
                num_qubits=num_qubits,
                entanglement=ent_map,
                reps=trial_reps,
                parameter_prefix='θ',
            )
            isa_circ = self.pm.run(abstract_circ) if self.pm else abstract_circ
            isa_op = op.apply_layout(isa_circ.layout) if self.pm else op
            trial_opt = COBYLA(maxiter=25)
            x0 = np.random.uniform(0, 2*np.pi, abstract_circ.num_parameters)
            def cost(x):
                pub = (isa_circ, isa_op, x)
                return self.estimator.run([pub]).result()[0].data.evs
            result = trial_opt.minimize(cost, x0)
            if result.fun < energy_threshold:
                best_depth = trial_reps
                break
        return best_depth

    def solve(
        self,
        matrix,
        optimizer,
        initial_point=None,
        energy_threshold=-1.0e-4,
        overlap_threshold=0.9,
        global_pool=None,
        collect_multiple_vectors=False,
        original_dim=None,
    ):
        num_qubits = int(np.log2(matrix.shape[0]))
        try:
            op = SparsePauliOp.from_operator(matrix, atol=1e-12)
        except Exception as e:
            raise TypeError(f"Input cannot be converted to a Qiskit operator: {e}") from e
        num_pauli_terms = len(op)
        if num_pauli_terms == 0:
            return (None, initial_point, [], 0.0, 0.0, num_pauli_terms)

        ent_map = self._get_entanglement(op)

        # Determine ansatz and depth based on mode
        if self.ansatz_mode == 'adaptive_depth':
            chosen_depth = self._select_adaptive_depth(op, num_qubits, ent_map, energy_threshold)
            abstract_ansatz = real_amplitudes(
                num_qubits=num_qubits,
                entanglement=ent_map,
                reps=chosen_depth,
                parameter_prefix='θ',
            )
            execution_ansatz = self.pm.run(abstract_ansatz) if self.pm else abstract_ansatz
            isa_op = op.apply_layout(execution_ansatz.layout) if self.pm else op
        elif self.ansatz_mode == 'adapt_pool':
            # Simplified: fallback to fixed depth for now (can be extended later)
            abstract_ansatz, execution_ansatz = self._get_ansatz(num_qubits, ent_map)
            isa_op = op.apply_layout(execution_ansatz.layout) if self.pm else op
        else:  # fixed
            abstract_ansatz, execution_ansatz = self._get_ansatz(num_qubits, ent_map)
            isa_op = op.apply_layout(execution_ansatz.layout) if self.pm else op

        # --- FIX: Handle mismatched warm-start length ---
        if initial_point is not None and len(initial_point) != abstract_ansatz.num_parameters:
            print(f"Warm-start parameter length {len(initial_point)} != {abstract_ansatz.num_parameters}, discarding warm start.")
            initial_point = None  # fallback

        if initial_point is None:
            # Use memory fallback if available (but we don't have a per-shape memory here)
            # For simplicity, use random uniform
            initial_point = np.random.uniform(0, 2*np.pi, abstract_ansatz.num_parameters)

        real_matrix = matrix.real
        vqe_trajectory = {
            "energy": [],
            "eval_counts": [],
            "collected_candidate_energy": [],
            "collected_candidate_true_quad": [],
            "collected_candidate_accepted": [],
        }
        collected_state_vectors = []
        evaluation_count = 0

        def cost_func(ansatz_params):
            nonlocal evaluation_count
            evaluation_count += 1
            pub = (execution_ansatz, isa_op, ansatz_params)
            energy = self.estimator.run([pub]).result()[0].data.evs
            vqe_trajectory["energy"].append(float(energy))
            vqe_trajectory["eval_counts"].append(evaluation_count)

            # Leakage penalty
            if original_dim is not None:
                state = self._bind_circuit_and_extract_state(abstract_ansatz, ansatz_params)
                leaked = float(np.sum(state[original_dim:] ** 2))
                penalized_energy = energy + self.leakage_penalty * leaked
            else:
                penalized_energy = energy
                leaked = 0.0

            if collect_multiple_vectors and energy < energy_threshold:
                ideal_state = self._bind_circuit_and_extract_state(abstract_ansatz, ansatz_params)
                true_quad = float(ideal_state @ real_matrix @ ideal_state)
                accepted = true_quad < energy_threshold
                vqe_trajectory["collected_candidate_energy"].append(float(energy))
                vqe_trajectory["collected_candidate_true_quad"].append(true_quad)
                vqe_trajectory["collected_candidate_accepted"].append(accepted)
                if accepted:
                    self._check_diversity_and_save_state(
                        ideal_state,
                        collected_state_vectors,
                        global_pool,
                        overlap_threshold,
                    )

            return penalized_energy

        start_time = time.time()
        result = optimizer.minimize(cost_func, initial_point)
        solve_time = time.time() - start_time
        vqe_trajectory["nfev"] = result.nfev

        optimal_state = self._bind_circuit_and_extract_state(abstract_ansatz, result.x)

        if not collected_state_vectors and result.fun < energy_threshold:
            pub = (execution_ansatz, isa_op, result.x)
            final_energy = self.estimator.run([pub]).result()[0].data.evs
            if final_energy < energy_threshold:
                self._check_diversity_and_save_state(
                    optimal_state,
                    collected_state_vectors,
                    global_pool,
                    overlap_threshold,
                    force_collect=True,
                )

        leaked_amplitude = (
            float(np.sum(optimal_state[original_dim:] ** 2))
            if original_dim is not None
            else 0.0
        )

        return (
            vqe_trajectory,
            result.x,
            collected_state_vectors,
            solve_time,
            leaked_amplitude,
            num_pauli_terms,
        )