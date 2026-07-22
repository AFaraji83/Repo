import time

import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
from qiskit_ibm_runtime import QiskitRuntimeService


class VQESubroutine:
    def __init__(self, ansatz_type='hardware_efficient', execution_mode='noiseless', backend_name=None, reps=2):
        """
        ansatz_type: 'hardware_efficient' or 'sparsity_aware'
        execution_mode: 'noiseless', 'noisy', or 'hardware'
        backend_name: e.g., 'ibm_brisbane' (required for noisy simulation/hardware execution)
        reps: Number of layers in the ansatz (for both HEA and SA)
        """
        self.ansatz_type = ansatz_type
        self.execution_mode = execution_mode
        self.reps = reps
        
        self.estimator = None
        self.pm = None
        self.ansatz_cache = {} # Topology/Ansatz caching layer
        
        self._setup_engine(backend_name)

    def _setup_engine(self, backend_name):
        """Initializes the correct V2 Estimator and PassManager."""
        if self.execution_mode == 'noiseless':
            self.estimator = StatevectorEstimator()
            
        elif self.execution_mode in ['noisy', 'hardware']:
            # Declare the channel and instance you want to use for noisy/hardware execution
            service = QiskitRuntimeService(channel="ibm_quantum_platform")
            
            # Use provided backend_name, or find the least busy one
            if backend_name is not None:
                real_backend = service.backend(backend_name)
                print(f"Using specified backend: {real_backend.name}")
            else:
                # Select the best available physical chip
                print("Finding the least busy hardware for the noise model...")
                real_backend = service.least_busy(operational=True, simulator=False)
                print(f"Auto-selected backend: {real_backend.name}")
            
            # Generate the PassManager for transpilation to the backend's ISA, which is needed for both noisy simulation and hardware execution to ensure the ansatz is compatible with the backend's qubit connectivity and gate set.
            self.pm = generate_preset_pass_manager(backend=real_backend)
            
            if self.execution_mode == 'noisy':
                noise_model = NoiseModel.from_backend(
                    real_backend,
                    thermal_relaxation=False, # Optionally exclude thermal relaxation noise to focus on gate and readout errors, which are often more dominant in current devices.
                    gate_error=True,          # Includes gate errors based on the backend's calibration data
                    readout_error=True        # Includes readout errors based on the backend's calibration data
                )
                self.estimator = AerEstimator(
                    options={"backend_options": {"noise_model": noise_model}}
                )
            else: # hardware
                self.estimator = RuntimeEstimator(mode=real_backend)

    def _get_entanglement(self, op: SparsePauliOp):
        """Determines the entanglement map based on the ansatz type."""
        if self.ansatz_type == 'hardware_efficient':
            return 'linear' # Or 'full', ignoring matrix sparsity
            
        # Sparsity-Aware (SA) Logic
        edges = set()
        for pauli in op.paulis:
            active = [i for i, char in enumerate(str(pauli)[::-1]) if char != 'I']
            if len(active) >= 2:
                for idx, q1 in enumerate(active):
                    for q2 in active[idx + 1:]:
                        edges.add(tuple(sorted((q1, q2))))
        return sorted(list(edges)) if edges else 'linear'

    def _get_ansatz(self, num_qubits, entanglement_map):
        """Builds, transpiles (if needed), and caches the ansatz."""
        # Use 'linear' as map key if HEA, otherwise tuple of the SA map
        map_key = entanglement_map if isinstance(entanglement_map, str) else tuple(entanglement_map)
        # Combine num_qubits and map_key into a single tuple to use as the cache key for both HEA and SA cases, ensuring that different qubit counts and entanglement maps are cached separately.
        cache_key = (num_qubits, map_key)
    
        # Check cache first to avoid redundant circuit generation and transpilation
        if cache_key not in self.ansatz_cache:
            abstract_circ = real_amplitudes(
                num_qubits=num_qubits, 
                entanglement=entanglement_map, 
                reps=self.reps, 
                parameter_prefix='θ'
            )
            # Transpile to ISA if we have a PassManager (Noisy/Hardware)
            isa_circ = self.pm.run(abstract_circ) if self.pm else abstract_circ
            # Cache BOTH the hardware-ready circuit and the mathematical ideal
            self.ansatz_cache[cache_key] = (isa_circ, abstract_circ)

        return self.ansatz_cache[cache_key]

    def _bind_circuit_and_extract_state(self, ansatz, params):
        """
        Binds the specified parameters to the given ansatz and extracts the statevector as a 
        real-valued numpy array for diversity checks.
        """
        return Statevector(ansatz.assign_parameters(params)).data.real

    def _check_diversity_and_save_state(self, state, collected_state_vectors, global_pool, overlap_threshold=0.9, force_collect=False):
        """
        Checks if a state vector is diverse against the global pool. Diverse states are appended to both
        collected_state_vectors (this run's cuts) and global_pool (seeds future diversity comparisons).
        When force_collect=True, the state is appended to collected_state_vectors even if it fails the
        diversity check (it is still only added to global_pool when diverse). This is used to guarantee
        that a genuine PSD violation is always reported as a cut, even when it duplicates a vector found
        in an earlier iteration -- the diversity filter exists to avoid redundant cuts, not to decide
        whether a matrix is PSD, so it must never be the sole reason a violation goes unreported.
        """
        is_diverse = True
        if global_pool is not None and len(global_pool) > 0:
            # Compute inner product against ALL vectors in pool
            overlaps = np.abs(np.dot(global_pool, state))
            # If all overlaps are strictly less than the threshold, the state vector passes the diversity check
            is_diverse = bool(np.all(overlaps < overlap_threshold))

        if is_diverse or force_collect:
            collected_state_vectors.append(state)
        if is_diverse and global_pool is not None:
            global_pool.append(state)
                            
    def solve(self, matrix, optimizer, initial_point=None, energy_threshold=-1.0e-6, overlap_threshold=0.9, global_pool=None, collect_multiple_vectors=False, original_dim=None):
        """
        global_pool: A list maintained by your outer algorithm
                     to store diverse, low-energy statevectors.
        original_dim: dimension of the matrix before zero-padding to the next power of two
                     (used to compute the leaked amplitude sitting in the padded subspace).
        """
        num_qubits = int(np.log2(matrix.shape[0]))
        # Convert the matrix to a Qiskit operator (SparsePauliOp)
        try:
            op = SparsePauliOp.from_operator(matrix, atol=1e-12)
        except Exception as e:
            raise TypeError(f"Input cannot be converted to a Qiskit operator: {e}") from e
        if len(op.coeffs) == 0:
            print("Operator is empty after conversion. Skipping.")
            print("Matrix = ", matrix)
            return None, initial_point, [], 0.0, 0.0 # Skip the optimization and return empty trajectory, the original initial point, an empty list of collected state vectors, zero solve time and zero leaked amplitude if operator is empty
            print(f"Number of Pauli terms: {len(op)}")
        
        # Get the entanglement map and corresponding ansatz (with caching)
        ent_map = self._get_entanglement(op)
        execution_ansatz, abstract_ansatz = self._get_ansatz(num_qubits, ent_map)

        # For noisy/hardware execution, we need to ensure the operator is transformed to match the qubit layout of the transpiled ansatz.
        isa_op = op.apply_layout(execution_ansatz.layout) if self.pm else op
                
        vqe_trajectory = {"energy": [], "eval_counts": []} # Local trajectory for this run
        collected_state_vectors = [] # To track which state vectors were collected during this run
        evaluation_count = 0 # Local counter for the number of cost function evaluations during this run
        
        # Define the cost function that the optimizer will minimize, which evaluates the energy for given parameters and tracks the trajectory.
        def cost_func(ansatz_params):

            nonlocal evaluation_count
            evaluation_count += 1

            # Create the parameterized circuit for this evaluation
            pub = (execution_ansatz, isa_op, ansatz_params)
            energy = self.estimator.run([pub]).result()[0].data.evs

            # Track trajectory
            vqe_trajectory["energy"].append(float(energy))
            vqe_trajectory["eval_counts"].append(evaluation_count)
            
            if collect_multiple_vectors and energy < energy_threshold:
                # Compute the statevector for this parameter set
                ideal_state = self._bind_circuit_and_extract_state(abstract_ansatz, ansatz_params)
                # Save the state to the global pool if it passes the diversity check
                self._check_diversity_and_save_state(ideal_state, collected_state_vectors, global_pool, overlap_threshold)

            return energy

        # Warm start logic: If an initial point is provided, we use it.
        if initial_point is None:
            # Fallback for the very first VQE call in the outer loop
            initial_point = np.random.uniform(0, 2*np.pi, abstract_ansatz.num_parameters)
        elif len(initial_point) != abstract_ansatz.num_parameters:
            raise ValueError("Warm start parameter array size does not match ansatz.")

        # Run the optimizer to minimize the cost function (energy) to find the optimal parameters
        start_time = time.time()
        result = optimizer.minimize(cost_func, initial_point)
        solve_time = time.time() - start_time
        vqe_trajectory["nfev"] = result.nfev

        # Compute the optimal statevector using the optimal parameters found (needed for the leaked-amplitude
        # check below regardless of collect_multiple_vectors, and for the force-collect check below).
        optimal_state = self._bind_circuit_and_extract_state(abstract_ansatz, result.x)

        # Guarantee that a genuine PSD violation is always reported as a cut. If nothing was collected
        # during the trajectory (e.g. every candidate duplicated an earlier cut and failed the diversity
        # check, or collect_multiple_vectors is False so no per-iteration check ran at all) but the
        # optimizer's final energy is still below threshold, force-collect it. Without this, a real
        # non-PSD witness that merely isn't "novel" relative to the pool would be silently dropped, and
        # the outer cutting-plane loop would misread the empty cut list as a false PSD certificate.
        if not collected_state_vectors and result.fun < energy_threshold:
            self._check_diversity_and_save_state(optimal_state, collected_state_vectors, global_pool, overlap_threshold, force_collect=True)

        # Fraction of probability mass sitting in the zero-padded junk subspace (indices >= original_dim).
        leaked_amplitude = float(np.sum(optimal_state[original_dim:] ** 2)) if original_dim is not None else 0.0

        # Return the trajectory, optimal parameters, collected state vectors, solve time and leaked amplitude so the outer loop can save them!
        return vqe_trajectory, result.x, collected_state_vectors, solve_time, leaked_amplitude