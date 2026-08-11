"""
Optimizer library for the VQE separation oracle.

Change #1 (Optimizer Benchmarking): six interchangeable optimizers behind one interface --
COBYLA, Powell, SPSA, QNSPSA, L-BFGS-B, SLSQP -- so optimizer_benchmark.py can swap between
them on identical instances and compare convergence rate, runtime, nfev, and oracle accuracy.

Change #2 (GPU utilization): every optimizer here that needs more than one energy per step
(SPSA/QNSPSA's perturbation pairs, L-BFGS-B/SLSQP's finite-difference gradients) requests
those evaluations through a *single* `batch_fun(list_of_param_vectors) -> list_of_costs` call
instead of one-at-a-time `fun` calls, whenever the caller supplies one. VQESubroutine wires
`batch_fun` to a batched/async cudaq.observe dispatch (see VQESubroutine._batch_observe), so
those evaluations are submitted to the GPU together instead of serially. If no `batch_fun` is
given, everything falls back to sequential `fun` calls -- so this module has no hard dependency
on CUDA-Q and can be unit-tested with a plain Python `fun`.

Uniform interface:
    optimizer.settings            -> dict, must contain "maxiter"
    optimizer.minimize(fun, x0, batch_fun=None, state_fn=None, callback=None, state=None)
        -> OptimizerResult(x, fun, nfev, nit, history, state)
    optimizer.supports_batch      -> bool, whether this optimizer will use batch_fun if given
    optimizer.needs_state_fn      -> bool, whether this optimizer requires state_fn (QNSPSA only)

state_fn(params) -> real ndarray statevector, used only by QNSPSA to estimate the Fubini-Study
metric tensor via exact state-overlaps (cheap on a simulator, since VQESubroutine already
extracts ideal statevectors for its diversity check -- see VQESubroutine._bind_circuit_and_extract_state).

callback(xk, fk) is called after every accepted step, when supported, and is how VQESubroutine
implements Change #5's "adaptive stopping criteria for the VQE": raising _EarlyStop(x, fun) from
inside a callback is caught by .minimize() and treated as ordinary convergence, not an error.

state carries optimizer-internal state across calls (currently only QNSPSA's running metric-
tensor average) so a fresh VQESubroutine.solve() call for the *same* SDP block in the next
cutting-plane iteration can continue refining the same estimate instead of restarting cold --
this is Change #4's "optimizer-state reuse".
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from scipy.optimize import minimize as scipy_minimize


class _EarlyStop(Exception):
    """Raised from inside a scipy callback to signal Change #5's adaptive VQE stopping
    criterion has been met; caught by .minimize() and treated as normal convergence."""
    def __init__(self, x, fun):
        self.x = x
        self.fun = fun


@dataclass
class OptimizerResult:
    x: np.ndarray
    fun: float
    nfev: int
    nit: int = 0
    history: list = field(default_factory=list)   # energy after each step, for convergence-rate comparisons
    state: Optional[dict] = None                   # optimizer-internal state to reuse next call (QNSPSA only)
    stopped_early: bool = False                     # True if the adaptive stopping criterion fired


def _make_scipy_callback(fun, history, adaptive_stop_below):
    """Builds a scipy-compatible callback(xk) that records history and, if adaptive_stop_below
    is set, raises _EarlyStop once fun(xk) is comfortably past the PSD-violation threshold."""
    def callback(xk):
        fk = fun(xk)
        history.append(fk)
        if adaptive_stop_below is not None and fk < adaptive_stop_below:
            raise _EarlyStop(np.asarray(xk, dtype=float), fk)
    return callback


class _ScipyOptimizer:
    """Shared implementation for the four scipy.optimize.minimize-backed optimizers
    (COBYLA, Powell, L-BFGS-B, SLSQP). Subclasses set `method` and any method-specific options."""
    method = None
    supports_batch = True   # via a custom batched finite-difference jacobian, see below
    needs_state_fn = False

    def __init__(self, maxiter=100, tol=None, adaptive_stop_below=None, **method_options):
        self.settings = {"maxiter": maxiter, "tol": tol, "adaptive_stop_below": adaptive_stop_below,
                          **method_options}

    def _batched_jac(self, fun, batch_fun, eps=1e-6):
        """Central-difference gradient where all 2*d perturbed evaluations for one gradient
        go out in a single batch_fun call -- Change #2's 'concurrent evaluation of multiple
        parameter vectors'. Falls back to None (scipy's own numerical jac) if no batch_fun."""
        if batch_fun is None:
            return None

        def jac(x):
            x = np.asarray(x, dtype=float)
            d = len(x)
            perturbed = []
            for i in range(d):
                e = np.zeros(d)
                e[i] = eps
                perturbed.append(x + e)
                perturbed.append(x - e)
            values = batch_fun(perturbed)
            grad = np.array([(values[2 * i] - values[2 * i + 1]) / (2 * eps) for i in range(d)])
            return grad
        return jac

    def minimize(self, fun, x0, batch_fun=None, state_fn=None, callback=None, state=None):
        history = []
        adaptive_stop_below = self.settings.get("adaptive_stop_below")
        cb = _make_scipy_callback(fun, history, adaptive_stop_below)

        options = {"maxiter": self.settings["maxiter"]}
        options.update({k: v for k, v in self.settings.items()
                         if k not in ("maxiter", "tol", "adaptive_stop_below")})
        jac = self._batched_jac(fun, batch_fun) if self.method in ("L-BFGS-B", "SLSQP") else None

        stopped_early = False
        try:
            result = scipy_minimize(fun, x0=np.asarray(x0, dtype=float), method=self.method,
                                     jac=jac, tol=self.settings["tol"], options=options, callback=cb)
            x_final, fun_final, nfev = result.x, float(result.fun), int(result.nfev)
            nit = int(getattr(result, "nit", len(history)))
        except _EarlyStop as stop:
            x_final, fun_final = stop.x, stop.fun
            nfev = len(history)  # best available count when we cut the run short via callback
            nit = len(history)
            stopped_early = True

        return OptimizerResult(x=x_final, fun=fun_final, nfev=nfev, nit=nit,
                                history=history, state=None, stopped_early=stopped_early)


class COBYLA(_ScipyOptimizer):
    """Gradient-free, robust to noise. No jac support (COBYLA doesn't use one), so batch_fun
    is accepted for interface uniformity but unused."""
    method = "COBYLA"
    supports_batch = False

    def __init__(self, maxiter=100, tol=None, disp=False, rhobeg=1.0, adaptive_stop_below=None):
        super().__init__(maxiter=maxiter, tol=tol, adaptive_stop_below=adaptive_stop_below,
                          disp=disp, rhobeg=rhobeg)


class Powell(_ScipyOptimizer):
    """Gradient-free direction-set search. Often needs fewer evaluations than COBYLA on
    smooth, low-dimensional landscapes like these small ansatze."""
    method = "Powell"
    supports_batch = False

    def __init__(self, maxiter=100, tol=None, adaptive_stop_below=None):
        super().__init__(maxiter=maxiter, tol=tol, adaptive_stop_below=adaptive_stop_below)


class LBFGSB(_ScipyOptimizer):
    """Quasi-Newton, gradient-based. Gradients come from a batched central-difference jac
    (see _batched_jac) when batch_fun is supplied -- one GPU dispatch per gradient instead of 2d."""
    method = "L-BFGS-B"

    def __init__(self, maxiter=100, tol=None, adaptive_stop_below=None):
        super().__init__(maxiter=maxiter, tol=tol, adaptive_stop_below=adaptive_stop_below)


class SLSQP(_ScipyOptimizer):
    """Sequential least-squares, gradient-based (also batched, like L-BFGS-B). Supports
    constraints if the caller ever needs them; unused here but free with the interface."""
    method = "SLSQP"

    def __init__(self, maxiter=100, tol=None, adaptive_stop_below=None):
        super().__init__(maxiter=maxiter, tol=tol, adaptive_stop_below=adaptive_stop_below)


class SPSA:
    """
    Simultaneous Perturbation Stochastic Approximation. Gradient-free, needs only 2 evaluations
    per step regardless of dimension -- well suited to noisy/shot-based cost functions.

    Both perturbation evaluations for a step go out in a single batch_fun call when available
    (Change #2), instead of two sequential fun() calls.
    """
    supports_batch = True
    needs_state_fn = False

    def __init__(self, maxiter=100, learning_rate=0.05, perturbation=0.05, seed=None,
                 adaptive_stop_below=None):
        self.settings = {"maxiter": maxiter, "learning_rate": learning_rate,
                          "perturbation": perturbation, "seed": seed,
                          "adaptive_stop_below": adaptive_stop_below}

    def minimize(self, fun, x0, batch_fun=None, state_fn=None, callback=None, state=None):
        rng = np.random.default_rng(self.settings["seed"])
        x = np.asarray(x0, dtype=float).copy()
        best_x, best_fun = x.copy(), float(fun(x))
        nfev = 1
        history = [best_fun]
        a, c = self.settings["learning_rate"], self.settings["perturbation"]
        adaptive_stop_below = self.settings.get("adaptive_stop_below")
        stopped_early = False

        for k in range(1, self.settings["maxiter"] + 1):
            ak = a / (k ** 0.602)
            ck = c / (k ** 0.101)
            delta = rng.choice([-1.0, 1.0], size=x.shape)

            if batch_fun is not None:
                f_plus, f_minus = batch_fun([x + ck * delta, x - ck * delta])
                nfev += 2
            else:
                f_plus = fun(x + ck * delta)
                f_minus = fun(x - ck * delta)
                nfev += 2

            ghat = (f_plus - f_minus) / (2.0 * ck) * delta
            x = x - ak * ghat

            current = float(fun(x))
            nfev += 1
            history.append(current)
            if current < best_fun:
                best_fun, best_x = current, x.copy()
            if callback is not None:
                callback(x, current)
            if adaptive_stop_below is not None and best_fun < adaptive_stop_below:
                stopped_early = True
                break

        return OptimizerResult(x=best_x, fun=best_fun, nfev=nfev, nit=k, history=history,
                                state=None, stopped_early=stopped_early)


class QNSPSA:
    """
    Quantum Natural SPSA (Gacon et al.): SPSA-style stochastic gradient, preconditioned by an
    SPSA-style estimate of the Fubini-Study metric tensor, so steps respect the ansatz's actual
    geometry rather than raw parameter space.

    Simplification vs. the hardware-oriented original: the metric estimator normally needs a
    small overlap/fidelity circuit per state pair, measured via shots. Here, since we're always
    running on a simulator that already extracts full statevectors for the diversity check
    (state_fn), the overlap |<psi(a)|psi(b)>|^2 is computed *exactly* from those statevectors
    instead of estimated from measurement statistics -- strictly better accuracy for the same
    cost, but it does mean this metric estimate has no shot noise even in 'shots'/'noisy'
    execution modes (only the energy evaluations feeding the gradient do).

    state (Change #4, optimizer-state reuse): carries a running exponential-moving-average of
    the metric tensor across calls, so a fresh call for the same SDP block next iteration
    continues refining that estimate instead of starting from a fresh identity metric.
    """
    supports_batch = True
    needs_state_fn = True

    def __init__(self, maxiter=100, learning_rate=0.05, perturbation=0.05, regularization=1e-3,
                 metric_averaging=0.9, seed=None, adaptive_stop_below=None):
        self.settings = {"maxiter": maxiter, "learning_rate": learning_rate,
                          "perturbation": perturbation, "regularization": regularization,
                          "metric_averaging": metric_averaging, "seed": seed,
                          "adaptive_stop_below": adaptive_stop_below}

    def minimize(self, fun, x0, batch_fun=None, state_fn=None, callback=None, state=None):
        if state_fn is None:
            raise ValueError("QNSPSA requires state_fn(params) -> statevector to estimate the "
                              "Fubini-Study metric; none was provided.")
        rng = np.random.default_rng(self.settings["seed"])
        x = np.asarray(x0, dtype=float).copy()
        d = len(x)
        best_x, best_fun = x.copy(), float(fun(x))
        nfev = 1
        history = [best_fun]
        a, c = self.settings["learning_rate"], self.settings["perturbation"]
        reg, avg_rate = self.settings["regularization"], self.settings["metric_averaging"]
        adaptive_stop_below = self.settings.get("adaptive_stop_below")
        stopped_early = False

        metric_avg = (state or {}).get("metric_avg", np.eye(d))
        if metric_avg.shape != (d, d):  # block signature changed size since last reuse; reset
            metric_avg = np.eye(d)

        for k in range(1, self.settings["maxiter"] + 1):
            ak = a / (k ** 0.602)
            ck = c / (k ** 0.101)

            # --- gradient, as in plain SPSA ---
            delta_g = rng.choice([-1.0, 1.0], size=d)
            if batch_fun is not None:
                f_plus, f_minus = batch_fun([x + ck * delta_g, x - ck * delta_g])
                nfev += 2
            else:
                f_plus, f_minus = fun(x + ck * delta_g), fun(x - ck * delta_g)
                nfev += 2
            ghat = (f_plus - f_minus) / (2.0 * ck) * delta_g

            # --- metric tensor, SPSA-style single-sample estimator over exact overlaps ---
            delta_1 = rng.choice([-1.0, 1.0], size=d)
            delta_2 = rng.choice([-1.0, 1.0], size=d)
            psi_0 = state_fn(x)
            psi_pp = state_fn(x + ck * delta_1 + ck * delta_2)
            psi_pm = state_fn(x + ck * delta_1 - ck * delta_2)
            psi_mp = state_fn(x - ck * delta_1 + ck * delta_2)
            psi_mm = state_fn(x - ck * delta_1 - ck * delta_2)

            def fid(a, b):
                return float(np.dot(a, b) ** 2)  # real statevectors -> overlap is just the dot product squared

            delta_f = fid(psi_0, psi_pp) - fid(psi_0, psi_pm) - fid(psi_0, psi_mp) + fid(psi_0, psi_mm)
            inv_delta_1 = 1.0 / (ck * delta_1)
            inv_delta_2 = 1.0 / (ck * delta_2)
            raw_metric = -0.25 * delta_f * np.outer(inv_delta_1, inv_delta_2)
            raw_metric = 0.5 * (raw_metric + raw_metric.T)  # symmetrize

            metric_avg = avg_rate * metric_avg + (1 - avg_rate) * raw_metric
            regularized = metric_avg + reg * np.eye(d)

            natural_grad = np.linalg.pinv(regularized, hermitian=True) @ ghat
            x = x - ak * natural_grad

            current = float(fun(x))
            nfev += 1
            history.append(current)
            if current < best_fun:
                best_fun, best_x = current, x.copy()
            if callback is not None:
                callback(x, current)
            if adaptive_stop_below is not None and best_fun < adaptive_stop_below:
                stopped_early = True
                break

        return OptimizerResult(x=best_x, fun=best_fun, nfev=nfev, nit=k, history=history,
                                state={"metric_avg": metric_avg}, stopped_early=stopped_early)


OPTIMIZER_REGISTRY = {
    "COBYLA": COBYLA, "Powell": Powell, "SPSA": SPSA, "QNSPSA": QNSPSA,
    "L-BFGS-B": LBFGSB, "SLSQP": SLSQP,
}
