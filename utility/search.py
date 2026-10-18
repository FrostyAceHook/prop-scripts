"""
Searches over a System for arbitrary targets, explorations, and optimisations.

Everything here works through `.tweak` and reading attributes, so it applies to
any System. A `param` is anything settable (an input, or an output with a
setter). A `metric` is anything readable: an output name, a dotted path like
"opening.t_start", or a function f(system) -> float. Values are SI throughout;
results display in each parameter's units.

    import search

    # Hit a target.
    valve = search.solve(valve, "D_pneumo", "margin_close", 50.0)

    # Explore relationships.
    search.sweep(valve, "k_spring", np.linspace(20e3, 80e3, 25),
            "margin_open", "margin_close").plot()
    search.sensitivity(valve, ["margin_open", "margin_close"]).report()

    # Worst case over tolerances. A plain float is a +- fraction of the current
    # value (0.1 = +-10%), a tuple is the two edges to try.
    search.corners(valve, "margin_close", {
            "Pmax_tank": 0.1,                  # 84 bar -> 75.6 & 92.4 bar
            "ell_open": (18.15e-3, 18.30e-3),  # -0.05/+0.1 mm
        }).report()

    # Optimise: minimise an objective over bounded params, subject to bounds on
    # other metrics. Append "log" to search a param on a log scale.
    opt = search.optimise(valve, "D_pneumo",
            params={"D_pneumo": (20e-3, 45e-3), "k_spring": (10e3, 120e3),
                    "tau_vent": (2e-3, 50e-3, "log")},
            constraints={"rmargin_open": (0.3, None),
                         "rmargin_close": (0.3, None)})
    opt.report()
    valve = opt.system
"""

import itertools
import time
from dataclasses import dataclass, field, replace
from math import nan, isfinite, log

import numpy as np
import matplotlib.pyplot as plt

from fmt import fmt_num, fmt_unit, table, style_ax


__all__ = (
    "solve",
    "sweep",
    "grid",
    "sensitivity",
    "corners",
    "optimise",
)


# Errors that indicate an un-realisable system (not a code bug).
_BAD = (ValueError, ArithmeticError)

# Penalty infeasible points.
_INFEASIBLE = 1e9



# ---- Helpers ------------------------------------------------------------------

def _name(metric):
    """ Display name of a metric (a name, or a function's name). """
    if isinstance(metric, str):
        return metric
    return getattr(metric, "__name__", repr(metric))

def _get(system, metric):
    """ Reads a metric as a float, or nan if it fails. """
    if system is None:
        return nan
    try:
        if callable(metric):
            value = metric(system)
        else:
            value = system
            for part in metric.split("."):
                value = getattr(value, part)
    except _BAD:
        return nan
    return nan if value is None else float(value)

def _tweak(system, **changes):
    """ `.tweak`, or None if the changes make an invalid system. """
    try:
        return system.tweak(**changes)
    except _BAD:
        return None

def _unit(system, metric):
    """ Display unit of a metric, if it's a plain parameter name. """
    if not isinstance(metric, str):
        return None
    if metric not in system.inputs() + system.outputs():
        return None
    return system.unit(metric)

def _show(system, metric, value):
    """ A value as a string in its display unit. """
    return fmt_unit(value, _unit(system, metric))

def _axis(system, metric):
    """ Axis label and SI -> display conversion for a metric. """
    unit = _unit(system, metric)
    if unit is None:
        return _name(metric), (lambda x: x)
    return f"{_name(metric)} [{unit.symbol}]", unit.nsi

def _violation(lo, hi, value):
    """ Total relative violation of lo <= value <= hi (0 if satisfied). """
    if not isfinite(value):
        return 1e3
    v = 0.0
    if lo is not None and value < lo:
        v += (lo - value) / (abs(lo) or 1.0)
    if hi is not None and value > hi:
        v += (value - hi) / (abs(hi) or 1.0)
    return v

def _assess(system, objective, constraints, sign):
    """
    Score a system for minimisation: the (signed) objective if feasible, else
    _INFEASIBLE + total violation. Returns (score, objective value,
    {constraint: (lo, hi, value)}).
    """
    value = _get(system, objective)
    cons = {m: (lo, hi, _get(system, m))
            for m, (lo, hi) in constraints.items()}
    viol = sum(_violation(*c) for c in cons.values())
    if not isfinite(value):
        viol += 1e3
    score = sign*value if viol == 0.0 else _INFEASIBLE + viol
    return score, value, cons

def _numeric_inputs(system, exclude=()):
    """ Float-valued inputs (the default params to perturb). """
    return [n for n in system.inputs()
            if n not in exclude and isinstance(getattr(system, n), float)
            and isfinite(getattr(system, n))]



# ---- Solve --------------------------------------------------------------------

def solve(system, param, metric, target, bracket=None, *, max_expand=40):
    """
    Returns a copy of `system` with `param` adjusted so `metric` == `target`.

    Without a `bracket` (lo, hi), steps outward from the current value until the
    error changes sign (geometrically for positive params), then refines with
    Brent's method.
    """
    from scipy.optimize import brentq

    def error(p):
        return _get(_tweak(system, **{param: p}), metric) - target

    if bracket is None:
        p0 = _get(system, param)
        e0 = error(p0)
        if e0 == 0.0:
            return system
        if not isfinite(e0):
            raise ValueError(f"{_name(metric)} is not finite at the current "
                    f"{param}")
        bracket = _bracket(error, p0, e0, max_expand)
    lo, hi = bracket
    if not error(lo) * error(hi) <= 0.0:
        raise ValueError(f"{_name(metric)} doesn't cross {fmt_num(target)} "
                f"for {param} in [{fmt_num(lo)}, {fmt_num(hi)}]")
    xtol = 1e-12 * max(abs(lo), abs(hi), 1e-300)
    p = brentq(error, lo, hi, xtol=xtol, rtol=1e-12, maxiter=200)
    return system.tweak(**{param: p})

def _bracket(error, p0, e0, max_expand, growth=1.5):
    # Step outward in both directions until the error changes sign. The
    # bracket is the last same-sign point and the first opposite-sign one.
    last = {+1: p0, -1: p0}
    step = abs(p0) or 1.0
    for k in range(1, max_expand + 1):
        for d in (+1, -1):
            if p0 > 0:
                p = p0 * growth**(d*k)
            else:
                p = p0 + d*step*(growth**k - 1)
            e = error(p)
            if not isfinite(e):
                continue
            if e == 0.0 or (e > 0.0) != (e0 > 0.0):
                return (min(last[d], p), max(last[d], p))
            last[d] = p
    raise ValueError("couldn't bracket a solution; pass bracket=(lo, hi)")



# ---- Sweep and grid -----------------------------------------------------------

@dataclass(eq=False)
class Sweep:
    system: object # base system
    param: str # swept param
    values: np.ndarray = field(repr=False) # swept values
    data: dict = field(repr=False) # metric name -> values

    def summary(self):
        rows = []
        for i, p in enumerate(self.values):
            rows.append([_show(self.system, self.param, p),
                    *(_show(self.system, m, d[i])
                      for m, d in self.data.items())])
        return f"Sweep of {self.param}:\n" + table(rows,
                [self.param, *self.data])
    def __str__(self):
        return self.summary()
    def report(self):
        print(self.summary())

    def plot(self, axes=None, label=None):
        """
        One subplot per metric. Pass axes from a previous plot to overlay.
        """
        names = list(self.data)
        if axes is None:
            fig, axes = plt.subplots(len(names), 1, sharex=True, squeeze=False,
                    figsize=(8, 1 + 2.2*len(names)), tight_layout=True)
            axes = axes[:, 0]
            for ax in axes:
                style_ax(ax)
        xlabel, x_of = _axis(self.system, self.param)
        base = _get(self.system, self.param)
        for ax, name in zip(axes, names):
            ylabel, y_of = _axis(self.system, name)
            y = y_of(self.data[name])
            ax.plot(x_of(self.values), y, label=label)
            if isfinite(base):
                ax.axvline(x_of(base), c=(0.4, 0.4, 0.4), lw=1.0, ls="--")
            finite = y[np.isfinite(y)]
            if finite.size and finite.min() < 0 < finite.max():
                ax.axhline(0, c="black", lw=1.0)
            ax.set_ylabel(ylabel)
        axes[0].set_title(f"Sweep of {self.param}")
        axes[-1].set_xlabel(xlabel)
        if label:
            axes[0].legend()
        axes[0].figure.align_ylabels(axes)
        return axes

def sweep(system, param, values, *metrics):
    """ Evaluates `metrics` at each of `values` of `param`. """
    values = np.asarray(values, dtype=float)
    data = {_name(m): np.full(len(values), nan) for m in metrics}
    for i, p in enumerate(values):
        s = _tweak(system, **{param: p})
        for m in metrics:
            data[_name(m)][i] = _get(s, m)
    return Sweep(system, param, values, data)


@dataclass(eq=False)
class Grid:
    system: object # base system
    param_x: str
    param_y: str
    metric: str
    values_x: np.ndarray = field(repr=False) # [SI]
    values_y: np.ndarray = field(repr=False) # [SI]
    Z: np.ndarray = field(repr=False) # metric, indexed [y, x]

    def plot(self, ax=None, levels=20, contours=(0.0,)):
        """
        Filled contours of the metric. `contours` are drawn as labelled lines
        (e.g. 0 to show where a margin runs out).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
            style_ax(ax)
        xlabel, x_of = _axis(self.system, self.param_x)
        ylabel, y_of = _axis(self.system, self.param_y)
        zlabel, z_of = _axis(self.system, self.metric)
        X, Y = np.meshgrid(x_of(self.values_x), y_of(self.values_y))
        Z = z_of(self.Z)
        filled = ax.contourf(X, Y, Z, levels=levels)
        plt.colorbar(filled, ax=ax, label=zlabel)
        finite = Z[np.isfinite(Z)]
        lines = sorted(z_of(c) for c in contours if finite.size
                       and finite.min() < z_of(c) < finite.max())
        if lines:
            drawn = ax.contour(X, Y, Z, levels=lines, colors="black",
                    linewidths=1.5)
            ax.clabel(drawn, fmt=lambda z: fmt_num(z))
        # Mark the base system, if it's inside the grid.
        bx = _get(self.system, self.param_x)
        by = _get(self.system, self.param_y)
        if self.values_x.min() <= bx <= self.values_x.max() \
                and self.values_y.min() <= by <= self.values_y.max():
            ax.plot(x_of(bx), y_of(by), "*", c="white", mec="black", ms=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{self.metric} over {self.param_x} and {self.param_y}")
        return ax

def grid(system, param_x, values_x, param_y, values_y, metric):
    """ Evaluates `metric` over a 2D grid of two params (param_y set first). """
    vx = np.asarray(values_x, dtype=float)
    vy = np.asarray(values_y, dtype=float)
    Z = np.full((len(vy), len(vx)), nan)
    for j, py in enumerate(vy):
        sy = _tweak(system, **{param_y: py})
        for i, px in enumerate(vx):
            s = None if sy is None else _tweak(sy, **{param_x: px})
            Z[j, i] = _get(s, metric)
    return Grid(system, param_x, param_y, _name(metric), vx, vy, Z)



# ---- Sensitivity --------------------------------------------------------------

@dataclass(eq=False)
class Sensitivity:
    system: object # base system
    params: list
    metrics: list
    base_params: np.ndarray = field(repr=False) # [SI]
    base_metrics: np.ndarray = field(repr=False) # [SI]
    derivative: np.ndarray = field(repr=False) # d metric/d param, [p, m]

    @property
    def elasticity(self): # [-] % change in metric per % change in param.
        with np.errstate(divide="ignore", invalid="ignore"):
            return (self.derivative * self.base_params[:, None]
                    / self.base_metrics[None, :])

    def ranked(self, metric):
        """ Params sorted by |elasticity| on one metric, largest first. """
        e = self.elasticity[:, self.metrics.index(_name(metric))]
        order = np.argsort(-np.nan_to_num(np.abs(e), nan=-1.0))
        return [(self.params[i], e[i]) for i in order]

    def summary(self):
        E = self.elasticity
        rows = [[p, *(f"{e:+.3f}" if isfinite(e) else "-" for e in E[i])]
                for i, p in enumerate(self.params)]
        return ("Sensitivity (% change in metric per % change in param):\n"
                + table(rows, ["param", *self.metrics]))
    def __str__(self):
        return self.summary()
    def report(self):
        print(self.summary())

def sensitivity(system, metrics, params=None, *, rel_step=1e-2, exclude=()):
    """
    Central-difference derivatives of `metrics` with respect to `params`
    (default: every float input, minus `exclude`). Simulated times are quantised
    to the time step, so keep `rel_step` big enough to span many steps.
    """
    metrics = list(metrics)
    params = list(params) if params is not None else \
            _numeric_inputs(system, exclude)
    p0 = np.array([_get(system, p) for p in params])
    m0 = np.array([_get(system, m) for m in metrics])
    D = np.full((len(params), len(metrics)), nan)
    for i, (name, p) in enumerate(zip(params, p0)):
        h = rel_step*abs(p) if p != 0 else rel_step
        up = _tweak(system, **{name: p + h})
        down = _tweak(system, **{name: p - h})
        for j, m in enumerate(metrics):
            D[i, j] = (_get(up, m) - _get(down, m)) / (2*h)
    return Sensitivity(system, params, [_name(m) for m in metrics], p0, m0, D)



# ---- Corners ------------------------------------------------------------------

@dataclass(eq=False)
class Corners:
    metric: str
    worst: str # "min" or "max"
    value: float # worst value found
    system: object # system at the worst corner
    changes: dict # param -> value at the worst corner
    nominal: float # metric at the nominal system

    def summary(self):
        s = self.system
        lines = [f"Worst-case ({self.worst}) {self.metric}:",
                f"  nominal  {_show(s, self.metric, self.nominal)}",
                f"  worst    {_show(s, self.metric, self.value)}",
                "  at:"]
        lines += [f"    {p:<18}{_show(s, p, v)}"
                  for p, v in self.changes.items()]
        return "\n".join(lines)
    def __str__(self):
        return self.summary()
    def report(self):
        print(self.summary())

def corners(system, metric, tolerances, worst="min"):
    """
    Worst case of `metric` over every combination of tolerance extremes (2^n
    evaluations). `tolerances` maps each param to either:
      - a number r: a +- fraction of the current value, trying
        current*(1 - r) and current*(1 + r). Zero stays zero, so give
        zero-valued params a tuple instead.
      - a tuple (lo, hi): the two values to try, in SI. These needn't be
        numbers, e.g. "orings": (low_friction_seals, high_friction_seals).
    Exact when the metric is monotonic in each param, HOWEVER may miss an
    interior worst case otherwise.
    """
    if worst not in ("min", "max"):
        raise ValueError("worst must be 'min' or 'max'")
    names = list(tolerances)
    extremes = []
    for name in names:
        r = tolerances[name]
        if isinstance(r, (int, float)):
            p = _get(system, name)
            r = (p*(1 - r), p*(1 + r))
        extremes.append(r)
    found = None
    for combo in itertools.product(*extremes):
        changes = dict(zip(names, combo))
        s = _tweak(system, **changes)
        value = _get(s, metric)
        if not isfinite(value): # a corner that breaks the design is the worst
            found = (value, s, changes)
            break
        if found is None or (value < found[0] if worst == "min"
                             else value > found[0]):
            found = (value, s, changes)
    value, s, changes = found
    return Corners(_name(metric), worst, value, s if s is not None else system,
            changes, _get(system, metric))



# ---- Optimise -----------------------------------------------------------------

class _Objective:
    """
    Penalised objective over the unit hypercube. A class rather than a closure
    so it can be pickled for parallel workers.
    """
    def __init__(self, system, params, objective, constraints, sign):
        self.system = system
        self.params = params # name -> (lo, hi, log)
        self.names = list(params)
        self.objective = objective
        self.constraints = constraints
        self.sign = sign
        self.cache = {}

    def to_params(self, u):
        out = {}
        for name, ui in zip(self.names, u):
            lo, hi, is_log = self.params[name]
            ui = min(max(float(ui), 0.0), 1.0)
            out[name] = lo * (hi/lo)**ui if is_log else lo + ui*(hi - lo)
        return out

    def to_u(self, system):
        u = []
        for name in self.names:
            lo, hi, is_log = self.params[name]
            p = _get(system, name)
            if not isfinite(p):
                ui = 0.5
            elif is_log:
                ui = log(p/lo) / log(hi/lo) if p > 0 else 0.0
            else:
                ui = (p - lo) / (hi - lo)
            u.append(min(max(ui, 0.0), 1.0))
        return np.array(u)

    def evaluate(self, u):
        """ (score, objective value, constraints) at `u`, cached. """
        key = tuple(np.round(np.asarray(u, dtype=float), 12))
        if key not in self.cache:
            s = _tweak(self.system, **self.to_params(u))
            if s is None:
                self.cache[key] = (2*_INFEASIBLE, nan, {})
            else:
                self.cache[key] = _assess(s, self.objective, self.constraints,
                        self.sign)
        return self.cache[key]

    def __call__(self, u):
        return self.evaluate(u)[0]

    def describe(self, u):
        score, value, _ = self.evaluate(u)
        if score < _INFEASIBLE:
            return f"{_name(self.objective)} = " \
                    f"{_show(self.system, self.objective, value)}"
        return f"infeasible (violation {fmt_num(score - _INFEASIBLE)})"


@dataclass(eq=False)
class Optimum:
    system: object # best system found
    objective: str
    maximise: bool
    value: float # objective at the optimum
    feasible: bool
    start: dict # param -> starting value
    params: dict # param -> optimum value
    bounds: dict # param -> (lo, hi)
    constraints: dict # metric -> (lo, hi, value at the optimum)
    n_evals: int
    elapsed: float # [s]

    def summary(self):
        s = self.system
        word = "maximise" if self.maximise else "minimise"
        lines = [f"Optimum ({word} {self.objective}): "
                f"{_show(s, self.objective, self.value)} "
                f"({'feasible' if self.feasible else 'INFEASIBLE'}, "
                f"{self.n_evals} evals, {fmt_num(self.elapsed, sig=3)} s)"]
        rows = []
        for p, v in self.params.items():
            lo, hi = self.bounds[p]
            at = "  <- at bound" if min(abs(v - lo), abs(v - hi)) \
                    < 1e-6*abs(hi - lo) else ""
            rows.append([p, _show(s, p, self.start[p]), _show(s, p, v),
                    f"[{_show(s, p, lo)}, {_show(s, p, hi)}]{at}"])
        lines.append(table(rows, ["param", "start", "optimum", "bounds"]))
        if self.constraints:
            rows = []
            for m, (lo, hi, v) in self.constraints.items():
                need = ("" if lo is None else f"{_show(s, m, lo)} <= ") + "x" \
                     + ("" if hi is None else f" <= {_show(s, m, hi)}")
                ok = "ok" if _violation(lo, hi, v) == 0.0 else "VIOLATED"
                rows.append([_name(m), _show(s, m, v), need, ok])
            lines.append(table(rows, ["constraint", "value", "required", ""]))
        return "\n".join(lines)
    def __str__(self):
        return self.summary()
    def report(self):
        print(self.summary())

def optimise(system, objective, params, constraints=None, *, maximise=False,
             method="global", polish=True, seed=None, maxiter=100, popsize=15,
             tol=1e-3, workers=1, verbose=True):
    """
    Minimises (or maximises) `objective` over bounded `params`, subject to
    bounds on other metrics.

    params      : param -> (lo, hi), or (lo, hi, "log") to search on a log
                  scale (for ranges spanning decades).
    constraints : metric -> (lo, hi), either side None for unbounded.
    method      : "global" = differential evolution, then a Nelder-Mead polish.
                  "local"  = Nelder-Mead from the current system only.
    tol         : the global search stops once every member is feasible and
                  the objective spread is within `tol` (relative), else after
                  `maxiter` generations.
    workers     : >1 (or -1 for all cores) evaluates in parallel processes;
                  metrics must then be names, not lambdas, and the calling
                  script needs an `if __name__ == "__main__":` guard.

    Constraints are feasibility-first: any feasible design beats every
    infeasible one, and infeasible designs are ranked by relative violation.
    Params are searched over [0, 1] internally, so mm and N/mm bounds behave
    equally well. Needs scipy >= 1.12.
    """
    from scipy.optimize import differential_evolution, minimize

    if method not in ("global", "local"):
        raise ValueError("method must be 'global' or 'local'")
    specs = {}
    for name, spec in params.items():
        lo, hi, *rest = spec
        is_log = bool(rest) and rest[0] == "log"
        if is_log and lo <= 0:
            raise ValueError(f"log bounds for {name} must be positive")
        specs[name] = (float(lo), float(hi), is_log)

    sign = -1.0 if maximise else 1.0
    # A fresh copy, so cached outputs (e.g. simulations) aren't sent to workers.
    obj = _Objective(replace(system), specs, objective, dict(constraints or {}),
            sign)
    n = len(specs)
    bounds = [(0.0, 1.0)] * n
    u0 = obj.to_u(system)
    t0 = time.perf_counter()
    n_evals = 0
    if verbose:
        word = "maximise" if maximise else "minimise"
        print(f"optimise: {word} {_name(objective)} over {', '.join(specs)}")
        print(f"  start   {obj.describe(u0)}")

    u = u0
    if method == "global":
        # scipy's own stopping test is relative to the mean score, which the
        # infeasibility offset swamps, so it's off and replaced by this one.
        last = [None]
        def callback(intermediate_result):
            r = intermediate_result
            if verbose and (r.fun != last[0] or r.nit % 10 == 0):
                print(f"  gen {r.nit:<3} {obj.describe(r.x)}")
                last[0] = r.fun
            E = r.population_energies
            return bool(np.all(E < _INFEASIBLE)
                    and np.std(E) <= tol*max(abs(np.mean(E)), 1e-300))
        result = differential_evolution(obj, bounds, x0=u0, seed=seed,
                maxiter=maxiter, popsize=popsize, tol=0.0, atol=0.0,
                polish=False, workers=workers, callback=callback,
                updating="deferred" if workers != 1 else "immediate")
        u, n_evals = result.x, n_evals + result.nfev

    if method == "local" or polish:
        result = minimize(obj, u, method="Nelder-Mead", bounds=bounds,
                options=dict(xatol=1e-6, fatol=1e-12, maxfev=300*n,
                             adaptive=n > 2))
        n_evals += result.nfev
        if result.fun <= obj(u):
            u = result.x
        if verbose:
            print(f"  polish  {obj.describe(u)}")

    score, value, cons = obj.evaluate(u)
    best = obj.to_params(u)
    return Optimum(
        system=system.tweak(**best),
        objective=_name(objective),
        maximise=maximise,
        value=value,
        feasible=score < _INFEASIBLE,
        start={p: _get(system, p) for p in specs},
        params=best,
        bounds={p: s[:2] for p, s in specs.items()},
        constraints={_name(m): c for m, c in cons.items()},
        n_evals=n_evals,
        elapsed=time.perf_counter() - t0,
    )
