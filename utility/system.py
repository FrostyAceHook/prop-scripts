"""
Store a system as its independent inputs, allowing nice interaction.

Subclass `System` and declare the independent inputs as annotated fields. Declare
any dependent outputs as @property, with a property setter to make it settable.
This setter must return a dictionary of input changes that produce the requested
value. Annotate fields and getters' return types with a display unit from `spec`:

    from math import pi, sqrt
    import spec as u
    from system import System, calc, derived

    Carea = lambda D: pi/4 * D**2
    Cdiam = lambda A: sqrt(4/pi * A)

    class Piston(System):
        # 1. Field: an independent input.
        D: u.mm
        P: u.bar
        stroke: u.mm

        # 2. derived: an output dependant on only one parameter.
        A = derived(u.mm2, "D", Carea, Cdiam)

        # 3. calc: any output (shorter form than property).
        swept = calc(u.cc,
                lambda s: s.A * s.stroke,
                lambda s, V: dict(stroke=V / s.A))

        # 4. property: any output.
        @property
        def F(self) -> u.N: # annotate unit here.
            return self.P * self.A
        @F.setter
        def F(self, F):
            return dict(A=F / self.P) # may set outputs, not just inputs.

    p = Piston(D=u.mm_(30, tol=0.02), P=u.bar_(10, env=(9, 11)),
               stroke=u.mm_(5))
    p.report()  # every input and output, in units
    p = p.tweak(F=1000) # F -> A -> D: D = 35.68 mm ±0.02
    p.tweak(D=u.mm_(32), swept=u.cc_(5)) # stroke = 6.217 mm

Systems nest: annotate a field with a System subclass, or return one from a calc.
Every name-taking method accepts a dotted path through the tree:

    class Rig(System):
        piston: Piston
        count: int
        F = calc(u.kN, lambda s: s.count * s.piston.F)

    r = Rig(piston=p, count=4)
    r["piston.A"]               # fetch, also r.fetch("piston.A")
    r.display("piston.F")       # "1000 N"
    r.unit("piston.D")          # mm
    r.tweak({"piston.P": u.bar_(10.5)})   # rebuilds piston, then rig
    r.report()                  # all members, flattened with dots

Inputs may carry spreads (see `spec`) for `variation`, `corners`, and a robust
`optimise`; `sweep` and `grid` explore the design.

A part is an output declared as a System, with a setter that writes it back to
the inputs, e.g. a seal auto-sized from a slot:

    seal = calc(PistonSeal, lambda s: ..., lambda s, seal: dict(slot=...))

`resolved()` writes every part back, so the design's choices become fixed
inputs with their tolerances. Statistics always run on it, so they vary the
parts as made rather than re-choosing them. Declare what a design must meet in
`requirements()`; `verify()` checks it, and `optimise` keeps to it.
"""

import ast
import dataclasses
import importlib
import inspect
import itertools
import math
import numbers
import sys
import textwrap
import time
import typing
import warnings
from functools import cache, cached_property

import numpy as np

from spec import Spread, Tol, Env, Bare, unit_obj_of, show, table, style_ax


__all__ = (
    "System",
    "output",
    "computed",
    "calc",
    "derived",
    "Metric",
    "metric",
    "corner",
    "mean",
    "sd",
    "Requirements",
)


if typing.TYPE_CHECKING or sys.version_info >= (3, 11):
    from typing import dataclass_transform
else: # For editor/type-checker support.
    def dataclass_transform(**kwargs):
        return lambda cls: cls


class _hybridmethod:
    """ Bound to the instance when called on one, else to the class. """
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        target = objtype if obj is None else obj
        return self.func.__get__(target, type(target))


@dataclass_transform(frozen_default=True)
class System:
    def __init_subclass__(cls, **options):
        # Class keywords pass through to dataclass, e.g.
        # `class X(System, eq=False)`. `frozen` is always implicitly true tho.
        super().__init_subclass__()
        dataclasses.dataclass(cls, **{**options, "frozen": True})



    ### NAMES ###################################################################

    @classmethod
    def inputs(cls, deep=False):
        """ Names of independent inputs. """
        if deep:
            return tuple(n.path for n in _tree(cls, include_outputs=False))
        return tuple(f.name for f in dataclasses.fields(cls))

    @classmethod
    def outputs(cls, deep=False):
        """ Names of dependent outputs. """
        if deep:
            return tuple(n.path for n in _tree(cls) if n.dependent)
        names = {}
        for klass in reversed(cls.__mro__):
            for name, attr in vars(klass).items():
                if not name.startswith("_") and \
                        isinstance(attr, (property, cached_property)):
                    names[name] = None
        return tuple(n for n in names
                if isinstance(getattr(cls, n), (property, cached_property)))

    @classmethod
    def parameters(cls, deep=False):
        """ Names of all inputs and outputs. """
        if deep:
            return tuple(n.path for n in _tree(cls))
        return _declaration_order(cls)

    @classmethod
    def settable(cls, deep=False):
        """ Names of all inputs and all outputs with a setter. """
        if deep:
            return tuple(n.path for n in _tree(cls) if n.settable)
        return tuple(n for n in cls.parameters() if cls._settable_here(n))

    @classmethod
    def parts(cls, views=False):
        """ Names of settable outputs declared as a System. If `views`, also
        includes non-settables. """
        return tuple(n for n in cls.outputs() if _system_type_of(cls, n)
                     and (views or cls._settable_here(n)))

    @classmethod
    def _settable_here(cls, name):
        if name in cls.inputs():
            return True
        prop = getattr(cls, name, None)
        return isinstance(prop, property) and prop.fset is not None



    ### GETTING #################################################################

    def fetch(self, name):
        """ Value at a dotted path, e.g. `s.fetch("load.F")`. """
        x = self
        for part in name.split("."):
            x = getattr(x, part)
        return x
    __getitem__ = fetch

    @_hybridmethod
    def unit(self_or_cls, name):
        """ Display unit of a parameter, or None if it isn't annotated (or is a
        System). On an instance, sub-system types come from the values; on the
        class, from annotations, like the listings. """
        if not isinstance(self_or_cls, type):
            owner, leaf = self_or_cls._resolve(name)
            return _unit_obj_of(type(owner), leaf)
        cls = self_or_cls
        *parents, leaf = name.split(".")
        for i, part in enumerate(parents):
            sub = _system_type_of(cls, part)
            if sub is None:
                raise AttributeError(f"{'.'.join(parents[:i+1])!r} is not "
                                     "declared as a System")
            cls = sub
        return _unit_obj_of(cls, leaf)



    ### SETTING #################################################################

    def tweak(self, _changes=None, /, **kwchanges):
        """
        Return a copy with the given inputs or settable outputs changed. Note
        these changes are applied in-order. Setters may themselves return other
        settable outputs, not just inputs. Names may be dotted paths into sub-
        systems, through inputs or settable outputs.
        """
        return self._tweak({**(_changes or {}), **kwchanges})

    def _tweak(self, changes, prefix=""):
        """ `tweak`, with `prefix` the path down to this system, for errors. """
        system = self
        for name, value in changes.items():
            head, _, rest = name.partition(".")
            where = f"can't tweak {prefix + name!r}: {type(system).__name__}"
            if head not in system.parameters():
                raise AttributeError(f"{where} has no parameter {head!r}")
            alias = getattr(getattr(type(system), head, None), "alias", None)
            if alias is not None:
                # An alias: tweak what it names instead (spreads included).
                system = system._tweak({alias + name[len(head):]: value},
                                       prefix)
                continue
            if rest:
                # Tweak the sub-system, then set it back on this one (a
                # replace for an input, the setter for an output).
                if not system._settable_here(head):
                    raise AttributeError(f"{where}.{head} is an output with "
                                         "no setter")
                if isinstance(value, Spread) and head not in system.inputs():
                    raise TypeError(f"{where}.{head} is an output, and spreads "
                                    "go on inputs only")
                sub = getattr(system, head)
                if not isinstance(sub, System):
                    raise AttributeError(f"{where}.{head} is not a System")
                new = sub._tweak({rest: value}, f"{prefix}{head}.")
                system = system._tweak({head: new}, prefix)
                if head not in system.inputs():
                    # An output's setter might not carry every change back.
                    got = system.fetch(name)
                    if not _close(got, new.fetch(rest)):
                        warnings.warn(f"tweak {prefix + name!r}: setter for "
                            f"{type(system).__name__}.{head} didn't carry the "
                            f"change through (got {got!r}, wanted "
                            f"{new.fetch(rest)!r})", stacklevel=3)
                continue
            if name in system.inputs():
                old = getattr(system, name)
                try:
                    value = _carry(old, value)
                except ValueError as e:
                    unit = type(system).unit(name)
                    if isinstance(old, Env):
                        raise ValueError(f"{where}.{name} can't move to "
                                f"{show(value, unit)}, outside its envelope "
                                f"{show(old, unit, limits=True)}") from None
                    raise ValueError(f"{where}.{name}: {e}") from None
                system = dataclasses.replace(system, **{name: value})
                continue
            if not system._settable_here(name):
                raise AttributeError(f"{where}.{name} is not settable "
                        f"(settable: {', '.join(system.settable())})")
            if isinstance(value, Spread):
                raise TypeError(f"{where}.{name} is an output, and spreads go "
                                "on inputs only")
            if isinstance(value, Bare):
                value = float(value)
            prop = getattr(type(system), name)
            system = system._tweak(prop.fset(system, value), prefix)
        return system

    def solve(self, param, metric, target, bracket=None, *, max_expand=40):
        """
        Return a copy with `param` adjusted so `metric` == `target`. Without a
        `bracket` (lo, hi), steps outward from the current value until the error
        changes sign (geometrically for positive params), then refines with
        Brent's method.
        """
        from scipy.optimize import brentq

        metric = Metric.of(metric)
        def error(p):
            return metric.measure(_try_tweak(self, {param: p})) - target

        current = self.fetch(param)
        held = isinstance(current, Env)
        note = f" ({param} is held to its envelope " \
               f"{show(current, self.unit(param), limits=True)})" \
                if held else ""
        if bracket is None:
            p0 = float(current)
            e0 = error(p0)
            if e0 == 0.0:
                return self
            if not math.isfinite(e0):
                raise ValueError(f"{metric.name} is not finite at the current "
                        f"{param}")
            limits = current.limits if held else (-math.inf, math.inf)
            bracket = _bracket(error, p0, e0, max_expand, limits, note)
        lo, hi = bracket
        if not error(lo) * error(hi) <= 0.0: # also catches nan
            raise ValueError(f"{metric.name} doesn't cross "
                    f"{target:.6g} for {param} in [{lo:.6g}, {hi:.6g}]{note}")
        xtol = 1e-12 * max(abs(lo), abs(hi), 1e-300)
        p = brentq(error, lo, hi, xtol=xtol, rtol=1e-12, maxiter=200)
        return self.tweak({param: p})


    def optimise(self, metric, params, constraints=None, *, maximise=False,
                 method="global", polish=True, seed=None, maxiter=100,
                 popsize=15, tol=1e-3, workers=1, verbose=True):
        """
        Minimise (or maximise) `metric` over bounded `params`, subject to bounds
        on other metrics. Returns an Optimum.

        params      : {path: (lo, hi)} in SI, or (lo, hi, "log") to search on
                      a log scale (for ranges spanning decades). Any settable
                      path; one with a spread keeps it, recentred.
        constraints : {metric: (lo, hi)}, either side None for unbounded.
                      Default: `self.requirements()`; {} for none.
        method      : "global" = differential evolution, then a Nelder-Mead
                      polish. "local" = Nelder-Mead from the current system.
        tol         : the global search stops once every member is feasible
                      and the metric spread is within `tol` (relative),
                      else after `maxiter` generations.
        workers     : >1 (or -1 for all cores) evaluates in parallel processes.

        Feasibility first: any feasible design beats every infeasible one, and
        infeasible ones rank by total relative violation. Params are searched
        over [0, 1] internally, so mm and N/mm bounds behave alike.
        """
        from scipy.optimize import differential_evolution, minimize

        if method not in ("global", "local"):
            raise ValueError("method must be 'global' or 'local'")
        specs = {}
        for path, spec in params.items():
            lo, hi, *rest = spec
            is_log = rest == ["log"]
            if rest and not is_log:
                raise ValueError(f"{path}: give (lo, hi) or (lo, hi, 'log')")
            if not lo < hi:
                raise ValueError(f"{path}: bounds need lo < hi")
            if is_log and lo <= 0:
                raise ValueError(f"{path}: log bounds must be positive")
            current = self.fetch(path)
            if isinstance(current, Env):
                for bound in (lo, hi):
                    try:
                        current.recentre(bound)
                    except ValueError:
                        raise ValueError(f"{path}: bounds go outside its "
                                "envelope " + show(current, self.unit(path),
                                limits=True) + "; tweak it bare first to "
                                "search past it") from None
            specs[path] = (float(lo), float(hi), is_log)

        metric = Metric.of(metric)
        if constraints is None:
            constraints = self.requirements()
        constraints = {Metric.of(m): bounds for m, bounds in constraints.items()}
        sign = -1.0 if maximise else 1.0
        # A fresh copy, so cached outputs (e.g. simulations) aren't sent to
        # workers.
        obj = _Objective(dataclasses.replace(self), specs, metric,
                         constraints, sign)
        n = len(specs)
        bounds = [(0.0, 1.0)] * n
        x = x0 = obj.to_unit(self)
        t0 = time.perf_counter()
        n_evals = 0
        if verbose:
            word = "maximise" if maximise else "minimise"
            print(f"optimise: {word} {metric.name} over "
                  f"{', '.join(specs)}")
            print(f"  start   {obj.describe(x0)}")

        if method == "global":
            # scipy's own stopping test is relative to the mean score, which
            # the infeasibility offset swamps, so it's off and replaced here.
            last = [None]
            def callback(intermediate_result):
                r = intermediate_result
                text = obj.describe(r.x)
                if verbose and (text != last[0] or r.nit % 10 == 0):
                    print(f"  gen {r.nit:<3} {text}")
                    last[0] = text
                E = r.population_energies
                return bool(np.all(E < _INFEASIBLE)
                        and np.std(E) <= tol*max(abs(np.mean(E)), 1e-300))
            result = differential_evolution(obj, bounds, x0=x0, seed=seed,
                    maxiter=maxiter, popsize=popsize, tol=0.0, atol=0.0,
                    polish=False, workers=workers, callback=callback,
                    updating="deferred" if workers != 1 else "immediate")
            x, n_evals = result.x, n_evals + result.nfev

        if method == "local" or polish:
            result = minimize(obj, x, method="Nelder-Mead", bounds=bounds,
                    options=dict(xatol=1e-6, fatol=1e-12, maxfev=300*n,
                                 adaptive=n > 2))
            n_evals += result.nfev
            if result.fun <= obj(x):
                x = result.x
            if verbose:
                print(f"  polish  {obj.describe(x)}")

        score, value, cons = obj.evaluate(x)
        best = obj.to_params(x)
        return Optimum(
            system=self.tweak(best),
            metric=metric,
            maximise=maximise,
            value=value,
            feasible=score < _INFEASIBLE,
            start={p: Metric.of(p).measure(self) for p in specs},
            params=best,
            bounds={p: spec[:2] for p, spec in specs.items()},
            constraints=cons,
            n_evals=n_evals,
            elapsed=time.perf_counter() - t0,
        )



    ### DESIGN ##################################################################

    def resolved(self):
        """
        This design with its choices made: each System input resolved, then each
        part written back through its setter (an auto-sized part becomes a fixed
        one, with its tolerances). Nominal values are unchanged. Statistics run
        on this, so they vary the parts as made.
        """
        system = self.tweak({n: r for n in self.inputs()
                             if isinstance(v := getattr(self, n), System)
                             and (r := v.resolved()) is not v})
        for name in type(system).parts():
            new = system.tweak({name: getattr(system, name)})
            if new != system: # unchanged parts keep the instance (and cache)
                system = new
        return system

    def requirements(self):
        """
        What the design must meet, as {metric: (lo, hi)} (either side None): a
        rule on a nominal value, or a `corner` for a worst case. Includes each
        sub-system's (System inputs and parts), under its path. Extend with
        `super().requirements() | {...}`.
        """
        cls, reqs = type(self), Requirements()
        # TODO: why not just parameters? why muddy the "part" concept with
        # "views"
        for name in cls.inputs() + cls.parts(views=True):
            sub = getattr(self, name)
            if isinstance(sub, System):
                reqs |= {_Sub(name, m): bounds
                         for m, bounds in sub.requirements().items()}
        return reqs

    def verify(self, reqs=None):
        """ Each requirement's value, σ margin, and whether it's met. """
        if reqs is None:
            reqs = self.requirements()
        return Verification(self,
                {m: (lo, hi, m.measure(self)) for m, (lo, hi) in reqs.items()},
                {m: _sigma(self, m, lo, hi) for m, (lo, hi) in reqs.items()})



    ### SPREADS #################################################################

    def spreads(self):
        """ {path: Tol or Env} of every input with a spread, once resolved. """
        return {n.path: v for n in _tree(self.resolved(),
                                         include_outputs=False)
                if isinstance(v := getattr(n.owner, n.leaf), Spread)}

    def unassessed(self):
        """ Paths of float inputs with no spread yet, once resolved. """
        return tuple(n.path for n in _tree(self.resolved(),
                                           include_outputs=False)
                     if isinstance(v := getattr(n.owner, n.leaf), float)
                     and not isinstance(v, Spread))



    ### STATISTICS ##############################################################

    def variation(self, metric, *, rel_step=0.01):
        """
        How `metric` varies over the Tol inputs. Each Tol is moved to its limits
        with the others at their means, giving its contribution to the sd, its
        share of the variance, and how non-linear `metric` is across its band.
        Also lists the unassessed inputs that `metric` responds to, by its change
        for +-`rel_step` of each (0 skips this). Envs stay at their nominal:
        tweak them first to look elsewhere. Runs on the resolved system.
        """
        space = _Space(self, metric)
        mid = space.measure(space.base)
        if not math.isfinite(mid):
            raise ValueError(f"{space.name} is not finite with every Tol at "
                             "its mean")
        tols = {}
        for path, t in space.tols.items():
            lo, hi = t.limits
            f_lo = space.measure(space.base, {path: lo})
            f_hi = space.measure(space.base, {path: hi})
            half = (f_hi - f_lo) / 2
            bend = (f_hi + f_lo) / 2 - mid # quadratic part, at the limits
            shift = bend * (2*t.sd / (hi - lo))**2 # its mean shift
            nonlin = abs(bend / half) if half else math.inf if bend else 0.0
            tols[path] = (half / (hi - lo) * 2*t.sd, shift, nonlin)
        sd = math.sqrt(sum(c**2 for c, _, _ in tols.values()))
        mean = mid + sum(s for _, s, _ in tols.values())
        tols = {p: (c, (c/sd)**2 if sd else 0.0, n)
                for p, (c, _, n) in tols.items()}
        responds = {}
        for path in space.system.unassessed() if rel_step else ():
            x = float(space.base.fetch(path))
            h = rel_step * abs(x)
            d = (space.measure(space.base, {path: x + h})
                 - space.measure(space.base, {path: x - h})) / 2 if h \
                else math.nan # zero: no relative step to take
            if d != 0:
                responds[path] = d
        return Variation(space.system, space.metric,
                         space.measure(space.system), mean, sd, tols, responds,
                         rel_step, space.evals)

    def corners(self, metric, worst="min", *, k=3.0, limits=False, step=1.0,
                max_iter=20, check=0, seed=None):
        """
        The worst case of `metric` at `k` sigma. Over every corner of the Env
        limits, finds the most probable combination of Tols that's `k` standard
        deviations from their means (inverse FORM, by HL-RF steps of `step`
        sigma), and evaluates it on the real system.

        With `limits`, every Tol goes to the limit that worsens `metric` instead
        (the absolute worst case). With `check` > 0, draws that many random Tol
        samples at the worst Env corner and counts how many are worse (Monte
        Carlo). Runs on the resolved system.
        """
        if worst not in ("min", "max"):
            raise ValueError("worst must be 'min' or 'max'")
        sign = 1.0 if worst == "min" else -1.0
        space = _Space(self, metric)
        found = None
        envs = list(space.envs.items())
        for combo in itertools.product(*(e.limits for _, e in envs)):
            at = dict(zip((p for p, _ in envs), combo))
            corner = _try_tweak(space.base, at)
            if limits:
                result = space.limits(corner, sign)
            else:
                result = space.form(corner, sign, k, step, max_iter)
            if found is None or not (sign*result[0] >= sign*found[0]):
                found = (*result, at, corner) # nan beats anything
        value, z, changes, iters, converged, at, corner = found
        if not converged:
            warnings.warn(f"corners of {space.name}: FORM didn't converge in "
                          f"{max_iter} iterations", stacklevel=2)
        worse = None
        if check:
            rng = np.random.default_rng(seed)
            worse = 0
            for zs in rng.standard_normal((check, len(space.tols))):
                v = space.measure(corner, space.values(zs))
                worse += not (sign*v >= sign*value) # nan counts as worse
        return Corners(space.system, space.metric, worst, k, limits,
                space.measure(space.system), space.measure(space.base),
                space.measure(corner), value,
                at, {p: (zi, changes[p]) for p, zi in zip(space.tols, z)},
                iters, converged, check, worse, space.system.unassessed(),
                space.evals)



    ### EXPLORING ###############################################################

    def sweep(self, param, values, *metrics):
        """ Evaluates `metrics` at all `values` of `param`, and returns a Sweep
        object. """
        values = np.asarray(values, dtype=float)
        metrics = {m.name: m for m in map(Metric.of, metrics)}
        data = {name: np.full(len(values), math.nan) for name in metrics}
        for i, p in enumerate(values):
            s = _try_tweak(self, {param: float(p)})
            for name, m in metrics.items():
                data[name][i] = m.measure(s)
        return Sweep(self, param, values, data, metrics)

    def grid(self, param_x, values_x, param_y, values_y, metric):
        """ Evaluates `metric` over a 2D grid of two params, and returns a Grid
        object. Each point sets `param_y` first, then `param_x`. """
        metric = Metric.of(metric)
        vx = np.asarray(values_x, dtype=float)
        vy = np.asarray(values_y, dtype=float)
        Z = np.full((len(vy), len(vx)), math.nan)
        for j, py in enumerate(vy):
            sy = _try_tweak(self, {param_y: float(py)})
            if sy is None:
                continue
            for i, px in enumerate(vx):
                Z[j, i] = metric.measure(_try_tweak(sy, {param_x: float(px)}))
        return Grid(self, param_x, param_y, metric, vx, vy, Z)



    ### DISPLAYING ##############################################################

    def display(self, name, fmt=None):
        """ Returns a string of the given (dotted) parameter in its units. """
        owner, leaf = self._resolve(name)
        value = getattr(owner, leaf)
        if isinstance(value, System):
            return type(value).__name__
        return show(value, type(owner).unit(leaf), fmt)

    def report(self, outputs=True, printed=True):
        """ Every parameter in the tree, flattened to dotted paths, in
        declaration order. Dependants (and anything under one) marked with -.
        """
        rows = []
        for n in _tree(self, outputs):
            if n.view is not None:
                text = f"{n.view.__name__} (view)"
            else:
                try:
                    text = n.owner.display(n.leaf)
                except Exception as e: # show it, don't kill the report.
                    text = f"<{type(e).__name__}: {e}>"
            alias = getattr(getattr(type(n.owner), n.leaf, None), "alias", None)
            if alias is not None: # its path from the top, as listed
                text += f"  (= {n.path[:-len(n.leaf)]}{alias})"
            rows.append((n.path, "-" if n.dependent else " ", text))
        width = min(max((len(p) for p, _, _ in rows), default=0), 36)
        lines = [type(self).__name__ + "\n"]
        if outputs:
            lines.append(" (- dependent)\n")
        for path, mark, text in rows:
            lines.append(f"  {mark} {path:<{width}}  {text}\n")
        s = "".join(lines)
        if printed:
            print(s)
        else:
            return s



    ### HELPERS #################################################################

    def _resolve(self, name):
        """ (owning system, leaf name) of a dotted path. """
        parent, _, leaf = name.rpartition(".")
        owner = self.fetch(parent) if parent else self
        if not isinstance(owner, System):
            raise AttributeError(f"{parent!r} is a {type(owner).__name__}, "
                                 "not a System")
        return owner, leaf




### PROPERTIES ##################################################################

class output(property):
    """ A property with a display unit. Cached per instance by default. An
    alias has `alias`: the path it names. """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None, *,
                 unit=None, cache=True, alias=None):
        super().__init__(fget, fset, fdel, doc)
        self.unit = unit
        self.cache = cache
        self.alias = alias
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None or not self.cache or self.name is None:
            return super().__get__(obj, objtype)
        # No try/except: computing inside an except block chains a KeyError
        # onto every error raised by a (nested) getter.
        d = obj.__dict__
        if self.name not in d:
            d[self.name] = self.fget(obj)
        return d[self.name]

    def _copy(self, **changes):
        kw = dict(fget=self.fget, fset=self.fset, fdel=self.fdel,
                  doc=self.__doc__, unit=self.unit, cache=self.cache,
                  alias=self.alias) | changes
        return type(self)(**kw)

    def getter(self, fget):
        return self._copy(fget=fget)

    def setter(self, fset):
        return self._copy(fset=fset)

def alias(unit, alias, *, doc=None):
    """ Another name for the parameter `alias`. Tweaking it tweaks `alias`,
    spreads included. """
    return output(lambda s: s.fetch(alias), lambda s, v: {alias: v},
                  doc=doc or f"Alias of {alias}.", unit=unit, cache=False,
                  alias=alias)

def computed(unit):
    """ Caching and unit annotation. """
    return lambda fget: output(fget, unit=unit)

def calc(unit, fget, fset=None, *, cache=True, doc=None):
    """ An output from a getter (and optional setter), with a display unit. """
    return output(fget, fset, doc=doc, unit=unit, cache=cache)

def derived(unit, source, derive, inverse=None, *, cache=True, doc=None):
    """ An output computed from only one input. """
    fset = (lambda s, v: {source: inverse(v)}) if inverse else None
    return calc(unit, lambda s: derive(s.fetch(source)), fset, cache=cache,
            doc=doc)


### METRICS #####################################################################

class Metric:
    """
    Something measurable on a system: a dotted path, or a function
    f(system) -> float. Anything that takes a metric takes either, or a
    Metric, which can also carry a display unit and a name:

        ratio = Metric(lambda s: s.margin_open / s.cyl.F, name="ratio")

        @metric(u.N)
        def gap(s):
            return s.margin_open - s.margin_close

    Calling one on a system measures it (nan if the system is unrealisable).
    """
    delta = False # values are differences: scaled, never offset

    def __init__(self, source, unit=None, name=None):
        if isinstance(source, Metric) or not (isinstance(source, str)
                                              or callable(source)):
            raise TypeError(f"a metric is a dotted path or a function "
                            f"f(system) -> float, not {source!r}")
        self.source = source
        self.unit = unit
        self._name = name

    @classmethod
    def of(cls, metric):
        """ `metric` as a Metric: a path or function is wrapped. """
        return metric if isinstance(metric, Metric) else cls(metric)

    @property
    def name(self):
        if self._name:
            return self._name
        if isinstance(self.source, str):
            return self.source
        return getattr(self.source, "__name__", repr(self.source))

    @property
    def path(self):
        """ The dotted path measured: the source path, else (a function) the
        name. """
        return self.source if isinstance(self.source, str) else self.name

    def unit_on(self, system):
        """ Display Unit on `system`: the one given, else a path's own. """
        if self.unit is not None:
            return unit_obj_of(self.unit)
        if isinstance(self.source, str):
            try:
                return system.unit(self.source)
            except AttributeError:
                return None
        return None

    def measure(self, system):
        """ The value on `system` as a float, nan if it's unrealisable. """
        if system is None:
            return math.nan
        try:
            value = self._value(system)
        except _BAD:
            return math.nan
        return math.nan if value is None else float(value)
    __call__ = measure

    def _value(self, system):
        if isinstance(self.source, str):
            return system.fetch(self.source)
        return self.source(system)

    def show(self, value, system, fmt=None):
        """ A value of this metric as text, in its unit on `system`. """
        if not math.isfinite(value):
            return str(value)
        return show(value, self.unit_on(system), fmt, delta=self.delta)

    def axis(self, system):
        """ Axis label, and SI -> display conversion, on `system`. """
        unit = self.unit_on(system)
        if unit is None:
            return self.name, (lambda x: x)
        conv = (lambda x: x / unit.scale) if self.delta else unit.from_si
        return f"{self.name} [{unit.symbol}]", conv

    def _kwargs(self):
        """ Constructor arguments: identity, pickling and equality. """
        return dict(source=self.source, unit=self.unit, name=self._name)
    def __eq__(self, other):
        return type(self) is type(other) and self._kwargs() == other._kwargs()
    def __hash__(self):
        return hash((type(self), tuple(self._kwargs().items())))
    def __repr__(self):
        if type(self) is Metric:
            return f"Metric({self.name!r})"
        return f"<{type(self).__name__}: {self.name}>"
    def __reduce__(self):
        # A decorated function's module name now holds this Metric, so
        # pickle it by that name, as pickle does for plain functions.
        if type(self) is Metric and _named(self.source) is self:
            return (_import, (self.source.__module__,
                              self.source.__qualname__))
        return (_build, (type(self), self._kwargs()))

def metric(unit=None, name=None):
    """ Decorator: a function f(system) -> float as a Metric. """
    return lambda func: Metric(func, unit, name)


class corner(Metric):
    """
    A metric: the value `System.corners` finds for `metric`, taking the same
    arguments. By default the `k`-sigma statistical bound at the worst Env
    corner; with `limits`, every Tol at its own limit. Cached per system;
    `.result(system)` gives the full Corners.
    """
    def __init__(self, metric, worst="min", *, k=3.0, limits=False,
                 step=1.0, max_iter=20):
        if worst not in ("min", "max"):
            raise ValueError("worst must be 'min' or 'max'")
        self.of = Metric.of(metric)
        self.worst = worst
        self.k = float(k)
        self.limits = bool(limits)
        self.step = float(step)
        self.max_iter = int(max_iter)

    @property
    def name(self):
        how = "limits" if self.limits else f"{self.k:g}σ"
        return f"{self.of.name} ({how} {self.worst})"

    @property
    def path(self):
        return self.of.path

    @property
    def delta(self):
        return self.of.delta

    def unit_on(self, system):
        return self.of.unit_on(system)

    def result(self, system):
        """ The Corners analysis behind the value. """
        return _cached(system, self, lambda: system.corners(self.of,
                self.worst, k=self.k, limits=self.limits, step=self.step,
                max_iter=self.max_iter))

    def _value(self, system):
        return self.result(system).value

    def _kwargs(self):
        return dict(metric=self.of, worst=self.worst, k=self.k,
                    limits=self.limits, step=self.step,
                    max_iter=self.max_iter)


class _FromVariation(Metric):
    """ A metric read from `System.variation` of another. """
    stat = None # the Variation attribute

    def __init__(self, metric):
        self.of = Metric.of(metric)

    @property
    def name(self):
        return f"{self.of.name} ({self.stat})"

    @property
    def path(self):
        return self.of.path

    def unit_on(self, system):
        return self.of.unit_on(system)

    def result(self, system):
        """ The Variation analysis behind the value, shared by mean and
        sd. """
        return _cached(system, ("variation", self.of),
                       lambda: system.variation(self.of, rel_step=0))

    def _value(self, system):
        return getattr(self.result(system), self.stat)

    def _kwargs(self):
        return dict(metric=self.of)

class mean(_FromVariation):
    """ A metric: `metric`'s expected value as the Tols scatter, Envs at
    nominal (as `System.variation`). Cached per system, shared with sd. """
    stat = "mean"

    @property
    def delta(self):
        return self.of.delta

class sd(_FromVariation):
    """ A metric: `metric`'s standard deviation as the Tols scatter, Envs at
    nominal (as `System.variation`). Cached per system, shared with mean. """
    stat = "sd"
    delta = True


class _WorstSigma(Metric):
    """
    A metric: the smallest σ margin over a system's requirements that are
    corners (nominal ones have none: keep them as constraints), at least 0 when
    they all pass. Filter which requirements count as `Requirements` does.
    Maximise it for the design furthest from failing, or closest to passing:

        s.optimise(worst_sigma.without("tube_seal"), params,
                   s.requirements().nominal(), maximise=True)
    """
    def __init__(self, without=(), only=None):
        self.without_paths = tuple(without)
        self.only_paths = None if only is None else tuple(only)

    @property
    def name(self):
        notes = ([f"only {', '.join(self.only_paths)}"]
                 if self.only_paths is not None else []) \
              + ([f"without {', '.join(self.without_paths)}"]
                 if self.without_paths else [])
        return "worst σ margin" + (f" ({'; '.join(notes)})" if notes else "")

    @property
    def path(self):
        return "worst_sigma"

    def unit_on(self, system):
        return None

    def without(self, *paths):
        """ This, ignoring requirements at or under `paths` too. """
        return type(self)(self.without_paths + paths, self.only_paths)

    def only(self, *paths):
        """ This, counting only requirements at or under `paths`. """
        return type(self)(self.without_paths, paths)

    def _value(self, system):
        reqs = system.requirements()
        if self.only_paths is not None:
            reqs = reqs.only(*self.only_paths)
        reqs = reqs.without(*self.without_paths)
        sigmas = [_sigma(system, m, lo, hi) for m, (lo, hi) in reqs.items()]
        sigmas = [-math.inf if math.isnan(b) else b
                  for b in sigmas if b is not None]
        return min(sigmas) if sigmas else math.nan

    def _kwargs(self):
        return dict(without=self.without_paths, only=self.only_paths)

worst_sigma = _WorstSigma()


class _Sub(Metric):
    """ A sub-system's metric: `metric` measured on the system at `path`. """
    def __init__(self, at, metric):
        self.at = at
        self.of = Metric.of(metric)

    @property
    def name(self):
        return f"{self.at}.{self.of.name}"

    @property
    def path(self):
        return f"{self.at}.{self.of.path}"

    @property
    def delta(self):
        return self.of.delta

    def unit_on(self, system):
        return self.of.unit_on(system.fetch(self.at))

    def _value(self, system):
        return self.of._value(system.fetch(self.at))

    def _kwargs(self):
        return dict(at=self.at, metric=self.of)


def _cached(system, key, compute):
    """ `compute()`, cached on the (immutable) `system`, as outputs are. """
    cache = system.__dict__.setdefault("_metric_cache", {})
    if key not in cache:
        cache[key] = compute()
    return cache[key]

def _named(func):
    """ What `func`'s module holds under its name, else None. """
    module = sys.modules.get(getattr(func, "__module__", None))
    obj = module
    for part in getattr(func, "__qualname__", "<>").split("."):
        obj = getattr(obj, part, None)
    return obj

def _import(module, qualname):
    """ The object named `qualname` in `module` (for unpickling). """
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj

def _build(cls, kwargs):
    """ cls(**kwargs) (for unpickling). """
    return cls(**kwargs)




### REQUIREMENTS ################################################################

class Requirements(dict):
    """
    {Metric: (lo, hi)}, as `System.requirements` gives: a plain dict (so
    `optimise` takes it), keyed by Metric, with filters on each metric's
    `path` (its dotted source path, under any sub-system; a function metric's
    leaf is its name). A path matches itself and everything below it:

        reqs.without("tube_seal")               # all of one sub-system
        reqs.without("outer_seal.ID_stretch")   # its stretch, both sides
        reqs.only("margin_actuate")
        reqs.nominal()                          # those without a corner
    """
    def __init__(self, items=()):
        super().__init__((Metric.of(m), b) for m, b in dict(items).items())

    def without(self, *paths):
        """ Those whose path is not at or under any of `paths`. """
        return Requirements({m: b for m, b in self.items()
                             if not _under(m.path, paths)})

    def only(self, *paths):
        """ Those whose path is at or under one of `paths`. """
        return Requirements({m: b for m, b in self.items()
                             if _under(m.path, paths)})

    def nominal(self):
        """ Those checked at nominal (not a `corner`): no σ margin. """
        return Requirements({m: b for m, b in self.items()
                             if not isinstance(_leaf(m), corner)})

    def __or__(self, other):
        return Requirements({**self, **Requirements(other)})
    def __ror__(self, other):
        return Requirements({**Requirements(other), **self})
    def __ior__(self, other):
        self.update(Requirements(other))
        return self


def _under(path, prefixes):
    return any(path == p or path.startswith(p + ".") for p in prefixes)

def _leaf(m):
    """ The metric inside any sub-system wrappers. """
    while isinstance(m, _Sub):
        m = m.of
    return m

def _sigma(system, m, lo, hi):
    """
    σ margin of requirement `m` in [lo, hi]: the sigma at which its bound is
    reached (from the middle: Tols at their means, at the worst Env corner),
    less the `k` it's required at. Positive passes with that to spare, negative
    fails. Linear from the `k`-sigma worst, so exact for a linear metric. None
    for a requirement checked at nominal.
    """
    while isinstance(m, _Sub):
        try:
            system = system.fetch(m.at)
        except _BAD:
            return math.nan
        m = m.of
    if not isinstance(m, corner) or m.limits:
        return None
    bound = lo if m.worst == "min" else hi
    if bound is None:
        return None
    try:
        c = m.result(system)
    except _BAD:
        return math.nan
    gap, reach = c.middle - bound, c.middle - c.value
    if reach == 0: # insensitive to every Tol
        meets = gap >= 0 if m.worst == "min" else gap <= 0
        return math.inf if meets else -math.inf
    return c.k * gap / reach - c.k



@dataclasses.dataclass(eq=False)
class Verification:
    """ Result of `System.verify`. """
    system: System
    results: dict = dataclasses.field(repr=False) # Metric -> (lo, hi, value)
    sigmas: dict = dataclasses.field(repr=False) # Metric -> σ margin or None

    @property
    def failed(self):
        """ {Metric: (lo, hi, value)} of every requirement not met. """
        return {m: r for m, r in self.results.items() if _violation(*r) > 0}

    @property
    def ok(self):
        return not self.failed

    def summary(self):
        verdict = "all met" if self.ok else f"{len(self.failed)} not met"
        sigma = lambda b: "" if b is None else f"{b:+.2f}σ"
        return (f"Requirements of {type(self.system).__name__} ({verdict}):\n"
                + _constraint_table(self.system, self.results, "requirement",
                    {m: sigma(b) for m, b in self.sigmas.items()}))
    def __str__(self):
        return self.summary()
    def report(self):
        print(self.summary())




### OPTIMISE ####################################################################

class _Objective:
    def __init__(self, system, specs, metric, constraints, sign):
        self.system = system
        self.specs = specs # path -> (lo, hi, log)
        self.metric = metric
        self.constraints = constraints
        self.sign = sign
        self.cache = {}

    def to_params(self, x):
        """ {path: SI value} at unit-cube point `x`. """
        out = {}
        for (path, (lo, hi, is_log)), xi in zip(self.specs.items(), x):
            xi = min(max(float(xi), 0.0), 1.0)
            out[path] = lo * (hi/lo)**xi if is_log else lo + xi*(hi - lo)
        return out

    def to_unit(self, system):
        """ The unit-cube point of `system`'s params (clipped to bounds). """
        x = []
        for path, (lo, hi, is_log) in self.specs.items():
            p = Metric.of(path).measure(system)
            if not math.isfinite(p):
                xi = 0.5
            elif is_log:
                xi = math.log(p/lo) / math.log(hi/lo) if p > 0 else 0.0
            else:
                xi = (p - lo) / (hi - lo)
            x.append(min(max(xi, 0.0), 1.0))
        return np.array(x)

    def evaluate(self, x):
        """ (score, metric value, {metric: (lo, hi, value)}), cached. """
        key = tuple(np.round(np.asarray(x, dtype=float), 12))
        if key not in self.cache:
            s = _try_tweak(self.system, self.to_params(x))
            if s is None:
                self.cache[key] = (2*_INFEASIBLE, math.nan, {})
            else:
                self.cache[key] = _assess(s, self.metric,
                                          self.constraints, self.sign)
        return self.cache[key]

    def __call__(self, x):
        return self.evaluate(x)[0]

    def describe(self, x):
        score, value, _ = self.evaluate(x)
        if score < _INFEASIBLE:
            m = self.metric
            return f"{m.name} = {m.show(value, self.system)}"
        if score >= 2*_INFEASIBLE:
            return "unrealisable"
        return f"infeasible (violation {score - _INFEASIBLE:.3g})"


@dataclasses.dataclass(eq=False)
class Optimum:
    """ Result of `System.optimise`. """
    system: System # best system found
    metric: Metric
    maximise: bool
    value: float # metric at the optimum
    feasible: bool
    start: dict = dataclasses.field(repr=False) # path -> starting value
    params: dict = dataclasses.field(repr=False) # path -> optimum value
    bounds: dict = dataclasses.field(repr=False) # path -> (lo, hi)
    constraints: dict = dataclasses.field(repr=False) # Metric ->
                                                      #   (lo, hi, value)
    n_evals: int = dataclasses.field(repr=False)
    elapsed: float = dataclasses.field(repr=False) # [s]

    def summary(self):
        s, obj = self.system, self.metric
        p_show = lambda p, v: Metric.of(p).show(v, s)
        word = "maximise" if self.maximise else "minimise"
        lines = [f"Optimum ({word} {obj.name}): "
                 f"{obj.show(self.value, s)} "
                 f"({'feasible' if self.feasible else 'INFEASIBLE'}, "
                 f"{self.n_evals} evals, {self.elapsed:.3g} s)"]
        rows = []
        for p, v in self.params.items():
            lo, hi = self.bounds[p]
            at = "@bound  " if min(abs(v - lo), abs(v - hi)) \
                    < 1e-3*abs(hi - lo) else ""
            rows.append([p, p_show(p, self.start[p]), p_show(p, v),
                         f"{at}[{p_show(p, lo)}, {p_show(p, hi)}]"])
        lines.append(table(rows, ["param", "start", "optimum", "bounds"]))
        if self.constraints:
            lines += ["", _constraint_table(s, self.constraints)]
        return "\n".join(lines)
    def __str__(self):
        return self.summary()
    def report(self):
        print(self.summary())


def _constraint_table(system, constraints, header="constraint", notes=None):
    """ Table of {Metric: (lo, hi, value)}: value, requirement, ok, and any
    `notes` {Metric: text} as a last column. """
    rows = []
    for m, (lo, hi, v) in constraints.items():
        need = ("" if lo is None else f"{m.show(lo, system)} <= ") + "x" \
             + ("" if hi is None else f" <= {m.show(hi, system)}")
        ok = "ok" if _violation(lo, hi, v) == 0.0 else "VIOLATED"
        rows.append([m.name, m.show(v, system), need, ok]
                    + ([notes.get(m, "")] if notes else []))
    return table(rows, [header, "value", "required", ""]
                 + (["σ margin"] if notes else []))

def _violation(lo, hi, value):
    """ Total relative violation of lo <= value <= hi (0 if satisfied). """
    if not math.isfinite(value):
        return 1e3
    v = 0.0
    if lo is not None and value < lo:
        v += (lo - value) / (abs(lo) or 1.0)
    if hi is not None and value > hi:
        v += (value - hi) / (abs(hi) or 1.0)
    return v

def _assess(system, metric, constraints, sign):
    """ Score for minimisation: sign*metric if feasible, else _INFEASIBLE
    plus the total violation. Returns (score, metric value,
    {Metric: (lo, hi, value)}). """
    value = metric.measure(system)
    cons = {m: (lo, hi, m.measure(system))
            for m, (lo, hi) in constraints.items()}
    viol = sum(_violation(*c) for c in cons.values())
    if not math.isfinite(value):
        viol += 1e3
    score = sign*value if viol == 0.0 else _INFEASIBLE + viol
    return score, value, cons

# Score offset for infeasible designs: any feasible score beats it.
_INFEASIBLE = 1e9



### STATISTICS ##################################################################

class _Space:
    """ The Tol inputs of a system as independent standard normals (z = 0 is
    every Tol at its mean), and `metric` measured over them. Works on the
    resolved system (`self.system`). """

    def __init__(self, system, metric):
        system = self.system = system.resolved()
        spreads = system.spreads()
        tols = {p: t for p, t in spreads.items() if isinstance(t, Tol)}
        self.base = system.tweak({p: t.mean for p, t in tols.items()})
        self.tols = {p: t for p, t in tols.items() if not t.is_exact}
        self.envs = {p: e for p, e in spreads.items() if isinstance(e, Env)}
        self.metric = Metric.of(metric)
        self.name = self.metric.name
        self.evals = 0

    def measure(self, system, changes=None):
        self.evals += 1
        if changes:
            system = None if system is None else _try_tweak(system, changes)
        return self.metric.measure(system)

    def values(self, z):
        """ {path: value} of every Tol at standard normal `z`. """
        return {p: t.at(zi) for (p, t), zi in zip(self.tols.items(), z)}

    def gradient(self, system, z, value, step):
        """ d metric / dz by central differences, or one-sided where a probe
        is unrealisable (`value` is the metric at `z`). """
        g = np.empty(len(z))
        for i in range(len(z)):
            dz = np.zeros(len(z))
            dz[i] = step
            up = self.measure(system, self.values(z + dz))
            down = self.measure(system, self.values(z - dz))
            if math.isnan(up):
                up, step_up = value, 0.0
            else:
                step_up = step
            if math.isnan(down):
                down, step_down = value, 0.0
            else:
                step_down = step
            span = step_up + step_down
            g[i] = (up - down) / span if span else math.nan
        return g

    def form(self, system, sign, k, step, max_iter):
        """ (value, z, values, iterations, converged) at the most probable
        point with |z| = k that minimises sign*metric. HL-RF steps, halved
        whenever one fails to worsen the metric (as bounded distributions
        make plain HL-RF oscillate). """
        z = np.zeros(len(self.tols))
        value = self.measure(system, self.values(z))
        g = self.gradient(system, z, value, step)
        damping = 1.0
        for i in range(1, max_iter + 1):
            norm = float(np.linalg.norm(g))
            if not math.isfinite(norm):
                return math.nan, z, self.values(z), i, False
            if norm == 0.0: # insensitive to every Tol (or there are none)
                return value, z, self.values(z), i, True
            new = z + damping * (-sign * k * g / norm - z)
            new *= k / np.linalg.norm(new)
            if np.linalg.norm(new - z) < 1e-3 * k:
                return value, z, self.values(z), i, True
            new_value = self.measure(system, self.values(new))
            if math.isnan(new_value): # unrealisable: as bad as it gets
                return new_value, new, self.values(new), i, True
            if i == 1 or sign*new_value < sign*value:
                z, value = new, new_value
                g = self.gradient(system, z, value, step)
            else:
                damping /= 2
        return value, z, self.values(z), max_iter, False

    def limits(self, system, sign):
        """ (value, z, values, 1, True) with every Tol at the limit that
        worsens sign*metric, judged by the slope at the means. """
        z = np.zeros(len(self.tols))
        g = self.gradient(system, z, self.measure(system, self.values(z)),
                          1.0)
        changes = {p: t.mean if gi == 0 else t.limits[bool(sign*gi < 0)]
                   for (p, t), gi in zip(self.tols.items(), g)}
        z = np.array([(changes[p] - t.mean) / t.sd if t.sd else 0.0
                      for p, t in self.tols.items()])
        return self.measure(system, changes), z, changes, 1, True


@dataclasses.dataclass(eq=False)
class Variation:
    """ Result of `System.variation`. """
    system: System # the resolved system analysed
    metric: Metric
    nominal: float # metric with every input at its nominal
    mean: float # to second order
    sd: float # to first order
    tols: dict = dataclasses.field(repr=False) # path -> (contribution,
                                               #   share, non-linearity)
    responds: dict = dataclasses.field(repr=False) # unassessed path ->
                                                   #   change for +-rel_step
    rel_step: float = dataclasses.field(repr=False)
    evals: int = dataclasses.field(repr=False)


    # Non-linearity within a band worth flagging: the quadratic part of the
    # change across the band, relative to the linear part.
    _NONLINEAR = 0.1

    def ranked(self):
        """ (path, contribution, share, non-linearity), largest share first. """
        return sorted(((p, *v) for p, v in self.tols.items()),
                      key=lambda r: -np.nan_to_num(r[2], nan=np.inf))

    def summary(self):
        s, m = self.system, self.metric
        unit = m.unit_on(s)
        d = lambda x: show(x, unit, delta=True)
        lines = [f"Variation of {m.name} ({self.evals} evaluations):",
                 f"  nominal  {m.show(self.nominal, s)}"]
        if math.isfinite(self.sd):
            lines += [f"  mean     {m.show(self.mean, s)}",
                      f"  sd       {d(self.sd)}"]
        else:
            lines.append("  mean, sd unknown: a Tol is unrealisable within "
                         "its band")
        rows = [[p, s.display(p), d(c),
                 f"{share:.1%}" if math.isfinite(share) else "",
                 f"{n:.0%}" + (" !" if n > self._NONLINEAR else "")]
                if math.isfinite(c) else
                [p, s.display(p), "unrealisable at a limit", "", ""]
                for p, c, share, n in self.ranked()]
        if rows:
            lines.append(table(rows, ["tol", "value", "1σ effect", "share",
                                      "non-linear"]))
        if self.responds:
            lines.append(f"Unassessed inputs it responds to (change for "
                         f"±{self.rel_step:.0%}):")
            rows = sorted(self.responds.items(),
                          key=lambda r: -np.nan_to_num(abs(r[1]), nan=np.inf))
            lines.append(table([[p, s.display(p), d(c)] for p, c in rows],
                               ["input", "value", "change"]))
        return "\n".join(lines)
    def __str__(self):
        return self.summary()
    def report(self):
        print(self.summary())


@dataclasses.dataclass(eq=False)
class Corners:
    """ Result of `System.corners`. """
    system: System # the resolved system analysed: `point`'s paths are its
    metric: Metric
    worst: str # "min" or "max"
    k: float
    limits: bool # arithmetic worst case, not statistical
    nominal: float # metric with every input at its nominal
    mean: float # metric with every Tol at its mean
    middle: float # metric with every Tol at its mean, at the worst Env corner
    value: float # metric at the worst point
    envs: dict = dataclasses.field(repr=False) # path -> value at the corner
    tols: dict = dataclasses.field(repr=False) # path -> (z, value)
    iterations: int = dataclasses.field(repr=False)
    converged: bool = dataclasses.field(repr=False)
    samples: int = dataclasses.field(repr=False) # Monte Carlo check, or 0
    worse: int | None = dataclasses.field(repr=False) # samples worse
    unassessed: tuple = dataclasses.field(repr=False)
    evals: int = dataclasses.field(repr=False)

    @property
    def point(self):
        """ {path: value} of every spread input at the worst point, as paths
        on `system`. """
        return self.envs | {p: v for p, (_, v) in self.tols.items()}

    def build(self):
        """ The worst build: `system` tweaked to `point` (spreads kept,
        recentred). """
        return self.system.tweak(self.point)

    def summary(self):
        s, m = self.system, self.metric
        how = "every Tol at its limit" if self.limits else \
                f"{self.k:g}σ, {self.iterations} iterations"
        lines = [f"Worst-case ({self.worst}) {m.name} ({how}, "
                 f"{self.evals} evaluations):",
                 f"  nominal  {m.show(self.nominal, s)}",
                 f"  at means {m.show(self.mean, s)}",
                 f"  worst    {m.show(self.value, s)}"
                 + (" (unrealisable)" if math.isnan(self.value) else "")]
        if self.samples:
            expect = 0.5 * math.erfc(self.k / math.sqrt(2))
            lines.append(f"  Monte Carlo: {self.worse} of {self.samples} "
                         f"samples worse ({self.worse/self.samples:.2%}; "
                         f"FORM expects {expect:.2%})")
        rows = [[p, show(v, s.unit(p)), "limit", ""]
                for p, v in self.envs.items()]
        k2 = sum(z**2 for z, _ in self.tols.values()) or 1.0
        rows += [[p, show(v, s.unit(p)), f"{z:+.2f}σ",
                  "" if self.limits else f"{z**2/k2:.0%}"]
                 for p, (z, v) in self.tols.items()]
        if rows:
            lines.append(table(rows, ["at", "value", "", "share"]))
        if self.unassessed:
            lines.append(f"  held at nominal (unassessed): "
                         f"{', '.join(self.unassessed)}")
        return "\n".join(lines)
    def __str__(self):
        return self.summary()
    def report(self):
        print(self.summary())




### SWEEP #######################################################################

@dataclasses.dataclass(eq=False)
class Sweep:
    """ Result of `System.sweep`. """
    system: System # base system
    param: str # swept param
    values: np.ndarray = dataclasses.field(repr=False) # swept values
    data: dict = dataclasses.field(repr=False) # metric name -> values
    metrics: dict = dataclasses.field(repr=False) # metric name -> Metric

    def summary(self):
        columns = [Metric.of(self.param), *self.metrics.values()]
        rows = [[m.show(v, self.system) for m, v in
                 zip(columns, (p, *(d[i] for d in self.data.values())))]
                for i, p in enumerate(self.values)]
        return f"Sweep of {self.param}:\n" + table(rows,
                [self.param, *self.data])
    def __str__(self):
        return self.summary()
    def report(self):
        print(self.summary())

    def plot(self, axes=None, label=None):
        """ One subplot per metric. Pass axes from a previous plot (and a
        `label`) to overlay. """
        import matplotlib.pyplot as plt
        names = list(self.data)
        if axes is None:
            fig, axes = plt.subplots(len(names), 1, sharex=True, squeeze=False,
                    figsize=(8, 1 + 2.2*len(names)), tight_layout=True)
            axes = axes[:, 0]
            for ax in axes:
                style_ax(ax)
        param = Metric.of(self.param)
        xlabel, x_of = param.axis(self.system)
        base = param.measure(self.system)
        for ax, name in zip(axes, names):
            ylabel, y_of = self.metrics[name].axis(self.system)
            y = y_of(self.data[name])
            ax.plot(x_of(self.values), y, label=label)
            if math.isfinite(base):
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



### GRID ########################################################################

@dataclasses.dataclass(eq=False)
class Grid:
    """ Result of `System.grid`. """
    system: System # base system
    param_x: str
    param_y: str
    metric: Metric
    values_x: np.ndarray = dataclasses.field(repr=False)
    values_y: np.ndarray = dataclasses.field(repr=False)
    Z: np.ndarray = dataclasses.field(repr=False) # metric, indexed [y, x]

    def plot(self, ax=None, levels=20, contours=(0.0,)):
        """ Filled contours of the metric. `contours` are drawn as labelled
        lines, e.g. 0 to show where a margin runs out. """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
            style_ax(ax)
        px, py = Metric.of(self.param_x), Metric.of(self.param_y)
        xlabel, x_of = px.axis(self.system)
        ylabel, y_of = py.axis(self.system)
        zlabel, z_of = self.metric.axis(self.system)
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
            ax.clabel(drawn, fmt=lambda z: f"{z:.4g}")
        # Mark the base system, if it's inside the grid.
        bx, by = px.measure(self.system), py.measure(self.system)
        if self.values_x.min() <= bx <= self.values_x.max() \
                and self.values_y.min() <= by <= self.values_y.max():
            ax.plot(x_of(bx), y_of(by), "*", c="white", mec="black", ms=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{self.metric.name} over {self.param_x} and "
                     f"{self.param_y}")
        return ax



### DUNGEON #####################################################################

def _is_system_type(t):
    return isinstance(t, type) and issubclass(t, System)

@cache
def _declared(cls, name):
    """ What a parameter is declared as: its field annotation, its output's
    unit, or its getter's return annotation. """
    if name in cls.inputs():
        return typing.get_type_hints(cls, include_extras=True).get(name)
    attr = getattr(cls, name, None)
    if isinstance(attr, output):
        return attr.unit
    fget = getattr(attr, "fget", None) or getattr(attr, "func", None)
    if fget is None:
        raise AttributeError(f"{cls.__name__} has no input or output {name!r}")
    return typing.get_type_hints(fget, include_extras=True).get("return")

@cache
def _system_type_of(cls, name):
    """ The System subclass a parameter is declared as, else None. """
    hint = _declared(cls, name)
    return hint if _is_system_type(hint) else None

@cache
def _unit_obj_of(cls, name):
    hint = _declared(cls, name)
    return None if _is_system_type(hint) else unit_obj_of(hint)

def _carry(old, new):
    """ The value to store over input `old`: a plain number keeps old's
    spread (recentred), Bare drops it, anything else is stored as given. """
    if isinstance(new, Bare):
        return float(new)
    if isinstance(old, Spread) and isinstance(new, numbers.Real) \
            and not isinstance(new, (bool, Spread)):
        return old.recentre(new)
    return new

def _close(a, b):
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-12)
    return a == b

class _Node(typing.NamedTuple):
    path: str
    owner: object # the System, or System class, holding `leaf`
    leaf: str
    dependent: bool # an output somewhere on the path
    settable: bool # settable at every step of the path
    view: type | None # a System type shown but not descended into

def _tree(target, include_outputs=True, prefix="", dependent=False,
          settable=True, chain=()):
    """ Yields a _Node per parameter below `target`, depth-first in
    declaration order. On an instance, sub-systems come from the values; on
    a class, from the annotations. """
    live = isinstance(target, System)
    cls = type(target) if live else target
    chain += (cls,)
    inputs = cls.inputs()
    for name in cls.parameters():
        is_input = name in inputs
        if not (include_outputs or is_input):
            continue
        hint = _declared(cls, name)
        sub = hint if _is_system_type(hint) else None
        nest = sub is not None and _nests(is_input, live, sub, chain)
        child = sub
        if live and (is_input or hint is None or nest):
            # Inputs are free to read; outputs are only evaluated when
            # undeclared or nested into, so declared views never run.
            try:
                child = getattr(target, name)
            except Exception:
                child = None # report() shows the error.
            sub = type(child) if isinstance(child, System) else None
            nest = sub is not None and _nests(is_input, live, sub, chain)
        dep = dependent or not is_input
        sett = settable and cls._settable_here(name)
        yield _Node(prefix + name, target, name, dep, sett,
                    None if nest else sub)
        if nest:
            yield from _tree(child, include_outputs, f"{prefix}{name}.", dep,
                             sett, chain)

def _nests(is_input, live, sub, chain):
    """ Whether to descend into sub-system type `sub`. A live input always
    nests: a frozen tree can't loop. Otherwise a type already above is a
    view. """
    return (is_input and live) or sub not in chain

@cache
def _declaration_order(cls):
    names = []
    for klass in reversed(cls.__mro__):
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(klass)))
        except (OSError, TypeError, SyntaxError):
            continue # no source (builtins, REPL): fall back below
        for node in tree.body[0].body:
            if (isinstance(node, ast.AnnAssign) and
                    isinstance(node.target, ast.Name)):
                names.append(node.target.id)  # D_seat: u.mm
            elif isinstance(node, ast.Assign):
                for target in node.targets:  # A_seat = ..., or a, b = ...
                    names += [t.id for t in getattr(target, "elts", [target])
                              if isinstance(t, ast.Name)]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)  # @property def F_pneumo(...)
    params = cls.inputs() + cls.outputs()
    ordered = [n for n in dict.fromkeys(names) if n in params]
    return tuple(ordered + [n for n in params if n not in ordered])


# Errors that mean an un-realisable system (not a code bug).
_BAD = (ValueError, ArithmeticError)

def _try_tweak(system, changes):
    """ `system.tweak(changes)`, or None if it makes an invalid system. """
    try:
        return system.tweak(changes)
    except _BAD:
        return None

def _bracket(error, p0, e0, max_expand, limits=(-math.inf, math.inf), note="",
             growth=1.5):
    """ Step outward from p0 in both directions, clamped to `limits`, until the
    error changes sign. Returns (lo, hi): the last same-sign point and the first
    opposite-sign one. """
    last = {+1: p0, -1: p0}
    step = abs(p0) or 1.0
    for k in range(1, max_expand + 1):
        for d in (+1, -1):
            if p0 > 0:
                p = p0 * growth**(d*k)
            else:
                p = p0 + d*step*(growth**k - 1)
            p = min(max(p, limits[0]), limits[1])
            if p == last[d]:
                continue # pinned at a limit
            e = error(p)
            if not math.isfinite(e):
                continue
            if e == 0.0 or (e > 0.0) != (e0 > 0.0):
                return (min(last[d], p), max(last[d], p))
            last[d] = p
    raise ValueError("couldn't bracket a solution; pass bracket=(lo, hi)"
                     + note)
