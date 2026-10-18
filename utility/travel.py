"""
Generic 1-DOF travel simulation.
"""

from dataclasses import dataclass, field
from math import inf, nan, floor, log10, isfinite
from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt

from fmt import fmt_num, fmt_mm, fmt_ms, fmt_m_s, style_ax


__all__ = (
    "Impact",
    "Journey",
    "Friction",
    "embark"
)


@dataclass(frozen=True, slots=True)
class Impact:
    stop: str # "min" or "max"
    t: float # [s] time of contact
    v: float # [m/s] velocity just before contact
    E: float # [J] kinetic energy at contact


@dataclass(eq=False)
class Journey:
    title: str # cosmetic title
    x_min: float # [m] lower hard stop position
    x_max: float # [m] upper hard stop position
    t: np.ndarray = field(repr=False)       # [s] time
    x: np.ndarray = field(repr=False)       # [m] position
    v: np.ndarray = field(repr=False)       # [m/s] velocity
    a: np.ndarray = field(repr=False)       # [m/s^2] acceleration
    m: np.ndarray = field(repr=False)       # [kg] mass
    forces: dict = field(repr=False)        # name -> [N] applied force history
    frictions: set = field(repr=False)      # names of the friction forces
    F_drive: np.ndarray = field(repr=False) # [N] sum of non-friction forces
    F_frictional: np.ndarray = field(repr=False) # [N] total friction magnitude
    impacts: list = field(default_factory=list) # [impact] impacts


    # ---- Key events -----------------------------------------------------------

    @property
    def t_start(self): # [s] When motion began, or None if it never moved
        if self.v[0] != 0.0:
            return 0.0
        moved = np.flatnonzero(self.x != self.x[0])
        return self.t[moved[0] - 1] if moved.size else None

    @property
    def first_impact(self): # [impact] First impact object.
        return self.impacts[0] if self.impacts else None

    @property
    def t_impact(self): # [s] Time of first impact.
        return self.impacts[0].t if self.impacts else None
    @property
    def v_impact(self): # [m/s] Speed of first impact.
        return self.impacts[0].v if self.impacts else None
    @property
    def E_impact(self): # [J] Energy of first impact.
        return self.impacts[0].E if self.impacts else None

    @property
    def stroke_time(self): # [s] Time for first impact after starting.
        if self.t_start is None or self.t_impact is None:
            return None
        return self.t_impact - self.t_start


    # ---- Reporting ------------------------------------------------------------

    def summary(self):
        lines = [f"{self.title}:"]
        lines.append(f"  starts at {fmt_mm(self.x[0])} "
                f"({fmt_mm(self.x_min)} & {fmt_mm(self.x_max)} stops)")
        if self.t_start is None:
            lines.append("  never moved within the simulated time")
            return "\n".join(lines)
        lines.append(f"  starts moving after {fmt_ms(self.t_start)}")
        if not self.impacts:
            lines.append("  did not reach a stop "
                    f"(ended at {fmt_mm(self.x[-1])})")
            return "\n".join(lines)
        imp = self.first_impact
        lines += [
            f"  hits {imp.stop} stop after {fmt_ms(imp.t)}",
            f"  stroke time   : {fmt_ms(self.stroke_time)}",
            f"  impact speed  : {fmt_m_s(abs(imp.v))}",
            f"  impact energy : {fmt_num(imp.E)} J",
        ]
        if len(self.impacts) > 1:
            lines.append(f"  ... with {len(self.impacts) - 1} further impact(s) "
                    "from bouncing")
        return "\n".join(lines)

    def __repr__(self):
        return f"<journey {self.title!r}>"

    def __str__(self):
        return self.summary()
    def plot(self, axes=None, label=None):
        """
        Plot position, velocity and forces. Pass axes from a previous plot to
        overlay runs.
        """
        # trim trailing data where the position no longer changes
        moves = np.flatnonzero(np.diff(self.x))
        N = moves[-1] + 2 if moves.size else len(self.t)
        t = self.t[:N] * 1e3
        x = self.x[:N]
        v = self.v[:N]
        forces = {name: arr[:N] for name, arr in self.forces.items()}
        F_drive = self.F_drive[:N]
        F_frictional = self.F_frictional[:N]

        if axes is None:
            fig, axes = plt.subplots(
                3, 1, sharex=True, figsize=(9, 7.5), tight_layout=True,
                gridspec_kw={"height_ratios": [4, 3, 8]},
            )
            for ax in axes:
                style_ax(ax)
        else:
            fig = axes[0].figure
        tag = (lambda s: f"{s} ({label})") if label else (lambda s: s)

        axes[0].plot(t, x * 1e3, label=label)
        for lim in (self.x_min, self.x_max):
            axes[0].axhline(lim * 1e3, c=(0.4, 0.4, 0.4), lw=1.0)
        axes[0].set_ylabel("Position [mm]")
        axes[0].set_title(self.title)

        axes[1].plot(t, v, label=label)
        axes[1].axhline(0, c="black", lw=1.0)
        axes[1].set_ylabel("Velocity [m/s]")

        for name, arr in forces.items():
            ls = "--" if name in self.frictions else "-"
            axes[2].plot(t, arr, ls=ls, label=tag(name))
        axes[2].axhline(0, c="black", lw=1.0)
        axes[2].set_ylabel("Force [N]")
        axes[2].set_xlabel("Time [ms]")
        axes[2].legend()

        for ax in axes:
            ax.set_xlim([-0.01*t[-1], 1.01*t[-1]])
            if self.t_impact is not None:
                ax.axvline(self.t_impact * 1e3, c="r", lw=1.0, alpha=0.5)

        if label:
            axes[0].legend()
        fig.align_ylabels(axes)
        return axes



def _fn(p):
    # Wrap a constant as f(t, x, v) so any input can be a number or a function.
    return p if callable(p) else (lambda t, x, v, _p=float(p): _p)

class Friction:
    """
    Marks a force as friction. Give its magnitude (>= 0) as a number or
    f(t, x, v). The solver applies it against the motion, or as a holding force
    up to this magnitude while stopped. While stuck it is called with v == 0
    exactly, so return the breakaway value there, e.g.
    lambda t, x, v: <breakaway> if v == 0 else <running>.
    """
    def __init__(self, magnitude):
        self.fn = _fn(magnitude)

def embark(forces, mass, x0, x_min, x_max, *,
        v0=0.0, restitution=0.0, dt=10e-6, t_end=0.5, t_after_impact=None,
        title="Journey"):
    """
    Simulate a body moving between two hard stops.

    forces         : [N] all forces, as dict of name -> (number, f(t, x, v), or
                         Friction).
    mass           : [kg] mass, as number or f(t, x, v).
    x0             : [m] initial position.
    x_min, x_max   : [m] stop positions.
    v0             : [m/s] initial velocity.
    restitution    : [-] coefficient of restitution for the stops.
    dt             : [s] simulation time step.
    t_end          : [s] simulation end time.
    t_after_impact : [s] if set, end the run this long after the first impact
                     instead of running to t_end.
    title          : cosmetic title.
    """
    if not x_min < x_max:
        raise ValueError("need x_min < x_max")
    if not x_min <= x0 <= x_max:
        raise ValueError("x0 must lie between x_min and x_max")
    if not isinstance(forces, dict):
        forces = {"force": forces}

    d_names = [k for k, f in forces.items() if not isinstance(f, Friction)]
    f_names = [k for k, f in forces.items() if isinstance(f, Friction)]
    d_fns = [_fn(forces[k]) for k in d_names]
    f_fns = [forces[k].fn for k in f_names]
    m_fn = _fn(mass)

    N = int(round(t_end / dt)) + 1
    xs, vs, ms, accs = [float(x0)], [float(v0)], [], []
    Fd_hist, Ff_hist = [], []
    d_hist = [[] for _ in d_fns]
    f_hist = [[] for _ in f_fns]
    impacts = []
    t_quit = inf
    xi, vi = float(x0), float(v0)
    # Direction of motion, kept for when there's no drive. Initialise to only
    # possible option when at one of the stops.
    last_s = 1.0 if (x0 == x_min) else -1.0
    last_hist_s = last_s

    for i in range(N):
        ti = i * dt
        mi = m_fn(ti, xi, vi)
        ms.append(mi)

        F_drive = 0.0
        for fn, h in zip(d_fns, d_hist):
            f = fn(ti, xi, vi)
            h.append(f)
            F_drive += f

        caps = [abs(fn(ti, xi, vi)) for fn in f_fns]
        F_fric = sum(caps)
        Fd_hist.append(F_drive)
        Ff_hist.append(F_fric)
        pinned = (xi >= x_max and F_drive > 0.0) \
              or (xi <= x_min and F_drive < 0.0)

        # Get frictional direction.
        if vi != 0.0:
            s = 1.0 if vi > 0.0 else -1.0
            hist_s = s
        elif F_drive != 0.0:
            s = 1.0 if F_drive > 0.0 else -1.0
            hist_s = last_hist_s # maintain prev plotted sign.
        else:
            s = last_s
            hist_s = last_hist_s
        last_s = s
        last_hist_s = hist_s

        for c, h in zip(caps, f_hist):
            h.append(-hist_s * c) # history/plotting only. full capacity.

        if vi == 0.0 and (pinned or abs(F_drive) <= F_fric):
            acc = 0.0 # a stop carries the load, or friction holds.
        else:
            acc = (F_drive - s * F_fric) / mi

        if i == N - 1 or ti >= t_quit:
            accs.append(nan)
            break

        v_new = vi + acc * dt
        if F_fric > 0.0 and vi != 0.0 and (v_new > 0.0) != (vi > 0.0):
            # Velocity crosses zero, so stop for at-least a tick so the next step
            # decides (via the stuck check) whether static friction holds it.
            v_new = 0.0
        x_new = xi + v_new * dt

        # Record genuine impacts, hold if already resting on the stop
        if x_new <= x_min and v_new < 0.0:
            if xi > x_min:
                impacts.append(Impact(
                    "min",
                    ti + dt,
                    v_new,
                    0.5 * mi * v_new*v_new
                ))
                v_new = -restitution * v_new
            else:
                v_new = 0.0
            x_new = x_min
        elif x_new >= x_max and v_new > 0.0:
            if xi < x_max:
                impacts.append(Impact(
                    "max",
                    ti + dt,
                    v_new,
                    0.5 * mi * v_new*v_new))
                v_new = -restitution * v_new
            else:
                v_new = 0.0
            x_new = x_max

        if impacts and t_after_impact is not None and t_quit == inf:
            t_quit = impacts[0].t + t_after_impact

        accs.append((v_new - vi) / dt)
        xi, vi = x_new, v_new
        xs.append(xi)
        vs.append(vi)

    hist = dict(zip(d_names, d_hist)) | dict(zip(f_names, f_hist))
    F = {name: np.asarray(hist[name]) for name in forces}

    return Journey(
        title=title,
        x_min=x_min, x_max=x_max,
        t=np.arange(len(xs)) * dt,
        x=np.asarray(xs),
        v=np.asarray(vs),
        a=np.asarray(accs),
        m=np.asarray(ms),
        forces=F,
        frictions=frozenset(f_names),
        F_drive=np.asarray(Fd_hist),
        F_frictional=np.asarray(Ff_hist),
        impacts=tuple(impacts)
    )
