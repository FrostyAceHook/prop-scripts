"""
Push-fit pneumatically-actuator quick-disconnect.
"""

import itertools
from math import pi, sqrt, exp, inf, nan, isnan
import matplotlib.pyplot as plt
import numpy as np

# Hack-in the utility module.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utility"))

import spec as u
from spec import show
from oring import *
from system import *
from travel import *
from helpers import *



class QuickDisconnect(System):
    # --- movement ---
    m: u.kg # Moving mass: piston mass
    stroke: u.mm # Piston stroke length.
    tau_fill: u.ms # Pneumo fill time constant.
    sim_dt: u.ms # Simulation time step.
    sim_t_end: u.ms # Simulation max sim time.

    # --- geometry ---
    tube: Circle
    piston: Circle

    # --- pneumo ---
    P_pneumo: u.bar
    A_pneumo = calc(u.mm2,
        lambda s: s.piston.A - s.tube.A,
        lambda s, A: {"piston.A": A + s.tube.A})
    F_pneumo = calc(u.N,
        lambda s: (s.P_pneumo - u.bar_(1))*s.A_pneumo,
        lambda s, F: dict(A_pneumo=F/(s.P_pneumo - u.bar_(1))))

    # --- seals ---
    # Each slot is a Groove (sized by hand) or an AutoGroove (sized for its
    # diameter whenever that changes, e.g. while optimising D_pneumo), with its
    # own friction scatter. The seals add D_tube or D_pneumo, P_pneumo and the
    # shared compound.
    gr_tube: Groove | AutoGroove
    gr_inner: Groove | AutoGroove
    gr_outer: Groove | AutoGroove

    def _seal(self, slot, kind, name, D, sliding=True):
        return slot.seal(kind, name, D, DP=self.P_pneumo, sliding=sliding)
    or_tube = calc(RodSeal, # Tube in the body (static).
        lambda s: s._seal(s.gr_tube, RodSeal, "Tube", s.tube.D, sliding=False),
        lambda s, seal: dict(gr_tube=Groove.of(seal)))
    or_inner = calc(RodSeal, # Piston, sliding on the tube.
        lambda s: s._seal(s.gr_inner, RodSeal, "Piston inner", s.tube.D),
        lambda s, seal: dict(gr_inner=Groove.of(seal)))
    or_outer = calc(PistonSeal, # Piston, sliding in the bore.
        lambda s: s._seal(s.gr_outer, PistonSeal, "Piston outer",
                s.piston.D),
        lambda s, seal: dict(gr_outer=Groove.of(seal)))

    def seals(self):
        return (self.or_tube, self.or_inner, self.or_outer)

    # --- friction (the tube seal is static) ---
    F_fric_running = calc(u.N, # Net sliding friction.
        lambda s: s.or_inner.F_friction_running
                + s.or_outer.F_friction_running)
    F_fric_breakaway = calc(u.N, # Net static friction.
        lambda s: s.or_inner.F_friction_breakaway
                + s.or_outer.F_friction_breakaway)


    # --- force balances (+ve margin = works) ---
    actuate = calc(Balance, lambda s: Balance.of(
        (s.F_pneumo, s.F_fric_breakaway),
        (s.F_pneumo, s.F_fric_running)))


    # --- requirements ---
    # Actuates at the worst supply pressure and the 3σ worst build. The seals'
    # own requirements come along under their paths.
    def requirements(self):
        return super().requirements() | {
            corner("actuate.margin"): (0.0, None),
        }



    # --- dynamics ---

    crack_time = calc(u.ms,
        lambda s: ifnan(s._actuating.t_start, inf))
    stroke_time = calc(u.ms,
        lambda s: ifnan(s._actuating.stroke_time, inf))
    t_actuate = calc(u.ms,
        lambda s: ifnan(s._actuating.t_impact, inf))
    v_impact_actuate = calc(u.m_s,
        lambda s: abs(ifnan(s._actuating.v_impact, nan)))
    E_impact_actuate = calc(u.J,
        lambda s: ifnan(s._actuating.E_impact, nan))

    def _forces(s):
        def pneumo(t, x, v):
            return s.F_pneumo * (1.0 - exp(-t/s.tau_fill))
        @Friction
        def o_ring(t, x, v):
            return s.F_fric_breakaway if v == 0 else s.F_fric_running
        return dict(pneumo=pneumo, o_ring=o_ring)

    def travel(self, **kw):
        opts = dict(
            dt=self.sim_dt,
            t_end=self.sim_t_end,
            t_after_impact=0.0,
            mass=self.m,
            x0=0.0,
            x_min=0.0,
            x_max=self.stroke,
            title="QD actuating",
        )
        opts.update(kw)
        return embark(self._forces(), **opts)

    # Cached runs. Private, so `.report()` doesn't print them.
    _actuating = computed(None)(lambda s: s.travel())
    def actuating(self):
        return self._actuating


    def plot(self):
        return self._actuating.plot()


def opts_oring():
    return dict(material="NBR", hardness=u.shoreA_(70, tol=5), lubricated=True)

def build():
    """ The design as drawn: every seal hand-sized. """
    mm, ms, bar = u.mm_, u.ms_, u.bar_
    return QuickDisconnect(
        m=0.00339039, stroke=mm(3.4),
        tau_fill=ms(5),
        sim_dt=ms(0.001), sim_t_end=ms(200),
        tube=Circle(D=mm(6, tol=(-0.1, 0.05))),
        piston=Circle(D=mm(15.2, tol=(0, 0.05))),
        P_pneumo=bar(10, env=(8, 10)),
        gr_tube=Groove(oring=finer("mor6x1"),
                root=mm(7.72, tol=0.05), Lz=mm(1.4, tol=0.1), **opts_oring()),
        gr_inner=Groove(oring=finer("mor6x1"),
                root=mm(7.72, tol=0.05), Lz=mm(1.4, tol=0.1), **opts_oring()),
        gr_outer=Groove(oring=finer("mor12.5x1.5"),
                root=mm(12.78, tol=0.05), Lz=mm(2.0, tol=0.1), **opts_oring()),
    )
def resizing(qd):
    return qd.tweak(dict(
        gr_tube =AutoGroove(squeeze=0.150, **opts_oring()),
        gr_inner=AutoGroove(squeeze=0.125, **opts_oring()),
        gr_outer=AutoGroove(squeeze=0.150, **opts_oring()),
    ))



def main():
    qd = build()
    qd.report()
    for seal in qd.seals():
        seal.outline(succinct=True)
    qd.verify().report()



def tolerances(qd):
    """ What drives the actuation margin, and its worst build. """
    qd.variation("actuate.margin").report()
    worst = qd.corners("actuate.margin", check=20000, seed=1)
    worst.report()
    build = worst.build()
    print("worst build:", build.display("actuate.margin"),
          build.display("F_fric_breakaway"), build.display("P_pneumo"))
    qd.corners("actuate.margin", limits=True).report() # arithmetic worst


def sweeps(qd):
    """ Margin against supply pressure (as built), and against bore (as
    designed: the outer seal re-sized for each bore). """
    k3 = corner("actuate.margin")
    # A shifting envelope, so the sweep can move the supply past 8-10 bar.
    shifting = qd.tweak(P_pneumo=u.bar_(10, env=(-2, 0), abs=True))
    P = [u.bar_(p) for p in (4, 5, 6, 7, 8, 9, 10, 11, 12)]
    sweep = shifting.sweep("P_pneumo", P, "actuate.margin", k3,
                           "F_fric_breakaway")
    sweep.report()
    sweep.plot()

    design = resizing(qd)
    D = u.mm_(np.linspace(12, 30, 100))
    sweep = design.sweep("piston.D", D, "actuate.margin", k3,
                         "or_outer.F_friction_breakaway")
    sweep.report()
    sweep.plot()


def grid(qd):
    """ Where the 3σ actuation margin runs out, over supply and bore. """
    design = resizing(qd).tweak(P_pneumo=u.bar_(10, env=(-2, 0), abs=True))
    P = u.bar_(np.linspace(3, 10, 10))
    D = u.mm_(np.linspace(10, 30, 50))
    g = design.grid("piston.D", D, "P_pneumo", P, corner("actuate.margin"))
    g.plot()
    return g


def optimise(qd):
    """ The smallest bore that actuates at 3σ with the outer seal's
    requirements met, then the most robust bore at 10 bar. """
    design = resizing(qd)
    reqs = design.requirements().without("or_tube", "or_inner")
    best = design.optimise("piston.D", {"piston.D": (u.mm_(10), u.mm_(20))},
                           reqs, seed=1, popsize=8, maxiter=20, verbose=False)
    best.report()
    robust = design.optimise(
            worst_sigma.without("or_tube", "or_inner"),
            {"piston.D": (u.mm_(10), u.mm_(20))}, reqs.nominal(),
            maximise=True, seed=1, popsize=8, maxiter=20, verbose=False)
    robust.report()
    robust.system.verify().report()
    return best, robust


if __name__ == "__main__":
    main()
    plt.show()
