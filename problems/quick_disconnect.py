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

import search
import units as u
from fmt import style_ax
from oring import *
from system import *
from travel import *


SQRT2 = sqrt(2)

def circ_area(D):
    return pi/4 * D*D
def circ_area_to_D(A):
    return sqrt(A*4/pi)
def circ_perim(D):
    return pi*D
def circ_perim_to_D(p):
    return p/pi

def calc_circ_D(name):
    return derived(u.mm, name, circ_area_to_D, circ_area)
def calc_circ_A(name):
    return derived(u.mm2, name, circ_area, circ_area_to_D)

def _value(x, default):
    """ `x`, or `default` if it's None. """
    return default if x is None else x
def _nanmax(*xs):
    """ max(), but nan if any value is nan. """
    return nan if any(isnan(x) for x in xs) else max(xs)



class QuickDisconnect(System):
    # --- movement ---
    m: u.kg # Moving mass: piston mass
    stroke: u.mm # Piston stroke length.
    tau_fill: u.ms # Pneumo fill time constant.
    sim_dt: u.ms # Simulation time step.
    sim_t_end: u.ms # Simulation max sim time.

    # --- tube ---
    D_tube: u.mm
    A_tube = calc_circ_A("D_tube")

    # --- pneumo ---
    P_pneumo: u.bar
    D_pneumo: u.mm
    A_pneumo = calc(u.mm2,
        lambda s: circ_area(s.D_pneumo) - s.A_tube,
        lambda s, A: dict(D_pneumo=circ_area_to_D(A + s.A_tube)))
    F_pneumo = calc(u.N,
        lambda s: s.P_pneumo*s.A_pneumo,
        lambda s, F: dict(A_pneumo=F/s.P_pneumo))

    # --- friction ---
    orings: tuple = None # O-rings: (tube, piston inner, piston outer)
    @computed(None)
    def all_seals(self):
        if self.orings is not None:
            return self.orings
        return (
            RodSeal.auto("Tube", self.D_tube,
                    squeeze=0.15, gland_fill=0.7,
                    DP=self.P_pneumo, CS=1.5e-3,
                    material="FKM", hardness=75, lubricated=True),
            RodSeal.auto("Piston inner", self.D_tube,
                    squeeze=0.125, gland_fill=0.7,
                    DP=self.P_pneumo, CS=1.5e-3,
                    material="FKM", hardness=75, lubricated=True),
            PistonSeal.auto("Piston outer", self.D_pneumo,
                    squeeze=0.15, gland_fill=0.7,
                    DP=self.P_pneumo, CS=1.5e-3,
                    material="FKM", hardness=75, lubricated=True),
        )
    dynamic_seals = computed(None)(lambda s: s.all_seals[1:])

    F_fric_running = calc(u.N, # Net sliding friction.
        lambda s: sum(o.F_friction_running for o in s.dynamic_seals))
    F_fric_breakaway = calc(u.N, # Net static friction.
        lambda s: sum(o.F_friction_breakaway for o in s.dynamic_seals))


    # --- force balances (+ve margin = works) ---
    @staticmethod
    def excess(drive, oppose):
        """ Fractional excess of drive over oppose (safety factor - 1). """
        return drive / oppose - 1 if oppose > 0 else float("inf")

    margin_actuate = calc(u.N,
        lambda s: s.F_pneumo - max(s.F_fric_breakaway, s.F_fric_running)
    )
    rmargin_actuate = calc(u.percent,
        lambda s: s.excess(s.F_pneumo, max(s.F_fric_breakaway, s.F_fric_running))
    )


    # --- tolerances ---
    def tolerances(self):
        return {
            "P_pneumo": 0.1,
            "orings": (
                tuple(o.loose for o in self.all_seals),
                tuple(o.tight for o in self.all_seals),
            )
        }
    worst_margin_actuate = calc(u.N,
        lambda s: search.corners(s, "margin_actuate", s.tolerances()).value)
    worst_rmargin_actuate = calc(u.percent,
        lambda s: search.corners(s, "rmargin_actuate", s.tolerances()).value)


    # --- optimisation ---
    cost = calc(None, lambda s: (s.P_pneumo/1e5)**2
                              + (s.D_pneumo/5e-3)**2
                            )


    # --- dynamics ---

    crack_time = calc(u.ms,
        lambda s: _value(s._actuating.t_start, inf))
    stroke_time = calc(u.ms,
        lambda s: _value(s._actuating.stroke_time, inf))
    t_actuate = calc(u.ms,
        lambda s: _value(s._actuating.t_impact, inf))
    v_impact_actuate = calc(u.m_s,
        lambda s: abs(_value(s._actuating.v_impact, nan)))
    E_impact_actuate = calc(u.J,
        lambda s: _value(s._actuating.E_impact, nan))

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



def main():
    qd = QuickDisconnect(
        m=0.00339039, stroke=3.4e-3,
        tau_fill=5e-3,
        sim_dt=0.001e-3, sim_t_end=200e-3,
        D_tube=6e-3,
        P_pneumo=10e5, D_pneumo=15.2e-3,
    )

    qd.report()
    for oring in qd.all_seals:
        oring.outline(succinct=True)
    qd.plot()


if __name__ == "__main__":
    main()
    plt.show()
