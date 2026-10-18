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
    orings: tuple # List of dynamic o-rings.
    F_fric_running = calc(u.N, # Net sliding friction.
        lambda s: sum(o.F_friction_running for o in s.orings))
    F_fric_breakaway = calc(u.N, # Net static friction.
        lambda s: sum(o.F_friction_breakaway for o in s.orings))
    seals: tuple # List of static o-rings.


    # --- force balances (+ve margin = works) ---
    @staticmethod
    def excess(drive, oppose):
        """ Fractional excess of drive over oppose (safety factor - 1). """
        return drive / oppose - 1 if oppose > 0 else float("inf")

    margin_close = calc(u.N,
        lambda s: s.F_pneumo - max(s.F_fric_breakaway, s.F_fric_running)
    )
    rmargin_close = calc(u.percent,
        lambda s: s.excess(s.F_pneumo, max(s.F_fric_breakaway, s.F_fric_running))
    )


    # --- tolerances ---
    def tolerances(self):
        seals = lambda level: tuple(o.tweak(friction_level=level)
                                    for o in self.orings)
        return {
            "P_pneumo": 0.1,
            "orings": (seals(OringFriction.LOW), seals(OringFriction.HIGH)),
        }
    worst_margin_close = calc(u.N,
        lambda s: search.corners(s, "margin_close", s.tolerances()).value)
    worst_rmargin_close = calc(u.percent,
        lambda s: search.corners(s, "rmargin_close", s.tolerances()).value)


    # --- optimisation ---
    cost = calc(None, lambda s: (s.P_pneumo/1e5)**2
                              + (s.D_pneumo/5e-3)**2
                            )


    # --- dynamics ---

    stroke_time = calc(u.ms,
        lambda s: _value(s._actuating.stroke_time, inf))

    t_crack_actuate = calc(u.ms,
        lambda s: _value(s._actuating.t_start, inf))
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
        """ Run the travel sim in the given `direction` ("open" or "close"). """
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
        return self.actuating.plot()



def main():
    qd = QuickDisconnect(
        m=0.00339039, stroke=3.4e-3,
        tau_fill=5e-3,
        sim_dt=0.001e-3, sim_t_end=200e-3,
        D_tube=6e-3,
        P_pneumo=10e5, D_pneumo=15.2e-3,
        orings=(
            # NOTE: seal DP values are not dynamically updated.
            RodSeal("Piston inner",
                oring_ID=6.0e-3, oring_CS=1.5e-3,
                gland_ID=6.0e-3, gland_OD=8.55e-3, gland_Lz=2e-3,
                DP=10e5, material="FKM", hardness=75, lubricated=True
            ),
            PistonSeal("Piston outer",
                oring_ID=12.5e-3, oring_CS=1.5e-3,
                gland_ID=12.78e-3, gland_OD=15.2e-3, gland_Lz=2e-3,
                DP=10e5, material="FKM", hardness=75, lubricated=True
            ),
        ),
        seals=(
            RodSeal("Entry",
                oring_ID=6.0e-3, oring_CS=1.5e-3,
                gland_ID=6.0e-3, gland_OD=8.55e-3, gland_Lz=2e-3,
                DP=10e5, material="FKM", hardness=75, lubricated=True
            ),
        )
    )

    qd.report()
    for oring in itertools.chain(qd.orings, qd.seals):
        oring.outline()



if __name__ == "__main__":
    main()
    plt.show()
