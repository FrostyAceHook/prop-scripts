"""
Normally-open spring-return pneumatically-closed main oxidiser valve.
  opens  : tank pressure on the sleeve, spring
  closes : pneumatic pressure on the sleeve
  resists: o-ring friction
"""

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

def frustum_area(h, D1, D2):
    return pi/2*(D1 + D2)*sqrt(h*h + (D1 - D2)**2/4)

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



class MainOxidiserValve(System):
    # --- movement ---
    m: u.kg # Moving mass: plunger mass + 1/3 spring mass
    stroke: u.mm # Plunger stroke length.
    chi_seat: u.mm # Un-seating length constant.
    tau_fill: u.ms # Pneumo fill time constant.
    tau_vent: u.ms # Pneumo vent time constant.
    sim_dt: u.ms # Simulation time step.
    sim_t_end: u.ms # Simulation max sim time.

    # --- sleeve ---
    ID_sleeve: u.mm
    th_US_sleeve: u.mm # tank-side thickness.
    th_DS_sleeve: u.mm # cc-side thickness.
    OD_US_sleeve = calc(u.mm,
        lambda s: s.ID_sleeve + 2*s.th_US_sleeve,
        lambda s, OD: dict(th_US_sleeve=(OD - s.ID_sleeve)/2))
    OD_DS_sleeve = calc(u.mm,
        lambda s: s.ID_sleeve + 2*s.th_DS_sleeve,
        lambda s, OD: dict(th_DS_sleeve=(OD - s.ID_sleeve)/2))
    A_US_sleeve = calc(u.mm2,
        lambda s: circ_area(s.OD_US_sleeve) - circ_area(s.ID_sleeve),
        lambda s, A: dict(OD_US_sleeve=
            circ_area_to_D(A + circ_area(s.ID_sleeve))))
    A_DS_sleeve = calc(u.mm2,
        lambda s: circ_area(s.OD_DS_sleeve) - circ_area(s.ID_sleeve),
        lambda s, A: dict(OD_DS_sleeve=
            circ_area_to_D(A + circ_area(s.ID_sleeve))))

    # --- tank side: opening+closing ---
    Pmin_tank: u.bar
    Pmax_tank: u.bar
    Fmin_tank_closed = calc(u.N,
        lambda s: s.Pmin_tank*s.A_US_sleeve,
        lambda s, F: dict(A_US_sleeve=F/s.Pmin_tank))
    Fmax_tank_closed = calc(u.N,
        lambda s: s.Pmax_tank*s.A_US_sleeve,
        lambda s, F: dict(A_US_sleeve=F/s.Pmax_tank))
    Fmin_tank_open = calc(u.N,
        lambda s: s.Pmin_tank*(s.A_US_sleeve - s.A_DS_sleeve),
        lambda s, F: dict(A_DS_sleeve=s.A_US_sleeve - F/s.Pmin_tank))
    Fmax_tank_open = calc(u.N,
        lambda s: s.Pmax_tank*(s.A_US_sleeve - s.A_DS_sleeve),
        lambda s, F: dict(A_DS_sleeve=s.A_US_sleeve - F/s.Pmax_tank))

    # --- pneumo side: closing ---
    P_pneumo: u.bar
    D_pneumo: u.mm
    A_pneumo = calc(u.mm2,
        lambda s: circ_area(s.D_pneumo) - circ_area(s.OD_DS_sleeve),
        lambda s, A: dict(D_pneumo=
            circ_area_to_D(A + circ_area(s.OD_DS_sleeve))))
    F_pneumo = calc(u.N,
        lambda s: s.P_pneumo*s.A_pneumo,
        lambda s, F: dict(A_pneumo=F/s.P_pneumo))

    # --- spring: opening ---
    k_spring: u.N_mm
    ell_relaxed: u.mm # Free length.
    ell_open: u.mm # Installed length with the valve open.
    ell_closed = calc(u.mm, # Installed length with the valve closed. Modifies
                            # stroke if set.
        lambda s: s.ell_open - s.stroke,
        lambda s, ell: dict(stroke=s.ell_open - ell))
    def F_spring(self, x):
        return max(0.0,
            self.k_spring * (self.ell_relaxed - (self.ell_closed + x)))
    F_spring_open = calc(u.N,
        lambda s: s.F_spring(s.stroke),
        lambda s, F: dict(ell_open=s.ell_relaxed - F / s.k_spring))
    F_spring_closed = calc(u.N,
        lambda s: s.F_spring(0.0),
        lambda s, F: dict(ell_closed=s.ell_relaxed - F / s.k_spring))

    # Generic die-spring limit: compression from free length, as a fraction of
    # free length. Long-life catalogue figures are roughly 20-30% depending on
    # grade (heavier grades lower); confirm with the spring maker.
    max_spring_compression = 0.4 # [-]
    spring_compression = calc(u.percent, # Most compressed (closed).
        lambda s: (s.ell_relaxed - s.ell_closed) / s.ell_relaxed)


    # --- friction ---
    orings: tuple = None # Seals. Sized from the geometry if not given.
    @computed(None)
    def seals(self):
        """ `orings` if given, else seals sized to suit the geometry. """
        if self.orings is not None:
            return self.orings
        return (
            RodSeal("Sleeve upstream", self.OD_US_sleeve, 3.53e-3,
                     DP=self.Pmax_tank),
            piston_seal("Sleeve downstream", self.D_pneumo, 3.53e-3,
                        DP=self.P_pneumo),
        )
    F_fric_running = calc(u.N, # Net sliding friction.
        lambda s: sum(o.F_friction_running for o in s.orings))
    F_fric_breakaway = calc(u.N, # Net static friction.
        lambda s: sum(o.F_friction_breakaway for o in s.orings))


    # --- force balances (+ve margin = works) ---
    @staticmethod
    def excess(drive, oppose):
        """ Fractional excess of drive over oppose (safety factor - 1). """
        return drive / oppose - 1 if oppose > 0 else float("inf")

    margin_hold_open = calc(u.N,
        lambda s: s.F_spring_open + s.Fmin_tank_open)
    rmargin_hold_open = calc(u.percent,
        lambda s: (s.F_spring_open + s.Fmin_tank_open + s.F_fric_breakaway)
                / s.F_pneumo)

    margin_hold_closed = calc(u.N,
        lambda s: s.F_pneumo - s.Fmax_tank_closed - s.F_spring_closed)
    rmargin_hold_closed = calc(u.percent,
        lambda s: s.excess(s.F_pneumo, s.Fmax_tank_closed + s.F_spring_closed))

    margin_close = calc(u.N,
        lambda s: s.F_pneumo - max(
            s.F_spring_open + s.F_fric_breakaway,
            s.F_spring_closed + s.F_fric_running
        ))
    rmargin_close = calc(u.percent,
        lambda s: s.excess(s.F_pneumo, max(
            s.F_spring_open + s.F_fric_breakaway,
            s.F_spring_closed + s.F_fric_running
        )))

    margin_open = calc(u.N,
        lambda s: min(
            s.Fmin_tank_closed + s.F_spring_closed - s.F_fric_breakaway,
            s.Fmin_tank_open + s.F_spring_open - s.F_fric_running
        ))
    rmargin_open = calc(u.percent,
        lambda s: min(
            s.excess(s.Fmin_tank_closed + s.F_spring_closed, s.F_fric_breakaway),
            s.excess(s.Fmin_tank_open + s.F_spring_open, s.F_fric_running)
        ))


    # --- tolerances ---
    def tolerances(self):
        seals = lambda level: tuple(o.tweak(friction_level=level)
                                    for o in self.orings)
        return {
            "k_spring": 0.1,
            "ell_open": (self.ell_open - 0.05e-3, self.ell_open + 0.05e-3),
            "stroke": (self.stroke - 0.05e-3, self.stroke + 0.05e-3),
            "P_pneumo": 0.1,
            "Pmax_tank": 0.1,
            "orings": (seals(OringFriction.LOW), seals(OringFriction.HIGH)),
        }
    worst_margin_hold_open = calc(u.N,
        lambda s: search.corners(s, "margin_hold_open", s.tolerances()).value)
    worst_rmargin_hold_open = calc(u.percent,
        lambda s: search.corners(s, "rmargin_hold_open", s.tolerances()).value)
    worst_margin_hold_closed = calc(u.N,
        lambda s: search.corners(s, "margin_hold_closed", s.tolerances()).value)
    worst_rmargin_hold_closed = calc(u.percent,
        lambda s: search.corners(s, "rmargin_hold_closed", s.tolerances()).value)
    worst_margin_close = calc(u.N,
        lambda s: search.corners(s, "margin_close", s.tolerances()).value)
    worst_rmargin_close = calc(u.percent,
        lambda s: search.corners(s, "rmargin_close", s.tolerances()).value)
    worst_margin_open = calc(u.N,
        lambda s: search.corners(s, "margin_open", s.tolerances()).value)
    worst_rmargin_open = calc(u.percent,
        lambda s: search.corners(s, "rmargin_open", s.tolerances()).value)


    # --- optimisation ---
    cost = calc(None, lambda s: (s.P_pneumo/1e5)**2
                              + (s.D_pneumo/10e-3)**2
                              + (s.th_US_sleeve/2e-3)**2
                              + (s.ell_open/10e-3)**2
                              + (s.ell_relaxed/5e-3)**2
                            )






def get_orings():
    TBO = 1e-5 # to-be optimised smile :)
    seat = PistonSeal("Seat (BS114)",
        oring_ID=15.54e-3, oring_CS=2.62e-3,
        gland_ID=TBO, gland_OD=20.2e-3, gland_Lz=TBO+1,
        DP=70e5, material="FKM", hardness=75, lubricated=True,
    )
    upstream = RodSeal("Upstream (BS218)",
        oring_ID=31.34e-3, oring_CS=2.62e-3,
        gland_ID=31.4e-3, gland_OD=TBO+1, gland_Lz=TBO+1,
        DP=70e5, material="FKM", hardness=75, lubricated=True,
    )
    downstream = RodSeal("Downstream (BS213)",
        oring_ID=23.4e-3, oring_CS=2.62e-3,
        gland_ID=23.4e-3, gland_OD=TBO+1, gland_Lz=TBO+1,
        DP=70e5, material="FKM", hardness=75, lubricated=True,
    )
    pneumo = PistonSeal("Pneumo (BS225)",
        oring_ID=47.23e-3, oring_CS=2.62e-3,
        gland_ID=TBO, gland_OD=50e-3, gland_Lz=TBO+1,
        DP=70e5, material="FKM", hardness=75, lubricated=True,
    )
    seat = seat.tweak(squeeze=0.17, gland_fill=0.75)
    upstream = upstream.tweak(squeeze=0.17, gland_fill=0.7)
    downstream = downstream.tweak(squeeze=0.13, gland_fill=0.7)
    pneumo = pneumo.tweak(squeeze=0.13, gland_fill=0.7)
    return (seat, upstream, downstream, pneumo)


def main():
    mov = MainOxidiserValve(
        m=0.05794474, stroke=15e-3,
        chi_seat=0.3e-3, tau_fill=5e-3, tau_vent=5e-3, # TODO:
        sim_dt=0.001e-3, sim_t_end=200e-3,
        ID_sleeve=20e-3, th_US_sleeve=6e-3, th_DS_sleeve=1.6e-3,
        Pmin_tank=45e5, Pmax_tank=70e5,
        P_pneumo=30e5, D_pneumo=50e-3,
        # k_spring=80e3, ell_relaxed=30e-3, ell_open=25e-3,
        k_spring=0.0, ell_relaxed=30.0, ell_open=30.0,
        orings=get_orings(), # NOTE: seal DP values are not dynamically updated.
    )

    def optimise(sys):
        params = {
            "D_pneumo": (10e-3, 90e-3),
            # "P_pneumo": (5e5, 30e5),
            "th_US_sleeve": (1.6e-3, 8e-3),
            # "k_spring": (5e3, 150e3),
            # "ell_relaxed": (35e-3, 80e-3),
            # "ell_open": (5e-3, 80e-3),
        }
        constraints = {
            # "worst_margin_hold_open": (100, None),
            # "worst_rmargin_hold_open": (0.1, None),
            # "worst_margin_hold_closed": (100, None),
            # "worst_rmargin_hold_closed": (0.15, None),
            # "worst_margin_close": (30, None),
            # "worst_rmargin_close": (0.05, None),
            # "worst_margin_open": (100, None),
            # "worst_rmargin_open": (0.15, None),

            "margin_hold_open": (60, None),
            "rmargin_hold_open": (0.2, None),
            "margin_hold_closed": (100, None),
            "rmargin_hold_closed": (0.2, None),
            "margin_close": (150, None),
            "rmargin_close": (0.4, None),
            "margin_open": (150, None),
            "rmargin_open": (0.4, None),
            "spring_compression": (None,
                    MainOxidiserValve.max_spring_compression),
        }
        coarse = sys.tweak(sim_dt=0.02e-3)
        opt = search.optimise(coarse, "cost", params, constraints,
                seed=0, maxiter=30, tol=1e-6, popsize=10, workers=-1)
        opt.report()
        best = opt.system.tweak(sim_dt=sys.sim_dt)
        best.report()
        return best
    # mov = optimise(mov)

    mov_30barpneumo = mov.tweak(
            P_pneumo=30e5, D_pneumo=46e-3,
            th_US_sleeve=5.6e-3,
        )
    mov_30barpneumo.report()
    for oring in mov_30barpneumo.orings:
        oring.outline()


if __name__ == "__main__":
    main()
    plt.show()
