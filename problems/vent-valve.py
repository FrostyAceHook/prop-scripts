"""
Spring-return vent valve.
  opens  : tank pressure under the seat, spring
  closes : pneumatic pressure on the piston
  resists: o-ring friction
"""

from math import pi, sqrt, exp, inf, nan
import matplotlib.pyplot as plt

# Hack-in the utility module.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utility"))

import travel
import units as u
from oring import PistonSeal, RodSeal
from system import System, calc, derived, computed


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


class VentValve(System):
    # --- movement ---
    m: u.kg # Moving mass: plunger mass + 1/3 spring mass
    stroke: u.mm # Plunger stroke length.
    chi_seat: u.mm # Un-seating length constant.
    tau_fill: u.ms # Pneumo fill time constant.
    tau_vent: u.ms # Pneumo vent time constant.
    sim_dt: u.ms # Simulation time step.
    sim_t_end: u.ms # Simulation max sim time.

    # --- tank side: opening ---
    Pmin_tank: u.bar
    Pmax_tank: u.bar
    D_ball: u.mm
    D_seat = derived(u.mm, "D_ball", lambda x: x/SQRT2, lambda x: x*SQRT2)
    A_seat = calc_circ_A("D_seat")
    A_ball = calc_circ_A("D_ball")
    Fmin_tank_open = calc(u.N,
        lambda s: s.Pmin_tank*s.A_ball,
        lambda s, F: dict(A_ball=F/s.Pmin_tank))
    Fmax_tank_open = calc(u.N,
        lambda s: s.Pmax_tank*s.A_ball,
        lambda s, F: dict(A_ball=F/s.Pmax_tank))
    Fmin_tank_closed = calc(u.N,
        lambda s: s.Pmin_tank*s.A_seat,
        lambda s, F: dict(A_seat=F/s.Pmin_tank))
    Fmax_tank_closed = calc(u.N,
        lambda s: s.Pmax_tank*s.A_seat,
        lambda s, F: dict(A_seat=F/s.Pmax_tank))

    # --- past-seal flow ---
    D_seat_bore: u.mm
    A_seat_bore = calc_circ_A("D_seat_bore")
    D_plunger_bore: u.mm
    A_plunger_bore = calc(u.mm2,
        lambda s: circ_area(s.D_plunger_bore) - circ_area(s.D_ball),
        lambda s, A: dict(D_plunger_bore=
            circ_area_to_D(A + circ_area(s.D_ball))))
    D_seat_cone = calc_circ_D("A_seat_cone")
    A_seat_cone = calc(u.mm2,
        lambda s: frustum_area(s.stroke/2, s.D_seat + s.stroke, s.D_seat),
        lambda s, A: dict(stroke=sqrt(s.D_seat**2 + 2*SQRT2*A/pi) - s.D_seat))

    # --- pneumo side: closing ---
    P_pneumo: u.bar
    D_pneumo: u.mm
    A_pneumo = calc_circ_A("D_pneumo")
    F_pneumo = calc(u.N,
        lambda s: s.P_pneumo*s.A_pneumo,
        lambda s, F: dict(A_pneumo=F/s.P_pneumo))

    # --- spring: opening ---
    k_spring: u.N_mm
    ell_relaxed: u.mm # Free length.
    ell_solid: u.mm # "Solid" length (for utilisation calc).
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
    spring_utilisation = calc(u.percent, # Fraction of available deflection used
                                         # when closed (keep below ~0.85).
        lambda s: (s.ell_relaxed - s.ell_closed) / (s.ell_relaxed - s.ell_solid))


    # --- friction ---
    orings: tuple # List of o-rings.
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
        lambda s: s.F_spring_open)
    rmargin_hold_open = calc(u.percent,
        lambda s: (s.F_spring_open + s.F_fric_breakaway) / s.F_pneumo)

    margin_hold_closed = calc(u.N,
        lambda s: s.F_pneumo - s.Fmax_tank_closed - s.F_spring_closed)
    rmargin_hold_closed = calc(u.percent,
        lambda s: s.excess(s.F_pneumo, s.Fmax_tank_closed + s.F_spring_closed))

    margin_close = calc(u.N,
        lambda s: s.F_pneumo - s.Fmax_tank_open - max(
            s.F_spring_open + s.F_fric_breakaway,
            s.F_spring_closed + s.F_fric_running
        ))
    rmargin_close = calc(u.percent,
        lambda s: s.excess(s.F_pneumo, s.Fmax_tank_open + max(
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


    # --- dynamics ---
    stroke_t_open = calc(u.ms, lambda s: s.opening.stroke_time or inf)
    t_open = calc(u.ms, lambda s: s.opening.t_impact or inf)
    t_crack_open = calc(u.ms, lambda s: s.opening.t_start or inf)
    v_impact_open = calc(u.m_s, lambda s: abs(s.opening.v_impact or nan))
    E_impact_open = calc(u.J, lambda s: s.opening.E_impact or nan)

    stroke_t_close = calc(u.ms, lambda s: s.closing.stroke_time or inf)
    t_close = calc(u.ms, lambda s: s.closing.t_impact or inf)
    t_crack_close = calc(u.ms, lambda s: s.closing.t_start or inf)
    v_impact_close = calc(u.m_s, lambda s: abs(s.closing.v_impact or nan))
    E_impact_close = calc(u.J, lambda s: s.closing.E_impact or nan)


    def _forces(s, is_opening=True):
        P_tank = s.Pmin_tank if is_opening else s.Pmax_tank
        def tank(t, x, v):
            t = min(max(x/s.chi_seat, 0.0), 1.0)
            return P_tank * (s.A_seat + t * (s.A_ball - s.A_seat))
        def pneumo(t, x, v):
            if is_opening:
                return -s.F_pneumo * exp(-t/s.tau_vent)
            else:
                return -s.F_pneumo * (1.0 - exp(-t/s.tau_fill))
        def spring(t, x, v):
            return s.F_spring(x)
        @travel.Friction
        def o_ring(t, x, v):
            return s.F_fric_breakaway if v == 0 else s.F_fric_running
        return dict(tank=tank, pneumo=pneumo, spring=spring, o_ring=o_ring)

    def travel(self, direction, **kw):
        """ Run the travel sim in the given `direction` ("open" or "close"). """
        if direction == "open":
            x0 = 0.0
        elif direction == "close":
            x0 = self.stroke
        else:
            raise ValueError("direction must be 'open' or 'close'")
        opts = dict(
            dt=self.sim_dt,
            t_end=self.sim_t_end,
            t_after_impact=0.0,
            mass=self.m,
            x0=x0,
            x_min=0.0,
            x_max=self.stroke,
            title="Valve opening" if direction == "open" else "Valve closing",
        )
        opts.update(kw)
        return travel.travel(self._forces(direction == "open"), **opts)

    opening = computed(None)(lambda s: s.travel("open"))
    closing = computed(None)(lambda s: s.travel("close"))


    def venting(self, V=11e-3, m0=8.0, T0=278.15):
        import numpy as np
        from CoolProp.CoolProp import PropsSI
        from scipy.integrate import solve_ivp

        @np.vectorize
        def N2O(*args, **kwargs):
            return PropsSI(*args, "N2O", **kwargs)

        Cd = 0.65 # [-]
        A = self.A_seat_bore # [m^2]

        def mdot_SPC(m, P_u, P_d, T_u, rho, y):
            # cheeky chat job. looks good tho.

            # 1. Calculate critical pressure ratio (choking point)
            # This determines if the flow is sonic (choked) or subsonic
            critical_ratio = (2 / (y + 1))**(y / (y - 1))
            actual_ratio = P_d / P_u

            # 2. Check for backflow or negligible pressure differential
            if actual_ratio >= 1.0:
                return 0.0

            # 4. Determine Flow Regime
            if actual_ratio <= critical_ratio:
                # --- CHOKED (SONIC) FLOW ---
                # The flow has reached Mach 1 at the throat.
                term = y * (2 / (y + 1))**((y + 1) / (y - 1))
                mdot = Cd * A * sqrt(P_u * rho * term)
            else:
                # --- SUBSONIC FLOW ---
                # The flow is governed by the pressure differential.
                factor = (2 / (y - 1))
                term = (actual_ratio**(2/y)) - (actual_ratio**((y + 1)/y))
                mdot = Cd * A * sqrt(y * P_u * rho * factor * term)
            return mdot

        def ode(t, Y):
            m, T = Y
            m = max(10e-3, m)
            rho = m/V
            P_u = N2O("P", "T", T, "D", rho)
            P_d = 101325.0 # lowwwwwkey an assumption
            rhosat = N2O("D", "T", T, "Q", 1)
            y = 1.27 # maybe?
            dm = -mdot_SPC(m, P_u, P_d, T, min(rho, rhosat), y)
            dT = (y - 1) * T * dm/m
            dT = 0.0 # assume equalizes with environment quickly.
            return dm, dT

        soln = solve_ivp(ode, t_span=(0, 60*30), max_step=10.0, y0=[m0, T0])
        mass = soln.y[0]
        temperature = soln.y[1]
        pressure = N2O("P", "T", soln.y[1], "D", soln.y[0]/V)

        _, axes = plt.subplots(2, 1, figsize=(8, 7), tight_layout=True,
                sharex=True)
        for ax in axes:
            travel.style_ax(ax)
        axes[0].set_title(f"Venting ({V*1e3}L tank, {m0}kg N2O, {T0-273.15}°C)")
        axes[0].plot(soln.t, mass)
        axes[0].set_ylabel("Mass [kg]")
        axes[1].plot(soln.t, u.nsi.bar(pressure))
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Pressure [bar]")
        return axes


    def plot(self):
        return self.opening.plot(), self.closing.plot(), self.venting()



def main():
    valve = VentValve(
        m=0.01936269 + 0.0089/3, stroke=1.5e-3,
        chi_seat=0.3e-3, tau_fill=5e-3, tau_vent=5e-3,
        sim_dt=0.001e-3, sim_t_end=200e-3,
        Pmin_tank=0.0, Pmax_tank=70e5, D_ball=6e-3,
        D_seat_bore=2e-3, D_plunger_bore=7e-3,
        P_pneumo=10e5, D_pneumo=31.5e-3,
        k_spring=25.5e3, ell_relaxed=25e-3, ell_open=18.2e-3, ell_solid=12.5e-3,
        orings=(
            # NOTE: seal DP values are not dynamically updated.
            PistonSeal("Pneumo side",
                    oring_ID=28.0e-3, oring_CS=2.0e-3,
                    gland_ID=28.5e-3, gland_OD=31.5e-3, gland_Lz=3e-3,
                    DP=10e5, material="FKM", hardness=75, lubricated=True),
            RodSeal("N2O side",
                    oring_ID=7.5e-3, oring_CS=2.0e-3,
                    gland_ID=7.9e-3, gland_OD=10.9e-3, gland_Lz=3e-3,
                    DP=70e5, material="FKM", hardness=75, lubricated=True),
        ),
    )
    # Give a little margin.
    valve = valve.tweak(Pmax_tank=valve.Pmax_tank*1.2,
                        P_pneumo=valve.P_pneumo/1.2)
    # Try stronger spring?
    valve = valve.tweak(k_spring=80e3)
    # Size controlling forces.
    valve = valve.tweak(F_pneumo=1200, F_spring_open=400)

    valve.report()
    valve.plot()
    plt.show()


if __name__ == "__main__":
    main()
