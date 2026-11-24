"""
Normally-open spring-return pneumatically-closed vent valve, on the N2O tank's
ullage. A loose hardened steel ball sits on a 90° cone seat in brass, over the
seat bore (the inlet). A shaft of the ball's diameter, running in the same
bore, pushes it closed; the shaft passes through the N2O-side shaft seal into a
dry chamber with the opening spring and the pneumatic piston.

  tank -> seat bore -> ball/seat -> cavity -> outlet lines (atmosphere)

  opens  : spring, tank pressure on the ball
  closes : pneumatic pressure on the piston
  resists: o-ring friction, ice (an allowance)

Pressures are absolute; forces use their difference to atmosphere. Closed, tank
pressure acts inside the seat line and the cavity is at atmosphere. Open, the
flow chokes in the seat bore and (sonic, but not limiting) again in the outlet
lines, so the cavity sits at P_tank*A_bore/A_outlets; that acts on the shaft,
and the seat bore's jet pushes the ball onto it. Travel (kinematics) isn't
modelled.
"""

from math import pi, sqrt

import matplotlib.pyplot as plt
import numpy as np

# Hack-in the utility module.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utility"))

import spec as u
from spec import show, style_ax, table
from oring import *
from system import *
from system import alias
from helpers import *


SQRT2 = sqrt(2)

# N2O vapour as an ideal gas: ratio of specific heats, the critical pressure
# ratio of choked flow, and a choked jet's thrust per throat area per upstream
# pressure.
GAMMA = 1.27
CHOKE = (2/(GAMMA + 1))**(GAMMA/(GAMMA - 1))
THRUST = (1 + GAMMA)*CHOKE

# Young's modulus [Pa] and Poisson's ratio: the ball (hardened steel) and the
# seat (brass).
BALL = (200e9, 0.29)
SEAT = (100e9, 0.34)



class VentValve(System):
    # --- conditions (absolute) ---
    P_tank: u.bar # Tank (ullage) pressure: empty to full.
    P_pneumo: u.bar # Pneumatic supply.
    F_ice: u.N # Ice to break, either way: an allowance.
    gap: u.mm # Largest radial extrusion gap at each seal.

    # --- geometry ---
    ball: Circle
    seat_bore: Circle # Under the seat, from the tank: the flow's choke.
    outlet: Circle # Each outlet line, from the cavity to atmosphere.
    outlets: int # How many.
    shaft: Circle # Pushes the ball; through the shaft seal.
    piston: Circle # Pneumatic piston bore.
    spring: Spring # Opening, around the shaft; long when open.

    # --- seals ---
    # Each slot is a Groove (sized by hand) or an AutoGroove (sized for its
    # diameter whenever that changes). The seals add shaft.D or piston.D, the
    # pressure each sees, and the extrusion gap.
    gr_shaft: Groove | AutoGroove
    gr_piston: Groove | AutoGroove

    stroke = alias(u.mm, "spring.travel")

    # --- seat and flow path ---
    seat = calc(Circle, # Contact line on a 90° cone. Sets ball.D.
        lambda s: Circle(D=s.ball.D/SQRT2),
        lambda s, seat: {"ball.D": seat.D*SQRT2})
    # (Rounded to the pm, so round dimensions meet round limits.)
    seat_land = calc(u.mm, # Seat cone outside the bore, radially.
        lambda s: round((s.seat.D - s.seat_bore.D)/2, 12))
    curtain_A = calc(u.mm2, # Past the lifted ball, as a frustum.
        lambda s: pi*(s.seat.D + s.stroke/2) * s.stroke/SQRT2)
    curtain_ratio = calc(u.none,
        lambda s: s.curtain_A / s.seat_bore.A)
    outlets_A = calc(u.mm2,
        lambda s: s.outlets * s.outlet.A)
    outlet_ratio = calc(u.none,
        lambda s: s.outlets_A / s.seat_bore.A)
    spring_inner_gap = calc(u.mm, # Spring ID over the shaft, radially.
        lambda s: round((s.spring.ID - s.shaft.D)/2, 12))
    spring_outer_gap = calc(u.mm, # Spring OD inside the piston bore, radially.
        lambda s: round((s.piston.D - s.spring.OD)/2, 12))

    # --- pressure forces (+ve opens) ---
    dP_tank = calc(u.bar,
        lambda s: s.P_tank - ATM)
    dP_pneumo = calc(u.bar,
        lambda s: s.P_pneumo - ATM)
    P_cavity = calc(u.bar, # Open: both restrictions choked.
        lambda s: max(ATM, s.P_tank * s.seat_bore.A/s.outlets_A))
    @computed(u.N)
    def F_jet(s): # Open: the seat bore's jet, all turned by the ball. Choked
                  # thrust, or 2 dP A when nearly unchoked (-> 0 with dP).
        dP = s.P_tank - s.P_cavity
        return max(0.0, min(2*dP, THRUST*s.P_tank - s.P_cavity)) \
                * s.seat_bore.A
    F_tank_closed = calc(u.N, # Inside the seat line.
        lambda s: s.dP_tank * s.seat.A)
    F_tank_open = calc(u.N, # Cavity over the shaft, and the jet.
        lambda s: (s.P_cavity - ATM)*s.shaft.A + s.F_jet)
    F_tank_seating = calc(u.N, # Closing onto the seat: tank inside the seat
                               # line, the cavity still pressurised outside it.
        lambda s: s.dP_tank*s.seat.A
                + (s.P_cavity - ATM)*(s.shaft.A - s.seat.A))
    F_pneumo = calc(u.N, # Closes.
        lambda s: s.dP_pneumo * s.piston.A)
    F_spring_open = alias(u.N, "spring.F_long")
    F_spring_closed = alias(u.N, "spring.F_short")

    # --- seals ---
    or_shaft = calc(RodSeal, # Shaft through the housing: cavity to dry side.
        lambda s: s.gr_shaft.seal(RodSeal, "Shaft (N2O side)", s.shaft.D,
                                  DP=s.P_cavity - ATM, gap=s.gap),
        lambda s, seal: dict(gr_shaft=Groove.of(seal)))
    or_piston = calc(PistonSeal,
        lambda s: s.gr_piston.seal(PistonSeal, "Piston", s.piston.D,
                                   DP=s.dP_pneumo, gap=s.gap),
        lambda s, seal: dict(gr_piston=Groove.of(seal)))

    def seals(self):
        return (self.or_shaft, self.or_piston)

    def F_fric(self, kind, opened, pneumo):
        """ Net `kind` friction, `opened` or closed, with the pneumatics on
        (`pneumo`) or vented. The cavity is at atmosphere when closed. """
        return (friction(self.or_shaft, kind,
                         self.P_cavity - ATM if opened else 0.0)
              + friction(self.or_piston, kind,
                         self.dP_pneumo if pneumo else 0.0))

    # --- force balances (+ve margin = works) ---
    # Ice is broken at breakaway. The gas isn't counted on to hold it open.
    holding = calc(Balance, lambda s: Balance.of( # Margin: seat load.
        (s.F_pneumo, s.F_tank_closed + s.F_spring_closed)))
    opening = calc(Balance, lambda s: Balance.of( # Pneumatics vented.
        (s.F_spring_closed + s.F_tank_closed,
         s.F_fric("breakaway", opened=False, pneumo=False) + s.F_ice),
        (s.F_spring_open,
         s.F_fric("running", opened=True, pneumo=False))))
    closing = calc(Balance, lambda s: Balance.of( # Against the tank.
        (s.F_pneumo,
         s.F_spring_open + s.F_tank_open
            + s.F_fric("breakaway", opened=True, pneumo=True) + s.F_ice),
        (s.F_pneumo,
         s.F_spring_closed + s.F_tank_seating
            + s.F_fric("running", opened=True, pneumo=True))))

    @computed(u.MPa)
    def seat_stress(s): # Peak Hertz pressure at the holding seat load: a line
        # contact of the ball (its radius in section) on the 45° cone.
        E = 1/((1 - BALL[1]**2)/BALL[0] + (1 - SEAT[1]**2)/SEAT[0])
        q = max(0.0, s.holding.margin) * SQRT2 / s.seat.P # normal, per length
        return 2*q / (pi*sqrt(4*q*s.ball.r/(pi*E))) if q else 0.0


    # --- requirements ---
    balances = ("holding", "opening", "closing")
    seat_load_min = u.N_(40) # Held closed, worst case: the seat's preload.
    opening_min = u.N_(80) # Margins beyond ice and friction, worst case.
    closing_min = u.N_(30)
    land_min = u.mm_(0.5) # Seat cone outside the bore, radially.
    flow_ratio_min = 2.0 # Curtain and outlets over the seat bore, so the seat
                         # bore stays the choke.
    spring_gap_min = u.mm_(0.25) # Spring to the shaft and the piston bore.

    def requirements(self):
        return super().requirements() | {
            corner("holding.margin"): (self.seat_load_min, None),
            corner("opening.margin"): (self.opening_min, None),
            corner("closing.margin"): (self.closing_min, None),
            corner("curtain_ratio"): (self.flow_ratio_min, None),
            "outlet_ratio": (self.flow_ratio_min, None),
            "seat_land": (self.land_min, None),
            "spring_inner_gap": (self.spring_gap_min, None),
            "spring_outer_gap": (self.spring_gap_min, None),
        }



def venting(vv, V=u.L_(11), m0=u.kg_(8), T0=u.degC_(5), Cd=0.65,
            t_end=u.minute_(30)):
    """
    Tank blowdown through the open valve: vapour choking in the seat bore,
    isothermal (optimistic: the boiling liquid really cools, so pressure
    falls). Needs CoolProp.
    """
    from CoolProp.CoolProp import PropsSI
    from scipy.integrate import solve_ivp

    N2O = np.vectorize(lambda *args: PropsSI(*args, "N2O"))
    A = vv.seat_bore.A

    def mdot(P, rho):
        ratio = ATM / P
        if ratio >= 1.0:
            return 0.0
        if ratio <= CHOKE:
            term = GAMMA * (2/(GAMMA + 1))**((GAMMA + 1)/(GAMMA - 1))
        else:
            term = 2*GAMMA/(GAMMA - 1) * (ratio**(2/GAMMA)
                                          - ratio**((GAMMA + 1)/GAMMA))
        return Cd * A * sqrt(P * rho * term)

    def ode(t, y):
        m = max(10e-3, y[0])
        P = N2O("P", "T", T0, "D", m/V)
        rho = min(m/V, N2O("D", "T", T0, "Q", 1)) # vapour leaves
        return [-mdot(P, rho)]

    soln = solve_ivp(ode, t_span=(0, t_end), y0=[m0], max_step=10.0)
    P = N2O("P", "T", T0, "D", soln.y[0]/V)
    _, axes = plt.subplots(2, 1, figsize=(8, 7), tight_layout=True,
                           sharex=True)
    for ax in axes:
        style_ax(ax)
    axes[0].set_title(f"Venting ({u.L_.from_si(V):g} L tank, "
                      f"{m0:g} kg N2O, {u.degC_.from_si(T0):g}°C)")
    axes[0].plot(soln.t, soln.y[0])
    axes[0].set_ylabel("Mass [kg]")
    axes[1].plot(soln.t, u.bar_.from_si(P))
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Pressure [bar]")
    return soln, axes



def opts_oring():
    return dict(material="FKM", hardness=u.shoreA_(75, tol=5), lubricated=True)

def conditions():
    """ Tank empty to full and the pneumatic supply (both absolute), the ice
    allowance, and the extrusion gap the fits must hold. """
    return dict(P_tank=u.bar_(70, env=(ATM/1e5, 70)),
                P_pneumo=u.bar_(10, env=(8, 10)),
                F_ice=u.N_(20), gap=u.mm_(0.05))

def build():
    """ The design: a 5 mm ball and shaft, a 24 mm piston, and a music wire
    spring (2.2 mm wire, 10.5 mm mean diameter, 6.75 active coils). """
    mm = u.mm_
    return VentValve(**conditions(),
        ball=Circle(D=mm(5, tol=0.0025)),
        seat_bore=Circle(D=mm(2, tol=(0, 0.05))),
        outlet=Circle(D=mm(3, tol=(0, 0.1))), outlets=2,
        shaft=Circle(D=mm(5, tol=(-0.022, -0.010))),
        piston=Circle(D=mm(24, tol=(0, 0.05))),
        spring=Spring(material="music wire", d=mm(2.2, tol=0.015),
                      D=mm(10.5, tol=0.15), n=6.75, L_free=mm(25.8, tol=0.4),
                      L_long=mm(21.5, tol=0.1), L_short=mm(20.5, tol=0.1)),
        gr_shaft=Groove(oring=bs1806(106),
                root=mm(9.38, tol=0.05), Lz=mm(3.4, tol=0.1), **opts_oring()),
        gr_piston=Groove(oring=finer("mor21x1.5"),
                root=mm(21.5, tol=0.05), Lz=mm(2.0, tol=0.1), **opts_oring()),
    )

def resizing(vv):
    """ The piston seal re-sized whenever the piston bore changes. """
    return vv.tweak(gr_piston=AutoGroove(**opts_oring()))

def excess(s):
    """ (opening, the tighter of closing and holding): each 3σ margin beyond
    its requirement. """
    return (corner("opening.margin")(s) - s.opening_min,
            min(corner("closing.margin")(s) - s.closing_min,
                corner("holding.margin")(s) - s.seat_load_min))

def refit(vv, overrun=0.3):
    """ vv with the spring's free length moved so `overrun` of its
    deflection is left to solid, keeping its forces and rate. """
    s = vv.spring
    deflection = s.L_free - s.L_short
    L_short = s.L_solid + overrun*deflection
    return vv.tweak({"spring.L_free": L_short + deflection,
                     "spring.L_short": L_short,
                     "spring.L_long": L_short + float(s.travel)})

def sized(vv, rounds=3, overrun=0.3):
    """ vv with just enough spring preload to open, then just enough piston
    to close and hold (each at its required 3σ margin), its piston seal
    re-sized and its spring refitted. """
    vv = resizing(vv)
    for _ in range(rounds):
        vv = vv.solve("F_spring_open", lambda s: excess(s)[0], 0.0)
        vv = refit(vv, overrun)
        vv = vv.solve("piston.D", lambda s: excess(s)[1], 0.0)
    return vv

def balanced(vv):
    """ vv with the spring preload that leaves opening as much spare 3σ
    margin as the tighter of closing and holding. """
    return vv.solve("F_spring_open", lambda s: excess(s)[0] - excess(s)[1],
                    0.0)

def with_spring(vv, material, d, D, k, overrun=0.3):
    """ vv with a spring of this wire, mean diameter and rate, preload
    balanced, and `overrun` of its deflection left to solid. """
    sp = vv.spring.tweak(dict(material=material, d=d, D=D))
    vv = vv.tweak(spring=sp).tweak({"spring.k": k})
    for _ in range(2):
        vv = refit(balanced(vv), overrun)
    return balanced(vv)


def main():
    vv = build()
    vv.report()
    for seal in vv.seals():
        seal.outline(succinct=True)
    vv.verify().report()
    for name in vv.balances:
        print(f"{name}: {vv.display(name + '.cases')}")


def tolerances(vv):
    """ What drives the opening and closing margins, and their worst builds. """
    for name in ("opening.margin", "closing.margin"):
        vv.variation(name).report()
        vv.corners(name, check=5000, seed=1).report()
        vv.corners(name, limits=True).report() # arithmetic worst


def sweeps(vv):
    """ The piston needed against the ice allowance (spring preload and piston
    re-solved for each), and the 3σ margins against the outlet lines' size
    (which sets the cavity pressure when open). """
    rows = []
    for F in u.N_(np.arange(0, 61, 10)):
        s = sized(vv.tweak(F_ice=F))
        rows.append([show(F, u.N), s.display("piston.D"),
                     s.display("F_spring_open"), s.display("F_spring_closed")])
    print("Ice allowance:\n" + table(rows, ["F_ice", "piston.D",
            "F_spring_open", "F_spring_closed"]))
    k3 = [corner(name + ".margin") for name in vv.balances]
    sweep = vv.sweep("outlet.D", u.mm_(np.arange(1.5, 4.01, 0.25)), *k3,
                     "P_cavity")
    sweep.report()
    sweep.plot()


def springs(vv, options=(("music wire", 2.2, 10.5, 30),
                          ("chrome silicon", 2.2, 10.5, 30),
                          ("music wire", 2.2, 10.5, 45),
                          ("302 stainless", 2.5, 10.5, 45))):
    """ Springs for vv's piston: (material, d [mm], D [mm], k [N/mm]) each
    with its preload balanced and 20% overrun to solid. Wire and coil
    diameter follow from solid-safety and clearing the shaft; then the rate
    trades length (low) against force tolerance (high). ~2 min each. """
    rows = []
    for material, d, D, k in options:
        s = with_spring(vv, material, u.mm_(d), u.mm_(D), u.N_mm_(k),
                        overrun=0.2)
        v, sp = s.verify(), s.spring
        r = {m.name.split(" ")[0]: x for m, (_, _, x) in v.results.items()}
        rows.append([material, f"{d:g}", f"{D:g}", f"{k:g}", f"{sp.n:.2f}",
                     sp.display("L_free"), f"{sp.index:.2f}",
                     f"{r['spring.solid_SF']:.2f}",
                     *(show(r[f"{b}.margin"], u.N) for b in s.balances),
                     "ok" if v.ok else "FAILS"])
    print(table(rows, ["material", "d", "D", "k", "n", "L_free", "index",
                       "solid SF", *s.balances, ""]))


if __name__ == "__main__":
    main()
    plt.show()
