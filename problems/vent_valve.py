"""
Normally-open spring-return pneumatically-closed vent valve, on the N2O tank's
ullage. A loose hardened steel ball sits on a 90° cone seat in brass, over the
seat bore (the inlet). A shaft, running in the same bore, pushes it closed; the
shaft passes through the N2O-side shaft seal into a dry chamber with the opening
spring and the pneumatic piston.

  tank -> choking bore -> seat bore -> ball/seat -> cavity -> outlet lines
  (atmosphere)

  opens  : spring, tank pressure on the ball
  closes : pneumatic pressure on the piston
  resists: o-ring friction, ice (not modelled)

Pressures are absolute; forces use their difference to atmosphere. Closed, tank
pressure acts inside the seat line and the cavity is at atmosphere. Open, the
flow chokes in the choking bore and (sonic, but not limiting) again in the outlet
lines, so the cavity sits at P_tank*A_choke/A_outlets; that acts on the shaft,
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

def _G_N2O(r):
    """ Mass flux function of pressure ratio r (downstream/upstream). """
    r = max(r, CHOKE)
    return sqrt(2*GAMMA/(GAMMA - 1)*(r**(2/GAMMA) - r**((GAMMA + 1)/GAMMA)))

def cavity_ratio(area_ratio):
    """ P_cavity/P_tank between an upstream restriction and choked downstream
    ones, `area_ratio` = upstream over downstream area: the same mass flow,
    psi(r) = r psi* area_ratio**-1. """
    if area_ratio <= CHOKE:
        return area_ratio # upstream choked too
    f = lambda r: _G_N2O(r) - r*_G_N2O(CHOKE)/area_ratio
    lo, hi = CHOKE, 1.0
    for _ in range(60):
        mid = (lo + hi)/2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi)/2


# Young's modulus [Pa] and Poisson's ratio: the ball (hardened steel) and the
# seat (brass).
BALL = (200e9, 0.29)
SEAT = (100e9, 0.34)
BODY = (69e9, 0.33) # 6061-T6.
BRASS_YIELD = 310e6 # C36000 half-hard.


class VentValve(System):
    # --- conditions (absolute) ---
    P_tank: u.bar # Tank (ullage) pressure: empty to full.
    P_pneumo: u.bar # Pneumatic supply.
    gap: u.mm # Largest radial extrusion gap at each seal.

    # --- geometry ---
    ball: Circle
    choke: Circle # Under the seat, from the tank: the flow's choke.
    outlet: Circle # Each outlet line, from the cavity to atmosphere.
    outlets: int # How many.
    shaft: Circle # Pushes the ball; through the shaft seal.
    piston: Circle # Pneumatic piston bore.
    spring: Spring # Opening, around the shaft; long when open.

    # --- seat insert: brass, pressed into the body from above ---
    pocket: Circle # The body's bore it's pressed into.
    insert: Annulus # Its press-fit diameter and bore.
    insert_L: u.mm # Press-fit length.
    land: Annulus # The annulus under the seat that seals the pocket.
    coin_width: u.mm # Seat band the ball coins into the cone, across it.
    mu_press: u.none # Friction between brass + Al.

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
    curtain_A = calc(u.mm2, # Past the lifted ball, as a frustum.
        lambda s: pi*(s.seat.D + s.stroke/2) * s.stroke/SQRT2)
    curtain_ratio = calc(u.none,
        lambda s: s.curtain_A / s.choke.A)
    outlets_A = calc(u.mm2,
        lambda s: s.outlets * s.outlet.A)
    outlet_ratio = calc(u.none,
        lambda s: s.outlets_A / s.choke.A)
    spring_inner_gap = calc(u.mm, # Spring ID over the shaft, radially.
        lambda s: (s.spring.ID - s.shaft.D)/2)
    spring_outer_gap = calc(u.mm, # Spring OD inside the piston bore, radially.
        lambda s: (s.piston.D - s.spring.OD)/2)

    # --- pressure forces (+ve opens) ---
    dP_tank = calc(u.bar,
        lambda s: s.P_tank - ATM)
    dP_pneumo = calc(u.bar,
        lambda s: s.P_pneumo - ATM)
    @computed(u.bar)
    def P_cavity(s): # Open: the outlets choked, the choke bore choked too
                     # (P_tank*A_choke/A_outlets) or, as the outlets shrink
                     # past it, subsonic, up to P_tank as they close.
        return max(ATM, s.P_tank*cavity_ratio(s.choke.A/s.outlets_A))
    @computed(u.N)
    def F_jet(s): # Open: the seat bore's jet, all turned by the ball. Choked
                  # thrust, or 2 dP A when nearly unchoked (-> 0 with dP).
        dP = s.P_tank - s.P_cavity
        return max(0.0, min(2*dP, THRUST*s.P_tank - s.P_cavity)) \
                * s.choke.A
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

    # --- seat insert assembly and coining ---
    interference = calc(u.um, # Diametral.
        lambda s: s.insert.OD - s.pocket.D)
    @computed(u.MPa)
    def fit_pressure(s): # Lamé: a brass tube in a thick 6061 body.
        d, di = s.insert.OD, s.insert.ID
        C = ((1 + BODY[1])/BODY[0]
             + ((d**2 + di**2)/(d**2 - di**2) - SEAT[1])/SEAT[0])
        return max(0.0, s.interference)/(d*C)
    F_fit = calc(u.N, # Friction pressing it in; also what holds it.
        lambda s: s.mu_press*s.fit_pressure * s.insert.Op*s.insert_L)
    F_land = calc(u.N, # Bottomed, to yield the land flat. (The floor, under
                       # so narrow a land, holds ~3x its own yield.)
        lambda s: BRASS_YIELD*s.land.A)
    F_insert = calc(u.N, # Press force to assemble it.
        lambda s: s.F_fit + s.F_land)
    F_coin = calc(u.N, # Ball's axial load to coin the seat band: fully
                       # plastic (~3x yield) across it, normal to the cone.
        lambda s: 3*BRASS_YIELD*s.coin_width * s.seat.p/SQRT2)
    P_coin = calc(u.bar, # Supply to coin the seat band.
        lambda s: s.P_supply(s.F_coin, spring=False))
    P_insert = calc(u.bar, # Supply to press the insert home and coin its land
                           # through the ball, instead of on a press.
        lambda s: s.P_supply(s.F_insert, spring=False))

    def P_supply(self, F, spring=True):
        """ Supply that puts axial load `F` on the ball, tank empty: against
        the closed spring and breakaway friction (which grows with it). """
        P = ATM
        for _ in range(8):
            P = ATM + (F + (self.F_spring_closed if spring else 0)
                       + friction(self.or_shaft, "breakaway", 0.0)
                       + friction(self.or_piston, "breakaway", P - ATM)
                       )/self.piston.A
        return P
    seat_pressure = calc(u.MPa, # Mean normal pressure on the coined band at
                                # the holding seat load. Coined at 3x yield.
        lambda s: max(0.0, s.holding.margin)*SQRT2 / (s.seat.p*s.coin_width))
    seat_load_ratio = calc(u.none, # Holding seat load over the coining load:
                                   # below 1, the band only reloads elastically.
        lambda s: max(0.0, s.holding.margin) / s.F_coin)

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
         s.F_fric("breakaway", opened=False, pneumo=False)),
        (s.F_spring_open,
         s.F_fric("running", opened=True, pneumo=False))))
    closing = calc(Balance, lambda s: Balance.of( # Against the tank.
        (s.F_pneumo,
         s.F_spring_open + s.F_tank_open
            + s.F_fric("breakaway", opened=True, pneumo=True)),
        (s.F_pneumo,
         s.F_spring_closed + s.F_tank_seating
            + s.F_fric("running", opened=True, pneumo=True))))


    # --- requirements ---
    balances = ("holding", "opening", "closing")
    seat_load_min = u.N_(40) # Held closed, worst case: the seat's preload.
    opening_min = u.N_(80) # Margins beyond ice and friction, worst case.
    closing_min = u.N_(30) # With the outlets clear.
    closing_blocked_min = u.N_(0) # Anywhere down to iced outlets.
    flow_ratio_min = 2.0 # Curtain and outlets over the seat bore, so the seat
                         # bore stays the choke.
    spring_gap_min = u.mm_(0.25) # Spring to the shaft and the piston bore.

    def requirements(self):
        return super().requirements() | {
            corner("holding.margin"): (self.seat_load_min, None),
            corner("opening.margin"): (self.opening_min, None),
            corner("closing.margin"): (self.closing_blocked_min, None),
            corner(closing_clear): (self.closing_min, None),
            corner("curtain_ratio"): (self.flow_ratio_min, None),
            "outlet_ratio": (self.flow_ratio_min, None),
            "spring_inner_gap": (self.spring_gap_min, None),
            "spring_outer_gap": (self.spring_gap_min, None),
        }

@metric(u.N, "closing.margin (outlets clear)")
def closing_clear(s):
    """ Closing margin with the outlets at their largest (clear of ice). """
    return s.tweak({"outlet.D": s.outlet.D.limits[1]}).closing.margin



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
    A = vv.choke.A

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


def thick_walls(vv):
    inlet = ThickWall(
        wall=Annulus(OD=u.mm_(25.4/4, tol=(-0.05, 0)), ID=u.mm_(3, tol=0.1)),
        Pi=corner("P_tank", "max")(vv),
        **AL6061_T6,
    )
    print("inlet ", end="")
    inlet.extents("SF", printed=True)
    piston = ThickWall(
        wall=Annulus(ID=vv.piston.D, th=u.mm_(2.5, tol=0.1)),
        Pi=corner("P_insert", "max")(vv),
        **AL6061_T6,
    )
    print("piston housing (during assembly) ", end="")
    piston.extents("SF", printed=True)





def opts_oring():
    return dict(material="FKM", hardness=u.shoreA_(75, tol=5), lubricated=True)

def conditions():
    """ Tank empty to full and the pneumatic supply (both absolute), and the
    extrusion gap the fits must hold. """
    return dict(P_tank=u.bar_(70, env=(ATM/1e5, 70)),
                P_pneumo=u.bar_(10, env=(9, 11)),
                gap=u.mm_(0.05))

def build(resizing=False):
    mm = u.mm_
    if resizing:
        orings = dict(
            gr_shaft=AutoGroove(sizes=FINER_METRIC,
                    **opts_oring()),
            gr_piston=AutoGroove(sizes=FINER_METRIC,
                    **opts_oring()),
        )
    else:
        orings = dict(
            gr_shaft=Groove(oring=finer("mor7x1.5"),
                    root=mm(9.67, tol=0.05), Lz=mm(2.0, tol=0.1),
                    **opts_oring()),
            gr_piston=Groove(oring=finer("mor26x1.5"),
                    root=mm(26.46, tol=0.05), Lz=mm(2.0, tol=0.1),
                    **opts_oring()),
        )
    vv = VentValve(**conditions(),
        ball=Circle(D=mm(5, tol=0.0025)),
        choke=Circle(D=mm(3, tol=(0.05, 0.1))),
        outlet=Circle(D=mm(3, env=(0.1, 3.0))), outlets=2, # ice maybe blocking.
        shaft=Circle(D=mm(7.1, iso286="f7")), # H7 bore
        piston=Circle(D=mm(29.0, iso286="H8")), # f7 shaft
        spring=Spring(material="music wire", d=mm(2.5, tol=0.015),
                      D=mm(13, tol=0.15), n=5.25, L_free=mm(24.2, tol=0.4),
                      L_long=mm(20.3, tol=0.1), L_short=mm(19.3, tol=0.1)),
        pocket=Circle(D=mm(6.5, iso286="H6", centred=True)),
        insert=Annulus(OD=mm(6.5, iso286="p6", centred=True),
                       ID=mm(2.0, tol=0.05)),
        insert_L=mm(1.5, tol=0.1),
        land=Annulus(ID=mm(4.0, tol=0.05),
                     th=mm(0.25, tol=0.05)),
        coin_width=mm(0.15, tol=0),
        mu_press=u.none_(0.25, tol=0.075, k=2),
        **orings,
    )
    vv = vv.solve("coin_width", "P_coin", u.bar_(18))
    return vv


def main():
    vv = build(resizing=False)
    vv.report()
    for seal in vv.seals():
        seal.outline(succinct=True)
    vv.verify().report()
    print("\n--- thick walls ---")
    thick_walls(vv)
    print("\n--- balances (nominal) ---")
    for name in vv.balances:
        print(f"{name}: {vv.display(name + '.cases')}")
    print("\n--- service ---")
    for name in ("F_pneumo", "P_cavity", "seat_stress"):
        vv.extents(name, printed=True)
    print("\n--- seat insert ---")
    for name in ("interference", "fit_pressure", "F_fit", "F_land",
                 "F_insert"):
        vv.extents(name, printed=True)
    print("\n--- coining with the piston, tank empty, no spring ---")
    for name in ("F_coin", "P_coin", "P_insert"):
        vv.extents(name, printed=True)
    print("\n--- insertion stress ---")
    A = corner("A", "min")\
        (Annulus(ID=u.mm_(5, tol=0.05), OD=u.mm_(6.372, tol=0.05)))
    F = corner("F_insert", "max")(vv)
    print("contact:", show(A, u.mm2))
    print("force:", show(F, u.N))
    print("stress:", show(F/A, u.MPa))
    print("\n--- removal stress ---")
    A = corner("A", "min")\
        (Annulus(ID=vv.insert.ID, OD=vv.choke.D))
    F = corner("F_fit", "max")(vv)
    print("contact:", show(A, u.mm2))
    print("force:", show(F, u.N))
    print("stress", show(F/A, u.MPa))



if __name__ == "__main__":
    main()
    plt.show()
