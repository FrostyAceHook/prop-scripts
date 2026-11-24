"""
Normally-open pneumatically-closed main oxidiser valve, between the N2O tank
and the injector: a sleeve sliding over a static egg-shaped plug.

  closed: the sleeve's mouth is over the egg, sealed by the egg's O-ring.
  open  : the sleeve has slid downstream; flow passes around the egg and
          through the sleeve's bore.

  opens  : tank pressure on the sleeve (and an optional spring)
  closes : pneumatic pressure on the sleeve's piston (downstream face)
  resists: o-ring friction

Seals: the seat (on the egg, into the sleeve's mouth), the upstream and
downstream rod seals (in the housing, on the sleeve's ODs), and the piston seal.
Between the upstream seal and the piston is a vented chamber.

Closed, tank pressure pushes the sleeve open over the annulus between the
upstream and seat seals. Open, flow pressure also pushes back on the sleeve's
downstream end (taken as tank pressure: conservative), leaving the annulus
between the two rod seals. Pressures are absolute, and forces use their
difference to atmosphere (the vented chamber, and the bore when closed). It
only closes with the tank empty. Travel (kinematics) isn't modelled.
"""

import matplotlib.pyplot as plt
import numpy as np

# Hack-in the utility module.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utility"))

import spec as u
from oring import *
from system import *
from helpers import *


AL6061_Ys = u.MPa_(241)


class MainOxidiserValve(System):
    # --- conditions (absolute) ---
    P_tank: u.bar # While opening (it only closes with the tank empty).
    P_pneumo: u.bar # Pneumatic supply.
    gap: u.mm # Largest radial extrusion gap at each seal.

    # --- geometry ---
    bore: Circle # Sleeve bore: the flow path.
    seat: Circle # Sleeve mouth, where the egg's seal slides in.
    sleeve_us: Circle # Sleeve OD through the upstream seal.
    sleeve_ds: Circle # Sleeve OD through the downstream seal.
    piston: Circle # Pneumatic piston bore.

    # --- seals ---
    # Each slot is a Groove (sized by hand) or an AutoGroove (sized for its
    # diameter whenever that changes). The seals add their sliding diameter,
    # the pressure each sees, and the extrusion gap.
    gr_seat: Groove | AutoGroove # On the egg.
    gr_us: Groove | AutoGroove # In the housing.
    gr_ds: Groove | AutoGroove # In the housing.
    gr_piston: Groove | AutoGroove # On the sleeve's piston.

    or_seat = calc(PistonSeal, # Tank to the (closed) bore.
        lambda s: s.gr_seat.seal(PistonSeal, "Seat (egg)", s.seat.D,
                                 DP=s.Pg_tank, gap=s.gap),
        lambda s, seal: dict(gr_seat=Groove.of(seal)))
    or_us = calc(RodSeal, # Tank to the vented chamber.
        lambda s: s.gr_us.seal(RodSeal, "Upstream", s.sleeve_us.D,
                               DP=s.Pg_tank, gap=s.gap),
        lambda s, seal: dict(gr_us=Groove.of(seal)))
    or_ds = calc(RodSeal, # Pneumatics to the bore (at tank pressure open).
        lambda s: s.gr_ds.seal(RodSeal, "Downstream", s.sleeve_ds.D,
                               DP=max(s.Pg_tank, s.Pg_pneumo), gap=s.gap),
        lambda s, seal: dict(gr_ds=Groove.of(seal)))
    or_piston = calc(PistonSeal, # Pneumatics to the vented chamber.
        lambda s: s.gr_piston.seal(PistonSeal, "Piston", s.piston.D,
                                   DP=s.Pg_pneumo, gap=s.gap),
        lambda s, seal: dict(gr_piston=Groove.of(seal)))

    def seals(self):
        return (self.or_seat, self.or_us, self.or_ds, self.or_piston)

    def F_fric(self, kind, seat=None, us=0.0, ds=0.0, piston=0.0):
        """ Net `kind` friction with these pressure differences across the
        seals (`seat` None: the seat seal isn't engaged). """
        F = (friction(self.or_us, kind, us) + friction(self.or_ds, kind, ds)
             + friction(self.or_piston, kind, piston))
        return F if seat is None else F + friction(self.or_seat, kind, seat)
    F_fric_breakaway = calc(u.N, lambda s: s.F_fric("breakaway", seat=s.Pg_tank, us=s.Pg_tank, ds=0.0, piston=s.Pg_pneumo))

    # --- walls ---
    # (Rounded to the pm, so round dimensions meet round limits.)
    wall_us = calc(u.mm, # Sleeve wall at the seat. Sets seat.D.
        lambda s: round((s.sleeve_us.D - s.seat.D)/2, 12),
        lambda s, w: {"seat.D": s.sleeve_us.D - 2*w})
    wall_ds = calc(u.mm, # Sleeve wall at the downstream seal.
        lambda s: round((s.sleeve_ds.D - s.bore.D)/2, 12))
    counterbore = calc(u.mm, # Seat outside the bore, radially.
        lambda s: round((s.seat.D - s.bore.D)/2, 12))
    piston_land = calc(u.mm, # Piston's groove root outside the upstream OD.
        lambda s: (s.or_piston.gland_root - s.sleeve_us.D)/2)
    @computed(u.none)
    def hoop_SF(s): # Thin-wall hoop stress at tank pressure: the mouth (outside
        # pressure, closed) and the downstream end (inside pressure, open).
        stress = s.P_tank * max(s.seat.r/s.wall_us,
                                s.bore.r/s.wall_ds)
        return AL6061_Ys / stress if stress > 0 else float("inf")

    # --- gauge pressures ---
    Pg_tank = calc(u.bar, lambda s: s.P_tank - ATM)
    Pg_pneumo = calc(u.bar, lambda s: s.P_pneumo - ATM)

    # --- areas ---
    A_closed = calc(u.mm2, # Tank pushes open, closed.
        lambda s: s.sleeve_us.A - s.seat.A)
    A_open = calc(u.mm2, # Tank pushes open, open (bore at tank pressure).
        lambda s: s.sleeve_us.A - s.sleeve_ds.A)
    A_pneumo = calc(u.mm2,
        lambda s: s.piston.A - s.sleeve_ds.A)

    # --- forces (+ve opens) ---
    spring: Spring | None = None # Opening, long when open. Preferably none.
    F_tank_closed = calc(u.N,
        lambda s: s.Pg_tank * s.A_closed)
    F_tank_open = calc(u.N,
        lambda s: s.Pg_tank * s.A_open)
    F_pneumo = calc(u.N, # Closes.
        lambda s: s.Pg_pneumo * s.A_pneumo)
    F_spring_open = calc(u.N,
        lambda s: s.spring.F_long if s.spring else 0.0)
    F_spring_closed = calc(u.N,
        lambda s: s.spring.F_short if s.spring else 0.0)

    # --- force balances (+ve margin = works) ---
    holding = calc(Balance, lambda s: Balance.of( # Closed, filling.
        (s.F_pneumo, s.F_tank_closed + s.F_spring_closed)))
    opening = calc(Balance, lambda s: Balance.of( # Pneumatics vented.
        # Breakaway closed, the seat sealing.
        (s.F_tank_closed + s.F_spring_closed,
         s.F_fric("breakaway", seat=s.Pg_tank, us=s.Pg_tank)),
        # Past the seat, the bore at tank pressure.
        (s.F_tank_open + s.F_spring_open,
         s.F_fric("running", us=s.Pg_tank, ds=s.Pg_tank))))
    closing = calc(Balance, lambda s: Balance.of( # Tank empty.
        (s.F_pneumo,
         s.F_spring_open
            + s.F_fric("breakaway", ds=s.Pg_pneumo, piston=s.Pg_pneumo)),
        # Onto the seat.
        (s.F_pneumo,
         s.F_spring_closed
            + s.F_fric("running", seat=0.0, ds=s.Pg_pneumo,
                       piston=s.Pg_pneumo))))


    # --- requirements ---
    balances = ("holding", "opening", "closing")
    hoop_SF_min = 6.0 # fat
    land_min = u.mm_(1.0) # Piston's groove root outside the upstream OD.

    def requirements(self):
        return super().requirements() | {
            corner("holding.margin"): (0.0, None),
            corner("opening.margin"): (0.0, None),
            corner("closing.margin"): (0.0, None),
            corner("hoop_SF", "min"): (self.hoop_SF_min, None),
            "counterbore": (0.0, None),
            "piston_land": (self.land_min, None),
        }



def opts_oring():
    return dict(material="FKM", hardness=u.shoreA_(75, tol=5), lubricated=True)

def conditions():
    """ Tank pressure while opening and the pneumatic supply (absolute), and
    the extrusion gap the fits must hold. """
    return dict(P_tank=u.bar_(55, env=(30, 75)),
                P_pneumo=u.bar_(10, env=(9, 10)),
                gap=u.mm_(0.04))

def build():
    """ The design: no spring, 10 bar pneumatics, 2 mm walls, a 50 mm
    piston. """
    mm = u.mm_
    # return MainOxidiserValve(**conditions(),
    #     bore=Circle(D=mm(20, tol=(0, 0.1))),
    #     seat=Circle(D=mm(21.8, tol=(0, 0.05))),
    #     sleeve_us=Circle(D=mm(25.8, tol=(-0.05, 0))),
    #     sleeve_ds=Circle(D=mm(24, tol=(-0.05, 0))),
    #     piston=Circle(D=mm(50, tol=(0, 0.05))),
    #     gr_seat=Groove(oring=finer("mor18x2"),
    #             root=mm(18.45, tol=0.05), Lz=mm(2.65, tol=0.1), **opts_oring()),
    #     gr_us=Groove(oring=finer("mor26x1.5"),
    #             root=mm(28.35, tol=0.05), Lz=mm(2.0, tol=0.1), **opts_oring()),
    #     gr_ds=Groove(oring=finer("mor24x1.5"),
    #             root=mm(26.55, tol=0.05), Lz=mm(2.0, tol=0.1), **opts_oring()),
    #     gr_piston=Groove(oring=finer("mor46x1.5"),
    #             root=mm(47.5, tol=0.05), Lz=mm(2.0, tol=0.1), **opts_oring()),
    # )
    return MainOxidiserValve(**conditions(),
        bore=Circle(D=mm(20, tol=(0, 0.1))),
        seat=Circle(D=mm(21.8, tol=(0, 0.05))),
        sleeve_us=Circle(D=mm(25.8, tol=(-0.05, 0))),
        sleeve_ds=Circle(D=mm(24, tol=(-0.05, 0))),
        piston=Circle(D=mm(50, tol=(0, 0.05))),
        gr_seat=Groove(oring=finer("mor18x2"),
                root=mm(18.45, tol=0.05), Lz=mm(2.65, tol=0.1), **opts_oring()),
        gr_us=Groove(oring=finer("mor26x1.5"),
                root=mm(28.35, tol=0.05), Lz=mm(2.0, tol=0.1), **opts_oring()),
        gr_ds=Groove(oring=finer("mor24x1.5"),
                root=mm(26.55, tol=0.05), Lz=mm(2.0, tol=0.1), **opts_oring()),
        gr_piston=Groove(oring=finer("mor46x2"),
                root=mm(46.65, tol=0.05), Lz=mm(2.65, tol=0.1), **opts_oring()),
    )

def resizing(mov):
    """ Every seal re-sized whenever its diameter changes. """
    auto = lambda **kw: AutoGroove(**opts_oring(), **kw)
    return mov.tweak(gr_seat=auto(), gr_us=auto(), gr_ds=auto(),
                     gr_piston=auto())

@metric(u.mm, "piston needed")
def piston_needed(s):
    """ The piston bore that just holds closed (zero 3σ holding margin). """
    return s.solve("piston.D", corner("holding.margin"), 0.0).piston.D


def main():
    mov = build()
    # mov = resizing(mov)
    # mov.report()
    # mov.verify().report()
    # return
    # mov.gr_piston.report()
    # mov.or_piston.report()
    # mov.or_piston.outline()
    # return
    mov.report()
    for seal in mov.seals():
        seal.outline(succinct=True)
    mov.verify().report()

    for name in mov.balances:
        print(f"{name}: {mov.display(name + '.cases')}")


def tolerances(mov):
    """ What drives the opening (friction) and holding (areas) margins, and
    their worst builds. """
    for name in ("opening.margin", "holding.margin"):
        mov.variation(name).report()
        mov.corners(name, check=5000, seed=1).report()


def sweeps(mov):
    """ The mouth wall sets the closed annulus, so the opening force and the
    piston needed to hold closed (seals re-sized). """
    walls = u.mm_(np.arange(1.0, 2.51, 0.25))
    sweep = resizing(mov).sweep("wall_us", walls, "A_closed",
                                corner("opening.margin"), piston_needed)
    sweep.report()
    sweep.plot()


def optimise(mov):
    """ The smallest piston bore at 3σ over the sleeve's diameters (seals
    re-sized, their catalogue fits left out as they come and go with the
    diameters). Round its diameters to ones whose rings fit after. ~25 min. """
    mm = u.mm_
    design = resizing(mov)
    reqs = design.requirements().without("or_seat", "or_us", "or_ds",
                                         "or_piston")
    params = {"seat.D": (mm(20), mm(26)),
              "sleeve_us.D": (mm(23), mm(30)),
              "sleeve_ds.D": (mm(23), mm(27)),
              "piston.D": (mm(30), mm(60))}
    best = design.optimise("piston.D", params, reqs, seed=1, popsize=6,
                           maxiter=25, polish=False, workers=-1)
    best.report()
    return best


if __name__ == "__main__":
    main()
    plt.show()