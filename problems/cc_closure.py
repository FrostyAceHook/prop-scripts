from abc import ABC, abstractmethod
from math import pi, sqrt, exp, inf, nan, isnan
import numpy as np

# Hack-in the utility module.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utility"))

import search
import units as u
from system import *



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




### FORCES ######################################################################

class Load(System):
    end: str # "nozzle" or "injector".
    count: int # retainer count.
    P: u.bar # internal pressure.
    A: u.mm2 # "effective" pressure area.
    sharing_factor: u.none = 0.85 # knock-down on yield strengths to model
                                  # imperfect load sharing.
    F = calc(u.kN, # axial force per retainer.
        lambda s: s.F_total/s.count,
        lambda s, F: dict(F_total=F*s.count))
    F_total = calc(u.kN, # total axial force.
        lambda s: s.A*s.P,
        lambda s, F: dict(P=F/s.A))

def injector(count, P, D_seal):
    return Load(end="injector", count=count, P=P, A=circ_area(D_seal))

def nozzle(count, P, D_seal, D_throat, Cf):
    return Load(end="nozzle", count=count, P=P,
            A=circ_area(D_seal) - Cf*circ_area(D_throat))




### BOLTS #######################################################################

# Metric bolt sizes, in coarse and fine: size -> (D, P, bore diameter)
#   size : size code, i.e. "M<size>(c|f)"
#   D : major diameter.
#   P : pitch.
#   bore diameter : hole diameter for tapping.
BOLT_SIZES = {
    "M3c":  ( 3.0, 0.50,  2.50),  "M3f":  ( 3, 0.35,  2.65),
    "M4c":  ( 4.0, 0.70,  3.30),  "M4f":  ( 4, 0.50,  3.50),
    "M5c":  ( 5.0, 0.80,  4.20),  "M5f":  ( 5, 0.50,  4.50),
    "M6c":  ( 6.0, 1.00,  5.00),  "M6f":  ( 6, 0.75,  5.20),
    "M8c":  ( 8.0, 1.25,  6.80),  "M8f":  ( 8, 1.00,  7.00),
    "M10c": (10.0, 1.50,  8.50),  "M10f": (10, 1.25,  8.80),
    "M12c": (12.0, 1.75, 10.20),  "M12f": (12, 1.25, 10.80),
    "M14c": (14.0, 2.00, 12.00),  "M14f": (14, 1.50, 12.50),
    "M16c": (16.0, 2.00, 14.00),  "M16f": (16, 1.50, 14.50),
    "M18c": (18.0, 2.50, 15.50),  "M18f": (18, 1.50, 16.50),
    "M20c": (20.0, 2.50, 17.50),  "M20f": (20, 1.50, 18.50),
    "M22c": (22.0, 2.50, 19.50),  "M22f": (22, 1.50, 20.50),
    "M24c": (24.0, 3.00, 21.00),  "M24f": (24, 2.00, 22.00),
    "M27c": (27.0, 3.00, 24.00),  "M27f": (27, 2.00, 25.00),
    "M30c": (30.0, 3.50, 26.50),  "M30f": (30, 2.00, 28.00),
    "M33c": (33.0, 3.50, 29.50),  "M33f": (33, 2.00, 31.00),
    "M36c": (36.0, 4.00, 32.00),  "M36f": (36, 3.00, 33.00),
    "M42c": (42.0, 4.50, 37.50),  "M42f": (42, 3.00, 39.00),
    "M48c": (48.0, 5.00, 43.00),  "M48f": (48, 3.00, 45.00),
    "M56c": (56.0, 5.50, 50.50),  "M56f": (56, 4.00, 52.00),
    "M64c": (64.0, 6.00, 58.00),  "M64f": (64, 4.00, 60.00),
}
BOLT_SIZES = {code: (D*1e-3, P*1e-3, BD*1e-3)
              for code, (D, P, BD) in BOLT_SIZES.items()}

# Metric bolt property classes: grade -> (Rp, Rm).
#   grade : grade code
#   Rp [MPa] : minimum 0.2 % proof (yield) strength.
#   Rm [MPa] : minimum tensile (ultimate) strength.
BOLT_GRADES = {
    "4.6":  ( 240,  400),
    "8.8":  ( 640,  800),
    "10.9": ( 940, 1040),
    "12.9": (1100, 1220),
    "14.9": (1260, 1400),
}
BOLT_GRADES = {code: (Us*1e6, Ys*1e6)
              for code, (Us, Ys) in BOLT_GRADES.items()}

# Shear yield strength as a fraction of Rp (von Mises shear yield criterion).
BOLT_SHEAR_YLD_RATIO = 1/sqrt(3)

# Shear ultimate strength as a fraction of Rm (empirical, 0.6--0.65 typical).
BOLT_SHEAR_ULT_RATIO = 0.60




### BEARING #####################################################################

# 6061-T6 bearing strengths: e/D -> (BYs, BUs)
#   e/D [-] : (hole centre to part edge, in the load direction) / bolt diameter.
#   BYs [MPa] : bearing yield stress (projected area basis, 2 % D hole set).
#   BUs [MPa] : bearing ultimate stress (projected area basis).
# MMPDS only publishes e/D = 1.5 and 2.0. Do not extrapolate below 1.5.
AL6061_BEARING = {
    1.5: (338, 462),
    2.0: (386, 607),
}
AL6061_BEARING = {e_d: (Ys*1e6, Us*1e6)
                  for e_d, (Ys, Us) in AL6061_BEARING.items()}

def al6061_BS(e_d=None):
    """ 6061-T6 bearing allowables (yield, ultimate) for a given e/D. """
    lo_y, lo_u = AL6061_BEARING[1.5]
    hi_y, hi_u = AL6061_BEARING[2.0]
    if e_d is None or e_d >= 2.0:
        return hi_y, hi_u
    if e_d < 1.5:
        raise ValueError(f"e/D = {e_d:.2f} < 1.5, no data")
    t = (e_d - 1.5) / 0.5
    return lo_y + t*(hi_y - lo_y), lo_u + t*(hi_u - lo_u)



### CASES #######################################################################

class Material(System):
    sigma = calc(u.MPa, lambda s: s.load.F / s.A)
    sfY = calc(u.none, lambda s: s.Ys*s.load.sharing_factor / s.sigma)
    sfU = calc(u.none, lambda s: s.Us / s.sigma)


class Bearing(Material):
    load: Load
    D: u.mm # bearing diameter.
    L: u.mm # bearing length.
    A = calc(u.mm2, lambda s: s.D*s.L)
    edge: u.mm = None # edge distance.
    e_d = calc(u.none, lambda s: None if s.edge is None else s.edge / s.D)

    Ys = calc(u.MPa, lambda s: al6061_BS(s.e_d)[0]) # bearing yield strength.
    Us = calc(u.MPa, lambda s: al6061_BS(s.e_d)[1]) # bearing ultimate strength.

class Shear(Material):
    load: Load
    A: u.mm2 # shear area.

    Ys: u.MPa # shear yield strength.
    Us: u.MPa # shear ultimate strength.




### PIN #########################################################################

class Pin(System):
    # --- situation ---

    load: Load # load.

    Ys: u.MPa # shear yield strength.
    Us: u.MPa # shear ultimate strength.

    D: u.mm # diameter.
    L: u.mm # length.
    A = calc_circ_A("D") # CS area.

    ell: u.mm # shear plane location.
    edge: u.mm # casing edge distance.
    BD: u.mm # bore diameter.


    # --- safety factors ---

    # 1. pin shear.
    pin_shear = calc(Shear,
        lambda s: Shear(load=s.load, A=s.A, Ys=s.Ys, Us=s.Us))
    # 2&3. bearing.
    casing_bearing = calc(Bearing,
        lambda s: Bearing(load=s.load, D=s.BD, L=s.ell, edge=s.edge))
    inner_bearing = calc(Bearing,
        lambda s: Bearing(load=s.load, D=s.BD, L=s.L - s.ell))

    min_sfY = calc(u.none,
        lambda s: min(s.pin_shear.sfY, s.casing_bearing.sfY,
                      s.inner_bearing.sfY))
    min_sfU = calc(u.none,
        lambda s: min(s.pin_shear.sfU, s.casing_bearing.sfU,
                      s.inner_bearing.sfU))


def pins(name, load, code, *, ell, L, edge=None, quiet=False):
    """ Model each of `load.count` bolts as a shear pin, print the report, and
    return the Pin.

    code  : size and grade, e.g. "M5c 12.9".
    ell   : casing engaged thread length (shear plane location).
    L     : total engaged thread length (casing + inner).
    edge  : casing hole-centre-to-edge distance in the load direction.

    Shear is taken across the external thread minor diameter (D3). Both
    bearing modes use the tapping drill diameter.
    """
    size, grade = code.split(" ")
    D, P, BD = BOLT_SIZES[size]
    Rp, Rm = BOLT_GRADES[grade]
    pin = Pin(load=load,
              Ys=BOLT_SHEAR_YLD_RATIO*Rp,
              Us=BOLT_SHEAR_ULT_RATIO*Rm,
              D=D - 1.226869*P,
              L=L, ell=ell, BD=BD, edge=edge)
    if quiet:
        return pin

    pp = pin.display # pretty print.
    modes = {
        "bolt shear": "pin_shear",
        "casing bearing": "casing_bearing",
        "inner bearing": "inner_bearing",
    }
    bar = lambda x: f"{x*load.P/1e5:.1f} bar" # pressure at a safety factor.

    print(f"{name}:")
    print(f"  {load.count}x {size} ({grade}) bolt  "
          f"(casing {pp('casing_bearing.L')}, "
          f"inner {pp('inner_bearing.L')} engaged)")
    print(f"  P = {pp('load.P')}, F = {pp('load.F_total')} "
          f"({pp('load.F')}/bolt)")
    print(f"  {'Mode':<15}  {'SF yld':<6}  {'P yld':<9}  {'SF ult':<6}  "
          f"{'P ult':<9}")
    for mode, key in modes.items():
        sy, su = pin.fetch(f"{key}.sfY"), pin.fetch(f"{key}.sfU")
        print(f"  {mode:<15}  {sy:<6.2f}  {bar(sy):<9}  "
              f"{su:<6.2f}  {bar(su):<9}")
    gy = min(modes, key=lambda m: pin.fetch(f"{modes[m]}.sfY"))
    gu = min(modes, key=lambda m: pin.fetch(f"{modes[m]}.sfU"))
    print(f"  reuse P:  {bar(pin.min_sfY)} ({gy})")
    print(f"  burst P:  {bar(pin.min_sfU)} ({gu})")
    print()
    pin.report()
    return pin


def main():
    # Chamber, all SI.
    P = 30e5            # chamber pressure.
    D_seal = 103e-3     # seal diameter.
    D_throat = 36e-3    # throat diameter.
    Cf = 1.4            # thrust coefficient.

    # Hole geometry (fresh, non-countersunk holes).
    th_wall = 7e-3      # casing wall thickness.
    inner_th = 9e-3     # inner part (nozzle/injector) radial thickness.
    break_edge = 0.5e-3 # casing ID edge break after tapping.
    edge = 13e-3        # casing axial edge distance.

    casing_ell = th_wall - break_edge
    inner_ell = inner_th - 1e-3 # estimate.
    geom = dict(ell=casing_ell, L=casing_ell + inner_ell, edge=edge)

    noz = lambda n: nozzle(n, P, D_seal, D_throat, Cf)
    inj = lambda n: injector(n, P, D_seal)

    pins("nozzle", noz(8), "M5c 12.9", **geom)
    pins("injector", inj(12), "M5c 12.9", **geom)


if __name__ == "__main__":
    main()
