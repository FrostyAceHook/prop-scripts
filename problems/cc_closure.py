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
from oring import *



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
    sharing_factor: u.none # multiplier on yield strengths to model imperfect
                           # load sharing.
    A: u.mm2 # "effective" pressure area.
    F = calc(u.kN, # axial force per retainer.
        lambda s: s.F_total/s.count,
        lambda s, F: dict(F_total=F*s.count))
    F_total = calc(u.kN, # total axial force.
        lambda s: s.A*s.P,
        lambda s, F: dict(P=F/s.A))

def injector(count, P, D_seal, sharing_factor):
    return Load(end="injector", count=count, P=P, sharing_factor=sharing_factor,
            A=circ_area(D_seal))

def nozzle(count, P, D_seal, sharing_factor, D_throat, Cf):
    return Load(end="nozzle", count=count, P=P, sharing_factor=sharing_factor,
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
SHEAR_YLD_RATIO = 1/sqrt(3)

# Shear ultimate strength as a fraction of Rm (empirical, 0.6--0.65 typical).
SHEAR_ULT_RATIO = 0.60




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



### CANTING #####################################################################

def embedment_modes(p1, p2, t1, t2, M):
    """ Lateral capacity of one stud in single shear across two members, with
    the stud free to cant (rotate) and to form plastic hinges. This is
    Johansen's yield model (the basis of the Eurocode 5 dowel equations, here
    without their empirical factors or rope effect).

    p1, p2 : bearing capacity per unit length (bore diameter x bearing stress)
             of member 1 (casing) and member 2 (inner part).
    t1, t2 : engaged lengths in member 1 and member 2.
    M      : plastic moment of the stud.

    Returns {mode: capacity}. The governing capacity is the minimum.

    Modes:
      casing uniform : member 1 crushes along its whole length, stud straight.
      inner uniform  : member 2 crushes along its whole length, stud straight.
      canted         : rigid stud rotates; both members crush, with reversed
                       bearing at the far end of each. Closed form of the
                       rigid-plastic cant model; = 0.41*p*t at t1 = t2.
      hinge in inner : casing side rotates rigidly, stud hinges in member 2.
      hinge in casing: inner side rotates rigidly, stud hinges in member 1.
      two hinges     : stud hinges in both members (long, slender stud).
    """
    b = p2 / p1                 # embedment strength ratio.
    r = t2 / t1                 # length ratio.
    return {
        "casing uniform": p1*t1,
        "inner uniform": p2*t2,
        "canted": p1*t1/(1 + b) * (sqrt(b + 2*b*b*(1 + r + r*r) + b**3*r*r)
                                   - b*(1 + r)),
        "hinge in inner": p1*t1/(2 + b) * (sqrt(2*b*(1 + b)
                                   + 4*b*(2 + b)*M/(p1*t1*t1)) - b),
        "hinge in casing": p1*t2/(1 + 2*b) * (sqrt(2*b*b*(1 + b)
                                   + 4*b*(1 + 2*b)*M/(p1*t2*t2)) - b),
        "two hinges": sqrt(2*b/(1 + b)) * sqrt(2*M*p1),
    }



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


class Embedment(System):
    """ Combined casing + inner bearing, with the stud free to cant and bend
    (see `embedment_modes`). Supersedes the two uniform bearing checks, which
    assume the stud stays square. """
    load: Load
    casing: Bearing
    inner: Bearing
    d: u.mm   # stud diameter for bending (thread root).
    Ys: u.MPa # stud tensile yield (for the yield-level plastic moment).
    Us: u.MPa # stud tensile ultimate (for the ultimate-level plastic moment).

    MY = calc(u.Nm, lambda s: s.Ys * s.d**3/6) # plastic moment, yield
    MU = calc(u.Nm, lambda s: s.Us * s.d**3/6) # plastic moment, ultimate.

    def _modes(s, which):
        Y = which == "Y"
        p1 = s.casing.D * (s.casing.Ys if Y else s.casing.Us)
        p2 = s.inner.D * (s.inner.Ys if Y else s.inner.Us)
        return embedment_modes(p1, p2, s.casing.L, s.inner.L,
                               s.MY if Y else s.MU)

    capY = calc(u.kN, lambda s: min(s._modes("Y").values())) # per stud.
    capU = calc(u.kN, lambda s: min(s._modes("U").values())) # per stud.
    modeY = calc(None, lambda s: min((m := s._modes("Y")), key=m.get))
    modeU = calc(None, lambda s: min((m := s._modes("U")), key=m.get))

    # Capacity relative to the uniform casing bearing check (1 = no effect).
    cant_factor = calc(u.none,
        lambda s: s.capU / (s.casing.A*s.casing.Us))

    sfY = calc(u.none, lambda s: s.capY*s.load.sharing_factor / s.load.F)
    sfU = calc(u.none, lambda s: s.capU / s.load.F)




### PIN #########################################################################

class Pin(System):
    # --- situation ---

    load: Load # load.

    SYs: u.MPa # shear yield strength.
    SUs: u.MPa # shear ultimate strength.
    TYs: u.MPa # tensile yield strength (stud bending).
    TUs: u.MPa # tensile ultimate strength (stud bending).

    D: u.mm # diameter.
    L: u.mm # length.
    A = calc_circ_A("D") # CS area.

    ell: u.mm # shear plane location.
    edge: u.mm # casing edge distance.
    BD: u.mm # bore diameter.


    # --- safety factors ---

    # 1. pin shear.
    pin_shear = calc(Shear,
        lambda s: Shear(load=s.load, A=s.A, Ys=s.SYs, Us=s.SUs))
    # 2&3. bearing.
    casing_bearing = calc(Bearing,
        lambda s: Bearing(load=s.load, D=s.BD, L=s.ell, edge=s.edge))
    inner_bearing = calc(Bearing,
        lambda s: Bearing(load=s.load, D=s.BD, L=s.L - s.ell))

    # 4. bearing with canting and stud bending (covers 2&3 as special cases).
    embedment = calc(Embedment,
        lambda s: Embedment(load=s.load, casing=s.casing_bearing,
                            inner=s.inner_bearing, d=s.D, Ys=s.TYs, Us=s.TUs))

    min_sfY = calc(u.none,
        lambda s: min(s.pin_shear.sfY, s.casing_bearing.sfY,
                      s.inner_bearing.sfY, s.embedment.sfY))
    min_sfU = calc(u.none,
        lambda s: min(s.pin_shear.sfU, s.casing_bearing.sfU,
                      s.inner_bearing.sfU, s.embedment.sfU))


def tapped_thru(load, code, *, ell_casing, ell_inner, edge=None, quiet=False):
    size, grade = code.split(" ")
    D, P, BD = BOLT_SIZES[size]
    Rp, Rm = BOLT_GRADES[grade]
    pin = Pin(load=load,
              SYs=SHEAR_YLD_RATIO*Rp, SUs=SHEAR_ULT_RATIO*Rm,
              TYs=Rp, TUs=Rm,
              D=D - 1.226869*P,
              L=ell_casing + ell_inner, ell=ell_casing, BD=BD, edge=edge)
    if quiet:
        return pin

    pp = pin.display # pretty print.
    modes = {
        "bolt shear": "pin_shear",
        "casing bearing": "casing_bearing",
        "inner bearing": "inner_bearing",
        "canted bearing": "embedment",
    }
    bar = lambda x: f"{x*load.P/1e5:.1f} bar" # pressure at a safety factor.

    print(f"{load.end}:")
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
    print(f"  canted bearing: yield by {pin.embedment.modeY}, "
          f"ult by {pin.embedment.modeU} "
          f"(x{pin.embedment.cant_factor:.2f} of uniform casing bearing)")
    print(f"  reuse P:  {bar(pin.min_sfY)} ({gy})")
    print(f"  burst P:  {bar(pin.min_sfU)} ({gu})")
    print()
    # pin.report()
    return pin



def pinned(load, code, *, ell_casing, ell_inner, edge=None, quiet=False):
    size, grade = code.split(" ")
    assert size[0] == "D", "invalid size"
    D = float(size[1:])*1e-3
    if grade == "4140HT":
        Rp = 750e6
        Rm = 930e6
    elif grade == "100Cr6":
        Rp = 1700e6
        Rm = 2200e6
    else:
        assert False, "invalid grade"
    pin = Pin(load=load,
              SYs=SHEAR_YLD_RATIO*Rp, SUs=SHEAR_ULT_RATIO*Rm,
              TYs=Rp, TUs=Rm,
              D=D, L=ell_casing + ell_inner, ell=ell_casing, BD=D, edge=edge)
    if quiet:
        return pin

    pp = pin.display # pretty print.
    modes = {
        "bolt shear": "pin_shear",
        "casing bearing": "casing_bearing",
        "inner bearing": "inner_bearing",
        "canted bearing": "embedment",
    }
    bar = lambda x: f"{x*load.P/1e5:.1f} bar" # pressure at a safety factor.

    print(f"{load.end}:")
    print(f"  {load.count}x {size} ({grade}) pin  "
          f"(casing {pp('casing_bearing.L')}, "
          f"inner {pp('inner_bearing.L')} engaged)")
    print(f"  P = {pp('load.P')}, F = {pp('load.F_total')} "
          f"({pp('load.F')}/pin)")
    print(f"  {'Mode':<15}  {'SF yld':<6}  {'P yld':<9}  {'SF ult':<6}  "
          f"{'P ult':<9}")
    for mode, key in modes.items():
        sy, su = pin.fetch(f"{key}.sfY"), pin.fetch(f"{key}.sfU")
        print(f"  {mode:<15}  {sy:<6.2f}  {bar(sy):<9}  "
              f"{su:<6.2f}  {bar(su):<9}")
    gy = min(modes, key=lambda m: pin.fetch(f"{modes[m]}.sfY"))
    gu = min(modes, key=lambda m: pin.fetch(f"{modes[m]}.sfU"))
    print(f"  canted bearing: yield by {pin.embedment.modeY}, "
          f"ult by {pin.embedment.modeU} "
          f"(x{pin.embedment.cant_factor:.2f} of uniform casing bearing)")
    print(f"  reuse P:  {bar(pin.min_sfY)} ({gy})")
    print(f"  burst P:  {bar(pin.min_sfU)} ({gu})")
    print()
    return pin


def main():
    P = 30e5               # chamber pressure.
    D_seal = 103e-3        # seal diameter.
    D_throat = 36e-3       # throat diameter.
    Cf = 1.4               # thrust coefficient.
    sharing_factor = 0.85  # -15% strength.

    inj = lambda n: injector(n, P, D_seal, sharing_factor)
    noz = lambda n: nozzle(n, P, D_seal, sharing_factor, D_throat, Cf)

    inj_geom = dict(ell_casing=3.23e-3, ell_inner=8.5e-3, edge=15.0)
    noz_geom = dict(ell_casing=3.01e-3, ell_inner=8.0e-3, edge=13.5)

    pinned(inj(12), "D8 100Cr6", **inj_geom)
    pinned(noz(8), "D6 100Cr6", **noz_geom)


    seal = PistonSeal(name="cc seals",
            oring_ID=94.84e-3, oring_CS=3.53e-3,
            gland_ID=97.2e-3, gland_OD=103.0e-3, gland_Lz=4.7e-3,
            DP=30e5,
            material="FKM", hardness=75, lubricated=False)
    seal.outline(succinct=True)


if __name__ == "__main__":
    main()
