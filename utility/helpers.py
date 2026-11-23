import math
from math import sqrt, pi

import spec as u
from spec import show, style_ax
from system import *


__all__ = (
    "ATM",
    "SPRING_WIRE",
    "friction",
    "least_margin",
    "Circle",
    "Balance",
    "Spring",
)


ATM = u.barg_(0) # [Pa] Standard atmosphere, absolute.


def friction(seal, kind, DP):
    """ A radial seal's `kind` friction ("running" or "breakaway") with `DP`
    across it, rather than its own DP. """
    seal = seal.tweak(DP=max(0.0, float(DP)))
    if kind == "running":
        return seal.F_friction_running
    if kind == "breakaway":
        return seal.F_friction_breakaway
    raise ValueError("kind must be 'running' or 'breakaway'")


@metric(u.N, "least margin")
def least_margin(s):
    """ The tightest of a system's force balances (paths in `s.balances`):
    its margin beyond what's required, at the requirement's worst case. """
    reqs = s.requirements().only(*s.balances)
    return min(m.measure(s) - lo for m, (lo, _) in reqs.items())


class Circle(System):
    D: u.mm
    r = derived(u.mm, "D", lambda D: D/2, lambda r: 2*r)
    A = derived(u.mm2, "D", lambda D: pi/4*D**2, lambda A: sqrt(A*4/pi))
    P = derived(u.mm, "D", lambda D: pi*D, lambda P: D/pi)



class _Cases(tuple):
    # ((drive, oppose), ...), shown as the balance it is.
    def __repr__(self):
        return " | ".join(f"{show(d, u.N, nosym=True)} > {show(o, u.N)}"
                          for d, o in self)

class Balance(System):
    """ Drive against oppose forces, per case. Positive margins work. """
    cases: tuple # ((drive, oppose), ...) [N]

    def __post_init__(self):
        cases = _Cases((float(d), float(o)) for d, o in self.cases)
        if not cases:
            raise ValueError("a Balance needs at least one case")
        object.__setattr__(self, "cases", cases)

    @classmethod
    def of(cls, *cases):
        """ From (drive, oppose) pairs. """
        return cls(cases=cases)

    margin = calc(u.N,
        lambda s: min(d - o for d, o in s.cases))
    excess = calc(u.percent,
        lambda s: min((d/o - 1 if o>0 else math.inf) for d, o in s.cases))



# Spring wire: tensile strength S_ut = A/d^m (A [MPa mm^m], d [mm]), shear
# modulus G [GPa], and the allowable static shear stress as a fraction of S_ut
# (set not removed). Shigley's Mechanical Engineering Design, Tables 10-4 to
# 10-6.
SPRING_WIRE = {
    "music wire": (2211, 0.145, 81.0, 0.45), # ASTM A228
    "302 stainless": (1867, 0.146, 69.0, 0.35), # ASTM A313, d 0.3-2.5 mm
    "chrome silicon": (1974, 0.108, 77.2, 0.50), # ASTM A401
}

class Spring(System):
    """ Round-wire helical compression spring, ends closed and ground, working
    between installed lengths L_long and L_short. """
    material: str # A key of SPRING_WIRE.
    d: u.mm # Wire diameter.
    D: u.mm # Mean coil diameter.
    n: u.none # Active coils.
    L_free: u.mm
    L_long: u.mm # Installed at the extended end of travel.
    L_short: u.mm # Installed at the compressed end of travel.

    # Proportions a spring maker can wind, and that stay stable.
    index_range = (4, 12) # D/d.
    n_min = 3 # Active coils.
    overrun_min = 0.15 # Travel left to solid, over the working deflection.
    solid_SF_min = 1.2 # Allowable stress over the stress at solid.
    slenderness_max = 5.26 # L_free/D: no buckling with flat, parallel ends.

    def __post_init__(self):
        if self.material not in SPRING_WIRE:
            raise ValueError(f"material must be one of {list(SPRING_WIRE)}")
        if not (0 < self.d < self.D and self.n > 0):
            raise ValueError("need 0 < d < D and n > 0")

    G = calc(u.MPa, # Shear modulus.
        lambda s: SPRING_WIRE[s.material][2]*1e9)
    S_ut = calc(u.MPa, # Wire tensile strength.
        lambda s: SPRING_WIRE[s.material][0]/(s.d*1e3)**SPRING_WIRE[
                s.material][1] * 1e6)
    tau_allow = calc(u.MPa, # Allowable static shear stress.
        lambda s: SPRING_WIRE[s.material][3] * s.S_ut)

    k = calc(u.N_mm, # Rate. Sets n.
        lambda s: s.G * s.d**4 / (8 * s.D**3 * s.n),
        lambda s, k: dict(n=s.G * s.d**4 / (8 * s.D**3 * k)))
    index = calc(u.none,
        lambda s: s.D / s.d)
    OD = calc(u.mm, lambda s: s.D + s.d)
    ID = calc(u.mm, lambda s: s.D - s.d)
    L_solid = calc(u.mm, # Closed and ground ends: 2 inactive coils.
        lambda s: (s.n + 2) * s.d)
    slenderness = calc(u.none,
        lambda s: s.L_free / s.D)

    def F(self, L):
        """ Force at installed length `L` (zero once free). """
        return max(0.0, self.k * (self.L_free - L))

    def tau(self, F):
        """ Shear stress at force `F`, with Wahl's curvature factor. """
        C = self.index
        wahl = (4*C - 1)/(4*C - 4) + 0.615/C
        return wahl * 8 * F * self.D / (pi * self.d**3)

    travel = calc(u.mm, # Sets L_short.
        lambda s: s.L_long - s.L_short,
        lambda s, x: dict(L_short=s.L_long - x))
    F_long = calc(u.N, # Sets both lengths, keeping travel.
        lambda s: s.F(s.L_long),
        lambda s, F: dict(L_long=s.L_free - F/s.k,
                          L_short=s.L_free - F/s.k - s.travel))
    F_short = calc(u.N, # Sets both lengths, keeping travel.
        lambda s: s.F(s.L_short),
        lambda s, F: dict(L_short=s.L_free - F/s.k,
                          L_long=s.L_free - F/s.k + s.travel))
    overrun = calc(u.percent, # Travel left to solid, over the deflection.
        lambda s: (s.L_short - s.L_solid) / (s.L_free - s.L_short))
    tau_short = calc(u.MPa,
        lambda s: s.tau(s.F_short))
    solid_SF = calc(u.none, # Survives being compressed solid with this SF.
        lambda s: s.tau_allow / s.tau(s.F(s.L_solid)))

    def requirements(self):
        return super().requirements() | {
            "index": self.index_range,
            "n": (self.n_min, None),
            corner("overrun", "min"): (self.overrun_min, None),
            corner("solid_SF", "min"): (self.solid_SF_min, None),
            "slenderness": (None, self.slenderness_max),
        }
