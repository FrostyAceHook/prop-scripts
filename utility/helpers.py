import math
import itertools
from math import sqrt, pi

import spec as u
from spec import show, style_ax, table
from system import *


__all__ = (
    "ATM",
    "SPRING_WIRE",
    "friction",
    "least_margin",
    "reseal",
    "seal_options",
    "Circle",
    "Annulus",
    "ThickWall",
    "AL6061_T6",
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


def reseal(s, name, round_root=u.mm_(0.01), round_Lz=u.mm_(0.05), **auto):
    """
    s with seal `name` (slot gr_<name>, seal or_<name>) re-sized as
    AutoGroove(**auto) would (e.g. CS=u.mm_(1.5), squeeze=0.16), then frozen
    as a Groove: its root rounded to `round_root` toward more squeeze, its
    width to `round_Lz` (None leaves either as sized). The slot's material,
    hardness, lubrication and friction level carry over.
    """
    from oring import AutoGroove, Groove
    slot = getattr(s, "gr_" + name)
    keep = dict(material=slot.material, hardness=slot.hardness,
                lubricated=slot.lubricated, friction_level=slot.friction_level)
    s = s.tweak({"gr_" + name: AutoGroove(**keep | auto)})
    seal = getattr(s, "or_" + name)
    groove = Groove.of(seal)
    if round_root:
        x = groove.root / round_root
        x = math.ceil(x - 1e-6) if seal.kind == "piston" \
            else math.floor(x + 1e-6)
        groove = groove.tweak(root=x * round_root)
    if round_Lz:
        groove = groove.tweak(Lz=round(groove.Lz / round_Lz) * round_Lz)
    return s.tweak({"gr_" + name: groove})


def seal_options(s, name, CS=(None,), squeeze=(0.15,), gland_fill=(0.7,),
                 **kw):
    """ A table of seal `name` re-sealed (see `reseal`) at every combination
    of CS, squeeze and gland fill: its ring, groove, 3σ ranges, friction, and
    which of its own requirements fail. Returns the passing systems. """
    rows, passing = [], []
    for cs in CS:
        for sq in squeeze:
            for fill in gland_fill:
                label = [show(cs, u.mm) if cs else "auto", f"{sq:.0%}",
                         f"{fill:.0%}"]
                try:
                    r = reseal(s, name, CS=cs, squeeze=sq, gland_fill=fill,
                               **kw)
                except ValueError:
                    rows.append(label + ["no ring fits"] + [""]*6)
                    continue
                seal = getattr(r, "or_" + name)
                ver = seal.verify()
                rng = lambda n: "{:.1%}..{:.1%}".format(*seal.bounds(n))
                rows.append(label + [
                    seal.oring.ref or "custom",
                    seal.display("gland_root"), seal.display("gland_Lz"),
                    rng("squeeze"), rng(seal.fit),
                    show(seal.F_friction_breakaway, u.N),
                    "ok" if ver.ok else ", ".join(m.name for m in ver.failed)])
                if ver.ok:
                    passing.append(r)
    print(f"Options for {name}:\n" + table(rows, ["CS", "squeeze", "fill",
            "ring", "root", "Lz", "squeeze (3σ)", "fit (3σ)", "breakaway",
            ""]))
    return passing


class Circle(System):
    D: u.mm
    r = derived(u.mm, "D", lambda D: D/2, lambda r: 2*r)
    A = derived(u.mm2, "D", lambda D: pi/4*D**2, lambda A: sqrt(A*4/pi))
    p = derived(u.mm, "D", lambda D: pi*D, lambda p: p/pi)


class Annulus(System):
    """ Inner + outer circle. Defined by two of ID, OD and th (not all three).
    Annulus(...) builds the matching form: AnnulusIDOD, AnnulusIDth or
    AnnulusODth. """

    def __new__(cls, *args, **given):
        if cls is not Annulus:
            return super().__new__(cls)
        if args:
            raise TypeError("need keywords only: two of ID=, OD=, and th=")
        form = _ANNULUS_FORMS.get(frozenset(given))
        if form is None:
            raise TypeError("need exactly two of ID=, OD=, and th=, got "
                            f"{', '.join(given) or 'none'}")
        if any(v is None for v in given.values()):
            raise TypeError("dimensions can't be None")
        return super().__new__(form)

    def __post_init__(self):
        if not (self.ID > 0 and self.th > 0):
            raise ValueError(f"need 0 < ID < OD (ID {self.ID}, OD {self.OD} m)")

    Ir = calc(u.mm, lambda s: s.ID/2)
    Or = calc(u.mm, lambda s: s.OD/2)
    A = calc(u.mm2, lambda s: pi/4*(s.OD**2 - s.ID**2))
    Ip = calc(u.mm, lambda s: pi*s.ID)
    Op = calc(u.mm, lambda s: pi*s.OD)

    inner = calc(Circle, lambda s: Circle(D=s.ID))
    outer = calc(Circle, lambda s: Circle(D=s.OD))

class AnnulusIDOD(Annulus, kw_only=True):
    """ Defined by ID and OD. Setting thickness keeps the ID. """
    ID: u.mm
    OD: u.mm
    th = calc(u.mm, lambda s: (s.OD - s.ID)/2,
              lambda s, th: dict(OD=s.ID + 2*th))

class AnnulusIDth(Annulus, kw_only=True):
    """ Defined by ID and thickness. Setting OD keeps the ID. """
    ID: u.mm
    th: u.mm
    OD = calc(u.mm, lambda s: s.ID + 2*s.th,
              lambda s, OD: dict(th=(OD - s.ID)/2))

class AnnulusODth(Annulus, kw_only=True):
    """ Defined by OD and thickness. Setting ID keeps the OD. """
    OD: u.mm
    th: u.mm
    ID = calc(u.mm, lambda s: s.OD - 2*s.th,
              lambda s, ID: dict(th=(s.OD - ID)/2))

_ANNULUS_FORMS = {frozenset(("ID", "OD")): AnnulusIDOD,
                  frozenset(("ID", "th")): AnnulusIDth,
                  frozenset(("OD", "th")): AnnulusODth}




# 6061-T6: Young's modulus, Poisson's ratio, 0.2% yield and ultimate [Pa].
AL6061_T6 = dict(E=68.9e9, nu=0.33, Ys=276e6, Us=310e6)

class ThickWall(System):
    """
    A thick-walled (Lamé) cylinder under internal and external pressure. Stresses
    are at the internal wall, where they all peak. `ends`:
      "open"    no axial loading.
      "closed"  full axial loading from the pressure.
      "plane"   held axially: plane strain
    """
    wall: Annulus
    E: u.MPa # Young's modulus.
    nu: u.none # Poisson's ratio.
    Ys: u.MPa # Yield strength.
    Us: u.MPa # Ultimate strength.
    Pi: u.bar # internal pressure.
    Po: u.bar = ATM # external pressure.
    ends: str = "closed"

    def __post_init__(self):
        if self.ends not in ("closed", "open", "plane"):
            raise ValueError(f"ends is 'closed', 'open' or 'plane', not "
                             f"{self.ends!r}")

    def stresses(self, r, Pi=None):
        """ (hoop, radial, axial) at radius `r` with internal pressure `P`. """
        p_i = (self.Pi if Pi is None else Pi) - ATM
        p_o = self.Po - ATM
        ri, ro = self.wall.Ir, self.wall.Or
        k = ro**2 - ri**2
        A = (p_i*ri**2 - p_o*ro**2)/k
        B = (p_i - p_o)*ri**2*ro**2/k
        axial = {"closed": A, "open": 0.0, "plane": 2*self.nu*A}[self.ends]
        return A + B/r**2, A - B/r**2, axial

    def _vm(self, r, Pi=None):
        h, rd, a = self.stresses(r, Pi)
        return sqrt(((h - rd)**2 + (rd - a)**2 + (a - h)**2)/2)

    dP = calc(u.bar, lambda s: s.Pi - s.Po)
    hoop = calc(u.MPa, lambda s: s.stresses(s.wall.Ir)[0])
    radial = calc(u.MPa, lambda s: s.stresses(s.wall.Ir)[1])
    axial = calc(u.MPa, lambda s: s.stresses(s.wall.Ir)[2])
    vm = calc(u.MPa, lambda s: s._vm(s.wall.Ir)) # von Mises.
    tresca = calc(u.MPa, # Stress intensity: twice the max shear.
        lambda s: max(abs(a - b) for a, b in itertools.combinations(
            s.stresses(s.wall.Ir), 2)))
    SF = calc(u.none, # On yield, by von Mises.
        lambda s: s.Ys/s.vm if s.vm else math.inf)

    @computed(u.bar)
    def Pi_yield(s): # Internal pressure (absolute) that first yields the bore.
        # The stresses are linear in it, so von Mises^2 is quadratic in it.
        f = lambda P: s._vm(s.wall.Ir, P)**2
        Po = s.Po
        c, a_b, b_a = f(Po), f(Po + 1e6), f(Po - 1e6)
        a = (a_b + b_a - 2*c)/2e12
        b = (a_b - b_a)/2e6
        c -= s.Ys**2
        return Po + (-b + sqrt(b*b - 4*a*c))/(2*a)

    Pi_burst = calc(u.bar, # Faupel's burst pressure (absolute), closed ends.
        lambda s: s.Po + 2/sqrt(3)*s.Ys*(2 - s.Ys/s.Us)
                  * math.log(s.wall.Or/s.wall.Ir))

    @computed(u.um)
    def dID(s): # Bore growth (diametral), as felt by a seal in it.
        h, rd, a = s.stresses(s.wall.Ir)
        return 2*s.wall.Ir*(h - s.nu*(rd + a))/s.E

    @computed(u.um)
    def dOD(s): # Outside diameter growth.
        h, rd, a = s.stresses(s.wall.Or)
        return 2*s.wall.Or*(h - s.nu*(rd + a))/s.E




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
