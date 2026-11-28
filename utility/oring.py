"""
O-ring calculator, for friction as well as stretch/squeeze/gland analysis.

Tolerances are spreads on the inputs (see `spec`): the ring's come with its
catalogue `OringSize`, the gland's are given (auto-sizing adds defaults), and
the friction coefficients' scatter is one on `friction_level`. Toleranced
ranges and requirements are 3-sigma FORM bounds from `System.corners`.

In an assembly, a radial seal's own parts are a `Groove` (sized by hand) or an
`AutoGroove` (sized for its diameter when built); the assembly adds the sliding
diameter and conditions. See `System.resolved` for how auto-sized seals are
frozen for statistics.
"""

from abc import ABC, abstractmethod
from itertools import groupby
from math import pi, sqrt, log, erf, isfinite, inf
from typing import Literal

import spec as u
from spec import show
from system import System, alias, calc, computed, corner
from oring_sizes import (OringSize, bs1806, finer, BS1806, FINER_BS,
        FINER_METRIC, MATCH_TOL)


__all__ = (
    "PistonSeal",
    "RodSeal",
    "InternalFaceSeal",
    "ExternalFaceSeal",
    "Groove",
    "AutoGroove",
    "OringFriction",
    "extrusion_gap",
    "OringSize",
    "bs1806",
    "finer",
    "BS1806",
    "FINER_BS",
    "FINER_METRIC",
)


MATERIAL_TYPE = Literal["FKM", "NBR", "EPDM"]

# Hardness range [Shore A] stocked for each O-ring material.
HARDNESS_RANGE = {
    "FKM": (70, 90),
    "NBR": (70, 90),
    "EPDM": (60, 90),
}

# Friction coefficients against metal, as (low, nominal, high), for
# (material, lubricated).
FRICTION = {
    ("FKM", False): {"running":   (0.40, 0.60, 0.80),
                     "pressure":  (0.40, 0.60, 0.80),
                     "breakaway": (0.80, 1.00, 1.25)},
    ("NBR", False): {"running":   (0.50, 0.70, 1.00),
                     "pressure":  (0.50, 0.70, 1.00),
                     "breakaway": (0.80, 1.00, 1.30)},
    ("EPDM", False): {"running":   (0.45, 0.65, 0.90),
                      "pressure":  (0.45, 0.65, 0.90),
                      "breakaway": (0.80, 1.00, 1.30)},
    ("FKM", True):  {"running":   (0.07, 0.12, 0.30),
                     "pressure":  (0.03, 0.05, 0.12),
                     "breakaway": (0.20, 0.35, 0.50)},
    ("NBR", True):  {"running":   (0.07, 0.12, 0.30),
                     "pressure":  (0.03, 0.05, 0.12),
                     "breakaway": (0.20, 0.35, 0.50)},
    ("EPDM", True):  {"running":   (0.07, 0.12, 0.30),
                      "pressure":  (0.03, 0.05, 0.12),
                      "breakaway": (0.20, 0.35, 0.50)},
}

# Friction level: a z-score of the coefficients' (log-normal) scatter, so LOW
# and HIGH are the tabulated values, taken as +-3 sigma. SPREAD (the default)
# is that scatter as a Tol (sd 1, limits LOW and HIGH); the others fix the
# level.
_ZBOUND = 3.0
class OringFriction: # exported on import *.
    LOW = u.exact(-_ZBOUND)
    NOM = u.exact(0.0)
    HIGH = u.exact(+_ZBOUND)
    SPREAD = u.tol(0.0, _ZBOUND)


# Largest radial extrusion gap [mm] an O-ring holds without back-up rings, as
# {hardness [Shore A]: {pressure up to [MPa]: (gap for CS up to 2, 3, 5, 7,
# and over 7 mm)}}. A common catalogue table (e.g. Hansa-Flex, "Permissible
# clearance gap of O-rings"). Past its last pressure, back-up rings are needed.
_GAP_CS = (2.0, 3.0, 5.0, 7.0, inf)
_GAP = {
    70: {3.5: (0.08, 0.09, 0.10, 0.13, 0.15),
         7.0: (0.05, 0.07, 0.08, 0.09, 0.10),
         10.5: (0.03, 0.04, 0.05, 0.07, 0.08)},
    80: {3.5: (0.10, 0.13, 0.15, 0.18, 0.20),
         7.0: (0.08, 0.09, 0.10, 0.13, 0.15),
         10.5: (0.05, 0.07, 0.08, 0.09, 0.10)},
    90: {3.5: (0.13, 0.15, 0.20, 0.23, 0.25),
         7.0: (0.10, 0.13, 0.15, 0.18, 0.20),
         10.5: (0.07, 0.09, 0.10, 0.13, 0.15),
         14.0: (0.05, 0.07, 0.08, 0.09, 0.10),
         17.5: (0.04, 0.05, 0.07, 0.08, 0.09)},
}

def extrusion_gap(DP, hardness, CS):
    """
    Largest radial gap [m] an O-ring of `hardness` [Shore A] and `CS` [m]
    holds against `DP` [Pa] without back-up rings; 0 means it needs them.
    Stepped in pressure and CS (a CS within 0.1 mm over a step counts as that
    step, so a 2 mm ring is a "2 mm" ring), linear in hardness (70-90).
    """
    col = next(i for i, edge in enumerate(_GAP_CS) if CS*1e3 <= edge + 0.1)
    def at(h):
        for p, gaps in _GAP[h].items():
            if DP <= p*1e6:
                return gaps[col]*1e-3
        return 0.0
    h = min(max(hardness, 70.0), 90.0)
    lo = 70 if h < 80 else 80
    f = (h - lo)/10
    return at(lo) + f*(at(lo + 10) - at(lo)) if f else at(lo)


def _span(lo, hi, unit):
    return f"{show(lo, unit)} to {show(hi, unit)}"


def _shift_gland(s, MD):
    # Move the gland to a new mid-diameter, keeping its radial size. Sets the
    # wall moving away first, so the gland never inverts partway through.
    ID, OD = MD - s.gland_Lr, MD + s.gland_Lr
    if MD > s.gland_MD:
        return dict(gland_OD=OD, gland_ID=ID)
    return dict(gland_ID=ID, gland_OD=OD)


class Oring(System, ABC):
    """
    Common geometry, material and force maths for an O-ring in a gland.
    Abstract: use PistonSeal, RodSeal, InternalFaceSeal or ExternalFaceSeal.
    """

    # [-] Squeeze limits.
    squeeze_min = 0.08
    squeeze_max = 0.30

    # [-] Gland fill limit.
    gland_fill_max = 0.85

    # [-] Recommended ranges, shown in the outline.
    squeeze_aim = (0.10, 0.20)
    gland_fill_aim = (0.60, 0.80)

    # [-] Assembly fit limits and the value auto-sizing aims for.
    ID_stretch_min = 0.01
    ID_stretch_target = 0.02
    ID_stretch_max = 0.05
    OD_compression_min = 0.01
    OD_compression_target = 0.02
    OD_compression_max = 0.03

    # The assembly fit that holds the ring in place and picks it when
    # auto-sizing: "ID_stretch" or "OD_compression".
    fit = None

    name: str # Cosmetic name.
    material: MATERIAL_TYPE
    hardness: u.shoreA
    lubricated: bool
    oring: OringSize
    gland_ID: u.mm
    gland_OD: u.mm
    gland_Lz: u.mm
    DP: u.bar # Pressure difference across the seal.
    friction_level: u.none = OringFriction.SPREAD # z-score: OringFriction.
    gap: u.mm = None # Largest radial extrusion gap (None: unchecked).

    def __init_subclass__(cls, **options):
        # Keep the short `__repr__` below for every seal type, rather than the
        # dataclass one listing every field.
        super().__init_subclass__(**{"repr": False, **options})

    def __post_init__(self):
        if not isinstance(self.oring, OringSize):
            raise TypeError("oring must be an OringSize")
        for v in (self.gland_ID, self.gland_OD, self.gland_Lz, self.DP,
                self.hardness, self.friction_level):
            if not isfinite(v):
                raise ValueError("inputs must be finite numbers")
        if not isinstance(self.lubricated, bool):
            raise TypeError("lubricated must be True or False")
        if isinstance(self.friction_level, bool):
            raise TypeError("friction_level must be a z-score")
        if self.material not in HARDNESS_RANGE:
            raise ValueError(f"material must be one of {list(HARDNESS_RANGE)}")
        if not 0 < self.hardness < 100:
            raise ValueError("hardness must be between 0 and 100 Shore A")
        if not 0 < self.gland_ID < self.gland_OD:
            raise ValueError("need 0 < gland_ID < gland_OD")
        if self.gland_Lz <= 0:
            raise ValueError("gland_Lz must be positive")
        if self.DP < 0:
            raise ValueError("DP must be zero or positive")
        if self.gap is not None and not (isfinite(self.gap)
                                         and self.gap >= 0):
            raise ValueError("gap must be None, or zero or positive")


    # ---- Gland geometry -------------------------------------------------------

    gland_Lr = calc(u.mm, # Set through gland_depth/gland_width instead.
        lambda s: (s.gland_OD - s.gland_ID)/2)
    gland_MD = calc(u.mm, # Moves the gland, keeping gland_Lr.
        lambda s: (s.gland_ID + s.gland_OD)/2,
        lambda s, MD: _shift_gland(s, MD))

    @computed(u.mm)
    @abstractmethod
    def gland_depth(self): ... # Squeeze direction.
    @computed(u.mm)
    @abstractmethod
    def gland_width(self): ...
    gland_area = calc(u.mm2, # Sets gland_width.
        lambda s: s.gland_depth * s.gland_width,
        lambda s, A: dict(gland_width=A / s.gland_depth))

    @computed(u.mm)
    @abstractmethod
    def contact_D(self): ... # Sealing line diameter.


    # ---- Material -------------------------------------------------------------

    @computed(u.MPa)
    def E(self): # Young's modulus. Sets hardness.
        # BS 903 relation, S = 100 erf(3.186e-4 sqrt(E)).
        lo, hi = 0.0, 3.0
        for _ in range(60):
            x = (lo + hi) / 2
            lo, hi = (x, hi) if erf(x) < self.hardness / 100 else (lo, x)
        return (x / 3.186e-4)**2
    @E.setter
    def E(self, E):
        return dict(hardness=100 * erf(3.186e-4 * sqrt(E)))
    nu = calc(u.none, # Poisson's constant.
        lambda s: 0.5) # essentially/assumed incompressible.

    def mu(self, kind):
        """ Friction coefficient of the given `kind`, at `friction_level`. """
        lo, nom, hi = FRICTION[(self.material, self.lubricated)][kind]
        z = self.friction_level
        return nom * ((hi if z > 0 else lo) / nom)**(abs(z) / _ZBOUND)
    mu_running = calc(u.none, lambda s: s.mu("running"))
    mu_pressure = calc(u.none, lambda s: s.mu("pressure"))
    mu_breakaway = calc(u.none, lambda s: s.mu("breakaway"))


    # ---- O-ring fit -----------------------------------------------------------

    @computed(u.mm)
    @abstractmethod
    def effective_MD(self): ...
    effective_CS = calc(u.mm,
        lambda s: s.oring.CS * sqrt(s.oring.MD / s.effective_MD))

    @computed(u.percent)
    def squeeze(self): # Sets gland_depth.
        return max(0.0, 1 - self.gland_depth/self.effective_CS)
    @squeeze.setter
    def squeeze(self, e):
        # Effective CS depends (weakly) on the gland, so iterate to converge.
        o = self
        for _ in range(20):
            o = o.tweak(gland_depth=o.effective_CS * (1 - e))
        return dict(gland_depth=float(o.gland_depth)) # may be a Tol'd input
    gland_fill = calc(u.percent, # Sets gland_width.
        lambda s: pi/4 * s.effective_CS**2 / s.gland_area,
        lambda s, f: dict(gland_width=
            pi/4 * s.effective_CS**2 / (f * s.gland_depth)))

    ID_stretch = calc(u.percent, # Sets oring.ID.
        lambda s: s.gland_ID/s.oring.ID - 1,
        lambda s, x: {"oring.ID": s.gland_ID / (1 + x)})
    OD_compression = calc(u.percent, # Sets oring.ID (keeps CS).
        lambda s: 1 - s.gland_OD/s.oring.OD,
        lambda s, x: {"oring.OD": s.gland_OD / (1 - x)})

    def _fit(self):
        """ (value, min, max, target) of the assembly fit. """
        return tuple(getattr(self, self.fit + suffix)
                     for suffix in ("", "_min", "_max", "_target"))


    # ---- Normal (contact) forces ----------------------------------------------

    @computed(u.N_mm)
    def q_normal_squeeze(self):
        # Lindley: normal force from squeeze per unit length of sealing line.
        e = self.squeeze
        return self.E * self.effective_CS * (1.25 * e**1.5 + 50 * e**6)

    @computed(u.mm)
    def contact_width(self):
        # Hertz line contact: the cross-section treated as a cylinder of radius
        # CS/2 pressed onto a flat wall with line load q, at zero pressure.
        # Elliptical pressure profile over a band of width b. Hertz assumes
        # b << CS; at typical squeezes b ~ CS/2, so this is extrapolated
        # (geometry suggests the real width is ~15% larger).
        E_star = self.E / (1 - self.nu**2)
        R = self.effective_CS / 2
        return 2 * sqrt(4 * self.q_normal_squeeze * R / (pi * E_star))

    @computed(u.MPa)
    def contact_peak_stress(self):
        # Peak of the Hertz profile at zero pressure: p0 = 4q / (pi b). Indicates
        # low-pressure / start-up sealing. Under pressure the rubber transmits DP
        # like a fluid, so contact is roughly p0 + DP.
        b = self.contact_width
        return 4/pi * self.q_normal_squeeze / b if (b > 0) else 0.0

    # Total normal force on the sliding surface from squeeze.
    F_normal_squeeze = calc(u.N,
        lambda s: s.q_normal_squeeze * pi * s.contact_D)
    # Extra normal force on the sliding surface: pressure is transmitted
    # through the rubber and acts over the contact band.
    F_normal_pressure = calc(u.N,
        lambda s: s.DP * s.contact_width * pi * s.contact_D)
    F_normal = calc(u.N,
        lambda s: s.F_normal_squeeze + s.F_normal_pressure)


    # ---- Extrusion ------------------------------------------------------------

    gap_max = calc(u.mm, # Largest radial extrusion gap, without back-ups.
        lambda s: extrusion_gap(s.DP, s.hardness, s.oring.CS))
    extrusion_margin = calc(u.mm, # Of gap_max over gap.
        lambda s: s.gap_max - s.gap if s.gap is not None else float("nan"))


    # ---- Auto-sizing ----------------------------------------------------------
    #
    # For every standard O-ring: build a rough gland (_auto_start), let .tweak
    # set the squeeze and then the gland fill (the setters move only the walls
    # they should), then keep the result if it is acceptable (_auto_ok) and
    # rank it (_auto_rank).
    #   Radial seals: the bore or rod is given. The ring is chosen by its
    #     assembly fit, ranked by closeness to the fit target.
    #   Face seals: the groove walls are free, so the groove is placed around
    #     each ring at the fit target; the given diameter is the opening the
    #     seal must go around (minimum groove ID). Smallest seal wins.
    # Gland walls without a spread (including a plain given diameter) get the
    # `gland_tol` band for their wall, as `u.tol(wall, band)`; None leaves
    # one unassessed.

    gland_tol = dict(gland_ID=0.05e-3, gland_OD=0.05e-3, gland_Lz=0.1e-3)

    # Rule-of-thumb CS grows with the sealing diameter D, roughly as sqrt(D):
    # CS ~ 0.42 sqrt(D), both in mm, with a floor. This follows the usual
    # pairings (1.78 mm around 10-20 mm, 2.62 mm around 25-45 mm, 3.53 mm around
    # 50-90 mm, 5.33 mm around 120-250 mm). Thinner rings are fragile and
    # tolerance-sensitive. Thicker rings add friction and bulk.
    CS_pref_k = 0.42 # [mm^0.5]

    @classmethod
    def preferred_CS(cls, D, CS_pref_k=None):
        if CS_pref_k is None:
            CS_pref_k = cls.CS_pref_k
        return CS_pref_k * sqrt(D*1e3) * 1e-3

    @classmethod
    @abstractmethod
    def _auto_start(cls, D, size, squeeze): ...
        # Rough gland for O-ring `size`, as field kwargs, or None if this ring
        # cannot work. Only a starting point for the tweaks.
    def _auto_ok(self, D):
        value, lo, hi, _ = self._fit()
        return lo <= value <= hi and self.squeeze > 0
    @abstractmethod
    def _auto_rank(self): ... # Minimised.
    @classmethod
    def _auto(cls, name, D, squeeze, gland_fill, *, CS=None, CS_pref_k=None,
            sizes=FINER_BS+FINER_METRIC, gland_tol=None, all_options=False,
            **seal):
        if CS is not None:
            sizes = [s for s in sizes if abs(s.CS - CS) <= MATCH_TOL]
        if gland_tol is None:
            gland_tol = cls.gland_tol
        CS_pref = cls.preferred_CS(D, CS_pref_k=CS_pref_k)
        CS_key = lambda CS: abs(log(CS / CS_pref))
        # Try the rings a CS at a time, closest to the preferred CS first.
        options = []
        by_CS = sorted(sizes, key=lambda s: CS_key(s.CS)) # stable: keeps order
        for _, group in groupby(by_CS, key=lambda s: CS_key(s.CS)):
            for size in group:
                start = cls._auto_start(D, size, squeeze)
                if start is None:
                    continue
                for wall, value in start.items():
                    band = gland_tol.get(wall)
                    if band is not None and not isinstance(value, u.Spread):
                        start[wall] = u.tol(value, band)
                try:
                    o = cls(name=name, oring=size, **start, **seal)
                    o = o.tweak(squeeze=squeeze, gland_fill=gland_fill)
                except ValueError: # this ring cannot make a valid gland here
                    continue
                if o._auto_ok(D):
                    options.append(o)
            if options and not all_options:
                break
        # Rank: CS closest to the preferred CS, then the seal type's own ranking
        # (fit, or smallest seal), then the reference, so near-identical rings
        # resolve the same way every time.
        options.sort(key=lambda o: (
                CS_key(o.oring.CS),
                o._auto_rank(),
                o.oring.ref,
            ))
        if all_options:
            return options
        if not options:
            raise ValueError(f"no standard O-ring fits this {cls.__name__} "
                    f"(D = {show(D, u.mm)}, squeeze {show(squeeze, u.percent)}"
                    f", gland fill {show(gland_fill, u.percent)}"
                    f"{f', CS {show(CS, u.mm)}' if CS else ''}); try another "
                    "CS or squeeze")
        return options[0]


    # ---- Tolerances -----------------------------------------------------------
    #
    # Ranges are 3-sigma FORM bounds from `System.corners` (`corner`'s default),
    # at the worst corner of any Env inputs: about 1 build in 740 lies beyond
    # each. The build at a bound is `s.corners(name, worst).build()`.

    def bounds(self, name):
        """ (low, high) of output `name` across tolerances. """
        return tuple(corner(name, worst)(self) for worst in ("min", "max"))


    # ---- Requirements ---------------------------------------------------------

    def requirements(self):
        _, fit_lo, fit_hi, _ = self._fit()
        reqs = super().requirements() | {
            corner("squeeze", "min"): (self.squeeze_min, None),
            corner("squeeze", "max"): (None, self.squeeze_max),
            corner("gland_fill", "max"): (None, self.gland_fill_max),
            corner(self.fit, "min"): (fit_lo, None),
            corner(self.fit, "max"): (None, fit_hi),
            "hardness": HARDNESS_RANGE[self.material], # stocked
        }
        if self.gap is not None:
            reqs |= {corner("extrusion_margin", "min"): (0.0, None)}
        return reqs

    def checks(self):
        """ Warnings: each requirement not met, and a non-catalogue ring. """
        warnings = []
        for m, (lo, hi, v) in self.verify().failed.items():
            side = (f"is below {m.show(lo, self)}" if lo is not None
                    and not v >= lo else f"is above {m.show(hi, self)}")
            warnings.append(f"{m.name} {m.show(v, self)} {side}")
        if self.oring.ref is None:
            warnings.append("O-ring isn't a catalogue size")
        return warnings


    # ---- Outline --------------------------------------------------------------
    #
    # Each outline_* gives a section of (label, text, brief) rows, which the
    # seal types extend. Brief rows make the succinct outline.

    def seal_name(self):
        return "bare o-ring"
    def _squeeze_aim(self):
        return self.squeeze_aim

    def _ranged(self, name, aim=None):
        # Output `name`, with its range across tolerances (and an aim range).
        unit = self.unit(name)
        aim = "" if aim is None else f"; aim {_span(*aim, unit)}"
        return (f"{self.display(name)} (toleranced "
                f"{_span(*self.bounds(name), unit)}{aim})")

    def outline_dimensions(self):
        ring = self.oring
        return [
            ("oring ID x CS", f"{self.display('oring.ID')} x "
                f"{self.display('oring.CS')} [{ring.ref or 'custom'}]", True),
            ("gland ID/OD/Lz", " / ".join(self.display(n)
                for n in ("gland_ID", "gland_OD", "gland_Lz")), True),
            ("gland W x D", f"{self.display('gland_width')} x "
                f"{self.display('gland_depth')}", False),
            ("effective CS", self.display("effective_CS"), False),
        ]
    def outline_fit(self):
        return [
            ("squeeze", self._ranged("squeeze", self._squeeze_aim()), True),
            ("gland fill", self._ranged("gland_fill", self.gland_fill_aim),
                True),
            (self.fit.replace("_", " "),
                self._ranged(self.fit, self._fit()[1:3]), True),
        ]
    def outline_contact(self):
        peak = self.contact_peak_stress
        rows = [
            ("contact load", self.display("q_normal_squeeze"), False),
            ("contact width", self.display("contact_width"), False),
            ("peak stress", f"{show(peak, u.MPa)} + ~DP = "
                f"~{show(peak + self.DP, u.MPa)}", False),
        ]
        if self.gap is not None:
            rows.append(("extrusion gap", f"{self.display('gap')} (no "
                f"back-ups: up to {show(self.bounds('gap_max')[0], u.mm)} "
                "toleranced)", True))
        return rows
    def outline_forces(self):
        return []

    def outline(self, succinct=False, printed=True):
        lube = "lubed" if self.lubricated else "dry"
        lines = [f"{self.name} ({self.seal_name()}): {self.material} "
                 f"{self.display('hardness')}, {lube}, "
                 f"{self.display('DP')} DP"]
        free = self.unassessed()
        sections = [self.outline_dimensions(), self.outline_fit(),
                    self.outline_contact(), self.outline_forces(),
                    [("untoleranced", ", ".join(free), False)] if free else []]
        sections = [[(label, text) for label, text, brief in rows
                     if brief or not succinct] for rows in sections]
        sections = [rows for rows in sections if rows]
        width = max(len(label) for rows in sections for label, _ in rows)
        for rows in sections:
            if not succinct:
                lines.append("  --")
            lines += [f"  {label:<{width}}  {text}" for label, text in rows]
        warnings = self.checks()
        if warnings and not succinct:
            lines.append("  --")
        lines += [f"  !! {warning}" for warning in warnings]
        s = "\n".join(lines) + "\n"
        if printed:
            print(s)
        else:
            return s

    def __repr__(self):
        return f"<o-ring {self.name!r}>"



class RadialSeal(Oring):
    """
    O-ring squeezed radially between two cylinders that slide axially relative to
    each other. Squeeze is across gland_Lr, width is across gland_Lz.

    Abstract: use PistonSeal or RodSeal.
    """

    # [-] Dynamic squeeze limits for a sliding seal, checked at nominal (and
    # its aim).
    squeeze_min_dynamic = 0.12
    squeeze_max_dynamic = 0.20

    sliding: bool = True # Moves relative to its sliding surface (else static).

    @property
    @abstractmethod
    def kind(self): ...

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.sliding, bool):
            raise TypeError("sliding must be True or False")

    gland_depth = calc(u.mm, # Sets the groove root (gland_ID for a piston,
                             # gland_OD for a rod), keeping the sliding surface.
        lambda s: s.gland_Lr,
        lambda s, d: dict(gland_ID=s.gland_OD - 2*d) if s.kind == "piston"
                else dict(gland_OD=s.gland_ID + 2*d))
    gland_width = calc(u.mm,
        lambda s: s.gland_Lz,
        lambda s, w: dict(gland_Lz=w))
    contact_D = calc(u.mm, # The sliding surface.
        lambda s: s.gland_OD if s.kind == "piston" else s.gland_ID)

    @classmethod
    def _auto_start(cls, D, size, squeeze):
        depth = size.CS * (1 - squeeze)
        if cls.kind == "piston":
            ID, OD = D - 2*depth, D
        else:
            ID, OD = D, D + 2*depth
        if ID <= 0:
            return None
        # Skip rings nowhere near fitting.
        rough = ID/size.ID - 1 if cls.kind == "piston" else 1 - OD/size.OD
        if not -0.03 <= rough <= 0.08:
            return None
        return dict(gland_ID=ID, gland_OD=OD, gland_Lz=1.5*size.CS)
    def _auto_rank(self):
        value, _, _, target = self._fit()
        return abs(value - target)

    @computed(u.mm)
    def effective_MD(self):
        # Squeezed between both gland walls, the cross-section is flattened
        # symmetrically, so its centroid sits at the gland mid-diameter.
        ring = self.oring
        CS = ring.CS * sqrt(ring.MD / self.gland_MD)
        if CS > self.gland_Lr:
            return self.gland_MD
        # Gland too deep to squeeze the O-ring: it stays seated.

        # Centreline diameter when touching at most one gland wall.
        CS = ring.CS
        for _ in range(40):
            if ring.ID < self.gland_ID:   # stretched onto the inner wall
                MD = self.gland_ID + CS
            elif ring.OD > self.gland_OD: # compressed into the outer wall
                MD = self.gland_OD - CS
            else:                         # loose
                MD = ring.MD
            if MD <= 0:
                raise ValueError("O-ring cannot fit: no centreline diameter "
                        "conserves its volume in this gland")
            CS = ring.CS * sqrt(ring.MD / MD)
        return MD


    # ---- Friction (along the sliding direction) -------------------------------

    F_friction_running = calc(u.N,
        lambda s: s.mu_running * s.F_normal_squeeze
                + s.mu_pressure * s.F_normal_pressure)
    F_friction_breakaway = calc(u.N,
        lambda s: s.mu_breakaway * s.F_normal)


    def requirements(self):
        reqs = super().requirements()
        if self.sliding:
            reqs |= {"squeeze": (self.squeeze_min_dynamic,
                                 self.squeeze_max_dynamic)}
        return reqs

    def seal_name(self):
        return ("dynamic " if self.sliding else "static ") + f"{self.kind} seal"
    def _squeeze_aim(self):
        if self.sliding:
            return (self.squeeze_min_dynamic, self.squeeze_max_dynamic)
        return super()._squeeze_aim()
    def outline_forces(self):
        return super().outline_forces() + [
            ("F_normal", f"{self.display('F_normal')} (squeeze "
                f"{self.display('F_normal_squeeze')} + pressure "
                f"{self.display('F_normal_pressure')})", False),
            ("F_fric breakaway", self._ranged("F_friction_breakaway"), True),
            ("F_fric running", self._ranged("F_friction_running"), True),
        ]


class PistonSeal(RadialSeal):
    """
    Groove cut into the plunger, O-ring slides on the housing bore.

    gland_ID = groove root diameter (on the plunger)
    gland_OD = bore diameter (sliding surface)
    gland_Lz = groove width
    Fit check: ID stretch onto the groove root.
    """
    kind = "piston"
    fit = "ID_stretch"

    gland_root = alias(u.mm, "gland_ID")
    gland_sliding = alias(u.mm, "gland_OD")

    @classmethod
    def auto(cls, name, bore_D, squeeze, gland_fill, **kwargs):
        return cls._auto(name, bore_D, squeeze, gland_fill, **kwargs)

class RodSeal(RadialSeal):
    """
    Groove cut into the housing bore, plunger slides through the O-ring.

    gland_ID = plunger diameter (sliding surface)
    gland_OD = groove root diameter (in the housing)
    gland_Lz = groove width
    Fit check: OD compression into the groove root.
    """
    kind = "rod"
    fit = "OD_compression"

    # Slightly tweaked OD compression limits (especially for small orings).
    OD_compression_min = 0.005
    OD_compression_max = 0.05

    gland_root = alias(u.mm, "gland_OD")
    gland_sliding = alias(u.mm, "gland_ID")

    @classmethod
    def auto(cls, name, rod_D, squeeze, gland_fill, **kwargs):
        return cls._auto(name, rod_D, squeeze, gland_fill, **kwargs)



class Groove(System):
    """ A radial seal's own parts, sized by hand; the assembly gives the rest
    (sliding diameter and conditions). """
    material: MATERIAL_TYPE
    hardness: u.shoreA
    lubricated: bool
    oring: OringSize
    root: u.mm # Groove root: gland_ID for a piston seal, gland_OD for a rod.
    Lz: u.mm # Groove width.
    friction_level: u.none = OringFriction.SPREAD

    @classmethod
    def of(cls, seal):
        """ The groove of a PistonSeal or RodSeal. """
        return cls(oring=seal.oring, root=seal.gland_root, Lz=seal.gland_Lz,
                material=seal.material, hardness=seal.hardness,
                lubricated=seal.lubricated, friction_level=seal.friction_level)

    def seal(self, kind, name, D, **conditions):
        """ A `kind` seal (PistonSeal or RodSeal) of this, sliding on `D`. """
        ID, OD = (self.root, D) if kind.kind == "piston" else (D, self.root)
        return kind(name=name, oring=self.oring, gland_ID=ID, gland_OD=OD,
                gland_Lz=self.Lz,
                material=self.material, hardness=self.hardness,
                lubricated=self.lubricated, friction_level=self.friction_level,
                **conditions)

class AutoGroove(System):
    """ A radial seal's own parts, left to `RadialSeal.auto` whenever the seal
    is built; the assembly gives the rest. """
    material: MATERIAL_TYPE
    hardness: u.shoreA
    lubricated: bool
    squeeze: u.percent = 0.15
    gland_fill: u.percent = 0.7
    CS: u.mm = None # None: picked by the sliding diameter.
    sizes: tuple = None
    friction_level: u.none = OringFriction.SPREAD

    def seal(self, kind, name, D, **conditions):
        """ An auto-sized `kind` seal (PistonSeal or RodSeal) sliding on `D`. """
        if self.sizes is not None:
            if "sizes" not in conditions:
                conditions["sizes"] = self.sizes
        return kind.auto(name, D, self.squeeze, self.gland_fill, CS=self.CS,
                material=self.material, hardness=self.hardness,
                lubricated=self.lubricated, friction_level=self.friction_level,
                **conditions)



class FaceSeal(Oring):
    """
    O-ring in a groove cut into a flat face, squeezed axially by a mating face.
    Static seal, so no friction.

    gland_ID, gland_OD = inner and outer walls of the groove
    gland_Lz           = groove depth (the squeeze direction)

    Abstract: use InternalFaceSeal or ExternalFaceSeal.
    """

    # [-] New squeeze limits.
    squeeze_min = 0.15

    # [-] New squeeze aim.
    squeeze_aim = (0.15, 0.25)

    @property
    @abstractmethod
    def kind(self): ...

    gland_depth = calc(u.mm,
        lambda s: s.gland_Lz,
        lambda s, d: dict(gland_Lz=d))
    gland_width = calc(u.mm, # Keeps the wall the O-ring is pressed onto
                             # (gland_OD internal, gland_ID external).
        lambda s: s.gland_Lr,
        lambda s, w: dict(gland_ID=s.gland_OD - 2*w) if s.kind == "internal"
                else dict(gland_OD=s.gland_ID + 2*w))
    contact_D = calc(u.mm,
        lambda s: s.effective_MD)

    @classmethod
    def _auto_start(cls, D, size, squeeze):
        # Put the pressure-side wall at the fit target; gland fill then moves
        # the other wall. Skip rings too small to go around the opening D.
        width = 1.4 * size.CS
        if cls.kind == "internal":
            OD = size.OD * (1 - cls.OD_compression_target)
            ID = OD - 2*width
        else:
            ID = size.ID * (1 + cls.ID_stretch_target)
            OD = ID + 2*width
        if ID < D:
            return None
        return dict(gland_ID=ID, gland_OD=OD, gland_Lz=size.CS*(1 - squeeze))
    def _auto_ok(self, D):
        return super()._auto_ok(D) and self.gland_ID >= D
    def _auto_rank(self):
        return self.gland_OD # smallest seal: least pressure force

    @computed(u.mm)
    def effective_MD(self):
        ring = self.oring
        CS = ring.CS
        for _ in range(40):
            if self.kind == "internal":
                MD = self.gland_OD - CS
            else:
                MD = self.gland_ID + CS
            if MD <= 0:
                raise ValueError("O-ring cannot fit: no centreline diameter "
                        "conserves its volume in this gland")
            CS = ring.CS * sqrt(ring.MD / MD)
        return MD

    F_axial_closure = calc(u.N,
        lambda s: s.F_normal_squeeze)
    @computed(u.N)
    def F_axial_pressure(self):
        # Assume a worst-case loading.
        D = self.gland_OD if self.kind == "internal" else self.gland_ID
        return self.DP * pi/4 * D**2
    @computed(u.N)
    def F_axial(self):
        direction = +1 if self.kind == "internal" else -1
        return self.F_axial_closure + direction*self.F_axial_pressure

    width_clearance = calc(u.mm, # Gland width beyond the O-ring CS.
        lambda s: s.gland_width - s.effective_CS)

    def requirements(self):
        return super().requirements() | {"width_clearance": (0.0, None)}

    def seal_name(self):
        return f"{self.kind} face seal"
    def outline_dimensions(self):
        return super().outline_dimensions() + [
            ("effective MD", self.display("effective_MD"), False),
        ]
    def outline_forces(self):
        clamp = self.display("F_axial_closure")
        return super().outline_forces() + [
            ("F_axial", f"{clamp} clamp + pressure = "
                f"{self.display('F_axial')} (upper bound)"
                if self.kind == "internal" else
                f"{clamp} clamp - pressure (geometry-dependent)", True),
        ]


class InternalFaceSeal(FaceSeal):
    """
    Pressure on the inside of the O-ring, pushing the O-ring onto the groove OD.
    Pressure acts on the area inside the sealing line, taken as gland_OD (upper
    bound), and pushes the faces apart.
    """
    kind = "internal"
    fit = "OD_compression"

    @classmethod
    def auto(cls, name, min_groove_ID, squeeze, gland_fill, **kwargs):
        return cls._auto(name, min_groove_ID, squeeze, gland_fill, **kwargs)

class ExternalFaceSeal(FaceSeal):
    """
    Pressure on the outside of the O-ring, pushing the O-ring onto the groove ID.
    Pressure acts on the area inside the sealing line, taken as gland_ID (lower
    bound), and pushes the faces together.
    """
    kind = "external"
    fit = "ID_stretch"

    @classmethod
    def auto(cls, name, min_groove_ID, squeeze, gland_fill, **kwargs):
        return cls._auto(name, min_groove_ID, squeeze, gland_fill, **kwargs)
