"""
O-ring calculator, for friction as well as stretch/squeeze/gland analysis.
"""

from abc import ABC, abstractmethod
from itertools import groupby, product
from math import pi, sqrt, log, erf, isfinite

import units as u
from fmt import fmt_num, fmt_mm, fmt_bar, fmt_MPa, fmt_N, fmt_pt
from system import System, calc, computed
from oring_sizes import (OringSize, bs1806, finer, BS1806, FINER_BS,
        FINER_METRIC)


__all__ = (
    "PistonSeal",
    "RodSeal",
    "InternalFaceSeal",
    "ExternalFaceSeal",
    "OringFriction",
    "OringSize",
    "bs1806",
    "finer",
    "BS1806",
    "FINER_BS",
    "FINER_METRIC",
)


# Allowed hardness range [Shore A] for each O-ring material.
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

class OringFriction: # exported on import *.
    LOW, NOM, HIGH = 0, 1, 2
LOW, NOM, HIGH = OringFriction.LOW, OringFriction.NOM, OringFriction.HIGH


def _aim(lo, hi):
    # "aim" range text for the outlines, e.g. "10--20%".
    return f"{fmt_pt(lo)[:-1]}--{fmt_pt(hi)}"


def _catalogue_size(ref):
    # The catalogue OringSize for a reference, or None if it isn't one.
    for lookup in (bs1806, finer):
        try:
            return lookup(ref)
        except (KeyError, ValueError):
            pass
    return None

def _catalogue_match(ID, CS):
    # The catalogue OringSize with this nominal ID x CS (to 0.001 mm), if any.
    # No two catalogue sizes share an ID x CS, so the match is unambiguous.
    tol = 0.001e-3
    for size in BS1806 + FINER_METRIC:
        if abs(size.ID - ID) <= tol and abs(size.CS - CS) <= tol:
            return size
    return None


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

    name: str # Cosmetic name.
    oring_ID: u.mm
    oring_CS: u.mm
    gland_ID: u.mm
    gland_OD: u.mm
    gland_Lz: u.mm
    DP: u.bar # Pressure difference across the seal.
    material: str
    hardness: u.shoreA
    lubricated: bool
    friction_level: int = NOM # Which friction coefficients: LOW, NOM or HIGH.
    oring_ref: str = "" # Size reference, e.g. "BS214".

    # +- tolerances. Ring tolerances default to the catalogue size's (from
    # `oring_ref`), else the rule of thumb in `ring_tolerances`.
    oring_ID_tol: u.mm = None
    oring_CS_tol: u.mm = None
    gland_ID_tol: u.mm = 0.05e-3
    gland_OD_tol: u.mm = 0.05e-3
    gland_Lz_tol: u.mm = 0.1e-3

    def __init_subclass__(cls, **options):
        # Keep the short `__repr__` below for every seal type, rather than the
        # dataclass one listing every field.
        super().__init_subclass__(**{"repr": False, **options})

    def __post_init__(self):
        for v in (self.oring_ID, self.oring_CS, self.gland_ID, self.gland_OD,
                self.gland_Lz, self.DP, self.hardness):
            if not isfinite(v):
                raise ValueError("inputs must be finite numbers")
        if not isinstance(self.lubricated, bool):
            raise ValueError("lubricated must be True or False")
        if self.material not in HARDNESS_RANGE:
            raise ValueError(f"material must be one of {list(HARDNESS_RANGE)}")
        low, high = HARDNESS_RANGE[self.material]
        if not low <= self.hardness <= high:
            raise ValueError(f"{self.material} hardness must be {low}--{high} "
                    "Shore A")
        if not 0 < self.gland_ID < self.gland_OD:
            raise ValueError("need 0 < gland_ID < gland_OD")
        if self.oring_ID <= 0 or self.oring_CS <= 0:
            raise ValueError("O-ring ID and CS must be positive")
        if self.gland_Lz <= 0:
            raise ValueError("gland_Lz must be positive")
        if self.DP < 0:
            raise ValueError("DP must be zero or positive")
        if (isinstance(self.friction_level, bool)
                or self.friction_level not in (LOW, NOM, HIGH)):
            raise ValueError("friction_level must be LOW, NOM or HIGH")
        for tol in (self.oring_ID_tol, self.oring_CS_tol, self.gland_ID_tol,
                self.gland_OD_tol, self.gland_Lz_tol):
            if tol is not None and not (isfinite(tol) and tol >= 0):
                raise ValueError("tolerances must be zero or positive")


    # ---- O-ring and gland geometry ---------------------------------------

    oring_OD = calc(u.mm, # Sets oring_ID (keeps CS).
        lambda s: s.oring_ID + 2*s.oring_CS,
        lambda s, OD: dict(oring_ID=OD - 2*s.oring_CS))
    oring_MD = calc(u.mm, # Sets oring_ID (keeps CS).
        lambda s: s.oring_ID + s.oring_CS,
        lambda s, MD: dict(oring_ID=MD - s.oring_CS))
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

    @computed(None)
    def catalogue_size(self):
        """ The catalogue size: by `oring_ref`, else by matching ID x CS. """
        if self.oring_ref:
            size = _catalogue_size(self.oring_ref)
            if size is not None:
                return size
        return _catalogue_match(self.oring_ID, self.oring_CS)


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
    nu = calc(u.ratio, # Poisson's constant.
        lambda s: 0.5) # essentially/assumed incompressible.

    def mu(self, kind):
        """ Friction coefficient of the given `kind`, at `friction_level`. """
        return FRICTION[(self.material, self.lubricated)][kind]\
                [self.friction_level]
    mu_running = calc(u.ratio, lambda s: s.mu("running"))
    mu_pressure = calc(u.ratio, lambda s: s.mu("pressure"))
    mu_breakaway = calc(u.ratio, lambda s: s.mu("breakaway"))


    # ---- O-ring fit -----------------------------------------------------------

    @computed(u.mm)
    @abstractmethod
    def effective_MD(self): ...
    effective_CS = calc(u.mm,
        lambda s: s.oring_CS * sqrt(s.oring_MD / s.effective_MD))

    @computed(u.percent)
    def squeeze(self): # Sets gland_depth.
        return max(0.0, 1 - self.gland_depth/self.effective_CS)
    @squeeze.setter
    def squeeze(self, e):
        # Effective CS depends (weakly) on the gland, so iterate to converge.
        o = self
        for _ in range(20):
            o = o.tweak(gland_depth=o.effective_CS * (1 - e))
        return dict(gland_depth=o.gland_depth)
    gland_fill = calc(u.percent, # Sets gland_width.
        lambda s: pi/4 * s.effective_CS**2 / s.gland_area,
        lambda s, f: dict(gland_width=
            pi/4 * s.effective_CS**2 / (f * s.gland_depth)))

    ID_stretch = calc(u.percent, # Sets oring_ID.
        lambda s: s.gland_ID/s.oring_ID - 1,
        lambda s, x: dict(oring_ID=s.gland_ID / (1 + x)))
    OD_compression = calc(u.percent, # Sets oring_ID.
        lambda s: 1 - s.gland_OD/s.oring_OD,
        lambda s, x: dict(oring_OD=s.gland_OD / (1 - x)))


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


    # ---- Auto-sizing ----------------------------------------------------------
    #
    # For every standard O-ring: build a rough gland (_auto_start), let .tweak
    # set the squeeze and then the gland fill (the setters move only the walls
    # they should), optionally round the machined dimensions, then keep the
    # result if it is acceptable (_auto_ok) and rank it (_auto_rank).
    #   Radial seals: the bore or rod is given. The ring is chosen by its
    #     assembly fit, ranked by closeness to the fit target.
    #   Face seals: the groove walls are free, so the groove is placed around
    #     each ring at the fit target; the given diameter is the opening the
    #     seal must go around (minimum groove ID). Smallest seal wins.

    # Gland wall given by the user, never rounded. None if all are free.
    given_wall = None

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

    def squeeze_range(self, CS_tol, machining_tol):
        """
        Worst-case (lowest, highest) squeeze for an O-ring CS tolerance and a
        +/- tolerance on the gland depth.
        """
        CS = self.effective_CS
        dCS = CS_tol * CS / self.oring_CS
        lo = 1 - (self.gland_depth + machining_tol) / (CS - dCS)
        hi = 1 - (self.gland_depth - machining_tol) / (CS + dCS)
        return max(0.0, lo), hi

    @abstractmethod
    def assembly_fit(self): ...
        # (value, low limit, high limit, target) of the fit that picks the ring.

    @classmethod
    @abstractmethod
    def _auto_start(cls, D, size, squeeze): ...
        # Rough gland for O-ring `size`, as field kwargs, or None if this ring
        # cannot work. Only a starting point for the tweaks.
    def _auto_ok(self, D):
        value, lo, hi, _ = self.assembly_fit()
        return lo <= value <= hi and self.squeeze > 0
    @abstractmethod
    def _auto_rank(self): ... # Minimised.
    @classmethod
    def _auto(cls, name, D, squeeze, gland_fill, *, CS=None, CS_pref_k=None,
            sizes=FINER_BS+FINER_METRIC, all_options=False, **seal):
        if CS is not None:
            sizes = [s for s in sizes if abs(s.CS - CS) <= 0.01e-3]
        CS_pref = cls.preferred_CS(D, CS_pref_k=CS_pref_k)
        CS_key = lambda CS: round(abs(log(CS / CS_pref)), 6)
        # Try the rings a CS at a time, closest to the preferred CS first.
        options = []
        by_CS = sorted(sizes, key=lambda s: CS_key(s.CS)) # stable: keeps order
        for _, group in groupby(by_CS, key=lambda s: CS_key(s.CS)):
            for size in group:
                start = cls._auto_start(D, size, squeeze)
                if start is None:
                    continue
                try:
                    o = cls(name=name, oring_ID=size.ID, oring_CS=size.CS,
                            oring_ref=size.ref, **start, **seal)
                    o = o.tweak(squeeze=squeeze).tweak(gland_fill=gland_fill)
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
                CS_key(o.oring_CS),
                o._auto_rank(),
                o.oring_ref,
            ))
        if all_options:
            return options
        if not options:
            raise ValueError(f"no standard O-ring fits this {cls.__name__} "
                    f"(D = {fmt_mm(D)}, squeeze {fmt_pt(squeeze)}, gland fill "
                    f"{fmt_pt(gland_fill)}{f', CS {fmt_mm(CS)}' if CS else ''})"
                    "; try another CS or squeeze")
        return options[0]


    # ---- Tolerances -----------------------------------------------------------
    #
    # Every combination of +- on the ring and gland dimensions is built once.
    # `tight` and `loose` are the builds with the most and least squeeze (then
    # gland fill), with HIGH and LOW friction coefficients: a thicker ring in a
    # shallower, narrower gland raises squeeze, fill and friction together, so
    # these two bound all three. The exception is an external face seal's
    # gland_ID, which both narrows the groove (more fill) and stretches the
    # ring thinner (less squeeze), so the fill range is taken over every build.

    @computed(None)
    def ring_tolerances(self):
        """ (ID tol, CS tol): given, else catalogue, else rule of thumb. """
        size = self.catalogue_size
        # Rule of thumb fitted to AS568 and metric catalogues (within ~15%):
        # +-(0.13 mm + 0.6% of ID) and +-(0.05 mm + 1.5% of CS).
        ID_tol = (size.ID_tol if size else 0.13e-3 + 0.006*self.oring_ID)
        CS_tol = (size.CS_tol if size else 0.05e-3 + 0.015*self.oring_CS)
        return (
            ID_tol if self.oring_ID_tol is None else self.oring_ID_tol,
            CS_tol if self.oring_CS_tol is None else self.oring_CS_tol,
        )

    @computed(None)
    def _tolerance_builds(self):
        ID_tol, CS_tol = self.ring_tolerances
        dims = {"oring_ID": ID_tol, "oring_CS": CS_tol,
                "gland_ID": self.gland_ID_tol, "gland_OD": self.gland_OD_tol,
                "gland_Lz": self.gland_Lz_tol}
        builds = []
        for signs in product((-1, 1), repeat=len(dims)):
            try:
                builds.append(self.tweak(**{k: getattr(self, k) + d*tol
                        for (k, tol), d in zip(dims.items(), signs)}))
            except ValueError: # this combination isn't a valid gland
                continue
        if not builds:
            raise ValueError("no combination of tolerances gives a valid gland")
        return builds

    tight = computed(None)(lambda s: max(s._tolerance_builds, # most squeeze
        key=lambda o: (o.squeeze, o.gland_fill)).tweak(friction_level=HIGH))
    loose = computed(None)(lambda s: min(s._tolerance_builds, # least squeeze
        key=lambda o: (o.squeeze, o.gland_fill)).tweak(friction_level=LOW))

    squeeze_lo = calc(u.percent, lambda s: s.loose.squeeze)
    squeeze_hi = calc(u.percent, lambda s: s.tight.squeeze)
    gland_fill_lo = calc(u.percent,
        lambda s: min(o.gland_fill for o in s._tolerance_builds))
    gland_fill_hi = calc(u.percent,
        lambda s: max(o.gland_fill for o in s._tolerance_builds))

    ID_stretch_lo = calc(u.percent,
        lambda s: min(o.ID_stretch for o in s._tolerance_builds))
    ID_stretch_hi = calc(u.percent,
        lambda s: max(o.ID_stretch for o in s._tolerance_builds))
    OD_compression_lo = calc(u.percent,
        lambda s: min(o.OD_compression for o in s._tolerance_builds))
    OD_compression_hi = calc(u.percent,
        lambda s: max(o.OD_compression for o in s._tolerance_builds))

    F_friction_running_lo = calc(u.N, lambda s: s.loose.F_friction_running)
    F_friction_running_hi = calc(u.N, lambda s: s.tight.F_friction_running)
    F_friction_breakaway_lo = calc(u.N, lambda s: s.loose.F_friction_breakaway)
    F_friction_breakaway_hi = calc(u.N, lambda s: s.tight.F_friction_breakaway)

    # ---- Checks ---------------------------------------------------------------

    def checks_ID_stretch(self, loose_message):
        if self.ID_stretch < self.ID_stretch_min:
            return [loose_message]
        elif self.ID_stretch_lo < self.ID_stretch_min:
            return [f"ID stretch falls to {fmt_pt(self.ID_stretch_lo)} at "
                "loose tolerances"]
        if self.ID_stretch > self.ID_stretch_max:
            return [f"ID stretch above {fmt_pt(self.ID_stretch_max)}"]
        elif self.ID_stretch_hi > self.ID_stretch_max:
            return [f"ID stretch reaches {fmt_pt(self.ID_stretch_hi)} at "
                    "tight tolerances"]
        return []
    def checks_OD_compression(self, loose_message):
        if self.OD_compression < self.OD_compression_min:
            return [loose_message]
        elif self.OD_compression_lo < self.OD_compression_min:
            return [f"OD compression falls to {fmt_pt(self.OD_compression_lo)} "
                  "at loose tolerances"]
        if self.OD_compression > self.OD_compression_max:
            return [f"OD compression above {fmt_pt(self.OD_compression_max)}"]
        elif self.OD_compression_hi > self.OD_compression_max:
            return [f"OD compression reaches {fmt_pt(self.OD_compression_hi)} "
                    "at tight tolerances"]
        return []

    def checks(self):
        warnings = []
        if self.squeeze <= 0:
            warnings.append("NO SQUEEZE")
        elif self.squeeze < self.squeeze_min:
            warnings.append(f"squeeze below {fmt_pt(self.squeeze_min)}")
        elif self.squeeze_lo < self.squeeze_min:
            warnings.append(f"squeeze falls to {fmt_pt(self.squeeze_lo)} at "
                    "loose tolerances")
        elif self.squeeze > self.squeeze_max:
            warnings.append(f"squeeze above {fmt_pt(self.squeeze_max)}")
        elif self.squeeze_hi > self.squeeze_max:
            warnings.append(f"squeeze reaches {fmt_pt(self.squeeze_hi)} at "
                    "tight tolerances")
        if self.gland_fill > self.gland_fill_max:
            warnings.append(f"gland fill above {fmt_pt(self.gland_fill_max)}")
        elif self.gland_fill_hi > self.gland_fill_max:
            warnings.append(f"gland fill reaches {fmt_pt(self.gland_fill_hi)} "
                    "at tight tolerances")
        size = self.catalogue_size
        if size is not None and (abs(size.ID - self.oring_ID) > 0.001e-3
                or abs(size.CS - self.oring_CS) > 0.001e-3):
            warnings.append(f"O-ring doesn't match {size.ref} "
                    f"({fmt_mm(size.ID)} x {fmt_mm(size.CS)})")
        if size is None:
            warnings.append("O-ring isn't a catalogue size")
        return warnings


    # ---- Summary --------------------------------------------------------------

    def seal_name(self):
        return "bare o-ring"
    def outline_environment(self, succinct=False):
        lube = "lubed" if self.lubricated else "dry"
        if succinct:
            return (
                f"{self.name} ({self.seal_name()}), {self.material} "
                    f"{fmt_num(self.hardness, 3)}A, {lube}, "
                    f"{fmt_bar(self.DP)} DP\n"
            )
        return (
            f"{self.name} ({self.seal_name()})\n"
            f"  material        {self.material} {fmt_num(self.hardness, 3)}A, "
                f"{lube}\n"
            f"  pressure diff   {fmt_bar(self.DP)}\n"
        )
    def outline_dimensions(self, succinct=False):
        if succinct:
            return (
                f"  oring (IDxCS)     "
                    f"{fmt_mm(self.oring_ID)} x {fmt_mm(self.oring_CS)}"
                    f"{' [' + self.oring_ref + ']' if self.oring_ref else ''}\n"
                f"  gland (ID/OD/Lz)  "
                    f"{fmt_mm(self.gland_ID)} / {fmt_mm(self.gland_OD)} / "
                    f"{fmt_mm(self.gland_Lz)}\n"
            )
        return (
            "  --\n"
            "  oring (IDxCS)   "
                f"{fmt_mm(self.oring_ID)} x {fmt_mm(self.oring_CS)}"
                f"{' [' + self.oring_ref + ']' if self.oring_ref else ''}\n"
            "  gland (ID/OD)   "
                f"{fmt_mm(self.gland_ID)} / {fmt_mm(self.gland_OD)}\n"
            "  gland (WxD)     "
                f"{fmt_mm(self.gland_width)} x {fmt_mm(self.gland_depth)}\n"
            f"  effective CS    {fmt_mm(self.effective_CS)}\n"
        )
    def outline_fit(self, succinct=False):
        if succinct:
            return (
                f"  squeeze       {fmt_pt(self.squeeze)} "
                    f"({_aim(self.squeeze_lo, self.squeeze_hi)})\n"
                f"  gland fill    {fmt_pt(self.gland_fill)} "
                    f"({_aim(self.gland_fill_lo, self.gland_fill_hi)})\n"
            )
        return (

            "  --\n"
            f"  squeeze         {fmt_pt(self.squeeze)} "
                f"({_aim(self.squeeze_lo, self.squeeze_hi)} toleranced, "
                f"aim {_aim(*self.squeeze_aim)})\n"
            f"  gland fill      {fmt_pt(self.gland_fill)} "
                f"({_aim(self.gland_fill_lo, self.gland_fill_hi)} toleranced, "
                f"aim {_aim(*self.gland_fill_aim)})\n"
        )
    def outline_contact(self, succinct=False):
        if succinct:
            return ""
        return (
            "  --\n"
            f"  contact load    {fmt_num(self.q_normal_squeeze*1e-3)} N/mm\n"
            f"  contact width   {fmt_mm(self.contact_width)}\n"
            f"  peak stress     {fmt_MPa(self.contact_peak_stress)} + ~DP = "
                f"~{fmt_MPa(self.contact_peak_stress + self.DP)}\n"
        )
    def outline_forces(self, succinct=False):
        if succinct:
            return ""
        return "  --\n"
    def outline(self, succinct=False, printed=True):
        warnings = ""
        checks = self.checks()
        if checks:
            warnings = "  --\n"
            warnings += "".join(f"  !! {warning}\n" for warning in checks)
        s = "".join([
            self.outline_environment(succinct=succinct),
            self.outline_dimensions(succinct=succinct),
            self.outline_fit(succinct=succinct),
            self.outline_contact(succinct=succinct),
            self.outline_forces(succinct=succinct),
            warnings
        ])
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

    # [-] Dynamic squeeze limits.
    squeeze_min_dynamic = 0.12
    squeeze_max_dynamic = 0.20

    # [-] Dynamic squeeze aim.
    squeeze_aim = (squeeze_min_dynamic, squeeze_max_dynamic)

    @property
    @abstractmethod
    def kind(self): ...

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

    def assembly_fit(self):
        # The O-ring grips the groove root, i.e. the part it does not slide
        # on: stretched onto it (piston), or compressed into it (rod).
        return (
            (self.ID_stretch, self.ID_stretch_min, self.ID_stretch_max,
              self.ID_stretch_target)
            if self.kind == "piston" else
            (self.OD_compression, self.OD_compression_min,
                self.OD_compression_max, self.OD_compression_target)
        )

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
        fit = ID/size.ID - 1 if cls.kind == "piston" else 1 - OD/size.OD
        if not -0.03 <= fit <= 0.08:
            return None
        return dict(gland_ID=ID, gland_OD=OD, gland_Lz=1.5*size.CS)
    def _auto_rank(self):
        value, _, _, target = self.assembly_fit()
        return abs(value - target)

    @computed(u.mm)
    def effective_MD(self):
        # Squeezed between both gland walls, the cross-section is flattened
        # symmetrically, so its centroid sits at the gland mid-diameter.
        CS = self.oring_CS * sqrt(self.oring_MD / self.gland_MD)
        if CS > self.gland_Lr:
            return self.gland_MD
        # Gland too deep to squeeze the O-ring: it stays seated.

        # Centreline diameter when touching at most one gland wall.
        CS = self.oring_CS
        for _ in range(40):
            if self.oring_ID < self.gland_ID:   # stretched onto the inner wall
                MD = self.gland_ID + CS
            elif self.oring_OD > self.gland_OD: # compressed into the outer wall
                MD = self.gland_OD - CS
            else:                               # loose
                MD = self.oring_MD
            if MD <= 0:
                raise ValueError("O-ring cannot fit: no centreline diameter "
                        "conserves its volume in this gland")
            CS = self.oring_CS * sqrt(self.oring_MD / MD)
        return MD


    # ---- Friction (along the sliding direction) -------------------------------

    F_friction_running = calc(u.N,
        lambda s: s.mu_running * s.F_normal_squeeze
                + s.mu_pressure * s.F_normal_pressure)
    F_friction_breakaway = calc(u.N,
        lambda s: s.mu_breakaway * s.F_normal)


    def checks(self):
        warnings = super().checks()
        s = self.squeeze
        if s < self.squeeze_min_dynamic:
            warnings.append("squeeze below dynamic seal minimum "
                    f"{fmt_pt(self.squeeze_min_dynamic)}")
        elif s > self.squeeze_max_dynamic:
            warnings.append("squeeze above dynamic seal maximum "
                    f"{fmt_pt(self.squeeze_max_dynamic)}")
        if self.kind == "piston":
            warnings += self.checks_ID_stretch(
                    "O-ring ID larger than gland root")
        else:
            warnings += self.checks_OD_compression(
                    "O-ring OD smaller than gland root")
        return warnings

    def seal_name(self):
        return f"{self.kind} seal"
    def outline_fit(self, succinct=False):
        if succinct:
            return super().outline_fit(succinct=succinct) + (
                (f"  ID stretch    {fmt_pt(self.ID_stretch)} "
                    f"({_aim(self.ID_stretch_lo, self.ID_stretch_hi)})\n"
                    if self.kind == "piston" else
                f"  OD compression {fmt_pt(self.OD_compression)} "
                    f"({_aim(self.OD_compression_lo, self.OD_compression_hi)})"
                    "\n")
            )
        return super().outline_fit(succinct=succinct) + (
            (f"  ID stretch      {fmt_pt(self.ID_stretch)} "
                f"({_aim(self.ID_stretch_lo, self.ID_stretch_hi)} toleranced, "
                f"aim {_aim(self.ID_stretch_min, self.ID_stretch_max)})\n"
                if self.kind == "piston" else
            f"  OD compression  {fmt_pt(self.OD_compression)} "
                f"({_aim(self.OD_compression_lo, self.OD_compression_hi)} "
                "toleranced, aim "
                f"{_aim(self.OD_compression_min, self.OD_compression_max)})\n")
        )
    def outline_forces(self, succinct=False):
        Fb = lambda o: fmt_N(o.F_friction_breakaway)
        Fr = lambda o: fmt_N(o.F_friction_running)
        lo, hi = self.loose, self.tight
        if succinct:
            return super().outline_forces(succinct=succinct) + (
                f"  F_fric breakaway  {Fb(self)} ({Fb(lo)[:-2]}--{Fb(hi)})\n"
                f"  F_fric running    {Fr(self)} ({Fr(lo)[:-2]}--{Fr(hi)})\n"
            )
        return super().outline_forces(succinct=succinct) + (
            f"  F_normal        {fmt_N(self.F_normal)} "
                f"(squeeze {fmt_N(self.F_normal_squeeze)} + "
                f"pressure {fmt_N(self.F_normal_pressure)})\n"
            f"  F_fric breakaway  {Fb(self)} ({Fb(lo)[:-2]}--{Fb(hi)})\n"
            f"  F_fric running  {Fr(self)} ({Fr(lo)[:-2]}--{Fr(hi)})\n"
        )


class PistonSeal(RadialSeal):
    """
    Groove cut into the plunger, O-ring slides on the housing bore.

    gland_ID = groove root diameter (on the plunger)
    gland_OD = bore diameter (sliding surface)
    gland_Lz = groove width
    Fit check: ID stretch onto the groove root.
    """
    kind = "piston"
    given_wall = "gland_OD"

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
    given_wall = "gland_ID"

    # Slightly tweaked OD compression limits (especially for small orings).
    OD_compression_min = 0.005
    OD_compression_max = 0.05

    @classmethod
    def auto(cls, name, rod_D, squeeze, gland_fill, **kwargs):
        return cls._auto(name, rod_D, squeeze, gland_fill, **kwargs)



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

    def assembly_fit(self):
        if self.kind == "internal":
            return (self.OD_compression, self.OD_compression_min,
                    self.OD_compression_max, self.OD_compression_target)
        return (self.ID_stretch, self.ID_stretch_min, self.ID_stretch_max,
                self.ID_stretch_target)

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
        CS = self.oring_CS
        for _ in range(40):
            if self.kind == "internal":
                MD = self.gland_OD - CS
            else:
                MD = self.gland_ID + CS
            if MD <= 0:
                raise ValueError("O-ring cannot fit: no centreline diameter "
                        "conserves its volume in this gland")
            CS = self.oring_CS * sqrt(self.oring_MD / MD)
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

    def checks(self):
        warnings = super().checks()
        if self.gland_width < self.effective_CS:
            warnings.append("gland narrower than the O-ring CS")
        if self.kind == "internal":
            warnings += self.checks_OD_compression(
                    "O-ring OD smaller than gland OD")
        else:
            warnings += self.checks_ID_stretch("O-ring ID larger than gland ID")
        return warnings

    def seal_name(self):
        return f"{self.kind} face seal"
    def outline_dimensions(self, succinct=False):
        if succinct:
            return super().outline_dimensions(succinct=succinct)
        return super().outline_dimensions(succinct=succinct) + (
            f"  effective MD    {fmt_mm(self.effective_MD)}\n"
        )
    def outline_fit(self, succinct=False):
        if succinct:
            return super().outline_fit(succinct=succinct) + (
                (f"  ID stretch    {fmt_pt(self.ID_stretch)} "
                    f"({_aim(self.ID_stretch_lo, self.ID_stretch_hi)})\n"
                    if self.kind == "external" else
                f"  OD compression {fmt_pt(self.OD_compression)} "
                    f"({_aim(self.OD_compression_lo, self.OD_compression_hi)})"
                    "\n")
            )
        return super().outline_fit(succinct=succinct) + (
            (f"  OD compression  {fmt_pt(self.OD_compression)} "
                f"({_aim(self.OD_compression_lo, self.OD_compression_hi)} "
                "toleranced, aim "
                f"{_aim(self.OD_compression_min, self.OD_compression_max)})\n"
                if self.kind == "internal" else
            f"  ID stretch      {fmt_pt(self.ID_stretch)} "
                f"({_aim(self.ID_stretch_lo, self.ID_stretch_hi)} toleranced, "
                f"aim {_aim(self.ID_stretch_min, self.ID_stretch_max)})\n")
        )
    def outline_forces(self, succinct=False):
        if succinct:
            return super().outline_forces(succinct=succinct) + (
                f"  F_axial: {fmt_N(self.F_axial_closure)} clamp, "
                    f"{fmt_N(self.F_axial)} total\n"
                    if self.kind == "internal" else
                f"  F_axial: {fmt_N(self.F_axial_closure)} clamp\n")
        return super().outline_forces(succinct=succinct) + (
            (f"  F_axial clamp   {fmt_N(self.F_axial_closure)} + pressure = "
                    f"{fmt_N(self.F_axial)} (upper-bound)\n"
                if self.kind == "internal" else
            f"  F_axial clamp   {fmt_N(self.F_axial_closure)} - pressure = "
                    "geom-dependent\n")
        )


class InternalFaceSeal(FaceSeal):
    """
    Pressure on the inside of the O-ring, pushing the O-ring onto the groove OD.
    Pressure acts on the area inside the sealing line, taken as gland_OD (upper
    bound), and pushes the faces apart.
    """
    kind = "internal"

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

    @classmethod
    def auto(cls, name, min_groove_ID, squeeze, gland_fill, **kwargs):
        return cls._auto(name, min_groove_ID, squeeze, gland_fill, **kwargs)
