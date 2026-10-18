"""
O-ring calculator, for friction as well as stretch/squeeze/gland analysis.
"""

from abc import ABC, abstractmethod
from math import pi, sqrt, floor, log10, erf, isfinite

import units as u
from fmt import fmt_num, fmt_mm, fmt_bar, fmt_MPa, fmt_N, fmt_pt
from system import System, calc, computed


__all__ = (
    "PistonSeal",
    "RodSeal",
    "InternalFaceSeal",
    "ExternalFaceSeal",
)


# Allowed hardness range [Shore A] for each O-ring material.
HARDNESS_RANGE = {
    "FKM": (70, 90),
    "NBR": (70, 90),
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
    ("FKM", True):  {"running":   (0.07, 0.12, 0.30),
                     "pressure":  (0.03, 0.05, 0.12),
                     "breakaway": (0.20, 0.35, 0.50)},
    ("NBR", True):  {"running":   (0.07, 0.12, 0.30),
                     "pressure":  (0.03, 0.05, 0.12),
                     "breakaway": (0.20, 0.35, 0.50)},
}
LOW, NOM, HIGH = 0, 1, 2


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

    # [-] Assembly fit limits.
    ID_stretch_max = 0.05
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

    def __init_subclass__(cls, **options):
        # Keep the short `__repr__` below for every seal type, rather than the
        # dataclass one listing every field.
        super().__init_subclass__(**{"repr": False, **options})

    def __post_init__(self):
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
        if self.friction_level not in (LOW, NOM, HIGH):
            raise ValueError("friction_level must be LOW, NOM or HIGH")
        for v in (self.oring_ID, self.oring_CS, self.gland_ID, self.gland_OD,
                self.gland_Lz, self.DP, self.hardness):
            if not isfinite(v):
                raise ValueError("inputs must be finite numbers")


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


    # ---- Checks ---------------------------------------------------------------

    def checks_ID_stretch(self, loose_message):
        if self.ID_stretch < 0:
            return [loose_message]
        if self.ID_stretch > self.ID_stretch_max:
            return [f"ID stretch above {fmt_pt(self.ID_stretch_max)} "
                    "- thins the O-ring and speeds up ageing"]
        return []
    def checks_OD_compression(self, loose_message):
        if self.OD_compression < 0:
            return [loose_message]
        if self.OD_compression > self.OD_compression_max:
            return [f"OD compression above {fmt_pt(self.OD_compression_max)} "
                    "- can pucker or twist in the gland"]
        return []

    def checks(self):
        warnings = []
        s = self.squeeze
        if s <= 0:
            warnings.append("NO CONTACT "
                    "- O-ring is not squeezed, it will not seal")
        elif s < self.squeeze_min:
            warnings.append(f"squeeze below {fmt_pt(self.squeeze_min)} "
                    "- may leak at low pressure or worst-case tolerances")
        elif s > self.squeeze_max:
            warnings.append(f"squeeze above {fmt_pt(self.squeeze_max)} "
                    "- risk of damage and overfill")
        if self.gland_fill > self.gland_fill_max:
            warnings.append(f"gland fill above {fmt_pt(self.gland_fill_max)} "
                    "- widen the gland")
        return warnings


    # ---- Summary --------------------------------------------------------------

    def seal_name(self):
        return "bare o-ring"
    def outline_environment(self):
        lube = "lubed" if self.lubricated else "dry"
        return (
            f"{self.name} ({self.seal_name()})\n"
            f"  material        {self.material} {self.hardness}A, {lube}\n"
            f"  pressure diff   {fmt_bar(self.DP)}\n"
        )
    def outline_dimensions(self):
        return (
            "  --\n"
            "  oring (IDxCS)   "
                f"{fmt_mm(self.oring_ID)} x {fmt_mm(self.oring_CS)}\n"
            "  gland (ID/OD)   "
                f"{fmt_mm(self.gland_ID)} / {fmt_mm(self.gland_OD)}\n"
            "  gland (WxD)     "
                f"{fmt_mm(self.gland_width)} x {fmt_mm(self.gland_depth)}\n"
            f"  effective CS    {fmt_mm(self.effective_CS)}\n"
        )
    def outline_fit(self):
        return (
            "  --\n"
            f"  squeeze         {fmt_pt(self.squeeze)} (aim 10--20%)\n"
            f"  gland fill      {fmt_pt(self.gland_fill)} (aim 60--80%)\n"
        )
    def outline_contact(self):
        return (
            "  --\n"
            f"  contact load    {fmt_num(self.q_normal_squeeze*1e-3)} N/mm\n"
            f"  contact width   {fmt_mm(self.contact_width)}\n"
            f"  peak stress     {fmt_MPa(self.contact_peak_stress)} + ~DP = "
                f"~{fmt_MPa(self.contact_peak_stress + self.DP)}\n"
        )
    def outline_forces(self):
        return "  --\n"
    def outline(self, printed=True):
        warnings = ""
        checks = self.checks()
        if checks:
            warnings = "  --\n"
            warnings += "".join(f"  !! {warning}\n" for warning in checks)
        s = "".join([
            self.outline_environment(),
            self.outline_dimensions(),
            self.outline_fit(),
            self.outline_contact(),
            self.outline_forces(),
            warnings
        ])
        if printed:
            print(s)
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
            warnings.append(f"squeeze below {fmt_pt(self.squeeze_min_dynamic)} "
                    "- O-ring may roll in the gland; "
                    "rolling resistance can far exceed the sliding estimate")
        elif s > self.squeeze_max_dynamic:
            warnings.append(f"squeeze above {fmt_pt(self.squeeze_max_dynamic)} "
                    "- high for a sliding seal")
        if self.kind == "piston":
            warnings += self.checks_ID_stretch(
                    "O-ring ID larger than gland root "
                    "- loose in gland, can roll or twist")
        else:
            warnings += self.checks_OD_compression(
                    "O-ring OD smaller than gland root "
                    "- loose in gland, can fall out or twist")
        return warnings

    def seal_name(self):
        return f"{self.kind} seal"
    def outline_fit(self):
        return super().outline_fit() + (
            (f"  ID stretch      {fmt_pt(self.ID_stretch)} (aim 1--5%)\n"
                if self.kind == "piston" else
            f"  OD compression  {fmt_pt(self.OD_compression)} (aim 1--3%)\n")
        )
    def outline_forces(self):
        lo = self.tweak(friction_level=LOW)
        hi = self.tweak(friction_level=HIGH)
        Fb = lambda o: fmt_N(o.F_friction_breakaway)
        Fr = lambda o: fmt_N(o.F_friction_running)
        return super().outline_forces() + (
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

class RodSeal(RadialSeal):
    """
    Groove cut into the housing bore, plunger slides through the O-ring.

    gland_ID = plunger diameter (sliding surface)
    gland_OD = groove root diameter (in the housing)
    gland_Lz = groove width
    Fit check: OD compression into the groove root.
    """
    kind = "rod"



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
            warnings.append("gland narrower than the O-ring CS "
                    "- no room to flatten")
        if self.kind == "internal":
            warnings += self.checks_OD_compression(
                    "O-ring OD smaller than gland OD "
                    "- pressure will stretch it outward (aim OD 1--3% larger)")
        else:
            warnings += self.checks_ID_stretch("O-ring ID larger than gland ID "
                    "- pressure will compress it inward (aim 1--5% stretch)")
        return warnings

    def seal_name(self):
        return f"{self.kind} face seal"
    def outline_dimensions(self):
        return super().outline_dimensions() + (
            f"  effective MD    {fmt_mm(self.effective_MD)}\n"
        )
    def outline_fit(self):
        return super().outline_fit() + (
            (f"  OD compression  {fmt_pt(self.OD_compression)} (aim 1--3%)\n"
                if self.kind == "internal" else
            f"  ID stretch      {fmt_pt(self.ID_stretch)} (aim 1--5%)\n")
        )
    def outline_forces(self):
        return super().outline_forces() + (
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

    Fit check: O-ring OD should be 1-3% larger than gland_OD.
    """
    kind = "internal"

class ExternalFaceSeal(FaceSeal):
    """
    Pressure on the outside of the O-ring, pushing the O-ring onto the groove ID.
    Pressure acts on the area inside the sealing line, taken as gland_ID (lower
    bound), and pushes the faces together.

    Fit check: O-ring ID should be stretched 1-5% onto gland_ID.
    """
    kind = "external"
