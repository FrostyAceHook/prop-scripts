"""
O-ring calculator, for friction as well as stretch/squeeze/gland analysis.
"""

from abc import ABC, abstractmethod
from math import pi, sqrt, floor, log10, erf, isfinite


# Formatting helpers.
def fmt_num(x, sig=4):
    if not isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    decimals = max(0, sig - 1 - floor(log10(abs(x))))
    s = f"{x:.{decimals}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s
fmt_mm = lambda x: f"{fmt_num(x*1e3)}mm"
fmt_bar = lambda x: f"{fmt_num(x*1e-5)} bar"
fmt_MPa = lambda x: f"{fmt_num(x*1e-6)} MPa"
fmt_N = lambda x: f"{fmt_num(x, sig=3)} N"
fmt_pt = lambda x: f"{fmt_num(x*100, sig=3)}%"


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


class Oring(ABC):
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

    def __init__(self, name, oring_ID, oring_CS, gland_ID, gland_OD, gland_Lz,
                DP, material="FKM", hardness=75, lubricated=False):
        if material not in HARDNESS_RANGE:
            raise ValueError(f"material must be one of {list(HARDNESS_RANGE)}")
        low, high = HARDNESS_RANGE[material]
        if not low <= hardness <= high:
            raise ValueError(f"{material} hardness must be {low}--{high} Shore "
                    "A")
        if not 0 < gland_ID < gland_OD:
            raise ValueError("need 0 < gland_ID < gland_OD")
        if oring_ID <= 0 or oring_CS <= 0:
            raise ValueError("O-ring ID and CS must be positive")
        if gland_Lz <= 0:
            raise ValueError("gland_Lz must be positive")
        if DP < 0:
            raise ValueError("DP must be zero or positive")
        for v in (oring_ID, oring_CS, gland_ID, gland_OD, gland_Lz, DP,
                hardness):
            if not isfinite(v):
                raise ValueError("inputs must be finite numbers")

        self.name = name
        self.oring_ID = oring_ID
        self.oring_CS = oring_CS
        self.gland_ID = gland_ID
        self.gland_OD = gland_OD
        self.gland_Lz = gland_Lz
        self.DP = DP
        self.material = material
        self.hardness = hardness
        self.lubricated = lubricated


    # ---- O-ring and gland geometry ---------------------------------------

    @property
    def oring_OD(self):
        return self.oring_ID + 2*self.oring_CS
    @property
    def oring_MD(self):
        return self.oring_ID + self.oring_CS
    @property
    def gland_Lr(self):
        return (self.gland_OD - self.gland_ID)/2
    @property
    def gland_MD(self):
        return (self.gland_ID + self.gland_OD)/2

    @property
    @abstractmethod
    def gland_depth(self): ...
    @property
    @abstractmethod
    def gland_width(self): ...

    def gland_area(self):
        return self.gland_depth * self.gland_width

    @property
    @abstractmethod
    def contact_D(self): ...


    # ---- Material -------------------------------------------------------------

    @property
    def E(self): # [Pa] Young's modulus.
        # BS 903 relation, S = 100 erf(3.186e-4 sqrt(E)).
        lo, hi = 0.0, 3.0
        for _ in range(60):
            x = (lo + hi) / 2
            lo, hi = (x, hi) if erf(x) < self.hardness / 100 else (lo, x)
        return (x / 3.186e-4)**2
    @property
    def nu(self): # [-] Poisson's constant.
        return 0.5 # essentially/assumed incompressible.

    def mu_running(self, level=NOM):
        return FRICTION[(self.material, self.lubricated)]["running"][level]
    def mu_pressure(self, level=NOM):
        return FRICTION[(self.material, self.lubricated)]["pressure"][level]
    def mu_breakaway(self, level=NOM):
        return FRICTION[(self.material, self.lubricated)]["breakaway"][level]


    # ---- O-ring fit -----------------------------------------------------------

    @abstractmethod
    def effective_MD(self): ...
    def effective_CS(self):
        return self.oring_CS * sqrt(self.oring_MD / self.effective_MD())

    def squeeze(self):
        CS = self.effective_CS()
        return max(0.0, 1 - self.gland_depth/CS)
    def gland_fill(self):
        return pi/4 * self.effective_CS()**2 / self.gland_area()

    def ID_stretch(self):
        return self.gland_ID/self.oring_ID - 1
    def OD_compression(self):
        return 1 - self.gland_OD/self.oring_OD


    # ---- Normal (contact) forces ----------------------------------------------

    def q_normal_squeeze(self):
        # Lindley: normal force from squeeze per unit length of sealing line.
        e = self.squeeze() # [N/m]
        return self.E * self.effective_CS() * (1.25 * e**1.5 + 50 * e**6)

    def contact_width(self):
        # Hertz line contact: the cross-section treated as a cylinder of radius
        # CS/2 pressed onto a flat wall with line load q, at zero pressure.
        # Elliptical pressure profile over a band of width b. Hertz assumes
        # b << CS; at typical squeezes b ~ CS/2, so this is extrapolated
        # (geometry suggests the real width is ~15% larger).
        E_star = self.E / (1 - self.nu**2)
        R = self.effective_CS() / 2
        return 2 * sqrt(4 * self.q_normal_squeeze() * R / (pi * E_star))

    def contact_peak_stress(self):
        # Peak of the Hertz profile at zero pressure: p0 = 4q / (pi b). Indicates
        # low-pressure / start-up sealing. Under pressure the rubber transmits DP
        # like a fluid, so contact is roughly p0 + DP.
        b = self.contact_width()
        return 4/pi * self.q_normal_squeeze() / b if (b > 0) else 0.0

    def F_normal_squeeze(self):
        # Total normal force on the sliding surface from squeeze.
        return self.q_normal_squeeze() * pi * self.contact_D
    def F_normal_pressure(self):
        # Extra normal force on the sliding surface: pressure is transmitted
        # through the rubber and acts over the contact band.
        return self.DP * self.contact_width() * pi * self.contact_D
    def F_normal(self):
        return self.F_normal_squeeze() + self.F_normal_pressure()


    # ---- Friction (along the sliding direction) -------------------------------

    def F_friction_running(self, level=NOM):
        return (self.mu_running(level) * self.F_normal_squeeze()
              + self.mu_pressure(level) * self.F_normal_pressure())
    def F_friction_breakaway(self, level=NOM):
        return self.mu_breakaway(level) * self.F_normal()


    # ---- Checks ---------------------------------------------------------------

    def checks_ID_stretch(self, loose_message):
        if self.ID_stretch() < 0:
            return [loose_message]
        if self.ID_stretch() > self.ID_stretch_max:
            return [f"ID stretch above {fmt_pt(self.ID_stretch_max)} "
                    "- thins the O-ring and speeds up ageing"]
        return []
    def checks_OD_compression(self, loose_message):
        if self.OD_compression() < 0:
            return [loose_message]
        if self.OD_compression() > self.OD_compression_max:
            return [f"OD compression above {fmt_pt(self.OD_compression_max)} "
                    "- can pucker or twist in the gland"]
        return []

    def checks(self):
        warnings = []
        s = self.squeeze()
        if s <= 0:
            warnings.append("NO CONTACT "
                    "- O-ring is not squeezed, it will not seal")
        elif s < self.squeeze_min:
            warnings.append(f"squeeze below {fmt_pt(self.squeeze_min)} "
                    "- may leak at low pressure or worst-case tolerances")
        elif s > self.squeeze_max:
            warnings.append(f"squeeze above {fmt_pt(self.squeeze_max)} "
                    "- risk of damage and overfill")
        if self.gland_fill() > self.gland_fill_max:
            warnings.append(f"gland fill above {fmt_pt(self.gland_fill_max)} "
                    "- widen the gland")
        return warnings


    # ---- Summary --------------------------------------------------------------

    def seal_name(self):
        return "bare o-ring"
    def summary_environment(self):
        lube = "lubed" if self.lubricated else "dry"
        return (
            f"{self.name} ({self.seal_name()})\n"
            f"  material        {self.material} {self.hardness}A, {lube}\n"
            f"  pressure diff   {fmt_bar(self.DP)}\n"
        )
    def summary_dimensions(self):
        return (
            "  --\n"
            "  oring (IDxCS)   "
                f"{fmt_mm(self.oring_ID)} x {fmt_mm(self.oring_CS)}\n"
            "  gland (ID/OD)   "
                f"{fmt_mm(self.gland_ID)} / {fmt_mm(self.gland_OD)}\n"
            "  gland (WxD)     "
                f"{fmt_mm(self.gland_width)} x {fmt_mm(self.gland_depth)}\n"
            f"  effective CS    {fmt_mm(self.effective_CS())}\n"
        )
    def summary_fit(self):
        return (
            "  --\n"
            f"  squeeze         {fmt_pt(self.squeeze())} (aim 10--20%)\n"
            f"  gland fill      {fmt_pt(self.gland_fill())} (aim 60--80%)\n"
        )
    def summary_contact(self):
        peak_stress = self.contact_peak_stress()
        return (
            "  --\n"
            f"  contact load    {fmt_num(self.q_normal_squeeze()*1e-3)} N/mm\n"
            f"  contact width   {fmt_mm(self.contact_width())}\n"
            f"  peak stress     {fmt_MPa(peak_stress)} + ~DP = "
                f"~{fmt_MPa(peak_stress + self.DP)}\n"
        )
    def summary_forces(self):
        return "  --\n"
    def summary(self):
        warnings = ""
        checks = self.checks()
        if checks:
            warnings = "  --\n"
            warnings += "".join(f"  !! {warning}\n" for warning in checks)
        return "".join([
            self.summary_environment(),
            self.summary_dimensions(),
            self.summary_fit(),
            self.summary_contact(),
            self.summary_forces(),
            warnings
        ])
    def report(self, *args, **kwargs):
        print(self.summary(*args, **kwargs))

    def __repr__(self):
        return f"<o-ring {self.name!r}>"

class ExceptionStaticSeal(Exception):
    pass



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

    @property
    def gland_depth(self):
        return self.gland_Lr
    @property
    def gland_width(self):
        return self.gland_Lz
    @property
    def contact_D(self):
        return self.gland_OD if self.kind == "piston" else self.gland_ID

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

    def checks(self):
        warnings = super().checks()
        s = self.squeeze()
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
    def summary_fit(self):
        return super().summary_fit() + (
            (f"  ID stretch      {fmt_pt(self.ID_stretch())} (aim 1--5%)\n"
                if self.kind == "piston" else
            f"  OD compression  {fmt_pt(self.OD_compression())} (aim 1--3%)\n")
        )
    def summary_forces(self):
        Fr = self.F_friction_running
        Fb = self.F_friction_breakaway
        return super().summary_forces() + (
            f"  F_normal        {fmt_N(self.F_normal())} "
                f"(squeeze {fmt_N(self.F_normal_squeeze())} + "
                f"pressure {fmt_N(self.F_normal_pressure())})\n"
            f"  F_fric breakaway  {fmt_N(Fb(NOM))} "
                    f"({fmt_N(Fb(LOW))[:-2]}--{fmt_N(Fb(HIGH))})\n"
            f"  F_fric running  {fmt_N(Fr(NOM))} "
                    f"({fmt_N(Fr(LOW))[:-2]}--{fmt_N(Fr(HIGH))})\n"
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

    @property
    def gland_depth(self):
        return self.gland_Lz
    @property
    def gland_width(self):
        return self.gland_Lr
    @property
    def contact_D(self):
        return self.effective_MD()

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

    def F_axial_closure(self):
        return self.F_normal_squeeze()
    def F_axial_pressure(self):
        # Assume a worst-case loading.
        D = self.gland_OD if self.kind == "internal" else self.gland_ID
        return self.DP * pi/4 * D**2
    def F_axial(self):
        direction = +1 if self.kind == "internal" else -1
        return self.F_axial_closure() + direction*self.F_axial_pressure()

    def F_friction_running(self, level=NOM):
        raise ExceptionStaticSeal()
    def F_friction_breakaway(self, level=NOM):
        raise ExceptionStaticSeal()

    def checks(self):
        warnings = super().checks()
        if self.gland_width < self.effective_CS():
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
    def summary_dimensions(self):
        return super().summary_dimensions() + (
            f"  effective MD    {fmt_mm(self.effective_MD())}\n"
        )
    def summary_fit(self):
        return super().summary_fit() + (
            (f"  OD compression  {fmt_pt(self.OD_compression())} (aim 1--3%)\n"
                if self.kind == "internal" else
            f"  ID stretch      {fmt_pt(self.ID_stretch())} (aim 1--5%)\n")
        )
    def summary_forces(self):
        return super().summary_forces() + (
            (f"  F_axial clamp   {fmt_N(self.F_axial_closure())} + pressure = "
                    f"{fmt_N(self.F_axial())} (upper-bound)\n"
                if self.kind == "internal" else
            f"  F_axial clamp   {fmt_N(self.F_axial_closure())} - pressure = "
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
