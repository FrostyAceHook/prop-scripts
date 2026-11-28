"""
Display units and spreads for System fields and getters. Conventionally imported
as `u`.

Values are always stored and computed in SI. A unit annotation only says how a
value is shown, and each alias is still a plain `float` to Python and type
checkers. Each unit has a converter of the same name with a trailing
underscore, taking a value in that unit to SI:

    import spec as u
    class Pipe(System):
        D: u.mm # stored in m, shown in mm
        L: u.m
        @property
        def area(self) -> u.mm2:
            return pi/4 * self.D**2

    Pipe(D=u.mm_(30), L=u.mm_(2000))
    u.mm_.from_si(0.03)                 # 30.0, back to display
    u.mm_(np.linspace(25, 35, 21))      # arrays too (without a spread)

A spread is a subclassed `float` equal to its nominal value, so every getter
works unchanged and arithmetic on one returns a plain float. Spreads mean:

    tol(x, ...)   statistical uncertainty.
    env(x, ...)   operating range.
    exact(x)      no uncertainty.
    x             (a plain float) not yet assessed.
    bare(x)       tagged to tell `System.tweak` to drop a spread.

A converter may also give the value a spread, written in its unit:

    u.mm_(30, tol=0)                    # exact
    u.mm_(30, tol=0.02)                 # +-0.02 mm, statistical (+-3 sigma)
    u.mm_(18.2, tol=(-0.05, 0.10))      # asymmetric offsets
    u.mm_(20, tol=(-0.020, -0.007))     # the nominal needn't be inside
    u.N_mm_(25, tol=0.1, rel=True)      # +-10%
    u.bar_(84, env=8.4)                 # +-8.4 bar, shifting
    u.bar_(84, env=(75.6, 92.4))        # fixed limits

or in SI with `u.tol`, `u.env`, `u.exact` and `u.bare`.
"""
from __future__ import annotations

import math
import numbers
import typing
from dataclasses import dataclass
from typing import Annotated, Literal




### UNITS #######################################################################

@dataclass(frozen=True)
class Unit:
    """
    Stores a display symbol and a scale+offset to allow displaying SI values.
    SI_value = displayed_value * scale + offset.

    Note that `offset`s make sense for absolute values. A difference value must
    use the matching offset-free unit.
    """
    symbol: str
    scale: float = 1.0
    offset: float = 0.0
    spaced: bool = True
    @property
    def display(self):
        return " "*self.spaced + self.symbol

    def to_si(self, value: float, *,
              tol: float | tuple[float, float] | None = None,
              env: float | tuple[float, float] | None = None,
              iso286: str | None = None, centred: bool = False,
              **options) -> float:
        """
        Display value (in this unit) -> SI value. `tol` and `env` allow spread
        creation also: see `Tol.create` and `Env.create`. A length can instead
        take an ISO 286 class (`iso286="H7"`) as its tol, where `centred` puts
        the value mid-band, with a symmetric tol.
        """
        if sum(x is not None for x in (tol, env, iso286)) > 1:
            raise ValueError("give at most one of tol, env or iso286")
        if centred and iso286 is None:
            raise TypeError("centred only applies with iso286")
        if iso286 is not None:
            from iso286 import deviations # (it imports this module)
            if self.offset:
                raise TypeError(f"iso286 needs a length unit, not {self.symbol}")
            lo, hi = (round(x/self.scale, 12) for x in
                      deviations(iso286, value*self.scale))
            if centred:
                value = round(value + (lo + hi)/2, 12)
                tol = round((hi - lo)/2, 12)
            else:
                tol = (lo, hi)
        if tol is not None:
            return Tol.create(value, tol, unit=self, **options)
        if env is not None:
            return Env.create(value, env, unit=self, **options)
        if options:
            raise TypeError(f"{', '.join(options)} only apply with tol or env")
        return value * self.scale + self.offset

    def from_si(self, si):
        """ SI value -> display value. """
        return (si - self.offset) / self.scale

def unit_obj_of(alias):
    """ Returns the Unit inside an annotation. """
    if isinstance(alias, Unit):
        return alias
    if typing.get_origin(alias) is Annotated:
        for meta in typing.get_args(alias)[1:]:
            if isinstance(meta, Unit):
                return meta
    return None


class Converter:
    """
    A unit's converter (`u.mm_` for `u.mm`): display value -> SI when called,
    with an optional spread (see `Unit.to_si`), and SI -> display with
    `from_si`.
    """
    __slots__ = ("unit",)

    def __init__(self, alias=None):
        if alias is None:
            self.unit = None
        else:
            self.unit = unit_obj_of(alias)
            if self.unit is None:
                raise TypeError(f"{alias!r} has no Unit")

    def __call__(self, *args, **kwargs):
        unit = self.unit or Unit("")
        return unit.to_si(*args, **kwargs)

    def from_si(self, si):
        return self.unit.from_si(si)

    def __repr__(self):
        return f"Converter({self.unit.symbol!r})"



### SPREADS #####################################################################

def show(x, unit=None, fmt=None, *, delta=False, limits=False, nosym=False):
    """
    SI `x` as text in `unit` (a Unit or an alias like `u.mm`), with its spread if
    it has one. `fmt` formats each number (default 5 sig. figs). With `delta`, a
    `x` is a difference, so unit will only be used for scaling. With `limits`, a
    spread shows only its limits. Non-numbers give their repr.
    """
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return repr(x)
    fmt = fmt or (lambda v: f"{v:.5g}")
    unit = unit_obj_of(unit)
    if isinstance(x, Spread):
        conv = unit.from_si if unit else (lambda v: v)
        sym = unit.display if (unit and not nosym) else ""
        if limits:
            return x._limits_text(conv, fmt) + sym
        return x._show(conv, unit.scale if unit else 1.0, sym, fmt)
    if unit is None:
        return fmt(x)
    value = x / unit.scale if delta else unit.from_si(x)
    return f"{fmt(value)}{'' if nosym else unit.display}"

class Spread(float):
    """
    A float (`nominal`) carrying a band (`lo`&`hi`, lo <= hi). The band is
    anchored to the nominal one of three ways, which sets how it follows a new
    nominal in `recentre` (as `System.tweak` does for a plain number):

        "abs"    lo, hi are offsets from the nominal: the band shifts
        "rel"    lo, hi are proportions of |nominal|: the band scales
        "fixed"  lo, hi are fixed, absolute limits
    """
    __slots__ = ("_lo", "_hi", "_anchor")

    def __new__(cls, nominal, lo, hi, *, anchor="abs"):
        if anchor not in ("abs", "rel", "fixed"):
            raise ValueError("anchor must be 'abs', 'rel' or 'fixed'")
        lo, hi, nominal = float(lo), float(hi), float(nominal)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"band must be finite, got ({lo:g}, {hi:g})")
        if not lo <= hi:
            raise ValueError(f"band needs lo <= hi, got ({lo:g}, {hi:g})")
        if anchor == "rel" and nominal == 0.0 and (lo, hi) != (0.0, 0.0):
            raise ValueError("a relative band around a zero nominal is empty")
        self = super().__new__(cls, nominal)
        for name, value in (("_lo", lo), ("_hi", hi), ("_anchor", anchor)):
            object.__setattr__(self, name, value)
        return self


    @property
    def nominal(self):
        return float(self)

    @property
    def limits(self):
        """ (lo, hi) absolute. """
        if self._anchor == "fixed":
            return self._lo, self._hi
        nom = float(self)
        s = abs(nom) if self._anchor == "rel" else 1.0
        return nom + self._lo*s, nom + self._hi*s

    @property
    def offsets(self):
        """ (lo, hi) as offsets from the nominal. """
        lo, hi = self.limits
        return lo - float(self), hi - float(self)

    @property
    def width(self):
        lo, hi = self.limits
        return hi - lo


    def recentre(self, nominal):
        """ The same band around a new nominal, per its anchor. """
        return type(self)(nominal, **self._state())

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def _state(self):
        return dict(lo=self._lo, hi=self._hi, anchor=self._anchor)
    @classmethod
    def _rebuild(cls, nominal, state):
        return cls(nominal, **state)
    def __reduce__(self):
        return (type(self)._rebuild, (float(self), self._state()))
    def __copy__(self):
        return self
    def __deepcopy__(self, memo):
        return self


    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._state().items())
        return f"{type(self).__name__}({float(self)!r}, {args})"

    def show(self, *args, **kwargs):
        """ Alias for `show(self, *args, **kwargs)`. """
        return show(self, *args, **kwargs)

    def _show(self, conv, scale, sym, fmt):
        raise NotImplementedError
    def _offsets_shown(self, scale):
        if self._anchor == "rel":
            return 100*self._lo, 100*self._hi, "%"
        return self._lo/scale, self._hi/scale, ""
    def _resolution(self, conv, fmt):
        """ (nominal, lo, hi) converted, and the decimal places that resolve
        the band: fixed limits are the band, else the offsets are. """
        x, lo, hi = (conv(v) for v in (float(self), *self.limits))
        return x, lo, hi, (_places(fmt, lo, hi) if self._anchor == "fixed" else
                           _places(fmt, lo - x, hi - x))
    def _limits_text(self, conv, fmt):
        _, lo, hi, places = self._resolution(conv, fmt)
        return f"[{_resolved(lo, places, fmt)}…{_resolved(hi, places, fmt)}]"


class Tol(Spread):
    """
    A statistical tolerance: random part-to-part variation. Supports normal or
    uniform distributions. The nominal is only the band's reference, so it may
    sit outside, all statistics only care about the band. Allows an anchor of
    'rel' or 'abs', not 'fixed'.
    """
    __slots__ = ("_k", "_dist")

    def __new__(cls, nominal, lo, hi, *, anchor="abs", k=3.0, dist="normal"):
        if anchor == "fixed":
            raise ValueError("a Tol follows its nominal: anchor 'abs' or 'rel'")
        if dist not in ("normal", "uniform"):
            raise ValueError("dist must be 'normal' or 'uniform'")
        if not k > 0:
            raise ValueError("k must be positive")
        self = super().__new__(cls, nominal, lo, hi, anchor=anchor)
        object.__setattr__(self, "_k", float(k))
        object.__setattr__(self, "_dist", dist)
        return self

    @classmethod
    def create(cls, value: float, band: float | tuple[float, float], *,
               unit: Unit | None = None, rel: bool = False, k: float = 3.0,
               dist: Literal["normal", "uniform"] = "normal") -> "Tol":
        """
        A statistical tolerance: random part-to-part variation. `value` and
        `band` are in `unit`, or SI if None (as `u.tol`). Examples for mm:

            u.mm_(30, tol=0.02)                 # +-0.02 mm
            u.mm_(18.2, tol=(-0.05, 0.10))      # -0.05/+0.1 mm
            u.mm_(20, tol=(-0.020, -0.007))     # nominal may be outside
            u.mm_(30, tol=0)                    # exact
            u.mm_(30, tol=0.1, rel=True)        # +-10%
            u.mm_(30, tol=(-0.05, 0.1), rel=True) # -5%/+10%
            u.mm_(30, tol=0.02, k=2)            # limits are +-2 sigma
            u.mm_(30, tol=0.02, dist="uniform") # non-Normal distribution

        `band` is +-x or (lo, hi) offsets from the nominal. With `rel`, they're
        proportions of |nominal|. The limits are +-k standard deviations of a
        "normal" distribution, or the edges of a "uniform" distribution.
        """
        nominal = value if unit is None else unit.to_si(value)
        lo, hi = _band(band)
        if not rel and unit is not None:
            lo, hi = lo*unit.scale, hi*unit.scale
        return cls(nominal, lo, hi, anchor="rel" if rel else "abs", k=k,
                   dist=dist)


    @property
    def k(self):
        """ Standard deviations at each limit (normal only). """
        return self._k

    @property
    def dist(self):
        """ "normal" or "uniform" distribution shape. """
        return self._dist

    @property
    def is_exact(self):
        """ Is there no uncertainty? """
        return self._lo == self._hi

    @property
    def mean(self):
        """ Centre of the band. """
        lo, hi = self.limits
        return (lo + hi)/2

    @property
    def sd(self):
        """ Standard deviation. """
        return self.width / (
            math.sqrt(12) if self._dist == "uniform" else 2*self._k
        )

    def at(self, z):
        """ The value `z` standard deviations from the mean. """
        if self._dist == "normal":
            return self.mean + z*self.sd
        lo, hi = self.limits
        return lo + (hi - lo) * 0.5*(1 + math.erf(z/math.sqrt(2)))


    def _state(self):
        return super()._state() | dict(k=self._k, dist=self._dist)

    def _show(self, conv, scale, sym, fmt):
        lo, hi, pct = self._offsets_shown(scale)
        places = None if pct else _places(fmt, lo, hi)
        nom = _resolved(conv(float(self)), places, fmt)
        if self.is_exact:
            band = "" if lo == 0 else f" {_signed(lo, fmt)}{pct}"
        else:
            band = _band_text(lo, hi, pct, fmt)
        if self.is_exact:
            note = " (exact)"
        elif self._dist != "normal":
            note = f" ({self._dist})"
        elif self._k != 3.0:
            note = f" (±{self._k:g}σ)"
        else:
            note = ""
        return f"{nom}{sym}{band}{note}"


class Env(Spread):
    """
    An operating envelope: conditions the design must work across, so it's
    analysed worst-case at its limits, never statistically. Allows any anchor.
    The nominal is the typical operating point, and must lie within the limits.
    Attempting to recentre outside limits raises ValueError.
    """
    __slots__ = ()

    def __new__(cls, nominal, lo, hi, *, anchor="fixed"):
        self = super().__new__(cls, nominal, lo, hi, anchor=anchor)
        lo, hi = self.limits
        if not lo <= float(self) <= hi:
            raise ValueError(f"nominal {float(self):g} is outside its "
                             f"envelope [{lo:g}, {hi:g}]")
        return self

    @classmethod
    def create(cls, value: float, band: float | tuple[float, float], *,
               unit: Unit | None = None, abs: bool = False, rel: bool = False,
               fixed: bool = False,
               anchor: Literal["abs", "rel", "fixed"] | None = None
               ) -> "Env":
        """
        An operating envelope: the design must work across all of it. `value`
        and `band` are in `unit`, or SI if None (as `u.env`). Examples for bar:

            u.bar_(6, env=0.5)              # +-0.5 offset, shifts
            u.bar_(6, env=0.1, rel=True)    # +-10% offset, scales
            u.bar_(84, env=(75.6, 92.4))    # [75.6, 92.4] limits, fixed

        If `anchor` is given (or the equivalent bool), re-centering behaviour is
        locked to that anchor's meaning.
        """
        if (abs + rel + fixed) > 1:
            raise ValueError("give at most one of abs, rel, and fixed")
        elif abs:
            if anchor not in (None, "abs"):
                raise ValueError(f"abs=True contradicts anchor={anchor!r}")
            anchor = "abs"
        elif rel:
            if anchor not in (None, "rel"):
                raise ValueError(f"rel=True contradicts anchor={anchor!r}")
            anchor = "rel"
        elif fixed:
            if anchor not in (None, "fixed"):
                raise ValueError(f"fixed=True contradicts anchor={anchor!r}")
            anchor = "fixed"
        elif anchor is None:
            anchor = "fixed" if isinstance(band, tuple) else "abs"
        nominal = value if unit is None else unit.to_si(value)
        lo, hi = _band(band)
        if anchor == "fixed":
            if isinstance(band, tuple):
                if unit is not None:
                    lo, hi = unit.to_si(lo), unit.to_si(hi)
            else:
                if unit is not None:
                    lo, hi = lo*unit.scale, hi*unit.scale
                lo, hi = nominal + lo, nominal + hi
        elif anchor == "abs" and unit is not None:
            lo, hi = lo*unit.scale, hi*unit.scale
        return cls(nominal, lo, hi, anchor=anchor)

    def _show(self, conv, scale, sym, fmt):
        x, _, _, places = self._resolution(conv, fmt)
        nom = _resolved(x, places, fmt)
        limits = " " + self._limits_text(conv, fmt)
        if self._anchor == "fixed":
            return f"{nom}{sym}{limits}"
        band = _band_text(*self._offsets_shown(scale), fmt)
        return f"{nom}{sym}{band}{limits}"


class Bare(float):
    """ Simple marker to tell `System.tweak` to drop any spread. """
    __slots__ = ()
    def __repr__(self):
        return f"bare({float(self)!r})"


tol = Tol.create
env = Env.create

def exact(value):
    """ No uncertainty. """
    return Tol(value, 0.0, 0.0)

def bare(value):
    """ Tells `System.tweak` to drop any spread. """
    return Bare(value)




def _band(band):
    if isinstance(band, tuple):
        lo, hi = band
        return float(lo), float(hi)
    return -float(band), float(band)

def _signed(x, fmt):
    if x == 0:
        return "0"
    return fmt(x) if x < 0 else "+" + fmt(x)

def _band_text(lo, hi, suffix, fmt):
    if -lo == hi:
        return f" ±{fmt(hi)}{suffix}"
    return f" {_signed(lo, fmt)}/{_signed(hi, fmt)}{suffix}"

def _shown_places(text):
    mant, _, exp = text.partition("e")
    return len(mant.partition(".")[2]) - int(exp or 0)

def _places(fmt, *offsets):
    shown = [_shown_places(fmt(abs(x))) for x in offsets
             if x != 0 and math.isfinite(x)]
    return max(shown) if shown else None

def _resolved(x, places, fmt):
    text = fmt(x)
    if places is None or not math.isfinite(x) or _shown_places(text) >= places:
        return text
    if x == 0:
        return "0"
    sig = max(places + math.floor(math.log10(abs(x))) + 1, 1)
    return f"{x:.{sig}g}"




### FORMATTING ##################################################################

def table(rows, header, indent="  "):
    """
    Plain-text table with a header rule. First column left-aligned, the rest
    right-aligned. Cells are converted with str().
    """
    rows = [[str(c) for c in r] for r in rows]
    header = [str(c) for c in header]
    widths = [max(len(r[i]) for r in [header, *rows])
              for i in range(len(header))]
    line = lambda r: "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                               for i, (c, w) in enumerate(zip(r, widths)))
    return "\n".join([indent + line(header),
                      indent + "-"*len(line(header)),
                      *(indent + line(r) for r in rows)])


def style_ax(ax):
    """ Minor ticks, and major/minor gridlines. """
    import matplotlib.ticker
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(True, axis="both", which="major", lw=1.0, ls="-", alpha=0.8)
    ax.grid(True, axis="both", which="minor", lw=0.6, ls="--", alpha=0.5)



### ALIASES #####################################################################

if typing.TYPE_CHECKING:
    _Float = float
else:
    class _Float(float):
        def __new__(cls, *args, **kwargs):
            raise TypeError("units aren't converters: "
                            "use u.mm_(value), not u.mm(value)")

_ATM = 101325.0
_PI = 3.141592653589793
_ABSZ = 273.15


# Dimensionless.
none = Annotated[_Float, None]
none_ = Converter()
percent = Annotated[_Float, Unit("%", 1e-2, spaced=False)]
percent_ = Converter(percent)

# Length.
m = Annotated[_Float, Unit("m")]
m_ = Converter(m)
mm = Annotated[_Float, Unit("mm", 1e-3)]
mm_ = Converter(mm)
um = Annotated[_Float, Unit("µm", 1e-6)]
um_ = Converter(um)
inch = Annotated[_Float, Unit("in", 25.4e-3)]
inch_ = Converter(inch)

# Area.
m2 = Annotated[_Float, Unit("m²")]
m2_ = Converter(m2)
mm2 = Annotated[_Float, Unit("mm²", 1e-6)]
mm2_ = Converter(mm2)

# Volume.
m3 = Annotated[_Float, Unit("m³")]
m3_ = Converter(m3)
mm3 = Annotated[_Float, Unit("mm³", 1e-9)]
mm3_ = Converter(mm3)
L = Annotated[_Float, Unit("L", 1e-3)]
L_ = Converter(L)
cc = Annotated[_Float, Unit("cm³", 1e-6)]
cc_ = Converter(cc)

# Angle/rotation.
rad = Annotated[_Float, Unit("rad")]
rad_ = Converter(rad)
deg = Annotated[_Float, Unit("°", _PI/180, spaced=False)]
deg_ = Converter(deg)
rad_s = Annotated[_Float, Unit("rad/s")]
rad_s_ = Converter(rad_s)
rpm = Annotated[_Float, Unit("rpm", 2*_PI/60)]
rpm_ = Converter(rpm)
revs = Annotated[_Float, Unit("rev/s", 2*_PI)]
revs_ = Converter(revs)

# Speed/acceleration.
m_s = Annotated[_Float, Unit("m/s")]
m_s_ = Converter(m_s)
m_s2 = Annotated[_Float, Unit("m/s²")]
m_s2_ = Converter(m_s2)

# Mass.
kg = Annotated[_Float, Unit("kg")]
kg_ = Converter(kg)
g = Annotated[_Float, Unit("g", 1e-3)]
g_ = Converter(g)
mg = Annotated[_Float, Unit("mg", 1e-6)]
mg_ = Converter(mg)

# Time.
s = Annotated[_Float, Unit("s")]
s_ = Converter(s)
ms = Annotated[_Float, Unit("ms", 1e-3)]
ms_ = Converter(ms)
us = Annotated[_Float, Unit("µs", 1e-6)]
us_ = Converter(us)
minute = Annotated[_Float, Unit("min", 60.0)]
minute_ = Converter(minute)
hour = Annotated[_Float, Unit("hr", 3600.0)]
hour_ = Converter(hour)

# Force.
N = Annotated[_Float, Unit("N")]
N_ = Converter(N)
kN = Annotated[_Float, Unit("kN", 1e3)]
kN_ = Converter(kN)
lbf = Annotated[_Float, Unit("lbf", 4.4482216152605)]
lbf_ = Converter(lbf)

# Pressure.
Pa = Annotated[_Float, Unit("Pa")]
Pa_ = Converter(Pa)
kPa = Annotated[_Float, Unit("kPa", 1e3)]
kPa_ = Converter(kPa)
MPa = Annotated[_Float, Unit("MPa", 1e6)]
MPa_ = Converter(MPa)
bar = Annotated[_Float, Unit("bar", 1e5)]
bar_ = Converter(bar)
psi = Annotated[_Float, Unit("psi", 6894.757293168)]
psi_ = Converter(psi)
barg = Annotated[_Float, Unit("barg", 1e5, _ATM)]
barg_ = Converter(barg)
psig = Annotated[_Float, Unit("psig", 6894.757293168, _ATM)]
psig_ = Converter(psig)

# Moment.
Nm = Annotated[_Float, Unit("Nm")]
Nm_ = Converter(Nm)
Nmm = Annotated[_Float, Unit("Nmm", 1e-3)]
Nmm_ = Converter(Nmm)
kNm = Annotated[_Float, Unit("kNm", 1e3)]
kNm_ = Converter(kNm)

# Stiffness.
N_m = Annotated[_Float, Unit("N/m")]
N_m_ = Converter(N_m)
N_mm = Annotated[_Float, Unit("N/mm", 1e3)]
N_mm_ = Converter(N_mm)

# Energy/power.
J = Annotated[_Float, Unit("J")]
J_ = Converter(J)
mJ = Annotated[_Float, Unit("mJ", 1e-3)]
mJ_ = Converter(mJ)
W = Annotated[_Float, Unit("W")]
W_ = Converter(W)

# Temperature.
K = Annotated[_Float, Unit("K")]
K_ = Converter(K)
degC = Annotated[_Float, Unit("°C", 1.0, _ABSZ, spaced=False)]
degC_ = Converter(degC)

# Hardness.
shoreA = Annotated[_Float, Unit("shore A")]
shoreA_ = Converter(shoreA)
