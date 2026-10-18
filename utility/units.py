"""
Display units for Design fields and getters.

Values are always stored and computed in SI. A unit annotation only says how a
value is shown. Each alias is still a plain `float` to Python and type checkers:

    import units as u
    class Pipe(Design):
        D: u.mm # stored in m, shown in mm
        L: u.m
        @property
        def area(self) -> u.mm2:
            return pi/4 * self.D**2
"""
from dataclasses import dataclass
from typing import Annotated, Callable, get_args, get_origin, TYPE_CHECKING


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

    def si(self, value):
        """ Display value -> SI value. """
        return value * self.scale + self.offset

    def nsi(self, si):
        """ SI value -> display value. """
        return (si - self.offset) / self.scale

def unit_of(alias):
    """ Returns the Unit inside an annotation. """
    if isinstance(alias, Unit):
        return alias
    if get_origin(alias) is Annotated:
        for meta in get_args(alias)[1:]:
            if isinstance(meta, Unit):
                return meta
    return None


class _Convert:
    def __init__(self, direction):
        self._direction = direction

    def __getattr__(self, name) -> Callable[[float], float]:
        unit = unit_of(globals().get(name)) if not name.startswith("_") else None
        if unit is None:
            raise AttributeError(f"no unit named {name!r}")
        return getattr(unit, self._direction)

    def __dir__(self):
        return [n for n, v in globals().items() if unit_of(v) is not None]

si = _Convert("si")
nsi = _Convert("nsi")



if TYPE_CHECKING:
    _Float = float
else:
    class _Float(float):
        def __new__(cls, *args, **kwargs):
            raise TypeError("units aren't converters: "
                            "use u.si.mm(value), not u.mm(value)")

_ATM = 101325.0
_PI = 3.141592653589793
_ABSZ = 273.15


# Dimensionless.
ratio = Annotated[_Float, Unit("-")]
percent = Annotated[_Float, Unit("%", 1e-2)]

# Length.
m = Annotated[_Float, Unit("m")]
mm = Annotated[_Float, Unit("mm", 1e-3)]
um = Annotated[_Float, Unit("µm", 1e-6)]
inch = Annotated[_Float, Unit("in", 25.4e-3)]

# Area.
m2 = Annotated[_Float, Unit("m²")]
mm2 = Annotated[_Float, Unit("mm²", 1e-6)]

# Volume.
m3 = Annotated[_Float, Unit("m³")]
mm3 = Annotated[_Float, Unit("mm³", 1e-9)]
L = Annotated[_Float, Unit("L", 1e-3)]
cc = Annotated[_Float, Unit("cm³", 1e-6)]

# Angle/rotation.
rad = Annotated[_Float, Unit("rad")]
deg = Annotated[_Float, Unit("°", _PI/180)]
rad_s = Annotated[_Float, Unit("rad/s")]
rpm = Annotated[_Float, Unit("rpm", 2*_PI/60)]

# Speed/acceleration.
m_s = Annotated[_Float, Unit("m/s")]
m_s2 = Annotated[_Float, Unit("m/s²")]

# Mass.
kg = Annotated[_Float, Unit("kg")]
g = Annotated[_Float, Unit("g", 1e-3)]
mg = Annotated[_Float, Unit("mg", 1e-6)]

# Time.
s = Annotated[_Float, Unit("s")]
ms = Annotated[_Float, Unit("ms", 1e-3)]
us = Annotated[_Float, Unit("µs", 1e-6)]
minute = Annotated[_Float, Unit("min", 60.0)]
hour = Annotated[_Float, Unit("h", 3600.0)]

# Force.
N = Annotated[_Float, Unit("N")]
kN = Annotated[_Float, Unit("kN", 1e3)]
lbf = Annotated[_Float, Unit("lbf", 4.4482216152605)]

# Pressure.
Pa = Annotated[_Float, Unit("Pa")]
kPa = Annotated[_Float, Unit("kPa", 1e3)]
MPa = Annotated[_Float, Unit("MPa", 1e6)]
bar = Annotated[_Float, Unit("bar", 1e5)]
psi = Annotated[_Float, Unit("psi", 6894.757293168)]
barg = Annotated[_Float, Unit("barg", 1e5, _ATM)]
psig = Annotated[_Float, Unit("psig", 6894.757293168, _ATM)]

# Stiffness.
N_m = Annotated[_Float, Unit("N/m")]
N_mm = Annotated[_Float, Unit("N/mm", 1e3)]

# Energy/power.
J = Annotated[_Float, Unit("J")]
mJ = Annotated[_Float, Unit("mJ", 1e-3)]
W = Annotated[_Float, Unit("W")]

# Temperature.
K = Annotated[_Float, Unit("K")]
degC = Annotated[_Float, Unit("°C", 1.0, _ABSZ)]
degF = Annotated[_Float, Unit("°F", 5/9, _ABSZ - 32 * 5/9)]
degdF = Annotated[_Float, Unit("Δ°F", 5/9)]
