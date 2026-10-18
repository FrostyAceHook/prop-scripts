"""
Store a system as its independent inputs, allowing nice interaction.

Subclass `System` and declare the independent inputs as annotated fields. Declare
any dependent outputs as @property, with a property setter to make it settable.
This setter must return a dictionary of input changes that produce the requested
value. Annotate fields and getters' return types with a display unit from
`units`:

    from math import pi, sqrt
    import units as u
    from system import System, calc, derived

    circ_area = lambda D: pi/4 * D**2
    circ_diam = lambda A: sqrt(4/pi * A)

    class Piston(System):
        # 1. Field: an independent input.
        D: u.mm
        P: u.bar
        stroke: u.mm

        # 2. derived: an output dependant on only one input.
        A = derived(u.mm2, "D", circ_area, circ_diam)

        # 3. calc: any output (shorter form than property).
        swept = calc(u.cc,
                lambda s: s.A * s.stroke,
                lambda s, V: dict(stroke=V / s.A))

        # 4. property: any output.
        @property
        def F(self) -> u.N: # annotate unit here.
            return self.P * self.A
        @F.setter
        def F(self, F):
            return dict(A=F / self.P) # setters may set outputs, not just inputs.

    p = Piston(D=u.si.mm(30), P=u.si.bar(10), stroke=u.si.mm(5))
    print(p.summary())  # every input and output, in units
    p = p.tweak(F=1000) # F -> A -> D: D = 35.68 mm
    print(p.summary())  # note that `.tweak` returned a new object.
    p.tweak(D=u.si.mm(32), swept=u.si.cc(5)) # stroke = 6.217 mm
"""

import ast
import dataclasses
import inspect
import sys
import textwrap
import typing
from functools import cache, cached_property

from units import unit_of


__all__ = (
    "System",
    "output",
    "computed",
    "calc",
    "derived",
)


if typing.TYPE_CHECKING or sys.version_info >= (3, 11):
    from typing import dataclass_transform
else: # For editor/type-checker support.
    def dataclass_transform(**kwargs):
        return lambda cls: cls


@dataclass_transform(frozen_default=True)
class System:
    def __init_subclass__(cls, **options):
        # Class keywords pass through to dataclass, e.g.
        # `class X(System, eq=False)`. `frozen` is always implicitly true tho.
        super().__init_subclass__()
        dataclasses.dataclass(cls, **{**options, "frozen": True})

    @classmethod
    def inputs(cls):
        """ Names of independent inputs. """
        return tuple(f.name for f in dataclasses.fields(cls))

    @classmethod
    def outputs(cls):
        """ Names of dependent outputs. """
        names = {}
        for klass in reversed(cls.__mro__):
            for name, attr in vars(klass).items():
                if not name.startswith("_") and \
                        isinstance(attr, (property, cached_property)):
                    names[name] = None
        return tuple(n for n in names
                if isinstance(getattr(cls, n), (property, cached_property)))

    @classmethod
    def parameters(cls):
        """ Names of all inputs and outputs, in declaration order. """
        return _declaration_order(cls)

    @classmethod
    def settable(cls):
        """ Names of settable parameters. """
        return cls.inputs() + tuple(name for name in cls.outputs()
                if getattr(getattr(cls, name), "fset", None) is not None)


    def display(self, name, fmt=lambda x: f"{x:.4g}"):
        """ Returns a string of the given parameter in its units. """
        value = getattr(self, name)
        unit = self.unit(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return repr(value)
        if unit is None:
            return fmt(value)
        return f"{fmt(unit.nsi(value))} {unit.symbol}"

    def summary(self, outputs=True):
        """ Every parameter in declaration order; dependants marked with *. """
        inputs = self.inputs()
        names = [n for n in self.parameters() if outputs or n in inputs]
        width = min(max(map(len, names)) + 2, 18)
        lines = [type(self).__name__ + "\n"]
        if outputs:
            lines.append(" (- dependent)\n")
        for n in names:
            mark = " " if n in inputs else "-"
            lines.append(f"  {mark} {n:<{width-2}}  {self.display(n)}\n")
        return "".join(lines)
    def report(self, *args, **kwargs):
        """ Prints `.summary()`. """
        print(self.summary(*args, **kwargs))


    def tweak(self, **changes):
        """
        Return a copy with the given inputs or settable outputs changed. Note
        these changes are applied in-order. Setters may themselves return other
        settable outputs, not just inputs.
        """
        system = self
        for name, value in changes.items():
            if name in system.inputs():
                system = dataclasses.replace(system, **{name: value})
                continue
            prop = getattr(type(system), name, None)
            if not isinstance(prop, property) or prop.fset is None:
                raise AttributeError(
                    f"{type(system).__name__}.{name} is not settable. "
                    f"Settable: {', '.join(system.settable())}"
                )
            system = system.tweak(**prop.fset(system, value))
        return system


    @classmethod
    def unit(cls, name):
        """ Display unit of a parameter, or None if it isn't annotated. """
        return _unit_of(cls, name)


class output(property):
    """ A property with a display unit. Cached per instance by default. """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None, *,
                 unit=None, cache=True):
        super().__init__(fget, fset, fdel, doc)
        self.unit = unit
        self.cache = cache
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None or not self.cache or self.name is None:
            return super().__get__(obj, objtype)
        try:
            return obj.__dict__[self.name]
        except KeyError:
            value = obj.__dict__[self.name] = self.fget(obj)
            return value

    def _copy(self, **changes):
        kw = dict(fget=self.fget, fset=self.fset, fdel=self.fdel,
                  doc=self.__doc__, unit=self.unit, cache=self.cache) | changes
        return type(self)(**kw)

    def getter(self, fget):
        return self._copy(fget=fget)

    def setter(self, fset):
        return self._copy(fset=fset)

def computed(unit):
    """ Caching and unit annotation. """
    return lambda fget: output(fget, unit=unit)

def calc(unit, fget, fset=None, *, cache=True, doc=None):
    """ An output from a getter (and optional setter), with a display unit. """
    return output(fget, fset, doc=doc, unit=unit, cache=cache)

def derived(unit, source, forward, inverse=None, *, cache=True, doc=None):
    """ An output computed from only one input. """
    fset = (lambda s, v: {source: inverse(v)}) if inverse else None
    return calc(unit, lambda s: forward(getattr(s, source)), fset, cache=cache,
            doc=doc)



@cache
def _unit_of(cls, name):
    if name in cls.inputs():
        hint = typing.get_type_hints(cls, include_extras=True).get(name)
    else:
        attr = getattr(cls, name, None)
        if isinstance(attr, output):
            return unit_of(attr.unit)
        fget = getattr(attr, "fget", None) or getattr(attr, "func", None)
        if fget is None:
            raise AttributeError(f"{cls.__name__} has no input or output "
                                 f"{name!r}")
        hint = typing.get_type_hints(fget, include_extras=True).get("return")
    return unit_of(hint)

@cache
def _declaration_order(cls):
    names = []
    for klass in reversed(cls.__mro__):
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(klass)))
        except (OSError, TypeError, SyntaxError):
            continue # no source (builtins, REPL): fall back below
        for node in tree.body[0].body:
            if (isinstance(node, ast.AnnAssign) and
                    isinstance(node.target, ast.Name)):
                names.append(node.target.id)  # D_seat: u.mm
            elif isinstance(node, ast.Assign):
                names += [t.id for t in node.targets
                          if isinstance(t, ast.Name)]  # A_seat = ...
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)  # @property def F_pneumo(...)
    params = cls.inputs() + cls.outputs()
    ordered = [n for n in dict.fromkeys(names) if n in params]
    return tuple(ordered + [n for n in params if n not in ordered])
