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

        # 2. derived: an output dependant on only one parameter.
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
    p.report()  # every input and output, in units
    p = p.tweak(F=1000) # F -> A -> D: D = 35.68 mm
    p.report()  # note that `.tweak` returned a new object.
    p.tweak(D=u.si.mm(32), swept=u.si.cc(5)) # stroke = 6.217 mm

Systems nest: annotate a field with a System subclass, or return one from a
calc. Every name-taking method accepts a dotted path through the tree:

    class Rig(System):
        piston: Piston
        count: int
        F = calc(u.kN, lambda s: s.count * s.piston.F)

    r = Rig(piston=p, count=4)
    r["piston.A"]               # fetch, also r.fetch("piston.A")
    r.display("piston.F")       # "1000 N"
    r.unit("piston.D")          # mm
    r.tweak({"piston.P": u.si.bar(20)})   # rebuilds piston, then rig
    r.report()                  # all members, flattened with dots
"""

import ast
import dataclasses
import inspect
import math
import sys
import textwrap
import typing
import warnings
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


class _hybridmethod:
    """ Bound to the instance when called on one, else to the class. """
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        target = objtype if obj is None else obj
        return self.func.__get__(target, type(target))


@dataclass_transform(frozen_default=True)
class System:
    def __init_subclass__(cls, **options):
        # Class keywords pass through to dataclass, e.g.
        # `class X(System, eq=False)`. `frozen` is always implicitly true tho.
        super().__init_subclass__()
        dataclasses.dataclass(cls, **{**options, "frozen": True})


    @classmethod
    def inputs(cls, deep=False):
        """ Names of independent inputs. """
        flat = tuple(f.name for f in dataclasses.fields(cls))
        return _expand(cls, flat, "inputs") if deep else flat

    @classmethod
    def outputs(cls, deep=False):
        """ Names of dependent outputs. """
        if deep:
            inputs = set(cls.inputs(deep=True))
            return tuple(n for n in cls.parameters(deep=True)
                         if n not in inputs)
        names = {}
        for klass in reversed(cls.__mro__):
            for name, attr in vars(klass).items():
                if not name.startswith("_") and \
                        isinstance(attr, (property, cached_property)):
                    names[name] = None
        return tuple(n for n in names
                if isinstance(getattr(cls, n), (property, cached_property)))

    @classmethod
    def parameters(cls, deep=False):
        """ Names of all inputs and outputs. """
        flat = _declaration_order(cls)
        return _expand(cls, flat, "parameters") if deep else flat

    @classmethod
    def settable(cls, deep=False):
        """ Names `tweak` accepts: inputs and outputs with a setter. """
        flat = tuple(n for n in cls.parameters() if cls._settable_here(n))
        return _expand(cls, flat, "settable") if deep else flat


    def display(self, name, fmt=lambda x: f"{x:.4g}"):
        """ Returns a string of the given (dotted) parameter in its units. """
        owner, leaf = self._resolve(name)
        value = getattr(owner, leaf)
        if isinstance(value, System):
            return type(value).__name__
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return repr(value)
        unit = type(owner).unit(leaf)
        if unit is None:
            return fmt(value)
        return f"{fmt(unit.nsi(value))} {unit.symbol}"


    def report(self, outputs=True, printed=True):
        """ Every parameter in the tree, flattened to dotted paths, in
        declaration order. Dependants (and anything under one) marked with -.
        """
        rows = []
        for path, owner, leaf, dependent, alias in self._walk(outputs):
            if alias is not None:
                text = f"= {alias}"
            else:
                try:
                    text = owner.display(leaf)
                except Exception as e: # show it, don't kill the report.
                    text = f"<{type(e).__name__}: {e}>"
            rows.append((path, "-" if dependent else " ", text))
        width = min(max((len(p) for p, _, _ in rows), default=0), 36)
        lines = [type(self).__name__ + "\n"]
        if outputs:
            lines.append(" (- dependent)\n")
        for path, mark, text in rows:
            lines.append(f"  {mark} {path:<{width}}  {text}\n")
        s = "".join(lines)
        if printed:
            print(s)
        else:
            return s


    def tweak(self, changes=None, /, **kwchanges):
        """
        Return a copy with the given inputs or settable outputs changed. Note
        these changes are applied in-order. Setters may themselves return other
        settable outputs, not just inputs. Names may be dotted paths into sub-
        systems, through inputs or settable outputs.
        """
        changes = {**(changes or {}), **kwchanges}
        system = self
        for name, value in changes.items():
            head, _, rest = name.partition(".")
            if rest:
                # Tweak the sub-system, then set it back on this one (a
                # replace for an input, the setter for an output).
                if not system._settable_here(head):
                    raise AttributeError(f"can't tweak {name!r}: "
                            f"{type(system).__name__}.{head} is an output "
                            "with no setter")
                sub = getattr(system, head)
                if not isinstance(sub, System):
                    raise AttributeError(f"can't tweak {name!r}: "
                            f"{type(system).__name__}.{head} is not a System")
                new = sub.tweak({rest: value})
                system = system.tweak({head: new})
                if head not in system.inputs():
                    # An output's setter might not carry every change back.
                    got = system.fetch(name)
                    if not _close(got, new.fetch(rest)):
                        warnings.warn(f"tweak {name!r}: setter for "
                            f"{type(system).__name__}.{head} didn't carry the "
                            f"change through (got {got!r}, wanted "
                            f"{new.fetch(rest)!r})", stacklevel=2)
                continue
            if name in system.inputs():
                system = dataclasses.replace(system, **{name: value})
                continue
            if not system._settable_here(name):
                raise AttributeError(
                    f"{type(system).__name__}.{name} is not settable. "
                    f"Settable: {', '.join(system.settable())}"
                )
            prop = getattr(type(system), name)
            system = system.tweak(prop.fset(system, value))
        return system

    @classmethod
    def _settable_here(cls, name):
        """ Whether a (non-dotted) name is an input or an output with a
        setter. """
        if name in cls.inputs():
            return True
        prop = getattr(cls, name, None)
        return isinstance(prop, property) and prop.fset is not None


    @_hybridmethod
    def unit(self_or_cls, name):
        """ Display unit of a parameter, or None if it isn't annotated (or is a
        System). On an instance, sub-system types come from the values; on the
        class, from annotations, like the listings. """
        if not isinstance(self_or_cls, type):
            owner, leaf = self_or_cls._resolve(name)
            return _unit_of(type(owner), leaf)
        cls = self_or_cls
        *parents, leaf = name.split(".")
        for i, part in enumerate(parents):
            sub = _system_type_of(cls, part)
            if sub is None:
                raise AttributeError(f"{'.'.join(parents[:i+1])!r} is not "
                                     "declared as a System")
            cls = sub
        return _unit_of(cls, leaf)


    def fetch(self, name):
        """ Value at a dotted path, e.g. `s.fetch("load.F")`. """
        x = self
        for part in name.split("."):
            x = getattr(x, part)
        return x
    __getitem__ = fetch


    def _resolve(self, name):
        """ (owning system, leaf name) of a dotted path. """
        parent, _, leaf = name.rpartition(".")
        owner = self.fetch(parent) if parent else self
        if not isinstance(owner, System):
            raise AttributeError(f"{parent!r} is a {type(owner).__name__}, "
                                 "not a System")
        return owner, leaf

    def _walk(self, outputs, prefix="", dependent=False, seen=None):
        """ Yields (path, owner, leaf, dependent, alias) depth-first. `alias`
        is the first path of an already-visited sub-system, else None. """
        if seen is None:
            seen = {id(self): "(self)"}
        inputs = self.inputs()
        for n in self.parameters():
            if not outputs and n not in inputs:
                continue
            path = prefix + n
            dep = dependent or n not in inputs
            try:
                value = getattr(self, n)
            except Exception:
                value = None # report() will show the error.
            if isinstance(value, System):
                if id(value) in seen:
                    yield path, self, n, dep, seen[id(value)]
                    continue
                seen[id(value)] = path
                yield path, self, n, dep, None
                yield from value._walk(outputs, path + ".", dep, seen)
            else:
                yield path, self, n, dep, None



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
        # No try/except: computing inside an except block chains a KeyError
        # onto every error raised by a (nested) getter.
        d = obj.__dict__
        if self.name not in d:
            d[self.name] = self.fget(obj)
        return d[self.name]

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

def derived(unit, source, derive, inverse=None, *, cache=True, doc=None):
    """ An output computed from only one input. """
    fset = (lambda s, v: {source: inverse(v)}) if inverse else None
    return calc(unit, lambda s: derive(s.fetch(source)), fset, cache=cache,
            doc=doc)



def _is_system_type(t):
    return isinstance(t, type) and issubclass(t, System)

@cache
def _declared(cls, name):
    """ What a parameter is declared as: its field annotation, its output's
    unit, or its getter's return annotation. """
    if name in cls.inputs():
        return typing.get_type_hints(cls, include_extras=True).get(name)
    attr = getattr(cls, name, None)
    if isinstance(attr, output):
        return attr.unit
    fget = getattr(attr, "fget", None) or getattr(attr, "func", None)
    if fget is None:
        raise AttributeError(f"{cls.__name__} has no input or output {name!r}")
    return typing.get_type_hints(fget, include_extras=True).get("return")

@cache
def _system_type_of(cls, name):
    """ The System subclass a parameter is declared as, else None. """
    hint = _declared(cls, name)
    return hint if _is_system_type(hint) else None

@cache
def _unit_of(cls, name):
    hint = _declared(cls, name)
    return None if _is_system_type(hint) else unit_of(hint)

def _close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-12)
    return a == b

def _expand(cls, flat, listing, chain=()):
    """ `flat` names of `cls`, each followed by the dotted deep `listing`
    (a System classmethod name) of its sub-system type, if it has one. """
    chain += (cls,)
    names = []
    for n in flat:
        names.append(n)
        sub = _system_type_of(cls, n)
        if sub is not None and sub not in chain:
            sub_flat = getattr(sub, listing)()
            names += [f"{n}.{m}" for m in _expand(sub, sub_flat, listing, chain)]
    return tuple(names)

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
