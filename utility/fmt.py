"""
Formatting helpers for printing and plotting.
"""

from math import floor, log10, isfinite

from units import unit_of


__all__ = (
    "fmt_num",
    "fmt_mm", "fmt_ms", "fmt_m_s", "fmt_N", "fmt_bar", "fmt_MPa", "fmt_pt",
    "fmt_unit",

    "table",

    "style_ax",
)


# ---- Numbers ------------------------------------------------------------------

def fmt_num(x, sig=5):
    """ `x` to `sig` significant figures, without trailing zeros. """
    if not isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    decimals = max(0, sig - 1 - floor(log10(abs(x))))
    s = f"{x:.{decimals}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s

# Fixed units, from SI.
fmt_mm = lambda x: f"{fmt_num(x*1e3)}mm"
fmt_ms = lambda x: f"{fmt_num(x*1e3)} ms"
fmt_m_s = lambda x: f"{fmt_num(x)} m/s"
fmt_N = lambda x: f"{fmt_num(x, sig=3)} N"
fmt_bar = lambda x: f"{fmt_num(x*1e-5)} bar"
fmt_MPa = lambda x: f"{fmt_num(x*1e-6)} MPa"
fmt_pt = lambda x: f"{fmt_num(x*100, sig=3)}%"

def fmt_unit(x, unit, sig=4):
    """
    SI `x` in a display unit (a `Unit` or a unit alias like `u.mm`), e.g.
    '31.5 mm'. Plain number if `unit` is None; repr if `x` isn't a number.
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return repr(x)
    unit = unit_of(unit)
    if unit is None or not isfinite(x):
        return fmt_num(x, sig)
    return f"{fmt_num(unit.nsi(x), sig)} {unit.symbol}"



# ---- Tables -------------------------------------------------------------------

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



# ---- Plots --------------------------------------------------------------------

def style_ax(ax):
    """ Minor ticks, and major/minor gridlines. """
    import matplotlib.ticker
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(True, axis="both", which="major", lw=1.0, ls="-", alpha=0.8)
    ax.grid(True, axis="both", which="minor", lw=0.6, ls="--", alpha=0.5)
