import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from CoolProp.CoolProp import PropsSI
from matplotlib.ticker import MultipleLocator

@np.vectorize
def N2O(*args, **kwargs):
    try:
        return PropsSI(*args, "N2O", **kwargs)
    except Exception:
        return float("nan")
@np.vectorize
def CO2(*args, **kwargs):
    try:
        return PropsSI(*args, "CO2", **kwargs)
    except Exception:
        return float("nan")

ms = [0.1, 0.15, 0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8] # n2o mass.
V = 1e-3 # tank volume.

Tmin = -5
Tmax = 55
Pmin = 20
Pmax = 90


def coords_angle(X, Y, *, x=None, y=None):
    assert (x is None) != (y is None)

    if x is None:
        i = np.nanargmin(abs(Y - y))
    else:
        i = np.nanargmin(abs(X - x))
    if X[i] != X[i] or Y[i] != Y[i]:
        return np.nan, np.nan, np.nan
    dx = X[min(i+1, len(X)-1)] - X[max(i-1, 0)]
    dy = Y[min(i+1, len(Y)-1)] - Y[max(i-1, 0)]
    angle = np.degrees(np.arctan2(dy, dx))
    return X[i], Y[i], angle


def plotme(ax):
    cmap = plt.get_cmap("plasma")
    colors = [cmap(v) for v in np.linspace(0.2, 0.8, len(ms))]

    T = np.linspace(Tmin+273.15, Tmax+273.15, 200)
    Psat_N2O = N2O("P", "T", T, "Q", 0)/1e5
    Psat_CO2 = CO2("P", "T", T, "Q", 0)/1e5

    Tcrit_N2O = N2O("Tcrit")-273.15
    Pcrit_N2O = N2O("pcrit")/1e5
    Tcrit_CO2 = CO2("Tcrit")-273.15
    Pcrit_CO2 = CO2("pcrit")/1e5

    T -= 273.15

    col_N2O = "purple"
    # col_CO2 = (0.12, 0.12, 0.35)
    col_CO2 = "orange"

    ax.plot([Tcrit_N2O], [Pcrit_N2O],
            color=col_N2O, markersize=4.5,
            zorder=5,
            marker="o", ls="",
            label="N2O"
        )
    ax.annotate(f"crit. point\n{Tcrit_N2O:.1f}°C, {Pcrit_N2O:.1f} bar",
            xy=(Tcrit_N2O, Pcrit_N2O),
            xytext=(8, -8), textcoords="offset points", ha="left", va="top",
            color=col_N2O, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
        )
    ax.plot([Tcrit_CO2], [Pcrit_CO2],
            color=col_CO2, markersize=4.5,
            zorder=5,
            marker="o", ls="",
            label="CO2"
        )
    ax.annotate(f"crit. point\n{Tcrit_CO2:.1f}°C, {Pcrit_CO2:.1f} bar",
            xy=(Tcrit_CO2, Pcrit_CO2),
            xytext=(-8, 8), textcoords="offset points", ha="right", va="bottom",
            color=col_CO2, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
        )

    for Tc, Pc, col in [(Tcrit_N2O, Pcrit_N2O, col_N2O),
                    (Tcrit_CO2, Pcrit_CO2, col_CO2)]:
        ax.plot([Tc, Tmax], [Pc, Pc], color=col, lw=1.2, ls="--", zorder=4)
        ax.plot([Tc, Tc], [Pc, Pmax], color=col, lw=1.2, ls="--", zorder=4)

    ax.plot(T[Psat_N2O == Psat_N2O], Psat_N2O[Psat_N2O == Psat_N2O],
            color=col_N2O, lw=1.5, ls="-")
    ax.plot(T[Psat_CO2 == Psat_CO2], Psat_CO2[Psat_CO2 == Psat_CO2],
            color=col_CO2, lw=1.5, ls="-")

    # hatches = {
    #     # fluid: (gas region, liquid region)
    #     "N2O": ("....", "////"),
    #     "CO2": ("oo", "\\\\\\\\"),
    # }
    # for name, Psat, Tc, Pc, col in [("N2O", Psat_N2O, Tcrit_N2O, Pcrit_N2O, col_N2O),
    #                                 ("CO2", Psat_CO2, Tcrit_CO2, Pcrit_CO2, col_CO2)]:
    #     h_gas, h_liq = hatches[name]
    #     below = (T < Tc) & np.isfinite(Psat)

    #     # Gas region: under the sat curve up to Tc, then under Pc out to Tmax
    #     T_gas = np.concatenate([T[below], [Tc, Tmax]])
    #     P_gas = np.concatenate([Psat[below], [Pc, Pc]])
    #     ax.fill_between(T_gas, Pmin, P_gas,
    #                     facecolor="none", edgecolor=col, hatch=h_gas,
    #                     lw=0, alpha=0.5, zorder=1)

    #     # Liquid region: above the sat curve, left of Tc, up to Pmax
    #     T_liq = np.concatenate([T[below], [Tc]])
    #     P_liq = np.concatenate([Psat[below], [Pc]])
    #     ax.fill_between(T_liq, P_liq, Pmax,
    #                     facecolor="none", edgecolor=col, hatch=h_liq,
    #                     lw=0, alpha=0.5, zorder=1)

    ax.annotate("superheated vapour",
            xy=(40.0, 45.0),
            xytext=(0, 0), textcoords="offset points", ha="center", va="center",
            color=(0.3, 0.3, 0.3), fontweight="bold", fontsize=9,
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8),
        )
    ax.annotate("compressed liquid",
            xy=(10.0, 65.0),
            xytext=(0, 0), textcoords="offset points", ha="center", va="center",
            color=(0.3, 0.3, 0.3), fontweight="bold", fontsize=9,
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8),
        )
    ax.annotate("super critical",
            xy=(45.0, 82.0),
            xytext=(0, 0), textcoords="offset points", ha="center", va="center",
            color=(0.3, 0.3, 0.3), fontweight="bold", fontsize=9,
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8),
        )

    ax.minorticks_on()
    ax.set_aspect("equal")
    ax.set_xlabel("Temperature [°C]", fontsize=13)
    ax.set_ylabel("Pressure [bar]", fontsize=13)
    ax.set_title("N2O & CO2 Saturation Pressure", fontsize=14)
    ax.legend()

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    ax.grid(which="major", color="gray", lw=1.0, ls="-", alpha=0.65)
    ax.grid(which="minor", color="lightgray", lw=0.6, ls="--", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=10, width=1.0)
    ax.tick_params(axis="both", which="minor", width=0.8)

    ax.set_xlim(Tmin, Tmax)
    ax.set_ylim(Pmin, Pmax)





def main():
    plt.rcParams.update({
        "figure.autolayout": True,
        "figure.max_open_warning": 0,
        "savefig.dpi": 350,
    })
    plt.figure(figsize=(7, 9))

    plotme(plt.gca())

    plt.tight_layout()
    plt.gcf().savefig("n2o_vs_co2.png")
    plt.show()



if __name__ == "__main__":
    main()
