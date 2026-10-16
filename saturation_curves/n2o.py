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

ms = [0.1, 0.15, 0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8] # n2o mass.
V = 1e-3 # tank volume.

Tmin = -5
Tmax = 60
Pmin = 20
Pmax = 120

def props(m):
    T = np.linspace(Tmin+273.15, Tmax+273.15, 200)
    rho = m / V

    P = N2O("P", "T", T, "D", rho)
    Psat = N2O("P", "T", T, "Q", 0)

    Tcrit = N2O("Tcrit")
    Pcrit = N2O("pcrit")

    Psat_ueps = Psat * (1 + 1e-5)
    Psat_deps = Psat * (1 - 1e-5)
    idew = np.nanargmax((P > Psat_ueps) | (P < Psat_deps))
    Tdew = T[idew]
    Pdew = P[idew]
    # if idew == 0 and not (P[0] > Psat_eps[0]):
    #     if (P > Psat_eps).any():
    #         Tdew = T[0]
    #         Pdew = P[0]
    #     else:
    #         Tdew = Tcrit
    #         Pdew = Pcrit

    return {
        "T": T - 273.15,
        "P": P / 1e5,
        "Psat": Psat / 1e5,
        "Tcrit": Tcrit - 273.15,
        "Pcrit": Pcrit / 1e5,
        "Tdew": Tdew - 273.15,
        "Pdew": Pdew / 1e5,
    }

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

    for m, col in zip(ms, colors):
        p = props(m)
        T = p["T"]
        P = p["P"]
        Tsat = p["T"]
        Psat = p["Psat"]
        Tcrit = p["Tcrit"]
        Pcrit = p["Pcrit"]
        Tdew = p["Tdew"]
        Pdew = p["Pdew"]

        plot_P = P.copy()
        plot_P[~(T > Tdew)] = np.nan
        idx = (T > Tdew).argmax()
        idx = max(0, idx - 1)
        plot_P[idx] = P[idx]
        idx = min(len(P) - 1, idx + 1)
        plot_P[idx] = P[idx]
        ax.plot(T, plot_P, color=col, lw=1.5)

        t, p, angle = coords_angle(T, P, y=Pmax - 15)
        mstr = f"{m:.2f}".rstrip("0").rstrip(".")
        ax.annotate(f"{mstr} kg",
                xy=(t, p),
                rotation=angle,
                rotation_mode="anchor",
                color=col, fontweight="bold", fontsize=10,
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.2")
            )

    ax.plot([Tcrit], [Pcrit],
            color="blue", markersize=4.5,
            zorder=5,
            marker="o", ls="")
    ax.annotate(f"crit. point\n{Tcrit:.1f}°C, {Pcrit:.1f} bar",
            xy=(Tcrit, Pcrit),
            xytext=(8, -8), textcoords="offset points", ha="left", va="top",
            color="blue", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
        )

    ax.plot(Tsat, Psat, color="blue", lw=1.5, ls="-")
    t, p, angle = coords_angle(Tsat, Psat, x=(2+Tmin + Tdew)/2)
    ax.annotate("sat. pressure",
            xy=(t, p),
            rotation=angle,
            rotation_mode="anchor",
            color="blue", fontweight="bold", fontsize=9,
            ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.2")
        )

    ax.minorticks_on()
    ax.set_aspect("equal")
    ax.set_xlabel("Temperature [°C]", fontsize=13)
    ax.set_ylabel("Pressure [bar]", fontsize=13)
    ax.set_title("N2O Pressure", fontsize=14)

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
    plt.gcf().savefig("n2o_pressure.png")
    plt.show()



if __name__ == "__main__":
    main()
