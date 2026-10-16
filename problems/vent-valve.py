from math import pi, sqrt
import numpy as np
import matplotlib.pyplot as plt


# Hack-in the utility module.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utility"))

from oring import PistonSeal, RodSeal
import travel


m = 0.01936269 + 0.0089/3 # [kg] moving mass (plunger + piston + ~1/3 spring mass)

pressure = lambda P, D: P*pi/4*D**2

# bottom-side.
P_tank = 3.5e6 # [Pa]
x_tank = 0.5e-3 # [m] lift over which tank-pressure area grows from seat to full bore
F_tank_open = pressure(P_tank, 8e-3) # [N]
F_tank_closed = pressure(P_tank, 8e-3/sqrt(2)) # [N]
def F_tank_x(x):
    # tank force ramps from seat area to full bore area over the first x_seal of lift
    s = min(max(x / x_tank, 0.0), 1.0)
    return F_tank_closed + s*(F_tank_open - F_tank_closed)

# top-side.
D_pneumo = 31.5e-3 # [m]
P_pneumo = 1e6 # [Pa]
F_pneumo = pressure(P_pneumo, D_pneumo) # [N]
tau_pneumo_venting = 5e-3 # [s] pneumatic vent time constant
tau_pneumo_filling = 5e-3 # [s] pneumatic fill time constant
def F_pneumo_venting_t(t):
    return F_pneumo * np.exp(-t / tau_pneumo_venting)
def F_pneumo_filling_t(t):
    return F_pneumo * (1 - np.exp(-t / tau_pneumo_filling))

# spring
ell_relaxed = 25e-3 # [m]
ell_open = 18.2e-3 # [m]
stroke = 5e-3 # [m]
ell_closed = ell_open - stroke # [m]
k = 25.5e3 # [N/m]
# x=0 is closed, x=stroke is open.
def F_spring_x(x):
    # spring extends as the valve opens: ell goes from ell_closed to ell_open
    return k * (ell_relaxed - (ell_closed + x))
F_spring_open = F_spring_x(0) # [N]
F_spring_closed = F_spring_x(stroke) # [N]

F_oring_static = 237.0 # [N]
F_oring_sliding = 89.0 # [N]

oring_pneumo = PistonSeal("Pneumo side",
        oring_ID=28.0e-3, oring_CS=2.0e-3,
        gland_ID=28.5e-3, gland_OD=31.5e-3, gland_Lz=3e-3,
        DP=30e5,
        material="FKM", hardness=75, lubricated=True)
oring_n2o = RodSeal("N2O side",
        oring_ID=7.5e-3, oring_CS=2.0e-3,
        gland_ID=7.9e-3, gland_OD=10.9e-3, gland_Lz=3e-3,
        DP=70e5,
        material="FKM", hardness=75, lubricated=True)



def main():
    print(f"F_pneumo: {F_pneumo}")
    print(f"F_tank_open: {F_tank_open}")
    print(f"F_tank_closed: {F_tank_closed}")
    print(f"F_spring_open: {F_spring_open}")
    print(f"F_spring_closed: {F_spring_closed}")
    print(f"F_oring_static: {F_oring_static}")
    print(f"F_oring_sliding: {F_oring_sliding}")

    # valve open static:
    #  +tank pressure
    #  +spring
    print("valve open static:")
    print(f" {F_tank_open + F_spring_open}")
    # valve closed static:
    #  -tank pressure
    #  -spring
    #  +pneumo pressure
    print("valve closed static:")
    print(f" {-F_tank_closed - F_spring_closed + F_pneumo}")
    # valve open, lets close:
    #  -tank pressure
    #  -spring
    #  -oring friction
    #  +pneumo pressure
    #  -ice??
    print("valve open, let's close:")
    print(f" {-F_tank_open - F_spring_open - F_oring_sliding + F_pneumo}")
    # valve closed, lets open:
    #  +tank pressure
    #  +spring
    #  -oring friction
    #  -ice??
    print("valve closed, let's open:")
    print(f" {F_tank_closed + F_spring_closed - F_oring_sliding}")


    def oring_friction(t, x, v):
        if v == 0:
            return oring_pneumo.F_friction_breakaway() \
                 + oring_n2o.F_friction_breakaway()
        return oring_pneumo.F_friction_running() \
                + oring_n2o.F_friction_running()

    opening = travel.travel(
        {
            "tank": lambda t, x, v: F_tank_x(x),
            "spring": lambda t, x, v: F_spring_x(x),
            "pneumatic": lambda t, x, v: -F_pneumo_venting_t(t),
            "o-ring": travel.Friction(oring_friction),
        },
        mass=m, x0=0.0, x_min=0.0, x_max=stroke,
        dt=2e-6, t_end=0.1, t_after_impact=0, title="Valve opening",
    )

    closing = travel.travel(
        {
            "tank": lambda t, x, v: F_tank_x(x),
            "spring": lambda t, x, v: F_spring_x(x),
            "pneumatic": lambda t, x, v: -F_pneumo_filling_t(t),
            "o-ring": travel.Friction(oring_friction),
        },
        mass=m, x0=stroke, x_min=0.0, x_max=stroke,
        dt=2e-6, t_end=0.1, t_after_impact=0, title="Valve closing",
    )

    print(opening)
    print(closing)
    opening.plot()
    closing.plot()
    plt.show()

if __name__ == "__main__":
    main()
