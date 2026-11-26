"""
ISO 286 limits and fits, nominal sizes up to 500 mm: standard tolerance grades
IT1-IT14 and the fundamental deviations of the common shaft (c-u) and hole
(C-U) letters. The tables are in mm and µm; the functions take and return SI
(metres), like the rest of the framework.

    D = u.mm_(6)
    it(7, D)                -> 12e-6                  zone width [m]
    deviations("H7", D)     -> (0.0, 12e-6)           (lower, upper) [m]
    limits("s6", D)         -> mm(6, tol=(0.019, 0.027)), for build()s
    fit("H7/s6", D)         -> prints (in µm) and returns the clearance [m]
    table(D, "H7", "s6")    -> prints the limits of each class

Sizes are "over the lower step, up to and including the upper". IT01/IT0,
grades above 14, the letters a, b, j, v-zc (and A, B, J, V-ZC), and sizes over
500 mm aren't covered.
"""

from bisect import bisect_left
import re

import spec as u

__all__ = ["deviations", "limits", "fit"]


# Main size steps [mm] (upper bounds), for IT and the unsubdivided letters.
MAIN = (3, 6, 10, 18, 30, 50, 80, 120, 180, 250, 315, 400, 500)
# Fine steps, where c, r, s, t, u subdivide.
FINE = (3, 6, 10, 14, 18, 24, 30, 40, 50, 65, 80, 100, 120, 140, 160, 180,
        200, 225, 250, 280, 315, 355, 400, 450, 500)

# Standard tolerance grades [µm], per MAIN step.
IT = {
    1: (0.8, 1, 1, 1.2, 1.5, 1.5, 2, 2.5, 3.5, 4.5, 6, 7, 8),
    2: (1.2, 1.5, 1.5, 2, 2.5, 2.5, 3, 4, 5, 7, 8, 9, 10),
    3: (2, 2.5, 2.5, 3, 4, 4, 5, 6, 8, 10, 12, 13, 15),
    4: (3, 4, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20),
    5: (4, 5, 6, 8, 9, 11, 13, 15, 18, 20, 23, 25, 27),
    6: (6, 8, 9, 11, 13, 16, 19, 22, 25, 29, 32, 36, 40),
    7: (10, 12, 15, 18, 21, 25, 30, 35, 40, 46, 52, 57, 63),
    8: (14, 18, 22, 27, 33, 39, 46, 54, 63, 72, 81, 89, 97),
    9: (25, 30, 36, 43, 52, 62, 74, 87, 100, 115, 130, 140, 155),
    10: (40, 48, 58, 70, 84, 100, 120, 140, 160, 185, 210, 230, 250),
    11: (60, 75, 90, 110, 130, 160, 190, 220, 250, 290, 320, 360, 400),
    12: (100, 120, 150, 180, 210, 250, 300, 350, 400, 460, 520, 570, 630),
    13: (140, 180, 220, 270, 330, 390, 460, 540, 630, 720, 810, 890, 970),
    14: (250, 300, 360, 430, 520, 620, 740, 870, 1000, 1150, 1300, 1400,
         1550),
}

# Shaft upper deviations es [µm] (c-h), per MAIN step unless FINE.
ES = {
    "d": (-20, -30, -40, -50, -65, -80, -100, -120, -145, -170, -190, -210,
          -230),
    "e": (-14, -20, -25, -32, -40, -50, -60, -72, -85, -100, -110, -125,
          -135),
    "f": (-6, -10, -13, -16, -20, -25, -30, -36, -43, -50, -56, -62, -68),
    "g": (-2, -4, -5, -6, -7, -9, -10, -12, -14, -15, -17, -18, -20),
    "h": (0,)*13,
}
ES_FINE = {
    "c": (-60, -70, -80, -95, -95, -110, -110, -120, -130, -140, -150, -170,
          -180, -200, -210, -230, -240, -260, -280, -300, -330, -360, -400,
          -440, -480),
}
# Shaft lower deviations ei [µm] (k-u). k is for grades 4-7 (else 0).
EI = {
    "k": (0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5),
    "m": (2, 4, 6, 7, 8, 9, 11, 13, 15, 17, 20, 21, 23),
    "n": (4, 8, 10, 12, 15, 17, 20, 23, 27, 31, 34, 37, 40),
    "p": (6, 12, 15, 18, 22, 26, 32, 37, 43, 50, 56, 62, 68),
}
EI_FINE = {
    "r": (10, 15, 19, 23, 23, 28, 28, 34, 34, 41, 43, 51, 54, 63, 65, 68, 77,
          80, 84, 94, 98, 108, 114, 126, 132),
    "s": (14, 19, 23, 28, 28, 35, 35, 43, 43, 53, 59, 71, 79, 92, 100, 108,
          122, 130, 140, 158, 170, 190, 208, 232, 252),
    "t": (None,)*6 + (41, 48, 54, 66, 75, 91, 104, 122, 134, 146, 166, 180,
          196, 218, 240, 268, 294, 330, 360),
    "u": (18, 23, 28, 33, 33, 41, 48, 60, 70, 87, 102, 124, 144, 170, 190,
          210, 236, 258, 284, 315, 350, 390, 435, 490, 540),
}



def _step(D, steps):
    # Index of D's [mm] size step.
    if not 0 < D <= steps[-1]:
        raise ValueError(f"{D:g} mm is outside 0-{steps[-1]} mm")
    return bisect_left(steps, D)

def _mm(D):
    return round(D*1e3, 9) # m -> mm, clean at the step bounds

def _it(grade, D):
    # Standard tolerance [µm] of IT`grade` at D [mm].
    if grade not in IT:
        raise ValueError(f"IT{grade} isn't covered (IT1-IT14)")
    return IT[grade][_step(D, MAIN)]

def it(grade, D):
    """ Standard tolerance [m] of IT`grade` at D [m]. """
    return round(_it(grade, _mm(D))*1e-6, 12)

def _shaft(letter, grade, D):
    # (es, ei) [µm] of a shaft class at D [mm].
    T = _it(grade, D)
    if letter == "js":
        T2 = T - 1 if 7 <= grade <= 11 and T % 2 else T # odd -> even below
        return T2/2, -T2/2
    if letter in ES or letter in ES_FINE:
        es = (ES[letter][_step(D, MAIN)] if letter in ES
              else ES_FINE[letter][_step(D, FINE)])
        return es, es - T
    if letter in EI or letter in EI_FINE:
        ei = (EI[letter][_step(D, MAIN)] if letter in EI
              else EI_FINE[letter][_step(D, FINE)])
        if ei is None:
            raise ValueError(f"{letter}{grade} doesn't exist at {D} mm")
        if letter == "k" and not 4 <= grade <= 7:
            ei = 0
        return ei + T, ei
    raise ValueError(f"shaft letter {letter!r} isn't covered")

def _hole(letter, grade, D):
    # (ES, EI) [µm] of a hole class at D [mm], from the shaft letter's.
    T = _it(grade, D)
    low = letter.lower()
    if low == "js":
        es, ei = _shaft("js", grade, D)
        return es, ei
    if low in ES or low in ES_FINE: # A-H: EI = -es.
        EI_ = -_shaft(low, grade, D)[0]
        return EI_ + T, EI_
    # K-U: ES = -ei (general rule), + delta (special rule) for K, M, N up to
    # IT8 and P-U up to IT7. Delta is 0 up to 3 mm.
    if low == "k":
        ei = EI["k"][_step(D, MAIN)] if grade <= 8 else 0
    else:
        ei = _shaft(low, 7, D)[1] # ei doesn't depend on the grade here
    special = grade <= 8 if low in "kmn" else grade <= 7
    delta = _it(grade, D) - _it(grade - 1, D) if special and D > 3 else 0
    ES_ = -ei + delta
    if low == "n" and grade > 8 and D > 3:
        ES_ = 0
    return ES_, ES_ - T

def _parse(class_):
    m = re.fullmatch(r"([A-Za-z]{1,2})(\d{1,2})", class_.strip())
    if not m:
        raise ValueError(f"can't read tolerance class {class_!r}")
    return m.group(1), int(m.group(2))

def _deviations(class_, D):
    # (lower, upper) [µm] of `class_` at D [mm].
    letter, grade = _parse(class_)
    hi, lo = (_hole if letter[0].isupper() else _shaft)(letter, grade, D)
    return lo, hi

def deviations(class_, D):
    """ (lower, upper) deviations [m] of tolerance class `class_` (e.g. "H7",
    "s6", "JS7") at nominal size D [m]. """
    lo, hi = _deviations(class_, _mm(D))
    return round(lo*1e-6, 9), round(hi*1e-6, 9)

def limits(class_, D, centred=False):
    """ D [m] with `class_`'s limits as its tolerance (SI). `centred` puts the
    nominal mid-band, with a symmetric tolerance. """
    Dm = _mm(D)
    lo, hi = (x/1000 for x in _deviations(class_, Dm)) # mm
    if centred:
        return u.mm_(round(Dm + (lo + hi)/2, 6), tol=round((hi - lo)/2, 6))
    return u.mm_(Dm, tol=(round(lo, 6), round(hi, 6)))

def fit(spec_, D, show=True):
    """ (min, max) diametral clearance [m] of fit "HOLE/SHAFT" (e.g.
    "H7/s6") at D [m]: negative is interference. """
    if "/" not in spec_:
        raise ValueError("expected a hole/shaft fit for spec")
    hole, shaft = (c.strip() for c in spec_.split("/"))
    H, S = deviations(hole, D), deviations(shaft, D)
    lo, hi = round(H[0] - S[1], 9), round(H[1] - S[0], 9)
    if show:
        kind = ("clearance" if lo >= 0 else "interference" if hi <= 0
                else "transition")
        print(f"{spec_} at {_mm(D):g} mm: hole {H[0]*1e6:+g}/{H[1]*1e6:+g}, "
              f"shaft {S[0]*1e6:+g}/{S[1]*1e6:+g} µm; clearance "
              f"{lo*1e6:+g} to {hi*1e6:+g} µm ({kind})")
    return lo, hi

def table(D, *classes):
    """ Prints each class's limits at D [m], in µm and mm. """
    Dm = _mm(D)
    for class_ in classes:
        lo, hi = _deviations(class_, Dm)
        print(f"{class_:>5} at {Dm:g} mm: {lo:+8g} {hi:+8g} µm  "
              f"({Dm + lo/1000:.4f} - {Dm + hi/1000:.4f} mm)")
