"""
Flood and rainfall frequency terms used across Mayim Tools.

Follows the Mayim Consulting Engineers convention "Common terminology used
in drainage design and waterway management" (1 March 2025):

    EY            exceedances per year
    AEP           annual exceedance probability, reported as a percentage
                  ("5% AEP")
    AEP (1 in x)  the reciprocal of the AEP
    ARI           average recurrence interval (years)

    (1) EY = 1 / AEP(1 in x)        (EY expressed from AEP (1 in x))
    (2) AEP (1 in x) = 1 / AEP
    (3) AEP (1 in x) = exp(EY) / (exp(EY) - 1)
    (4) ARI = 1 / (-ln(1 - AEP))
    (5) AEP = (exp(1/ARI) - 1) / exp(1/ARI)

The return period T of an annual-maximum series (every DDF table in this
suite: Design Rainfall, Precipitation data to DDF, SANRAL) is AEP (1 in x),
so AEP = 1/T exactly: T = 2 years is 50% AEP and T = 100 years is 1% AEP.
ARI is a different quantity (the mean interval between exceedances of a
partial series); converting between AEP and ARI uses equations (4) and (5),
and the two differ noticeably above about 5% AEP (50% AEP is 1.44 years
ARI; 2 years ARI is 39.35% AEP).

Every tool reports frequency as % AEP: "2% AEP" in CSV headers, layer
fields and names ("2pc AEP" where '%' is not allowed), "2% AEP (1 in 50)"
in reports. Frequencies are entered as AEP % ("50, 20, 10, 2, 1").
Internally the DDF tables keep T = AEP (1 in x), the annual-maximum
return period. Older DDF CSVs with '50yr Depth (mm)' headers still read.
"""

from __future__ import annotations

import math
import re


def aep_from_one_in_x(one_in_x: float) -> float:
    """AEP (%) of an annual-maximum return period T = AEP (1 in x): 100/T."""
    if one_in_x <= 0:
        raise ValueError("AEP (1 in x) must be greater than 0")
    return 100.0 / one_in_x


def one_in_x_from_aep(aep_percent: float) -> float:
    """AEP (1 in x) of an AEP given in %: 100/AEP."""
    if not 0 < aep_percent <= 100:
        raise ValueError("AEP must be greater than 0% and at most 100%")
    return 100.0 / aep_percent


def ari_from_aep(aep_percent: float) -> float:
    """Equation (4): ARI (years) = 1 / -ln(1 - AEP)."""
    p = aep_percent / 100.0
    if not 0 < p < 1:
        raise ValueError("AEP must be greater than 0% and less than 100%")
    return 1.0 / -math.log(1.0 - p)


def aep_from_ari(ari_years: float) -> float:
    """Equation (5): AEP (%) = (exp(1/ARI) - 1) / exp(1/ARI)."""
    if ari_years <= 0:
        raise ValueError("ARI must be greater than 0")
    e = math.exp(1.0 / ari_years)
    return 100.0 * (e - 1.0) / e


def ey_from_aep(aep_percent: float) -> float:
    """Exceedances per year of an AEP: -ln(1 - AEP), the inverse of
    equation (3); equals 1/ARI."""
    return 1.0 / ari_from_aep(aep_percent)


def aep_from_ey(ey: float) -> float:
    """Equation (3): AEP (%) = 100 / AEP(1 in x) = 100 (exp(EY) - 1) / exp(EY)."""
    if ey <= 0:
        raise ValueError("EY must be greater than 0")
    return 100.0 * (math.exp(ey) - 1.0) / math.exp(ey)


def format_aep(aep_percent: float) -> str:
    """The AEP number as shown in labels: whole numbers without decimals
    (50, 20, 1), otherwise up to two decimals for values of 1% and more
    (33.33, 18.13) and three significant figures below 1% (0.5, 0.2, 0.05).
    """
    v = float(aep_percent)
    if abs(v - round(v)) < 1e-9:
        return f"{round(v):d}"
    if v >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.3g}"


def aep_label(aep_percent: float) -> str:
    """'5% AEP'."""
    return f"{format_aep(aep_percent)}% AEP"


def aep_label_from_one_in_x(one_in_x: float) -> str:
    """'2% AEP' for T = 50."""
    return aep_label(aep_from_one_in_x(one_in_x))


def aep_label_long(aep_percent: float, nbsp: bool = False) -> str:
    """'2% AEP (1 in 50)' - the report form. With ``nbsp`` the two parts
    are held together by non-breaking spaces, so a narrow table header
    breaks only between '2% AEP' and '(1 in 50)'."""
    x = one_in_x_from_aep(aep_percent)
    text = f"{aep_label(aep_percent)} (1 in {_fmt_x(x)})"
    if nbsp:
        a, b = text.split(" (", 1)
        text = a.replace(" ", "\u00a0") + " (" + b.replace(" ", "\u00a0")
    return text


def aep_label_long_from_one_in_x(one_in_x: float, nbsp: bool = False) -> str:
    """'2% AEP (1 in 50)' for T = 50."""
    return aep_label_long(aep_from_one_in_x(one_in_x), nbsp)


def aep_header(one_in_x: float, n_columns: int = 1) -> str:
    """Report table column header of AEP (1 in x). Up to 5 frequency
    columns: '2% AEP' over '(1 in 50)' (a line break between them); more
    columns are too narrow for the bracket, so just '2%' over 'AEP' (the
    caption or the text gives the 1 in x)."""
    aep = aep_from_one_in_x(one_in_x)
    if n_columns > 5:
        return f"{format_aep(aep)}%\nAEP"
    return (
        aep_label(aep).replace(" ", "\u00a0")
        + "\n(1\u00a0in\u00a0"
        + _fmt_x(one_in_x_from_aep(aep))
        + ")"
    )


def aep_name(aep_percent: float) -> str:
    """'2pc AEP' - for names that must avoid '%' (HEC-HMS and SWMM
    element and file names)."""
    return f"{format_aep(aep_percent)}pc AEP"


def _fmt_x(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return f"{round(x):d}"
    return f"{x:.3g}"


def aep_depth_header(aep_percent: float, kind: str = "Depth") -> str:
    """DDF CSV column header: '2% AEP Depth (mm)', '2% AEP Lower (mm)'."""
    return f"{format_aep(aep_percent)}% AEP {kind} (mm)"


_AEP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*AEP", re.IGNORECASE)
_ONE_IN_RE = re.compile(r"\b1\s*(?:in|:)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_YR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:yr|year)", re.IGNORECASE)


def one_in_x_of_label(text: str) -> float | None:
    """AEP (1 in x) of a column header or label: '2% AEP Depth (mm)',
    '50yr Depth (mm)' (older files), '1 in 50' or '1:50'. None if the
    text names no frequency."""
    s = str(text)
    m = _AEP_RE.search(s)
    if m:
        t = 100.0 / float(m.group(1))
        # '33.33% AEP' is 1 in 3: undo the rounding of the label
        return float(round(t)) if abs(t - round(t)) < 0.005 * t else t
    m = _ONE_IN_RE.search(s)
    if m:
        return float(m.group(1))
    m = _YR_RE.search(s)
    if m:
        return float(m.group(1))
    return None


def parse_aep_list(text: str) -> list[float]:
    """AEPs (%) from a comma / space separated entry: '50, 20, 10, 2, 1'
    (a trailing '%' is allowed). Raises ValueError on a bad value."""
    out = []
    for part in re.split(r"[,;\s]+", str(text or "").strip()):
        if not part:
            continue
        v = float(part.rstrip("%"))
        if not 0 < v < 100:
            raise ValueError(f"AEP {part} is not between 0% and 100%")
        out.append(v)
    return out


def one_in_x_list_from_aeps(aeps) -> list[float]:
    """T = AEP (1 in x) of each AEP (%), with label rounding undone
    (33.33% -> 3)."""
    return [one_in_x_of_label(f"{format_aep(a)}% AEP") for a in aeps]
