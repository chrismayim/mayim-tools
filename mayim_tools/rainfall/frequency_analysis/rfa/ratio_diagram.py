"""
L-moment ratio diagram diagnostic (Hosking & Wallis 1997; Vogel &
Fennessey 1993, "L moment diagrams should replace product moment
diagrams").

Turns the usual qualitative/visual L-moment ratio diagram into a
quantitative goodness-of-fit ranking: for each candidate distribution,
compute the L-kurtosis (tau4) it WOULD predict given the sample's
ACTUAL L-skewness (tau3), then measure the vertical distance to the
sample's ACTUAL tau4. Smaller distance = better shape agreement.

This does not replace looking at the fitted quantiles - it answers a
narrower, useful question: "given how skewed this duration's annual
maxima are, which candidate distribution's kurtosis behaviour best
matches what's actually observed?"
"""

from __future__ import annotations

from dataclasses import dataclass

from .distributions import (
    GUMBEL_TAU4,
    gev_tau3_tau4_exact,
    glo_tau3_tau4_exact,
    pt3_tau4_from_tau3,
)


@dataclass
class RatioDiagramEntry:
    distribution: str
    tau4_theoretical: float
    tau4_sample: float
    distance: float  # |tau4_theoretical - tau4_sample|


@dataclass
class RatioDiagramResult:
    duration_label: str
    tau3_sample: float
    tau4_sample: float
    entries: list  # list[RatioDiagramEntry], sorted best-fit first
    recommended: str  # distribution name with the smallest distance


def compare_distributions(
    duration_label: str, tau3: float, tau4: float, kappa_gev: float | None
) -> RatioDiagramResult:
    """Computes each candidate distribution's theoretical tau4 given
    the sample's tau3, compares to the sample's actual tau4, and ranks.

    kappa_gev is passed in (already fitted elsewhere) rather than
    re-derived here, since GEV's kappa-from-tau3 approximation is only
    ever computed once per duration - no need to duplicate that fit.
    """
    entries = []

    # Gumbel: fixed point, no shape parameter - distance uses the FULL
    # (tau3, tau4) Euclidean-style comparison in the tau4 dimension
    # only (matching how the other distances are computed, so ranking
    # stays on a consistent basis), but note Gumbel's tau3 essentially
    # never matches the sample's tau3 unless the data is very close to
    # Gumbel-shaped - the tau4 distance alone can understate a Gumbel
    # mismatch when tau3 disagrees sharply. Flagged in the exported
    # diagnostic table (both tau3 and tau4 are reported, not just the
    # distance) so this is visible rather than hidden.
    entries.append(
        RatioDiagramEntry("GUMBEL", GUMBEL_TAU4, tau4, abs(GUMBEL_TAU4 - tau4))
    )

    if kappa_gev is not None:
        _, t4_gev = gev_tau3_tau4_exact(kappa_gev)
        entries.append(RatioDiagramEntry("GEV", t4_gev, tau4, abs(t4_gev - tau4)))

    kappa_glo = -tau3  # exact GLO relationship
    _, t4_glo = glo_tau3_tau4_exact(kappa_glo)
    entries.append(RatioDiagramEntry("GLO", t4_glo, tau4, abs(t4_glo - tau4)))

    t4_pt3 = pt3_tau4_from_tau3(tau3)
    entries.append(RatioDiagramEntry("LP3", t4_pt3, tau4, abs(t4_pt3 - tau4)))

    entries.sort(key=lambda e: e.distance)
    recommended = entries[0].distribution

    return RatioDiagramResult(
        duration_label=duration_label,
        tau3_sample=tau3,
        tau4_sample=tau4,
        entries=entries,
        recommended=recommended,
    )
