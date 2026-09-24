"""Plain-text catchment description for the annexure report - no QGIS or
python-docx dependency, so it can be unit-tested directly.

The description is deliberately guideline-neutral: it classifies the
catchment against widely used, citable geomorphometric classes and gives
general modelling considerations, and leaves the choice of a specific
design method to the relevant local guideline.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Classification thresholds (each with its source; see REFERENCES)
# ---------------------------------------------------------------------------

# Ponce (1989): small < 2.5 km2, midsize 2.5-250 km2, large > 250 km2.
SIZE_CLASSES = [
    (2.5, "small"),
    (250.0, "midsize"),
    (math.inf, "large"),
]

# Elongation ratio (Schumm 1956), descriptive classes as commonly applied.
ELONGATION_CLASSES = [
    (0.5, "more elongated"),
    (0.7, "elongated"),
    (0.8, "less elongated"),
    (0.9, "oval"),
    (math.inf, "circular"),
]

# Mean slope classes, % (FAO 2006).
MEAN_SLOPE_CLASSES = [
    (2.0, "level to very gently sloping"),
    (5.0, "gently sloping"),
    (10.0, "sloping"),
    (15.0, "strongly sloping"),
    (30.0, "moderately steep"),
    (60.0, "steep"),
    (math.inf, "very steep"),
]

REFERENCES = [
    "Bransby-Williams, G. (1922). Flood discharge and the dimensions of "
    "spillways in India. The Engineer, 134, 321-322.",
    "FAO (2006). Guidelines for Soil Description, 4th edition. Food and "
    "Agriculture Organization of the United Nations, Rome.",
    "Gravelius, H. (1914). Flusskunde. Göschen, Berlin.",
    "Horn, B.K.P. (1981). Hill shading and the reflectance map. Proceedings "
    "of the IEEE, 69(1), 14-47.",
    "Horton, R.E. (1932). Drainage-basin characteristics. Transactions of "
    "the American Geophysical Union, 13, 350-361.",
    "Horton, R.E. (1945). Erosional development of streams and their "
    "drainage basins. Geological Society of America Bulletin, 56, 275-370.",
    "Kirpich, Z.P. (1940). Time of concentration of small agricultural "
    "watersheds. Civil Engineering, 10(6), 362.",
    "Melton, M.A. (1965). The geomorphic and paleoclimatic significance of "
    "alluvial deposits in southern Arizona. Journal of Geology, 73, 1-38.",
    "Miller, V.C. (1953). A quantitative geomorphic study of drainage basin "
    "characteristics in the Clinch Mountain area, Virginia and Tennessee. "
    "Technical Report 3, Columbia University, New York.",
    "O'Callaghan, J.F. & Mark, D.M. (1984). The extraction of drainage "
    "networks from digital elevation data. Computer Vision, Graphics, and "
    "Image Processing, 28, 323-344.",
    "Pike, R.J. & Wilson, S.E. (1971). Elevation-relief ratio, hypsometric "
    "integral, and geomorphic area-altitude analysis. Geological Society of "
    "America Bulletin, 82, 1079-1084.",
    "Ponce, V.M. (1989). Engineering Hydrology: Principles and Practices. "
    "Prentice Hall, Englewood Cliffs, NJ.",
    "Schumm, S.A. (1956). Evolution of drainage systems and slopes in "
    "badlands at Perth Amboy, New Jersey. Geological Society of America "
    "Bulletin, 67, 597-646.",
    "Strahler, A.N. (1952). Hypsometric (area-altitude) analysis of "
    "erosional topography. Geological Society of America Bulletin, 63, "
    "1117-1142.",
    "Strahler, A.N. (1957). Quantitative analysis of watershed "
    "geomorphology. Transactions of the American Geophysical Union, 38(6), "
    "913-920.",
    "Strahler, A.N. (1958). Dimensional analysis applied to fluvially "
    "eroded landforms. Geological Society of America Bulletin, 69, 279-300.",
    "Strahler, A.N. (1964). Quantitative geomorphology of drainage basins "
    "and channel networks. In V.T. Chow (ed.), Handbook of Applied "
    "Hydrology, Section 4-II. McGraw-Hill, New York.",
    "USBR (1973). Design of Small Dams, 2nd edition. United States Bureau "
    "of Reclamation, Washington DC.",
    "USDA-NRCS (2007). National Engineering Handbook, Part 630 Hydrology, "
    "Chapter 16: Hydrographs. United States Department of Agriculture.",
    "Wilford, D.J., Sakals, M.E., Innes, J.L., Sidle, R.C. & Bergerud, W.A. "
    "(2004). Recognition of debris flow, debris flood and flood hazard "
    "through watershed morphometrics. Landslides, 1, 61-66.",
]


def classify(value, classes):
    """First label whose upper bound the value falls below."""
    if value is None:
        return None
    for upper, label in classes:
        if value < upper:
            return label
    return classes[-1][1]


def _fmt(value, digits=2, unit=""):
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}{unit}"


def _pct(value, digits=1):
    return "n/a" if value is None else f"{100.0 * value:.{digits}f} %"


def hypsometric_stage(hi):
    """Strahler (1952): > 0.60 youthful, 0.35-0.60 mature, < 0.35 old."""
    if hi is None:
        return None
    if hi > 0.60:
        return "youthful (in disequilibrium)"
    if hi >= 0.35:
        return "mature (in equilibrium)"
    return "old (monadnock) stage"


def melton_screening(melton, length_km):
    """Wilford et al. (2004): flood (< 0.30), debris flood (0.30-0.60, or
    > 0.60 with length >= 2.7 km), debris flow (> 0.60 and length < 2.7 km)."""
    if melton is None or length_km is None:
        return None
    if melton < 0.30:
        return "flood"
    if melton <= 0.60 or length_km >= 2.7:
        return "debris flood"
    return "debris flow"


def tc_values(attrs):
    names = [
        ("Kirpich", attrs.get("tc_kirp_mn")),
        ("USBR", attrs.get("tc_usbr_mn")),
        ("Bransby-Williams", attrs.get("tc_bw_mn")),
    ]
    return [(n, v) for n, v in names if v is not None]


def describe_catchment(attrs: dict, detail: dict) -> dict:
    """Returns {"overview": str, "geomorphology": [str], "modelling": [str]}."""
    a = attrs
    area = a["area_km2"]
    size = classify(area, SIZE_CLASSES)
    outlet = "pour line" if a["out_type"] == "line" else "pour point"
    parts = []

    overview = (
        f"Catchment {a['outlet_id']} drains {_fmt(area, 3)} km² "
        f"({_fmt(a['area_ha'], 1)} ha) to a {outlet} at "
        f"E {_fmt(a['x_out'], 1)}, N {_fmt(a['y_out'], 1)}"
        + (
            f", outlet elevation {_fmt(a['z_outlet'], 1)} m"
            if a["z_outlet"] is not None
            else ""
        )
        + f". Elevations range from {_fmt(a['z_min'], 1)} m to "
        f"{_fmt(a['z_max'], 1)} m (relief {_fmt(a['relief_m'], 1)} m, mean "
        f"{_fmt(a['z_mean'], 1)} m). The longest flow path is "
        f"{_fmt(a['lfp_km'], 3)} km long, with a 10-85 slope of "
        f"{_pct(a['s_1085'], 2)} and an equal-area slope of {_pct(a['s_ea'], 2)}."
    )
    if a["mode"] == "incremental" and a["ds_ids"]:
        overview += (
            f" It is an incremental sub-catchment draining into outlet(s) "
            f"{a['ds_ids']}."
        )

    # --- size ---------------------------------------------------------------
    size_text = {
        "small": "rainfall can usually be taken as uniform in space and time "
        "over the catchment, overland flow is a large part of the response and "
        "channel storage is small",
        "midsize": "rainfall can usually still be taken as uniform in space but "
        "not in time, and the response is increasingly governed by the channel "
        "network",
        "large": "rainfall varies in both space and time and channel storage "
        "and routing dominate the response",
    }[size]
    parts.append(
        f"Scale. By the classification of Ponce (1989), this is a {size} "
        f"catchment (small < 2.5 km², midsize 2.5-250 km², large > 250 km²). "
        f"For a {size} catchment, {size_text}."
    )

    # --- shape --------------------------------------------------------------
    re_ = a.get("elong_r")
    shape = classify(re_, ELONGATION_CLASSES)
    if shape:
        tendency = (
            "a relatively quick, peaked response, since flow from most of the "
            "area reaches the outlet at similar times"
            if re_ >= 0.8
            else "a flatter, longer hydrograph than a compact catchment of the "
            "same area, since travel times from different parts are spread out"
        )
        parts.append(
            f"Shape. The elongation ratio of {_fmt(re_, 2)} (Schumm 1956) "
            f"classes the catchment as {shape}; the form factor is "
            f"{_fmt(a['form_f'], 2)} (Horton 1932), the circularity ratio "
            f"{_fmt(a['circ_r'], 2)} (Miller 1953) and the Gravelius "
            f"compactness coefficient {_fmt(a['gravel_kc'], 2)}. This shape "
            f"suggests {tendency}. The basin length (outlet to the farthest "
            f"point of the catchment) is {_fmt(a['lb_km'], 3)} km. Circularity "
            "and compactness use the stepped cell outline, which lengthens the "
            "perimeter; treat them as indicative."
        )

    # --- relief and slope -----------------------------------------------------
    mean_slope = a.get("s_mean_pc")
    slope_class = classify(mean_slope, MEAN_SLOPE_CLASSES)
    classes = detail.get("slope_classes", [])
    steep = sum(c["pct"] for c in classes[4:]) if classes else None
    flat = classes[0]["pct"] if classes else None
    if slope_class:
        parts.append(
            f"Relief and slope. The mean catchment slope is "
            f"{_fmt(mean_slope, 1)} %, which is {slope_class} in the FAO (2006) "
            f"classes; {_fmt(flat, 0)} % of the area is flatter than 2 % and "
            f"{_fmt(steep, 0)} % is steeper than 15 %. The relief ratio "
            f"(Schumm 1956) is {_fmt(a['relief_r'], 4)} and the average slope "
            f"of the longest flow path is {_pct(a['s_avg'], 2)}."
            + (
                " Much of the catchment is steep, so expect high flow "
                "velocities and a short time to peak."
                if steep is not None and steep >= 30
                else ""
            )
            + (
                " Much of the catchment is flat, so flow paths in the DEM are "
                "sensitive to small elevation errors, and ponding or sheet flow "
                "may matter."
                if flat is not None and flat >= 50
                else ""
            )
        )

    # --- hypsometry -----------------------------------------------------------
    stage = hypsometric_stage(a.get("hyps_int"))
    if stage:
        parts.append(
            f"Hypsometry. The hypsometric integral is {_fmt(a['hyps_int'], 3)}, "
            f"which Strahler (1952) associates with a {stage} landscape. "
            + (
                "A large share of the area sits high in the catchment, which "
                "often goes with active incision and higher sediment supply."
                if a["hyps_int"] > 0.60
                else (
                    "Area is spread fairly evenly over the elevation range."
                    if a["hyps_int"] >= 0.35
                    else "Most of the area sits low in the catchment, typical of a "
                    "well-eroded landscape with gentle lower slopes."
                )
            )
        )

    # --- drainage network -----------------------------------------------------
    thr = detail.get("stream_threshold_km2")
    if a.get("strm_ord"):
        bif = a.get("bif_r")
        bif_text = ""
        if bif is not None:
            bif_text = (
                f" The mean bifurcation ratio is {_fmt(bif, 2)}; values of about "
                "3 to 5 are typical of networks without strong geological "
                "control (Strahler 1964)"
                + (
                    ", so the higher value here may indicate structural control "
                    "or an elongated form."
                    if bif > 5
                    else "."
                )
            )
        parts.append(
            f"Drainage network. With streams defined as cells draining at least "
            f"{_fmt(thr, 3)} km², the network reaches Strahler order "
            f"{a['strm_ord']} with {a['n_strm']} stream segments and "
            f"{_fmt(a['strm_km'], 2)} km of channel: a drainage density of "
            f"{_fmt(a['dd_kmkm2'], 2)} km/km² and a stream frequency of "
            f"{_fmt(a['strm_freq'], 2)} per km².{bif_text} The mean length of "
            f"overland flow (Horton 1945) is about {_fmt(a['lo_km'], 3)} km. "
            "Because the network comes from a DEM threshold, density, order and "
            "frequency change with that threshold; compare them between "
            "catchments only at the same threshold and resolution."
        )

    # --- flow path ----------------------------------------------------------
    if a.get("sinuosity") is not None:
        parts.append(
            f"Main flow path. The longest flow path has a sinuosity of "
            f"{_fmt(a['sinuosity'], 2)} (path length over straight-line "
            f"distance). The distance from the outlet to the point on the path "
            f"nearest the centroid (Lca) is {_fmt(a['lca_km'], 3)} km. D8 paths "
            "step between cell centres in eight directions, so they run a few "
            "per cent longer than the true channel."
        )

    # --- aspect ---------------------------------------------------------------
    aspects = [x for x in detail.get("aspect", []) if x["label"] != "Flat"]
    if aspects:
        top = max(aspects, key=lambda x: x["pct"])
        parts.append(
            f"Aspect. The most common slope aspect is {top['label']} "
            f"({_fmt(top['pct'], 0)} % of the area)."
        )

    # --- debris screening ---------------------------------------------------
    screen = melton_screening(a.get("melton_r"), a.get("lb_km"))
    if screen:
        parts.append(
            f"Process screening. The Melton ruggedness ratio is "
            f"{_fmt(a['melton_r'], 3)} and the ruggedness number is "
            f"{_fmt(a['rugged_n'], 3)} (Strahler 1958). With the thresholds of "
            f"Wilford et al. (2004), the catchment screens as {screen}-prone. "
            "That method was calibrated on specific regions and is an "
            "indicative screen only."
        )

    return {
        "overview": overview,
        "geomorphology": parts,
        "modelling": modelling_considerations(a, size, screen),
    }


def modelling_considerations(a: dict, size: str, screen: str | None) -> list[str]:
    """General, guideline-neutral modelling points for this catchment."""
    points = []
    method = {
        "small": "Peak-flow methods that assume uniform rainfall over the "
        "catchment (rational-type methods) and single lumped hydrograph methods "
        "are generally suitable at this scale, subject to the limits set by the "
        "applicable local guideline.",
        "midsize": "A hydrograph method (unit hydrograph or runoff-routing) with "
        "the catchment split into sub-catchments linked by channel routing is "
        "generally preferred at this scale; single peak-flow formulae become "
        "less reliable as area grows.",
        "large": "A semi-distributed or distributed model with channel routing, "
        "spatially varying rainfall and areal reduction of point rainfall is "
        "generally needed at this scale.",
    }[size]
    points.append(method)

    tcs = tc_values(a)
    if tcs:
        lo = min(v for _, v in tcs)
        hi = max(v for _, v in tcs)
        listing = ", ".join(f"{n} {v:.1f} min" for n, v in tcs)
        points.append(
            f"Time of concentration estimates: {listing}. Estimates range from "
            f"{lo:.1f} to {hi:.1f} min"
            + (
                f" (a factor of {hi / lo:.1f}), so the choice of method matters"
                if lo > 0 and hi / lo >= 1.5
                else ""
            )
            + ". Each empirical formula is only valid within the conditions it "
            "was derived for (Kirpich 1940 comes from small, steep rural "
            "catchments), and none of these values separates overland from "
            "channel flow. Select or combine methods to suit the catchment and "
            "the governing guideline."
        )
        points.append(
            f"Design storm durations should bracket the critical duration; as a "
            f"starting range, test from about {0.5 * lo:.0f} min to "
            f"{2.0 * hi:.0f} min. For hydrograph models, use a computational "
            f"time step of no more than about 0.1 to 0.2 x Tc, i.e. "
            f"{0.1 * lo:.1f} to {0.2 * lo:.1f} min here (compare the unit "
            "hydrograph guidance of USDA-NRCS 2007, dt <= 0.29 x lag)."
        )

    if a.get("hole_ha", 0) > 0:
        points.append(
            f"{a['hole_ha']:.2f} ha inside the outline does not drain to the "
            "outlet in the DEM (depressions or unconditioned pits). Check whether "
            "these are real storages (dams, pans, sinks) to model explicitly, or "
            "DEM artefacts to remove by conditioning."
        )
    if size != "small" or (a.get("strm_ord") or 0) >= 4:
        points.append(
            "Consider sub-dividing the catchment at tributary confluences, "
            "changes in land use or slope, and points of interest; the tool's "
            "incremental mode produces non-overlapping sub-catchments with "
            "their downstream links for this."
        )
    if screen in ("debris flood", "debris flow"):
        points.append(
            "The morphometric screen points to sediment-laden flows. Consider "
            "bulking factors or a sediment/debris assessment where the "
            "consequences warrant it."
        )
    points.append(
        "Check the delineated boundary against imagery, survey and drainage "
        "infrastructure. Culverts, bridges, embankments and urban drainage are "
        "not in a bare-earth DEM and can move the divide."
    )
    return points
