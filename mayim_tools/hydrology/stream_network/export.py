"""Field definitions and the sectioned CSV for the stream network tool (no
QGIS dependency). Numbers are written plainly, without thousands
separators."""

from __future__ import annotations

import csv
import math

from .core import NetworkResult

# (field, type code s/i/d, unit, description) - shared by the vector layer,
# the CSV and the report's definitions.
REACH_FIELDS = [
    ("link_id", "i", "-", "reach identifier"),
    ("ds_link", "i", "-", "reach this one flows into (empty at the outlet)"),
    (
        "network_id",
        "s",
        "-",
        "drainage network (root pour point, or N1, N2... by area)",
    ),
    (
        "outlet_id",
        "s",
        "-",
        "first pour point downstream (network id without pour points)",
    ),
    ("strahler", "i", "-", "Strahler (1957) stream order"),
    ("shreve", "i", "-", "Shreve (1966) magnitude: channel heads upstream"),
    ("horton", "i", "-", "Horton (1945) order: main stream keeps the downstream order"),
    ("hack", "i", "-", "Hack (1957) order: main stem 1, its tributaries 2, ..."),
    ("is_main", "i", "-", "1 if the reach is on the network's main stem (Hack 1)"),
    ("links_to_out", "i", "-", "number of reaches to the outlet, this one included"),
    (
        "dist_out_m",
        "d",
        "m",
        "channel distance from the reach's downstream end to the outlet",
    ),
    ("length_m", "d", "m", "reach length along the D8 path"),
    ("up_len_m", "d", "m", "longest channel length upstream, this reach included"),
    ("us_km2", "d", "km²", "upstream area at the reach's downstream end"),
    ("z_start", "d", "m", "elevation at the upstream end"),
    ("z_end", "d", "m", "elevation at the downstream end (junction or outlet)"),
    ("drop_m", "d", "m", "z_start - z_end"),
    ("slope", "d", "m/m", "drop / length"),
    ("sinuosity", "d", "-", "length / straight-line distance between the ends"),
    (
        "v_ms",
        "d",
        "m/s",
        "routing velocity (constant, or Manning with the minimum slope)",
    ),
    ("tt_min", "d", "min", "travel time length / velocity"),
    ("musk_k_h", "d", "h", "Muskingum K (= travel time)"),
    ("musk_x", "d", "-", "Muskingum X (weighting factor, user setting)"),
    ("n_sub", "i", "-", "Muskingum sub-reaches for the time step (about K / dt)"),
    (
        "dt_check",
        "s",
        "-",
        "Muskingum stability check 2KX <= dt <= 2K(1-X) per sub-reach",
    ),
]

NODE_FIELDS = [
    ("node_id", "i", "-", "node identifier"),
    ("node_type", "s", "-", "channel head, confluence, pour point or outlet"),
    ("network_id", "s", "-", "drainage network"),
    ("link_id", "i", "-", "reach starting here (or ending here, for the outlet)"),
    ("x", "d", "m", "easting"),
    ("y", "d", "m", "northing"),
    ("z", "d", "m", "elevation"),
    ("us_km2", "d", "km²", "upstream area"),
    ("n_inflow", "i", "-", "number of inflowing reaches"),
    ("strahler", "i", "-", "Strahler order of the reach"),
    ("shreve", "i", "-", "Shreve magnitude of the reach"),
]

NETWORK_FIELDS = [
    ("network_id", "-", "drainage network"),
    ("outlet_x", "m", "outlet easting"),
    ("outlet_y", "m", "outlet northing"),
    ("outlet_z", "m", "outlet elevation"),
    ("area_km2", "km²", "drainage area at the outlet"),
    ("n_reaches", "-", "number of reaches"),
    ("n_heads", "-", "number of channel heads (first-order sources)"),
    ("max_strahler", "-", "highest Strahler order"),
    ("magnitude", "-", "Shreve magnitude at the outlet"),
    ("total_length_km", "km", "total stream length"),
    ("drainage_density", "km/km²", "total stream length / area"),
    ("stream_frequency", "1/km²", "channel heads / area"),
    ("main_length_km", "km", "main-stem length (longest stream)"),
    ("main_relief_m", "m", "main-stem head elevation - outlet elevation"),
    ("main_slope_avg", "m/m", "main-stem relief / length"),
    ("main_slope_1085", "m/m", "main-stem 10-85 slope"),
    ("main_sinuosity", "-", "main-stem length / straight-line distance"),
    ("main_tt_min", "min", "sum of reach travel times along the main stem"),
    ("hack_h", "-", "Hack's law exponent h in L = c A^h (reaches)"),
    ("hack_c", "-", "Hack's law coefficient c (L km, A km²)"),
    ("hack_r2", "-", "coefficient of determination of the Hack fit"),
    ("z_max_stream", "m", "highest point of the main stem"),
]

ORDER_FIELDS = [
    ("network_id", "-", "drainage network"),
    ("order", "-", "Strahler order u"),
    ("n_streams", "-", "number of Strahler streams N_u"),
    ("total_length_km", "km", "total length of order-u streams"),
    ("mean_length_km", "km", "mean length of order-u streams"),
    ("mean_area_km2", "km²", "mean upstream area at the end of order-u streams"),
    ("mean_slope", "m/m", "mean slope of order-u streams"),
    ("mean_drop_m", "m", "mean drop of order-u streams"),
    ("rb_next", "-", "bifurcation ratio N_u / N_u+1"),
    ("rl_next", "-", "length ratio L_u+1 / L_u"),
    ("ra_next", "-", "area ratio A_u+1 / A_u"),
    ("rs_next", "-", "slope ratio S_u / S_u+1"),
]

FIT_FIELDS = [
    ("network_id", "-", "drainage network"),
    ("max_order", "-", "highest Strahler order"),
    ("rb_mean", "-", "mean consecutive bifurcation ratio"),
    ("rl_mean", "-", "mean consecutive length ratio"),
    ("ra_mean", "-", "mean consecutive area ratio"),
    ("rs_mean", "-", "mean consecutive slope ratio"),
    ("rb_fit", "-", "bifurcation ratio from log-linear fit of N_u on u"),
    ("rb_r2", "-", "R² of that fit"),
    ("rl_fit", "-", "length ratio from fit of mean length on u"),
    ("rl_r2", "-", "R² of that fit"),
    ("ra_fit", "-", "area ratio from fit of mean area on u"),
    ("ra_r2", "-", "R² of that fit"),
    ("rs_fit", "-", "slope ratio from fit of mean slope on u"),
    ("rs_r2", "-", "R² of that fit"),
]

DROP_FIELDS = [
    ("threshold_km2", "km²", "stream threshold tested"),
    ("max_order", "-", "highest Strahler order at this threshold"),
    ("n_first_order", "-", "number of first-order streams"),
    ("n_higher_order", "-", "number of higher-order streams"),
    ("mean_drop_first_m", "m", "mean drop of first-order streams"),
    ("mean_drop_higher_m", "m", "mean drop of higher-order streams"),
    ("sd_drop_first_m", "m", "standard deviation, first order"),
    ("sd_drop_higher_m", "m", "standard deviation, higher orders"),
    ("t_statistic", "-", "Welch t statistic for the difference in means"),
    (
        "passes",
        "-",
        "|t| < 2: no significant difference (consistent with constant drop)",
    ),
]


# Key to the short forms used in the column headings (written at the top
# of the CSV). Full per-field definitions are in the last section.
ABBREVIATIONS = [
    ("Units", "", ""),
    ("_m", "metres", "length_m, z_start, drop_m"),
    ("_km", "kilometres", "main_length_km, total_length_km"),
    ("_km2", "square kilometres (km²)", "us_km2, area_km2"),
    ("_ms", "metres per second (m/s)", "v_ms"),
    ("_min", "minutes", "tt_min"),
    ("_h", "hours", "musk_k_h"),
    ("Quantities", "", ""),
    ("z", "elevation (m)", "z_start, z_end, outlet_z"),
    ("x / y", "easting / northing (m)", "x, y, outlet_x"),
    ("us", "upstream: area draining to the point", "us_km2"),
    ("ds", "downstream: the reach flowed into", "ds_link"),
    ("up_len", "longest channel length upstream", "up_len_m"),
    ("dist_out", "channel distance to the network outlet", "dist_out_m"),
    ("links_to_out", "number of reaches to the outlet, this one included", ""),
    ("is_main", "1 = on the main stem (longest stream), 0 = not", ""),
    ("n_", "number (count) of", "n_streams, n_reaches, n_inflow"),
    ("n_sub", "number of Muskingum sub-reaches", ""),
    ("avg", "average (relief / length)", "main_slope_avg"),
    (
        "1085",
        "10-85 slope: between 10 % and 85 % of length from the outlet",
        "main_slope_1085",
    ),
    ("sd", "standard deviation", "sd_drop_first_m"),
    (
        "first / higher",
        "first-order / all higher-order Strahler streams",
        "mean_drop_first_m",
    ),
    ("Ordering", "", ""),
    ("strahler", "Strahler (1957) stream order", ""),
    ("shreve", "Shreve (1966) magnitude (channel heads upstream)", ""),
    ("horton", "Horton (1945) order", ""),
    ("hack", "Hack (1957) order; main stem = 1", ""),
    ("u", "Strahler order number (in definitions)", ""),
    ("Routing", "", ""),
    ("v", "velocity", "v_ms"),
    ("tt", "travel time", "tt_min"),
    ("musk_k / musk_x", "Muskingum K (storage constant) / X (weighting factor)", ""),
    ("dt", "computational time step", "dt_check"),
    ("Horton ratios and fits", "", ""),
    ("rb", "bifurcation ratio R_b = N_u / N_u+1", "rb_next, rb_fit"),
    ("rl", "length ratio R_L = L_u+1 / L_u", "rl_next, rl_fit"),
    ("ra", "area ratio R_A = A_u+1 / A_u", "ra_next, ra_fit"),
    ("rs", "slope ratio R_S = S_u / S_u+1", "rs_next, rs_fit"),
    ("_next", "ratio between this order and the next", "rb_next"),
    ("_mean", "mean of the consecutive-order ratios", "rb_mean"),
    ("_fit", "ratio from the log-linear regression over all orders", "rb_fit"),
    ("r2", "coefficient of determination R² of a fit", "rb_r2, hack_r2"),
    ("hack_h / hack_c", "exponent h / coefficient c in Hack's law L = c A^h", ""),
    ("Drop test", "", ""),
    ("t_statistic", "Welch two-sample t for first- vs higher-order mean drops", ""),
    ("passes", "yes when |t| < 2 (Tarboton et al. 1991)", ""),
]


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if abs(value) >= 1:
            text = f"{value:.4f}".rstrip("0").rstrip(".")
        else:
            text = f"{value:.6g}"
        return text
    return str(value)


def write_sectioned_csv(path: str, result: NetworkResult, context: dict) -> None:
    """One CSV with headed sections: run summary, drop test, networks,
    Horton statistics, Horton ratios, reaches (with routing inputs), nodes,
    main-stem profiles and field definitions."""
    s = result.settings
    routing = s.get("routing")
    with open(
        path, "w", newline="", encoding="utf-8-sig"
    ) as f:  # BOM: Excel shows ² correctly
        w = csv.writer(f)

        def section(title, fields, rows):
            w.writerow([title])
            w.writerow([name for name, *_ in fields])
            for row in rows:
                w.writerow([_fmt(row.get(name)) for name, *_ in fields])
            w.writerow([])

        w.writerow(["STREAM NETWORK EXTRACTION - RESULTS"])
        w.writerow([])
        w.writerow(
            ["ABBREVIATIONS USED IN COLUMN HEADINGS (full definitions in section 9)"]
        )
        w.writerow(["Abbreviation", "Meaning", "Example columns"])
        for abbr, meaning, example in ABBREVIATIONS:
            if not meaning:
                w.writerow([f"-- {abbr} --"])
            else:
                w.writerow([abbr, meaning, example])
        w.writerow([])
        w.writerow(["1. RUN SUMMARY"])
        w.writerow(["Item", "Value"])
        summary = [
            (
                "Tool",
                f"{context.get('tool', 'Stream Network')} {context.get('version', '')}",
            ),
            ("Run", context.get("run_time", "")),
            ("DEM", context.get("dem_path", "")),
            ("D8 pointer", context.get("pointer_path", "")),
            ("Pointer encoding", context.get("encoding", "")),
            ("CRS", context.get("crs", "")),
            ("Cell size (m)", f"{s['cell_size'][0]:g} x {s['cell_size'][1]:g}"),
            ("Grid (columns x rows)", f"{s['grid'][0]} x {s['grid'][1]}"),
            ("Pour points used", s.get("n_pour_points", 0)),
            ("Pour point snap radius (cells)", s.get("snap_radius_cells")),
            ("Stream threshold (km²)", s.get("threshold_km2")),
            (
                "Drop-test threshold (km², smallest with |t| < 2)",
                result.drop_test_threshold_km2,
            ),
            ("Minimum network area kept (km²)", s.get("min_network_km2")),
            ("Velocity method", routing.method if routing else ""),
            ("Constant velocity (m/s)", routing.velocity_ms if routing else ""),
            ("Manning n", routing.manning_n if routing else ""),
            ("Hydraulic radius (m)", routing.hydraulic_radius_m if routing else ""),
            ("Minimum routing slope (m/m)", routing.min_slope if routing else ""),
            ("Muskingum X", routing.muskingum_x if routing else ""),
            ("Computational time step (min)", routing.time_step_min if routing else ""),
            ("Networks", len(result.networks)),
            ("Reaches", len(result.reaches)),
        ]
        for k, v in summary:
            w.writerow([k, _fmt(v)])
        for msg in result.warnings:
            w.writerow(["Warning", msg])
        w.writerow([])

        section(
            "2. CONSTANT-DROP TEST (Tarboton, Bras & Rodriguez-Iturbe 1991)",
            DROP_FIELDS,
            result.drop_test,
        )
        section("3. NETWORK SUMMARY", NETWORK_FIELDS, result.networks)
        section(
            "4. HORTON ORDER STATISTICS (Strahler streams per order)",
            ORDER_FIELDS,
            result.order_stats,
        )
        section(
            "5. HORTON RATIOS (consecutive means and log-linear regression fits)",
            FIT_FIELDS,
            result.horton_fits,
        )
        section(
            "6. REACHES - geometry, stream orders and routing inputs",
            REACH_FIELDS,
            result.reaches,
        )
        section(
            "7. NODES - channel heads, confluences, pour points, outlets",
            NODE_FIELDS,
            result.nodes,
        )
        profile_rows = [row for rows in result.profiles.values() for row in rows]
        section(
            "8. MAIN-STEM LONGITUDINAL PROFILES",
            [("network_id",), ("dist_from_outlet_m",), ("z_m",)],
            profile_rows,
        )

        w.writerow(["9. FIELD DEFINITIONS"])
        w.writerow(["Section", "Field", "Unit", "Definition"])
        for title, fields in (
            ("Constant-drop test", DROP_FIELDS),
            ("Network summary", NETWORK_FIELDS),
            ("Horton order statistics", ORDER_FIELDS),
            ("Horton ratios", FIT_FIELDS),
        ):
            for name, unit, desc in fields:
                w.writerow([title, name, unit, desc])
        for title, fields in (("Reaches", REACH_FIELDS), ("Nodes", NODE_FIELDS)):
            for name, _t, unit, desc in fields:
                w.writerow([title, name, unit, desc])
        w.writerow(
            [
                "Main-stem profile",
                "dist_from_outlet_m",
                "m",
                "channel distance from the outlet",
            ]
        )
        w.writerow(["Main-stem profile", "z_m", "m", "DEM elevation"])
