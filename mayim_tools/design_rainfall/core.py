"""
Core design rainfall calculation engine.

Implements the Smithers & Schulze regional L-moment design rainfall
methodology (WRC Report K5/1060, 2002) for South Africa, reconstructed
from the rainfall3 dataset and validated against:
    - the worked Cedara example in the original rainfall2 user manual
      (Appendix A, Figure 82) for medians and bounds across short,
      1-day, and 2-7 day durations, and
    - the source report's own equations (Ch. 4-5) for the formula shapes.

All formulae below are CONFIRMED (median depths validated to <0.1mm,
confidence bounds validated to <1mm, against the manual's published
Cedara example at LATM=1772, LONGM=1817).

--------------------------------------------------------------------
1. Median (design) depth - Report Equation 33
--------------------------------------------------------------------
    DRE[i,j] = GC[1day, j] * L1[i]

    where i = duration, j = return period. The SAME growth curve set
    (derived from the 1-day duration, hence 'gc_1day.dbf') is reused
    for every duration - this is a deliberate simplification in the
    source methodology, not an approximation on our part.

--------------------------------------------------------------------
2. Confidence bounds - Report Equations 34-37
--------------------------------------------------------------------
    U90[DRE[i,j]] = GCU[1day,j] * L1[i] * (1 + Pi[i]/100)
    L90[DRE[i,j]] = GCL[1day,j] * L1[i] * (1 - Pi[i]/100)

    where Pi[i] is the duration-specific prediction-interval percentage
    for the L1 index itself (Report Eq. 36). Pi[i] is PRE-COMPUTED per
    grid point and duration in SAgrid.dbf's P1D-P7D / P5-P1200 fields.
    (Earlier assumption that these were scaling proportions was WRONG -
    they are prediction-interval percentages; confirmed by exact
    reproduction of the manual's published bounds.)

--------------------------------------------------------------------
3. Index value (L1) by duration
--------------------------------------------------------------------
    1 day:       L1 = ADJ_L1_1D                              (grid field, direct)

    5min-24h:    L1_24h = ADJ_L1_1D * ratio24h[s_cluster]     (24h21dratios.dbf, MEDIAN col)
                 L1_dur = XCOEF[s_cluster,dur]*L1_24h + CONST[s_cluster,dur]
                                                               (ShortDurationL_1Regressions.dbf, Eq.16)

    2-7 day:     psi_D   = THETA + TAU * D^SIGMA              (Eq. 14, slope regression)
                 omega_D = UPSILON + KAPPA * D^RHO            (Eq. 13, intercept regression)
                 L1_D    = psi_D * L1_1day + omega_D          (Eq. 12)
                 (region from grid's AV_CLUSTER field, matches
                 Daily2-7dayL1regressions7Regions.dbf's REGION)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

DEFAULT_GPKG = Path(__file__).parent / "data" / "design_rainfall.gpkg"

SHORT_DURATIONS = [
    5,
    10,
    15,
    30,
    45,
    60,
    90,
    120,
    240,
    360,
    480,
    600,
    720,
    960,
    1200,
    1440,
]
DAILY_DURATIONS = [1, 2, 3, 4, 5, 6, 7]
RETURN_PERIODS = [2, 5, 10, 20, 50, 100, 200]

_SHORT_PI_FIELD = {
    5: "P5",
    10: "P10",
    15: "P15",
    30: "P30",
    45: "P45",
    60: "P60",
    90: "P90",
    120: "P120",
    240: "P240",
    360: "P360",
    480: "P480",
    600: "P600",
    720: "P720",
    960: "P960",
    1200: "P1200",
}
_DAILY_PI_FIELD = {1: "P1D", 2: "P2D", 3: "P3D", 4: "P4D", 5: "P5D", 6: "P6D", 7: "P7D"}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


@dataclass
class DesignRainfallResult:
    duration_label: str
    return_period: int
    depth: float
    lower: float | None = None
    upper: float | None = None


@dataclass
class PointEstimate:
    latitude: float
    longitude: float
    map_mm: float
    altitude_m: float
    cluster: int
    s_cluster: int
    av_cluster: int
    source: str
    results: list[DesignRainfallResult] = field(default_factory=list)


class DesignRainfallEngine:
    """Loads the bundled GeoPackage once and serves point/station queries."""

    def __init__(self, gpkg_path: str | Path = DEFAULT_GPKG):
        self.gpkg_path = Path(gpkg_path)
        if not self.gpkg_path.exists():
            raise FileNotFoundError(
                f"Design rainfall GeoPackage not found at {self.gpkg_path}. "
                "Run data_prep/convert_to_gpkg.py first."
            )
        self._grid = None
        self._stations = None
        self._short_regr = None
        self._multiday_regr = None
        self._growth_curves = None
        self._ratio_24h = None

    @property
    def grid(self) -> gpd.GeoDataFrame:
        if self._grid is None:
            self._grid = gpd.read_file(self.gpkg_path, layer="sa_grid")
            _ = self._grid.sindex  # warm up spatial index
        return self._grid

    @property
    def stations(self) -> gpd.GeoDataFrame:
        if self._stations is None:
            self._stations = gpd.read_file(self.gpkg_path, layer="stations")
        return self._stations

    @property
    def short_regr(self) -> pd.DataFrame:
        if self._short_regr is None:
            self._short_regr = gpd.read_file(
                self.gpkg_path, layer="short_duration_regr"
            )
        return self._short_regr

    @property
    def multiday_regr(self) -> pd.DataFrame:
        if self._multiday_regr is None:
            self._multiday_regr = gpd.read_file(self.gpkg_path, layer="multiday_regr")
        return self._multiday_regr

    @property
    def growth_curves(self) -> pd.DataFrame:
        if self._growth_curves is None:
            self._growth_curves = gpd.read_file(self.gpkg_path, layer="growth_curves")
        return self._growth_curves

    @property
    def ratio_24h(self) -> pd.DataFrame:
        if self._ratio_24h is None:
            self._ratio_24h = gpd.read_file(self.gpkg_path, layer="ratio_24h")
        return self._ratio_24h

    def nearest_grid_point(self, lat: float, lon: float) -> pd.Series:
        pt = Point(lon, lat)
        idx = self.grid.sindex.nearest(pt)[1][0]
        return self.grid.iloc[idx]

    def find_station(
        self, name: str | None = None, saws_number: str | None = None
    ) -> pd.DataFrame:
        df = self.stations
        if saws_number:
            return df[df["FSTN_NO"].str.contains(saws_number, case=False, na=False)]
        if name:
            return df[df["STN_NAME"].str.contains(name, case=False, na=False)]
        raise ValueError("Provide either name or saws_number")

    def nearest_stations(self, lat: float, lon: float, n: int = 5) -> gpd.GeoDataFrame:
        """N closest daily rainfall stations to a point, with great-circle
        distance in km (matches the original tool's reference listing)."""
        df = self.stations.copy()
        df["DIST_KM"] = df.geometry.apply(lambda g: _haversine_km(lat, lon, g.y, g.x))
        return df.nsmallest(n, "DIST_KM")

    def _growth_curve_row(self, cluster: int) -> pd.Series:
        gc = self.growth_curves
        row = gc[gc["CLUSTER"] == cluster]
        if row.empty:
            raise ValueError(f"No growth curve for CLUSTER={cluster}")
        return row.iloc[0]

    def _ratio_24h_median(self, s_cluster: int) -> float:
        r = self.ratio_24h
        row = r[r["CLUSTER"] == s_cluster]
        if row.empty:
            raise ValueError(f"No 24h ratio for S_CLUSTER={s_cluster}")
        return float(row.iloc[0]["MEDIAN"])

    def _short_duration_l1(
        self, s_cluster: int, duration_min: int, l1_24h: float
    ) -> float:
        r = self.short_regr
        row = r[(r["CLUSTER"] == s_cluster) & (r["DURATION"] == duration_min)]
        if row.empty:
            raise ValueError(
                f"No short-duration regression for S_CLUSTER={s_cluster}, dur={duration_min}"
            )
        return float(row.iloc[0]["XCOEF"]) * l1_24h + float(row.iloc[0]["CONST"])

    def _multiday_l1(self, region: int, D: int, l1_1day: float) -> float:
        r = self.multiday_regr
        row = r[r["REGION"] == region]
        if row.empty:
            raise ValueError(f"No multiday regression for REGION={region}")
        row = row.iloc[0]
        psi = float(row["THETA"]) + float(row["TAU"]) * (D ** float(row["SIGMA"]))
        omega = float(row["UPSILON"]) + float(row["KAPPA"]) * (D ** float(row["RHO"]))
        return psi * l1_1day + omega

    @staticmethod
    def _depth_with_bounds(
        l1: float, pi_pct: float, gc_row: pd.Series, return_periods: list[int]
    ):
        out = []
        for rt in return_periods:
            gc = gc_row.get(f"GC{rt}")
            gcl = gc_row.get(f"GCL{rt}")
            gcu = gc_row.get(f"GCU{rt}")
            if gc is None:
                continue
            median = l1 * float(gc)
            upper = l1 * float(gcu) * (1 + pi_pct / 100) if gcu is not None else None
            lower = l1 * float(gcl) * (1 - pi_pct / 100) if gcl is not None else None
            out.append((rt, median, lower, upper))
        return out

    def estimate_short_durations(
        self,
        grid_row: pd.Series,
        durations: list[int] = SHORT_DURATIONS,
        return_periods: list[int] = RETURN_PERIODS,
    ) -> list[DesignRainfallResult]:
        adj_l1_1d = float(grid_row["ADJ_L1_1D"])
        cluster = int(grid_row["CLUSTER"])
        s_cluster = int(grid_row["S_CLUSTER"])
        gc_row = self._growth_curve_row(cluster)
        ratio = self._ratio_24h_median(s_cluster)
        l1_24h = adj_l1_1d * ratio

        results = []
        for dur in durations:
            l1_dur = self._short_duration_l1(s_cluster, dur, l1_24h)
            pi_field = _SHORT_PI_FIELD.get(dur)
            pi_pct = (
                float(grid_row[pi_field]) if pi_field and pi_field in grid_row else 0.0
            )
            if dur < 60:
                label = f"{dur} min"
            elif dur % 60 == 0:
                label = f"{dur // 60} h"
            else:
                label = f"{dur / 60:g} h"
            for rt, median, lower, upper in self._depth_with_bounds(
                l1_dur, pi_pct, gc_row, return_periods
            ):
                results.append(
                    DesignRainfallResult(
                        duration_label=label,
                        return_period=rt,
                        depth=round(median, 1),
                        lower=round(lower, 1) if lower is not None else None,
                        upper=round(upper, 1) if upper is not None else None,
                    )
                )
        return results

    def estimate_1day(
        self,
        grid_row: pd.Series,
        return_periods: list[int] = RETURN_PERIODS,
    ) -> list[DesignRainfallResult]:
        adj_l1_1d = float(grid_row["ADJ_L1_1D"])
        cluster = int(grid_row["CLUSTER"])
        gc_row = self._growth_curve_row(cluster)
        pi_pct = float(grid_row["P1D"]) if "P1D" in grid_row else 0.0

        results = []
        for rt, median, lower, upper in self._depth_with_bounds(
            adj_l1_1d, pi_pct, gc_row, return_periods
        ):
            results.append(
                DesignRainfallResult(
                    duration_label="1 day",
                    return_period=rt,
                    depth=round(median, 1),
                    lower=round(lower, 1) if lower is not None else None,
                    upper=round(upper, 1) if upper is not None else None,
                )
            )
        return results

    def estimate_multiday(
        self,
        grid_row: pd.Series,
        durations: list[int] = DAILY_DURATIONS[1:],
        return_periods: list[int] = RETURN_PERIODS,
    ) -> list[DesignRainfallResult]:
        adj_l1_1d = float(grid_row["ADJ_L1_1D"])
        cluster = int(grid_row["CLUSTER"])
        region = int(grid_row["AV_CLUSTER"])
        gc_row = self._growth_curve_row(cluster)

        results = []
        for D in durations:
            l1_d = self._multiday_l1(region, D, adj_l1_1d)
            pi_field = _DAILY_PI_FIELD.get(D)
            pi_pct = (
                float(grid_row[pi_field]) if pi_field and pi_field in grid_row else 0.0
            )
            for rt, median, lower, upper in self._depth_with_bounds(
                l1_d, pi_pct, gc_row, return_periods
            ):
                results.append(
                    DesignRainfallResult(
                        duration_label=f"{D} day",
                        return_period=rt,
                        depth=round(median, 1),
                        lower=round(lower, 1) if lower is not None else None,
                        upper=round(upper, 1) if upper is not None else None,
                    )
                )
        return results

    def estimate_at_point(
        self,
        lat: float,
        lon: float,
        short_durations_min: list[int] | None = None,
        daily_durations: list[int] | None = None,
        return_periods: list[int] | None = None,
    ) -> PointEstimate:
        """Full grid-based estimate at an arbitrary point: short durations
        (5min-24h), 1-day, and 2-7 day, all with 90% confidence bounds."""
        row = self.nearest_grid_point(lat, lon)
        short_durations_min = (
            short_durations_min if short_durations_min is not None else SHORT_DURATIONS
        )
        daily_durations = (
            daily_durations if daily_durations is not None else DAILY_DURATIONS[1:]
        )
        return_periods = return_periods or RETURN_PERIODS

        results = []
        if short_durations_min:
            results += self.estimate_short_durations(
                row, short_durations_min, return_periods
            )
        results += self.estimate_1day(row, return_periods)
        if daily_durations:
            results += self.estimate_multiday(row, daily_durations, return_periods)

        return PointEstimate(
            latitude=float(row.geometry.y),
            longitude=float(row.geometry.x),
            map_mm=float(row["MAP"]),
            altitude_m=float(row["ALTITUDE"]),
            cluster=int(row["CLUSTER"]),
            s_cluster=int(row["S_CLUSTER"]),
            av_cluster=int(row["AV_CLUSTER"]),
            source="grid",
            results=results,
        )
