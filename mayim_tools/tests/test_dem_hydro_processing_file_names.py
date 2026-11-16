"""DEM Hydrological Processing output names:
<project number>_<DEM description>_<description>, each part only when given
(Chris, 2026-11-16: e.g. M1234_COP30_Corrected_DEM)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("qgis.core")

from mayim_tools.hydrology.dem_hydro_processing.dem_hydro_processing_algorithm import (  # noqa: E402
    SPEC_BY_KEY,
    file_prefix,
)


@pytest.mark.parametrize(
    "project, dem, expected",
    [
        ("M1234", "COP30", "M1234_COP30_Corrected_DEM.tif"),
        ("M1234", "", "M1234_Corrected_DEM.tif"),
        ("", "COP30", "COP30_Corrected_DEM.tif"),
        ("", "", "Corrected_DEM.tif"),
        (" M1234 ", "Lidar 1 m", "M1234_Lidar_1_m_Corrected_DEM.tif"),
    ],
)
def test_file_names(project, dem, expected, tmp_path):
    spec = SPEC_BY_KEY["OUTPUT_DEM"]
    path = spec.path({spec.kind: str(tmp_path)}, file_prefix(project, dem))
    assert Path(path).name == expected
