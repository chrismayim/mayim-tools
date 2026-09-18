"""
Mayim Tools - Processing Provider
=================================

Registers Mayim Tools algorithms in the QGIS Processing Toolbox.
"""

from __future__ import annotations

from pathlib import Path

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from mayim_tools.data.grib_to_csv.grib_to_csv_algorithm import GribToCsvAlgorithm
from mayim_tools.rainfall.chirps.chirps_extract_algorithm import ChirpsExtractAlgorithm
from mayim_tools.rainfall.cmorph_extract.cmorph_extract_algorithm import (
    CmorphExtractAlgorithm,
)
from mayim_tools.rainfall.ddf_to_hyetographs.ddf_to_hyetographs_algorithm import (
    DdfToHyetographsAlgorithm,
)
from mayim_tools.rainfall.design_rainfall.design_rainfall_algorithm import (
    DesignRainfallPointAlgorithm,
)
from mayim_tools.rainfall.frequency_analysis.rainfall_frequency_algorithm import (
    RainfallFrequencyAlgorithm,
)
from mayim_tools.rainfall.huff_curves.huff_curves_algorithm import HuffCurvesAlgorithm
from mayim_tools.rainfall.imerg_extract.imerg_extract_algorithm import (
    ImergExtractAlgorithm,
)
from mayim_tools.rainfall.merra2_extract.merra2_extract_algorithm import (
    Merra2ExtractAlgorithm,
)


class MayimToolsProvider(QgsProcessingProvider):
    """Mayim Tools Processing provider."""

    def id(self) -> str:  # noqa: A003 - required by the QGIS API.
        """Return the unique provider ID."""
        return "mayimtools"

    def name(self) -> str:
        """Return the provider display name."""
        return "Mayim Tools"

    def icon(self) -> QIcon:
        """Return the provider icon."""
        logo = Path(__file__).resolve().parents[1] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def loadAlgorithms(self) -> None:  # noqa: N802 - required by QGIS.
        """Register Mayim Tools algorithms."""
        self.addAlgorithm(DesignRainfallPointAlgorithm())
        self.addAlgorithm(ChirpsExtractAlgorithm())
        self.addAlgorithm(CmorphExtractAlgorithm())
        self.addAlgorithm(ImergExtractAlgorithm())
        self.addAlgorithm(Merra2ExtractAlgorithm())
        self.addAlgorithm(RainfallFrequencyAlgorithm())
        self.addAlgorithm(HuffCurvesAlgorithm())
        self.addAlgorithm(GribToCsvAlgorithm())
        self.addAlgorithm(DdfToHyetographsAlgorithm())
