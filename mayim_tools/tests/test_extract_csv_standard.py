"""Mayim output standard for the point-extraction CSVs (Extract: CHIRPS,
CMORPH, IMERG, MERRA2, PERSIANN CDR and ERA5 precipitation): UTF-8 with a
byte-order mark, no '#' key on the time series (the rainfall tools read
it directly), a column key on the ERA5 anomalies audit. No QGIS needed."""

from __future__ import annotations

import pandas as pd

BOM = b"\xef\xbb\xbf"


def _series_reads_in_rainfall_tools(path, time_col, depth_col):
    """The DDF / Huff / Adjust-to-DDF tools read the extract CSV with plain
    pandas; the Design Storm Ensembles loader uses read_csv_any_delimiter."""
    from mayim_tools.rainfall._common.csvio import read_csv_any_delimiter

    df = pd.read_csv(path)
    assert list(df.columns)[:3] == ["Site", time_col, depth_col]
    df2 = read_csv_any_delimiter(str(path))
    assert list(df2.columns)[0] == "Site"


def test_chirps_csv_has_bom_and_plain_header(tmp_path):
    from mayim_tools.rainfall.chirps.core import ChirpsResult
    from mayim_tools.rainfall.chirps.export import write_chirps_csv

    df = pd.DataFrame(
        {"Date": ["2020-01-01", "2020-01-02"], "PrecipitationMM": [1.5, float("nan")]}
    )
    path = tmp_path / "c.csv"
    n = write_chirps_csv({"Site A": ChirpsResult(dataframe=df)}, path)
    assert n == 2
    raw = path.read_bytes()
    assert raw.startswith(BOM)
    assert raw.decode("utf-8-sig").splitlines()[0] == "Site,Date,PrecipitationMM"
    _series_reads_in_rainfall_tools(path, "Date", "PrecipitationMM")


def test_era5_csv_bom_and_anomalies_key(tmp_path):
    from mayim_tools.rainfall.era5_extract.core import Era5Result
    from mayim_tools.rainfall.era5_extract.export import (
        ANOMALIES_KEY,
        write_era5_anomalies,
        write_era5_csv,
    )

    df = pd.DataFrame(
        {
            "ValidTime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
            "PrecipitationMM": [1.5, 2.5],
        }
    )
    anomalies = pd.DataFrame(
        {
            "reference_time": [pd.Timestamp("2020-01-01 06:00")],
            "step_hours": [2],
            "valid_time": [pd.Timestamp("2020-01-01 08:00")],
            "accumulated_mm": [2.0],
            "precipitation_mm_raw": [-8.0],
        }
    )
    result = Era5Result(dataframe=df, product="era5", anomalies=anomalies)
    main = tmp_path / "e.csv"
    write_era5_csv({"Site A": result}, main)
    assert main.read_bytes().startswith(BOM)
    assert main.read_text(encoding="utf-8-sig").startswith("Site,ValidTime,")
    _series_reads_in_rainfall_tools(main, "ValidTime", "PrecipitationMM")

    audit = tmp_path / "a.csv"
    assert write_era5_anomalies({"Site A": result}, audit) == 1
    lines = audit.read_text(encoding="utf-8-sig").splitlines()
    assert audit.read_bytes().startswith(BOM)
    assert lines[0] == "# Column key"
    header = next(ln for ln in lines if not ln.startswith("#")).split(",")
    assert set(header) == set(ANOMALIES_KEY)
    back = pd.read_csv(audit, comment="#")
    assert back["PrecipitationMMRaw"].iloc[0] == -8.0


def test_dataframe_writer_bom(tmp_path):
    """CMORPH / IMERG / MERRA2 / PERSIANN write their combined DataFrame with
    the shared writer; it needs QGIS only for the module import, so the
    behaviour (utf-8-sig, no index) is checked with pandas directly."""
    df = pd.DataFrame(
        {
            "Site": ["A", "A"],
            "Timestamp": ["2020-01-01 00:00", "2020-01-01 01:00"],
            "PrecipitationMMHR": [0.5, 1.0],
        }
    )
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")  # = write_dataframe_csv
    assert path.read_bytes().startswith(BOM)
    _series_reads_in_rainfall_tools(path, "Timestamp", "PrecipitationMMHR")
