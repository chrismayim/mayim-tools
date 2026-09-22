"""Output generation for Design Storm Ensembles - zero QGIS dependency,
like core.py."""

import csv


def write_ensemble_csv(ensemble_rows, path):
    """Main ensemble CSV: Duration, TimeStep, AEP, Shape, then one
    column per increment (padded with blanks for rows with fewer
    increments than the widest row)."""
    max_increments = max((len(r["Increments"]) for r in ensemble_rows), default=0)
    header = ["Duration", "TimeStep", "AEP", "Shape"] + [
        f"Increment_{i + 1}" for i in range(max_increments)
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in ensemble_rows:
            row = [r["Duration"], r["TimeStep"], r["AEP"], r["Shape"]]
            row += r["Increments"]
            row += [""] * (max_increments - len(r["Increments"]))
            writer.writerow(row)


def write_catalogue_csv(catalogue_rows, path):
    if not catalogue_rows:
        header = [
            "duration_minutes",
            "event_id",
            "start",
            "end",
            "depth_mm",
            "aep_percent",
            "aep_band",
            "aep_method",
            "shape",
        ]
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        return
    header = list(catalogue_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(catalogue_rows)


def write_bin_summary_csv(bin_summary_rows, path):
    if not bin_summary_rows:
        header = [
            "Duration",
            "TimeStep",
            "AEP",
            "Shape",
            "n_storms",
            "n_storms_own_duration",
            "status",
            "pooled_from_durations",
        ]
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        return
    header = list(bin_summary_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(bin_summary_rows)
