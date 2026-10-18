"""
Stochastic depression classification (Monte Carlo persistence).

Lindsay, J.B. and Creed, I.F. (2005). "Removal of artifact
depressions from digital elevation models: towards a minimum impact
approach." Hydrological Processes, 19(16), 3113-3126.
Lindsay, J.B. and Creed, I.F. (2006). "Distinguishing actual and
artefact depressions in digital elevation data." Computers &
Geosciences, 32(8), 1192-1204.

Of the five approaches to distinguishing real depressions from
artefacts compared in the 2006 paper (ground inspection, examining
source data, statistical classification against ground-truthed
training data, simple size/depth/volume thresholds, and Monte Carlo
stochastic simulation), stochastic simulation was the best-performing
of the four automatable methods (75% correct in their controlled
comparison) - and simple thresholds, the most common approach in
practice, performed WORST, wrongly removing several genuine small
depressions. This module implements the stochastic approach: perturb
the DEM with a plausible random error field (matching the DEM's own
known or estimated vertical accuracy, with spatial autocorrelation,
since real DEM error is not independent from cell to cell), re-run
depression filling on each perturbed realisation, and see what
fraction of realisations a given depression survives in. A depression
that only exists because of a specific, unlikely noise pattern
disappears across most realisations; a genuine one persists
regardless of plausible noise. The paper's own stated limitation is
kept in mind, not solved here: convergent/bottleneck terrain can
produce artificially high persistence scores even for genuine
artefacts - this is one line of evidence among several this whole
project is meant to combine (see the project's own dev README), not
a standalone verdict.

WINDOWED, NOT WHOLE-DEM (a deliberate departure from the published
method, agreed for this project's basin-scale requirement - see chat
discussion 2026-09-23): Lindsay & Creed's original method perturbs and
re-floods the ENTIRE DEM per iteration (304 iterations in their own
test) - impractical at basin scale, run once per candidate depression.
Here, each depression's persistence is assessed within a local window
around it (large enough to capture its real spill behaviour, not the
whole basin) - this is this project's OWN adaptation for tractability,
not a technique described in the cited papers, and the window-margin
heuristic below should be understood as a practical judgement call,
not something with a precise theoretical derivation. A genuine
limitation of windowing, worth stating plainly: a depression whose
only real escape route lies OUTSIDE its window will appear to persist
even when it might not with a larger window - the margin needs to be
generous enough to capture a depression's true relevant surroundings.

NOISE GENERATION: Lindsay & Creed's own method used a Monte Carlo
turning-bands approach to generate spatially autocorrelated error
fields calibrated against a target spatial autocorrelation structure
(Moran's I). This implementation uses a simpler, well-established
alternative for the same purpose - i.i.d. Gaussian noise passed
through a Gaussian smoothing filter, then rescaled to the target
standard deviation (smoothing reduces variance, so rescaling is
necessary) - chosen for straightforward independent implementation
rather than reproducing turning-bands specifically. The smoothing
kernel's sigma (in cells) controls the noise's spatial correlation
length; there is no published formula linking this directly to a
target Moran's I value, so `correlation_length_cells` is exposed as a
tunable parameter with a documented, reasoned default rather than a
value derived from the papers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .depression_hierarchy import DepressionHierarchy
from .priority_flood import priority_flood_fill

# A small tolerance for "did this cell get raised by filling" - guards
# against floating-point noise from the perturbation/filling arithmetic
# registering as a spurious "persisted" result for a cell that was
# never really a depression to begin with.
_PERSISTENCE_EPS = 1e-9

DEFAULT_N_ITERATIONS = (
    100  # a practical default, not derived from the papers - see module docstring
)
# Changed from an initial 3.0 after measuring its effect (see dev/README.md): with
# 10 cm vertical error, a 5 cm single-cell pit "persisted" in 88% of realisations
# at correlation length 3 (would be called natural), vs 52% at length 1. Strongly
# correlated noise barely changes the RELATIVE relief between neighbouring cells,
# so it flatters small pits. Still a judgement call, not a published value - it
# should be set from the DEM's real error autocorrelation when that is known.
DEFAULT_CORRELATION_LENGTH_CELLS = 1.0


def generate_correlated_noise(
    shape: tuple[int, int],
    std: float,
    correlation_length_cells: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generates a single realisation of a spatially autocorrelated
    Gaussian random field of the given shape, with the given target
    standard deviation. See module docstring for the method and why
    it was chosen over turning-bands.
    """
    white = rng.standard_normal(shape)
    if correlation_length_cells > 0:
        smoothed = gaussian_filter(
            white, sigma=correlation_length_cells, mode="reflect"
        )
    else:
        smoothed = white
    current_std = smoothed.std()
    if current_std > 0:
        smoothed = smoothed * (std / current_std)
    return smoothed


def _default_window_margin(
    hierarchy: DepressionHierarchy, leaf_id: int, dem: np.ndarray | None = None
) -> int:
    mask = (
        hierarchy.footprint_of(leaf_id, dem)
        if dem is not None
        else hierarchy.cells_of(leaf_id)
    )
    if not mask.any():
        mask = hierarchy.cells_of(leaf_id)
    rows, cols = np.nonzero(mask)
    diameter = int(max(rows.max() - rows.min(), cols.max() - cols.min())) + 1
    # a floor of 10 cells regardless of how small the depression itself is,
    # since a tiny depression's relevant surrounding terrain (where a real
    # or alternative escape route might be) is not itself tiny
    return max(10, diameter)


def compute_persistence_probability(
    dem: np.ndarray,
    hierarchy: DepressionHierarchy,
    leaf_id: int,
    vertical_error_std: float,
    correlation_length_cells: float = DEFAULT_CORRELATION_LENGTH_CELLS,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    window_margin: int | None = None,
    nodata: float | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimates the probability that `leaf_id`'s depression is a
    genuine feature of the terrain, rather than an artefact of DEM
    noise, by Monte Carlo perturbation within a local window (see
    module docstring for the full method and its departures from the
    published, whole-DEM approach).

    Returns a value in [0, 1]: the fraction of `n_iterations` random,
    plausible perturbations of the local terrain (given
    `vertical_error_std`, the DEM's own assumed/estimated vertical
    accuracy) under which the depression's own pit cell still gets
    raised by depression filling. Values close to 1 indicate a
    depression robust to plausible noise (more likely genuine); values
    close to 0 indicate a depression that only exists under the
    original, unperturbed elevations (more likely an artefact of a
    few noisy cells). This is ONE line of evidence for classification,
    not a final verdict on its own - see this depression's other
    metrics (size, depth, stream-crossing evidence where available)
    for the fuller picture this project is meant to combine.
    """
    if rng is None:
        rng = np.random.default_rng()

    dem64 = dem.astype(np.float64)
    mask = hierarchy.footprint_of(leaf_id, dem64)
    if not mask.any():
        mask = hierarchy.cells_of(leaf_id)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError(f"leaf {leaf_id} has no cells - cannot assess persistence")

    if window_margin is None:
        window_margin = _default_window_margin(hierarchy, leaf_id, dem64)

    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())
    r0 = max(0, r_min - window_margin)
    r1 = min(dem64.shape[0], r_max + window_margin + 1)
    c0 = max(0, c_min - window_margin)
    c1 = min(dem64.shape[1], c_max + window_margin + 1)

    window = dem64[r0:r1, c0:c1].copy()
    window_is_nodata = (
        (window == nodata) if nodata is not None else np.zeros_like(window, dtype=bool)
    )

    leaf_elevs = dem64[rows, cols]
    min_idx = int(np.argmin(leaf_elevs))
    pit_r_local = int(rows[min_idx]) - r0
    pit_c_local = int(cols[min_idx]) - c0

    persisted_count = 0
    for _ in range(n_iterations):
        noise = generate_correlated_noise(
            window.shape, vertical_error_std, correlation_length_cells, rng
        )
        perturbed = window.copy()
        valid = ~window_is_nodata
        perturbed[valid] = window[valid] + noise[valid]

        filled = priority_flood_fill(perturbed, nodata=nodata)
        raised_amount = (
            filled[pit_r_local, pit_c_local] - perturbed[pit_r_local, pit_c_local]
        )
        if raised_amount > _PERSISTENCE_EPS:
            persisted_count += 1

    return persisted_count / n_iterations


def _smooth_control_surface(
    window: np.ndarray, feature_mask: np.ndarray, valid: np.ndarray, sweeps: int = 200
) -> np.ndarray:
    """The surface the terrain would plausibly have WITHOUT the feature:
    the feature's cells replaced by harmonic (Laplace) interpolation of
    their surroundings, computed by Jacobi relaxation with the
    surrounding cells held fixed. On a uniform slope this reproduces
    the slope exactly; on flat ground it reproduces the flat.

    Deliberately NOT simply "the depression filled to its spill
    elevation": a first version of the baseline did that, and measured
    a steep-terrain baseline of 0.30 where the true no-feature figure
    is ~0.01 - the filled surface leaves the pit cell exactly level
    with its downhill neighbour, a tie that noise breaks into a pit
    roughly half the time, so it is a surface balanced on the edge of a
    depression rather than a surface without one.
    """
    u = window.copy()
    ring_vals = window[valid & ~feature_mask]
    if ring_vals.size == 0 or not feature_mask.any():
        return u
    u[feature_mask] = float(np.median(ring_vals))
    weight = valid.astype(np.float64)
    for _ in range(sweeps):
        padded_u = np.pad(np.where(valid, u, 0.0), 1)
        padded_w = np.pad(weight, 1)
        total = np.zeros_like(u)
        count = np.zeros_like(u)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                total += padded_u[
                    1 + dr : 1 + dr + u.shape[0], 1 + dc : 1 + dc + u.shape[1]
                ]
                count += padded_w[
                    1 + dr : 1 + dr + u.shape[0], 1 + dc : 1 + dc + u.shape[1]
                ]
        new_vals = np.divide(total, count, out=u.copy(), where=count > 0)
        u[feature_mask] = new_vals[feature_mask]
    return u


@dataclass
class PersistenceResult:
    raw: float  # fraction of realisations in which the pit cell lies in a depression
    baseline: (
        float  # same, with THIS depression removed - how often noise alone does it
    )
    adjusted: float  # chance-corrected: (raw - baseline) / (1 - baseline), floored at 0


def compute_persistence_with_baseline(
    dem: np.ndarray,
    hierarchy: DepressionHierarchy,
    leaf_id: int,
    vertical_error_std: float,
    correlation_length_cells: float = DEFAULT_CORRELATION_LENGTH_CELLS,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    window_margin: int | None = None,
    nodata: float | None = None,
    rng: np.random.Generator | None = None,
) -> PersistenceResult:
    """Persistence corrected for how often noise ALONE would put the
    same cell inside a depression.

    WHY THIS EXISTS (this project's own addition - not in Lindsay &
    Creed): raw persistence has a terrain-dependent floor. Measured
    directly during development, with 10 cm vertical error and no real
    feature present at all, noise alone placed a given cell inside some
    depression in 42% of realisations on flat ground, 22% on a
    0.02 m/cell slope, and ~1% on a 0.10 m/cell slope. So on flat
    terrain - floodplains, exactly where artefact pits are common and
    where hydrological modelling matters most - a raw score of 0.4
    carries no evidence whatsoever, while the same score on a steep
    slope would be strongly meaningful. Comparing against a control
    removes that terrain dependence.

    The control is the same window with THIS depression's cells replaced
    by a smooth interpolation of their surroundings (see
    _smooth_control_surface for why not simply "filled"), perturbed with the SAME noise realisations as the
    real run (paired, which cuts sampling variance considerably versus
    two independent runs) and scored identically. The correction is the
    standard chance-correction (raw - baseline) / (1 - baseline).
    """
    if rng is None:
        rng = np.random.default_rng()

    dem64 = dem.astype(np.float64)
    node = hierarchy.nodes[leaf_id]
    # footprint (cells below the spill elevation), not the leaf's catchment -
    # the catchment can include a whole rising valley or hillside, which would
    # make the window huge and the control surface flatten real terrain
    mask = hierarchy.footprint_of(leaf_id, dem64)
    if not mask.any():
        mask = hierarchy.cells_of(leaf_id)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError(f"leaf {leaf_id} has no cells - cannot assess persistence")

    if window_margin is None:
        diameter = int(max(rows.max() - rows.min(), cols.max() - cols.min())) + 1
        window_margin = max(10, diameter)

    r0 = max(0, int(rows.min()) - window_margin)
    r1 = min(dem64.shape[0], int(rows.max()) + window_margin + 1)
    c0 = max(0, int(cols.min()) - window_margin)
    c1 = min(dem64.shape[1], int(cols.max()) + window_margin + 1)

    window = dem64[r0:r1, c0:c1].copy()
    window_is_nodata = (
        (window == nodata) if nodata is not None else np.zeros_like(window, dtype=bool)
    )
    valid = ~window_is_nodata

    local_mask = mask[r0:r1, c0:c1]
    control = _smooth_control_surface(window, local_mask, valid)

    pit_idx = int(np.argmin(dem64[rows, cols]))
    pr, pc = int(rows[pit_idx]) - r0, int(cols[pit_idx]) - c0

    raw_hits = 0
    base_hits = 0
    for _ in range(n_iterations):
        noise = generate_correlated_noise(
            window.shape, vertical_error_std, correlation_length_cells, rng
        )
        for surface, is_control in ((window, False), (control, True)):
            perturbed = surface.copy()
            perturbed[valid] = surface[valid] + noise[valid]
            filled = priority_flood_fill(perturbed, nodata=nodata)
            if filled[pr, pc] - perturbed[pr, pc] > _PERSISTENCE_EPS:
                if is_control:
                    base_hits += 1
                else:
                    raw_hits += 1

    raw = raw_hits / n_iterations
    baseline = base_hits / n_iterations
    adjusted = 0.0 if baseline >= 1.0 else max(0.0, (raw - baseline) / (1.0 - baseline))
    return PersistenceResult(raw=raw, baseline=baseline, adjusted=adjusted)
