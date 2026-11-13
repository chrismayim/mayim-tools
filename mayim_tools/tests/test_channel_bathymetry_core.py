"""Tests for DEM Channel Bathymetry: core on synthetic rivers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from mayim_tools.hydrology.channel_bathymetry.core import (  # noqa: E402
    ChannelError,
    Settings,
    bank_slope_depth,
    densify,
    half_parabola_split,
    run_channel_bathymetry,
    running_nanmedian,
)

CELL = 1.0


def river(
    length=600.0,
    amp=0.0,
    wavelength=300.0,
    w_bf=30.0,
    d_bf=3.0,
    stage=1.5,
    split_bf=0.5,
    s0=0.001,
    noise=0.02,
    island=False,
    seed=1,
):
    """A river whose bankfull section is two half-parabolas (left half-width
    split_bf * w_bf), so the banks above the water continue the shape below
    it. The water is ``stage`` m deep and hydro-flattened (10 m steps).
    Returns (dem, gt, centreline, true bed, water mask)."""
    rng = np.random.default_rng(seed)
    margin = 60.0
    nx = int(length + 2 * margin)
    ny = int(2 * (amp + w_bf + margin))
    gt = (0.0, CELL, 0.0, float(ny), 0.0, -CELL)
    xs = np.arange(nx) + 0.5
    ys = ny - (np.arange(ny) + 0.5)
    X, Y = np.meshgrid(xs, ys)
    cx = np.linspace(margin, margin + length, 3000)
    cy = ny / 2 + amp * np.sin(2 * np.pi * (cx - margin) / wavelength)
    line = np.column_stack([cx, cy])
    from scipy.spatial import cKDTree

    xy, ch, t = densify(line, 0.25)
    dist, idx = cKDTree(xy).query(np.column_stack([X.ravel(), Y.ravel()]))
    nvec = np.column_stack([-t[idx, 1], t[idx, 0]])
    off = np.einsum("ij,ij->i", np.column_stack([X.ravel(), Y.ravel()]) - xy[idx], nvec)
    off = off.reshape(X.shape)
    s = ch[idx].reshape(X.shape)
    a_l, a_r = split_bf * w_bf, (1 - split_bf) * w_bf
    centre = (a_l - a_r) / 2  # thalweg offset so the edges sit at +a_l / -a_r
    thal = -centre * 0  # thalweg at the centreline
    u = np.where(off >= thal, (off - thal) / a_l, (thal - off) / a_r)
    top = 100.0 - s0 * s
    bed = np.where(u < 1, top - d_bf * (1 - u**2), top + 0.002 * (np.abs(off) - w_bf))
    if island:
        isl = (np.abs(off) < 3) & (np.abs(s - length / 2) < 30)
        bed = np.where(isl, top - d_bf + stage + 0.6, bed)
    wse = top - d_bf + stage
    wse_flat = 100.0 - s0 * (np.floor(s / 10) * 10) - d_bf + stage
    water = bed < wse
    dem = np.where(water, wse_flat, bed + rng.normal(0, noise, bed.shape))
    return dem, gt, line, np.where(water, bed, np.nan), water


def depth_error(res, true_bed, water):
    both = water & np.isfinite(res.beds["central"])
    return res.beds["central"][both] - true_bed[both]


# ----------------------------------------------------------------- helpers
def test_half_parabola_split():
    d, a = half_parabola_split(0.4, 0.4)
    assert d == pytest.approx(0.1) and a == pytest.approx(0.5)
    d, a = half_parabola_split(0.8, 0.2)  # steep left bank: thalweg left
    assert a == pytest.approx(0.2)
    assert bank_slope_depth(0.4, np.nan, 20) == pytest.approx(2.0)
    assert np.isnan(bank_slope_depth(np.nan, np.nan, 20))


def test_running_nanmedian():
    v = [1, np.nan, 3, 100, 5]
    out = running_nanmedian(v, 3)
    assert out[1] == pytest.approx(2) and out[3] == pytest.approx(5)


# ----------------------------------------------------------- whole channel
def test_straight_parabola_recovers_depth():
    dem, gt, line, true_bed, water = river()
    res = run_channel_bathymetry(dem, None, gt, [line])
    iou = (res.water & water).sum() / (res.water | water).sum()
    assert iou > 0.9
    mid = [s for s in res.stations if 100 < s.chainage < 500]
    dc = np.median([s.depth["central"] for s in mid])
    # the bank line is a chord over the band above the water, a little
    # steeper than the parabola's tangent at the edge: about +15% here
    assert dc == pytest.approx(1.5, rel=0.2)
    assert np.median([s.width for s in mid]) == pytest.approx(30 * 0.5**0.5, rel=0.03)
    err = depth_error(res, true_bed, water)
    assert np.sqrt(np.mean(err**2)) < 0.3
    for s in mid:
        assert s.depth["low"] <= s.depth["central"] <= s.depth["high"]
    assert all(s.basis == "banks" for s in mid)


def test_meander_and_volume():
    dem, gt, line, true_bed, water = river(amp=40)
    res = run_channel_bathymetry(dem, None, gt, [line])
    true_vol = np.nansum(np.where(water, dem - true_bed, 0)) * CELL**2
    vol = res.summary()["volume_central_m3"]
    assert vol == pytest.approx(true_vol, rel=0.2)


def test_steep_left_bank_moves_the_thalweg_left():
    dem, gt, line, true_bed, water = river(split_bf=0.3)
    res = run_channel_bathymetry(dem, None, gt, [line])
    mid = [s for s in res.stations if 100 < s.chainage < 500]
    assert np.median([s.split for s in mid]) < 0.45
    assert np.median([s.sl for s in mid]) > np.median([s.sr for s in mid])


def test_caps():
    dem, gt, line, _, _ = river()
    res = run_channel_bathymetry(dem, None, gt, [line], Settings(max_depth=1.0))
    mid = [s for s in res.stations if 100 < s.chainage < 500]
    assert max(s.depth["high"] for s in mid) <= 1.0 + 1e-9
    assert all(s.cap == "max" for s in mid)
    assert np.nanmax(res.depth["central"]) <= 1.0 + 1e-6
    res = run_channel_bathymetry(dem, None, gt, [line], Settings(min_ratio=40))
    mid = [s for s in res.stations if 100 < s.chainage < 500]
    assert all(s.depth["central"] <= s.width / 40 + 1e-9 for s in mid)
    assert all(s.cap == "ratio" for s in mid)


def test_bed_meets_banks_without_a_step():
    dem, gt, line, _, water = river()
    res = run_channel_bathymetry(dem, None, gt, [line])
    from scipy import ndimage

    edge = res.water & ~ndimage.binary_erosion(res.water)
    edge[:, :110] = False  # not the cut ends of the reach
    edge[:, 610:] = False
    assert np.nanmax(res.depth["central"][edge]) < 0.6
    assert np.all(res.beds["central"][res.water] <= dem[res.water] + 1e-9)


def test_island_flags_more_than_one_strip():
    dem, gt, line, _, _ = river(island=True)
    res = run_channel_bathymetry(dem, None, gt, [line])
    assert any(s.multi for s in res.stations)
    assert np.isfinite(res.beds["central"][res.water]).all()


def test_user_mask_and_errors():
    dem, gt, line, _, water = river()
    res = run_channel_bathymetry(dem, None, gt, [line], user_mask=water)
    assert res.water.sum() == water.sum()
    with pytest.raises(ChannelError, match="water polygon"):
        run_channel_bathymetry(dem, None, gt, [line], user_mask=np.zeros_like(water))
    with pytest.raises(ChannelError, match="No centreline overlaps|outside"):
        run_channel_bathymetry(dem, None, gt, [line + 1e5])


def test_no_banks_uses_fallback_ratio():
    dem, gt, line, _, water = river()
    flat = np.where(water, dem, np.nanmax(dem[water]) + 0.001)  # banks flat
    with pytest.raises(ChannelError, match="No bed"):
        run_channel_bathymetry(flat, None, gt, [line])
    res = run_channel_bathymetry(
        flat, None, gt, [line], Settings(fallback_ratios=(15, 10, 7))
    )
    mid = [s for s in res.stations if 100 < s.chainage < 500]
    assert all(s.basis == "ratio" for s in mid)
    assert mid[0].depth["central"] == pytest.approx(mid[0].width / 10)


def test_line_drawn_upstream_is_reversed():
    dem, gt, line, _, _ = river()
    res = run_channel_bathymetry(dem, None, gt, [line[::-1]])
    assert any("reversed" in w for w in res.warnings)
    wse = [s.wse for s in res.stations if np.isfinite(s.wse)]
    assert wse[0] > wse[-1]
