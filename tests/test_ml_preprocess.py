import numpy as np
from ml_detection.preprocess import resample_to_100hz, offset_global, offset_window, FS


def test_resample_produces_uniform_100hz_grid():
    # Irregular 2 s of data; resample must land on a 0.01 s grid.
    t = np.array([0.0, 0.017, 0.031, 0.5, 1.0, 2.0])
    y = np.array([10.0, 9.0, 8.0, 5.0, 5.0, 5.0])
    tr, yr = resample_to_100hz(t, y)
    assert FS == 100
    dt = np.diff(tr)
    assert np.allclose(dt, 0.01, atol=1e-9)
    assert tr[0] == 0.0
    assert yr.shape == tr.shape
    # Verify linear interpolation is working correctly: test on a perfectly linear source.
    t_linear = np.array([0.0, 1.0, 2.0])
    y_linear = np.array([0.0, 10.0, 20.0])  # y = 10*t exactly
    tr_linear, yr_linear = resample_to_100hz(t_linear, y_linear)
    np.testing.assert_allclose(yr_linear, 10.0 * tr_linear, atol=1e-9)


def test_resample_stable_at_absolute_epoch_time():
    # Regression: recording timestamps can be absolute Unix epoch seconds (~1.7e9). Building the
    # 100 Hz grid at that magnitude with a naive cumulative step loses float precision and drifts
    # by up to ~1 sample. Verify the grid built on absolute-epoch time matches the same grid built
    # on zero-based time to well under one sample (1e-6 s), i.e. the epoch offset is handled
    # without cancellation error.
    epoch = 1.7e9
    # 5 s of regular 100 Hz-ish source samples on an absolute epoch base.
    t_abs = epoch + np.arange(0, 5.0, 1.0 / FS)
    # A ramp so interpolation is exact and any grid drift shows up as a value error.
    y = (t_abs - epoch) * 3.0
    tr_abs, yr_abs = resample_to_100hz(t_abs, y)
    tr_rel, yr_rel = resample_to_100hz(t_abs - epoch, y)
    # The grids must agree once the epoch offset is removed, to far better than one sample.
    assert np.max(np.abs((tr_abs - epoch) - tr_rel)) < 1e-6
    # And the interpolated values must match the exact ramp (no drift-induced error).
    np.testing.assert_allclose(yr_abs, 3.0 * (tr_abs - epoch), atol=1e-4)


def test_offsets_put_max_at_zero():
    y = np.array([-3.0, -1.0, -7.0])
    assert offset_global(y).max() == 0.0
    assert offset_window(y).max() == 0.0
    np.testing.assert_allclose(offset_global(y), y - y.max())
