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


def test_offsets_put_max_at_zero():
    y = np.array([-3.0, -1.0, -7.0])
    assert offset_global(y).max() == 0.0
    assert offset_window(y).max() == 0.0
    np.testing.assert_allclose(offset_global(y), y - y.max())
