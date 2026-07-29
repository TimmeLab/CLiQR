import numpy as np
import data_analysis as da


class _StubNet:
    def eval(self): return self


def test_ml_algorithm_writes_expected_datasets(monkeypatch, tmp_path):
    import h5py
    # Two animals with simple traces; stub detect_licks to return fixed times.
    # save_filtered_data (shared with basic_algorithm/hilbert_algorithm) flags
    # missing_data=True for non-control animals lacking 'consumed_vol'/'weight',
    # exactly as it would for real filter_data() output, so both are included
    # here to isolate this test to the ML-specific contract being verified.
    data_by_animal = {
        "A1": {"cap_data": np.zeros(1000), "time_data": np.linspace(0, 10, 1000),
               "used_start_idx": 0, "used_stop_idx": 999,
               "consumed_vol": 0.5, "weight": 20.0},
    }
    monkeypatch.setattr(da, "_load_ml_nets", lambda ckpt: (_StubNet(), _StubNet()))
    monkeypatch.setattr(da, "detect_licks",
                        lambda t, c, b, p: np.array([1.0, 2.0, 3.0]))
    out = tmp_path / "filtered.h5"
    with h5py.File(out, "w") as f:
        missing = da.ml_algorithm(data_by_animal, f, str(tmp_path / "log.txt"))
    assert missing is False
    # num_licks is an in-memory field on the data dict, not a dataset persisted
    # to HDF5 -- basic_algorithm/hilbert_algorithm never write it via
    # save_filtered_data either, and ml_algorithm mirrors that contract exactly.
    assert data_by_animal["A1"]["num_licks"] == 3
    with h5py.File(out, "r") as f:
        assert np.allclose(f["A1"]["lick_times"][()], [1.0, 2.0, 3.0])
