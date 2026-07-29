import os
import numpy as np
import torch
import pytest
from ml_detection.weights_io import load_matlab_nets

REFS = "ml_detection/checkpoints/parity_refs.npz"
MAT = "ML Detection MATLAB Code/lickNets.mat"


@pytest.mark.skipif(not os.path.exists(REFS), reason="parity refs not generated")
def test_pytorch_matches_matlab_to_1e4():
    refs = np.load(REFS)
    bout, point = load_matlab_nets(MAT)
    with torch.no_grad():
        pb = torch.softmax(bout(torch.tensor(refs["bout_in"]).unsqueeze(1)), dim=1).numpy()
        pp = torch.softmax(point(torch.tensor(refs["point_in"]).unsqueeze(1)), dim=1).numpy()
    assert np.max(np.abs(pb - refs["bout_out"])) < 1e-4
    assert np.max(np.abs(pp - refs["point_out"])) < 1e-4
