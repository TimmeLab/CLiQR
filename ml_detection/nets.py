"""
PyTorch reproductions of the MATLAB lick-detection networks.

MATLAB fakes 1-D convolution with a `convolution2dLayer([1 k], nFilters)` acting on a
`1 x N x 1` "image". That is exactly an `nn.Conv1d` with kernel length k. We reproduce
the two networks layer-for-layer so weights ported from `lickNets.mat` are meaningful.

Both networks begin with a zscore normalization using a SINGLE scalar mean/std, matching
MATLAB's `imageInputLayer(..., 'Normalization','zscore')` as trained. The scalars are stored
as buffers so they save/load with the model and can be refit during transfer learning.
"""
import torch
import torch.nn as nn


class LickBoutNet(nn.Module):
    """Coarse gate: does a 3 s (300-sample) window contain >=1 lick in its central 1 s?"""

    def __init__(self):
        super().__init__()
        # zscore constants (scalar); identity by default until real weights are loaded.
        self.register_buffer("norm_mean", torch.zeros(1))
        self.register_buffer("norm_std", torch.ones(1))

        # conv2d[1x15]x16 -> BN -> ReLU -> maxpool[1x4] stride4 -> conv2d[1x15]x32 -> BN -> ReLU
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, padding="same")
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, padding="same")
        self.bn2 = nn.BatchNorm1d(32)
        # After pool: 300 -> 75 samples, 32 channels -> flatten 32*75 = 2400
        self.fc1 = nn.Linear(32 * 75, 32)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = (x - self.norm_mean) / self.norm_std
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class LickPointNet(nn.Module):
    """Fine classifier: is the CENTER sample of a 21-sample window a lick?"""

    def __init__(self):
        super().__init__()
        self.register_buffer("norm_mean", torch.zeros(1))
        self.register_buffer("norm_std", torch.ones(1))

        # conv2d[1x5]x16 -> BN -> ReLU -> conv2d[1x5]x32 -> BN -> ReLU (NO pooling: needs precise localization)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding="same")
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(32)
        # No pool: 21 samples preserved, 32 channels -> flatten 32*21 = 672
        self.fc1 = nn.Linear(32 * 21, 32)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = (x - self.norm_mean) / self.norm_std
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
