import torch
from ml_detection.nets import LickBoutNet, LickPointNet


def test_bout_net_output_shape():
    net = LickBoutNet()
    x = torch.randn(4, 1, 300)          # [batch, channels, samples]
    y = net(x)
    assert y.shape == (4, 2)            # 2-class logits per window


def test_point_net_output_shape():
    net = LickPointNet()
    x = torch.randn(7, 1, 21)
    y = net(x)
    assert y.shape == (7, 2)


def test_norm_buffers_apply():
    # With mean=0, std=1 the normalization is identity; a hand-set mean shifts input.
    net = LickPointNet()
    net.norm_mean.fill_(0.0)
    net.norm_std.fill_(1.0)
    x = torch.zeros(1, 1, 21)
    # forward must run without error and produce finite logits
    y = net(x)
    assert torch.isfinite(y).all()
