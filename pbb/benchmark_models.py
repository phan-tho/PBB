"""Matched CNN/WRN/ResNet architectures and diagonal Gaussian posteriors.

Unlike the original example models, KL is computed from live parameters and is
never cached by forward(). This also makes DataParallel replica handling safe.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class MNISTCNN(nn.Module):
    """PBB's original CNNet4l, with dropout disabled and raw logits returned."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


class WideBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.shortcut = (nn.Identity() if in_channels == out_channels and stride == 1
                         else nn.Conv2d(in_channels, out_channels, 1, stride, bias=False))

    def forward(self, x):
        pre = F.relu(self.bn1(x))
        shortcut = x if isinstance(self.shortcut, nn.Identity) else self.shortcut(pre)
        return shortcut + self.conv2(F.relu(self.bn2(self.conv1(pre))))


class WideResNet28x4(nn.Module):
    """Zero-dropout WRN-28-4 following the authors' models/wide-resnet.lua.

    Reference: https://github.com/szagoruyko/wide-residual-networks
    Convolution initialization follows models/utils.lua (fan-in, no bias).
    """
    def __init__(self, classes):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        blocks = []
        previous = 16
        for group, width in enumerate((64, 128, 256)):
            for block in range(4):
                stride = 2 if group > 0 and block == 0 else 1
                blocks.append(WideBlock(previous, width, stride))
                previous = width
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256, classes)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = F.relu(self.bn(self.blocks(self.conv(x))))
        return self.fc(F.avg_pool2d(x, 8).flatten(1))


def make_model(dataset):
    if dataset == 'mnist':
        return MNISTCNN()
    if dataset == 'cifar10':
        return WideResNet28x4(10)
    if dataset == 'cifar100':
        return WideResNet28x4(100)
    raise ValueError(f'Unsupported dataset: {dataset}')


def imagenet_resnet18(classes, weights_path=None, pretrained=True):
    """Standard torchvision ImageNet-1K ResNet-18 with a fresh CIFAR head.

    The new classifier is intentionally initialized after loading ImageNet
    weights. Its random seed is set by the caller before reading CIFAR data,
    making the complete parameter prior independent of the downstream sample.
    """
    if classes not in (10, 100):
        raise ValueError(f'ResNet-18 transfer supports 10 or 100 classes, got {classes}')
    try:
        if not pretrained:
            model = resnet18(weights=None)
        elif weights_path is None:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = resnet18(weights=None)
            state = torch.load(weights_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError) as error:
        source = 'the Torchvision cache' if weights_path is None else str(weights_path)
        raise RuntimeError(
            f'Could not load ImageNet-1K ResNet-18 weights from {source}. '
            'Attach/download the official state-dict and pass --imagenet-weights PATH.'
        ) from error
    model.fc = nn.Linear(model.fc.in_features, classes)
    return model


class GaussianParameter(nn.Module):
    def __init__(self, value, sigma):
        super().__init__()
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('Prior standard deviation must be positive and finite')
        self.mu = nn.Parameter(value.detach().clone())
        self.rho = nn.Parameter(torch.full_like(value, math.log(math.expm1(sigma))))
        self.register_buffer('prior_mu', value.detach().clone())
        # Use exactly the same floating-point sigma at initialization, giving KL=0.
        self.register_buffer('prior_sigma', F.softplus(self.rho.detach()).clone())
        self.sampling = True

    def forward(self):
        if not self.sampling:
            return self.mu
        return self.mu + F.softplus(self.rho) * torch.randn_like(self.mu)

    def kl(self, cpu_double=False):
        sigma = F.softplus(self.rho)
        mu, prior_mu, prior_sigma = self.mu, self.prior_mu, self.prior_sigma
        if cpu_double:
            mu, sigma, prior_mu, prior_sigma = (
                value.detach().cpu().double() for value in (mu, sigma, prior_mu, prior_sigma))
        t = 2 * (sigma.log() - prior_sigma.log())
        return 0.5 * (((mu - prior_mu) / prior_sigma).square() + torch.expm1(t) - t).sum()


class GaussianLayer(nn.Module):
    def __init__(self, original, sigma):
        super().__init__()
        self.weight = GaussianParameter(original.weight, sigma)
        self.bias = None if original.bias is None else GaussianParameter(original.bias, sigma)
        if isinstance(original, nn.Conv2d):
            self.kind = 'conv'
            self.options = (original.stride, original.padding, original.dilation, original.groups)
        elif isinstance(original, nn.Linear):
            self.kind = 'linear'
        elif isinstance(original, nn.BatchNorm2d):
            self.kind = 'bn'
            self.eps = original.eps
            self.register_buffer('running_mean', original.running_mean.detach().clone())
            self.register_buffer('running_var', original.running_var.detach().clone())
        elif isinstance(original, nn.LayerNorm):
            self.kind = 'ln'
            self.options = (original.normalized_shape, original.eps)
        else:
            raise TypeError(type(original))

    def forward(self, x):
        weight = self.weight()
        bias = None if self.bias is None else self.bias()
        if self.kind == 'conv':
            return F.conv2d(x, weight, bias, *self.options)
        if self.kind == 'linear':
            return F.linear(x, weight, bias)
        if self.kind == 'bn':
            # No updates from B, including while the posterior is in train mode.
            return F.batch_norm(x, self.running_mean, self.running_var, weight, bias,
                                training=False, momentum=0.0, eps=self.eps)
        shape, eps = self.options
        return F.layer_norm(x, shape, weight, bias, eps)


class GaussianNetwork(nn.Module):
    def __init__(self, prior, sigma):
        super().__init__()
        self.net = copy.deepcopy(prior)

        def convert(module):
            for name, child in list(module.named_children()):
                if isinstance(child, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)):
                    setattr(module, name, GaussianLayer(child, sigma))
                else:
                    convert(child)
        convert(self.net)

    def forward(self, x, sample=True):
        for module in self.modules():
            if isinstance(module, GaussianParameter):
                module.sampling = sample
        return self.net(x)

    def compute_kl(self, cpu_double=False):
        return sum(module.kl(cpu_double) for module in self.modules()
                   if isinstance(module, GaussianParameter))
