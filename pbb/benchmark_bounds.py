"""Finite-sample certificates for a fixed posterior using independent MC blocks."""
import math

import torch
from torch import nn


def binary_kl(q, p):
    if q == p:
        return 0.0
    if p == 0 or p == 1:
        return math.inf
    return ((q * math.log(q / p) if q else 0.0)
            + ((1 - q) * math.log((1 - q) / (1 - p)) if q < 1 else 0.0))


def inverse_kl_upper(q, budget):
    if not math.isfinite(q) or not 0 <= q <= 1 or not math.isfinite(budget) or budget < 0:
        raise ValueError(f'Invalid inverse KL arguments: {q}, {budget}')
    if q == 1 or budget == 0:
        return q
    if q == 0:
        return -math.expm1(-budget)
    lo, hi = q, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if binary_kl(q, mid) <= budget:
            lo = mid
        else:
            hi = mid
    # Return the conservative upper bracket, not the lower numerical approximation.
    return hi


def certificate(empirical_mc, kl, n_bound, mc_draws, delta_pb, delta_mc):
    if n_bound < 2 or mc_draws < 1 or kl < 0 or not math.isfinite(kl):
        raise ValueError('Invalid certificate sample sizes or KL')
    if not 0 < delta_pb < 1 or not 0 < delta_mc < 1 or delta_pb + delta_mc >= 1:
        raise ValueError('Confidence failure probabilities must be positive and sum to less than one')
    # Each draw is a bounded independent block mean, NOT an independent image.
    empirical_upper = inverse_kl_upper(empirical_mc, math.log(1 / delta_mc) / mc_draws)
    complexity = (kl + math.log(2 * math.sqrt(n_bound) / delta_pb)) / n_bound
    endpoint = inverse_kl_upper(empirical_upper, complexity)
    return dict(empirical_gibbs_risk_mc=empirical_mc, empirical_gibbs_risk_upper=empirical_upper,
                kl_nats=kl, kl_per_bound_example=kl / n_bound, n_bound=n_bound,
                mc_draws=mc_draws, delta_pb=delta_pb, delta_mc=delta_mc,
                confidence=1 - delta_pb - delta_mc, certificate=endpoint,
                certificate_percent=100 * endpoint)


class IndependentBlockRisk(nn.Module):
    """One fresh Gaussian network per replica and independently sampled data block.

    Each call returns a single block mean per replica. With replacement sampling
    from B, these means are iid in [0,1] with expectation empirical Gibbs risk.
    A Chernoff/kl bound applies with the NUMBER OF BLOCKS as sample size.
    """
    def __init__(self, posterior):
        super().__init__()
        self.posterior = posterior

    def forward(self, x, y):
        logits = self.posterior(x, sample=True)
        return (logits.argmax(1) != y).float().mean().reshape(1)
