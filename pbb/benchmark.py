"""python -m pbb.benchmark --help"""
import argparse
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from pbb.bounds import PBBobj
from pbb.benchmark_bounds import IndependentBlockRisk, certificate
from pbb.benchmark_data import Images, NORMALIZATION, load_arrays, split_hash, split_indices
from pbb.benchmark_models import GaussianNetwork, make_model


PRESETS = {
    'mnist': dict(prior_epochs=30, prior_optimizer='adam', prior_lr=0.001,
                  prior_weight_decay=0.0001, sigma_prior=0.03),
    'cifar10': dict(prior_epochs=200, prior_optimizer='sgd', prior_lr=0.1,
                    prior_weight_decay=0.0005, sigma_prior=0.005),
    'cifar100': dict(prior_epochs=200, prior_optimizer='sgd', prior_lr=0.1,
                     prior_weight_decay=0.0005, sigma_prior=0.005),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', choices=PRESETS)
    p.add_argument('--data-root', help='Read-only mounted data directory; recursive discovery is supported')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    p.add_argument('--data-parallel', action='store_true', help='Use all visible CUDA GPUs, or --gpu-ids')
    p.add_argument('--gpu-ids', help='Visible CUDA device indices, e.g. 0,1')
    p.add_argument('--amp', action='store_true', help='CUDA mixed precision for training only; certification stays FP32')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--prior-fraction', type=float, choices=[0.5], default=0.5)
    p.add_argument('--batch-size', type=int, default=256, help='Total across GPUs')
    p.add_argument('--prior-epochs', type=int, default=None, help='30 MNIST; 200 CIFAR')
    p.add_argument('--prior-lr', type=float, default=None)
    p.add_argument('--posterior-epochs', type=int, default=100)
    p.add_argument('--posterior-lr', type=float, default=0.005)
    p.add_argument('--posterior-momentum', type=float, default=0.95)
    p.add_argument('--sigma-prior', type=float, default=None, help='0.03 MNIST; 0.005 CIFAR; fixed before B')
    p.add_argument('--pmin', type=float, default=1e-5)
    p.add_argument('--delta-pb', type=float, default=0.04)
    p.add_argument('--delta-mc', type=float, default=0.01)
    p.add_argument('--mc-draws', type=int, default=20000, help='Independent network/data blocks, rounded up to GPU count')
    p.add_argument('--mc-batch-size', type=int, default=32, help='Images per independent network draw, per GPU')
    p.add_argument('--test-draws', type=int, default=3, help='Diagnostic stochastic test passes')
    p.add_argument('--resume', action='store_true', help='Repeat original command with this flag to resume saved epochs')
    p.add_argument('--certify-only', action='store_true', help='Load completed posterior and saved configuration from --out')
    p.add_argument('--train-only', action='store_true', help='Save final posterior without running certification')
    p.add_argument('--smoke-test', action='store_true', help='Two batches per training stage; tiny MC/test checks, not a paper result')
    return p


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_checkpoint(path, **state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    # Distinct device streams are essential when counting replica MC blocks as
    # independent observations. manual_seed_all alone gives identical streams.
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                torch.cuda.manual_seed(seed + 1009 * (index + 1))
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def rng_state(device):
    state = dict(cpu=torch.get_rng_state())
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state_all()
    if device.type == 'mps':
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    torch.set_rng_state(state['cpu'])
    if device.type == 'cuda' and 'cuda' in state and len(state['cuda']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state['cuda'])
    if device.type == 'mps' and 'mps' in state:
        torch.mps.set_rng_state(state['mps'])


def select_device(args):
    kind = args.device
    if kind == 'auto':
        kind = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if kind == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; enable the Kaggle GPU accelerator')
    if kind == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable to this Python process')
    if args.data_parallel and kind != 'cuda':
        raise ValueError('--data-parallel requires CUDA')
    if args.gpu_ids and not args.data_parallel:
        raise ValueError('--gpu-ids requires --data-parallel')
    ids = [int(x) for x in args.gpu_ids.split(',')] if args.gpu_ids else list(range(torch.cuda.device_count()))
    if kind == 'cuda':
        if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= torch.cuda.device_count() for i in ids):
            raise ValueError('Invalid --gpu-ids')
        ids = ids if args.data_parallel else ids[:1]
        torch.cuda.set_device(ids[0])
        return torch.device(f'cuda:{ids[0]}'), ids
    return torch.device(kind), []


class TrainingPrecision(nn.Module):
    def __init__(self, model, amp):
        super().__init__()
        self.model, self.amp = model, amp

    def forward(self, x):
        # Enter autocast inside each DataParallel worker thread.
        with torch.autocast(device_type=x.device.type, dtype=torch.float16,
                            enabled=self.amp and x.is_cuda):
            return self.model(x)


def parallel(model, ids):
    return nn.DataParallel(model, device_ids=ids) if len(ids) > 1 else model


def loader(dataset, args, device, seed, shuffle=False, sampler=None, batch_size=None):
    return DataLoader(dataset, batch_size=batch_size or args.batch_size, shuffle=shuffle,
                      sampler=sampler, num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                      generator=torch.Generator().manual_seed(seed))


def fit(model, dataset, phase, args, device, ids, log):
    is_posterior = phase == 'posterior'
    epochs = args.posterior_epochs if is_posterior else args.prior_epochs
    lr = args.posterior_lr if is_posterior else args.prior_lr
    if is_posterior:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.posterior_momentum)
    elif args.prior_optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.prior_weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=args.prior_weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    forward = parallel(TrainingPrecision(model, args.amp), ids)
    pb = PBBobj(objective='fquad', pmin=args.pmin, delta=args.delta_pb,
                n_posterior=len(dataset), n_bound=len(dataset))
    start = 0
    progress = args.out / 'training.pt'
    if args.resume and progress.exists():
        state = torch.load(progress, map_location='cpu', weights_only=True)
        if state['phase'] == phase:
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            if state['scaler'] and scaler.is_enabled():
                scaler.load_state_dict(state['scaler'])
            restore_rng(state['rng'], device)
            start = state['epoch']
            log(event='resume', phase=phase, completed_epochs=start)
    for epoch in range(start, epochs):
        forward.train()
        begin = time.time()
        total, total_loss, total_errors = 0, 0.0, 0
        train_loader = loader(dataset, args, device, args.seed + epoch + (10000 if is_posterior else 0), shuffle=True)
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = forward(x)
            if is_posterior:
                bounded_ce = F.nll_loss(F.log_softmax(logits.float(), 1).clamp_min(math.log(args.pmin)), y) / math.log(1 / args.pmin)
                # Source-module KL once, outside autocast and replica forwards.
                loss = pb.bound(bounded_ce, model.compute_kl(), len(dataset))
            else:
                loss = F.cross_entropy(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite {phase} loss at epoch {epoch + 1}, step {step}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # GradScaler must be allowed to skip an overflowing AMP step and
            # lower its scale. FP32 non-finite gradients are genuine failures.
            nn.utils.clip_grad_norm_(model.parameters(), 10.0,
                                     error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer)
            scaler.update()
            total += len(y)
            total_loss += loss.item() * len(y)
            total_errors += (logits.argmax(1) != y).sum().item()
            if args.smoke_test and step == 1:
                break
        scheduler.step()
        log(event='epoch', phase=phase, epoch=epoch + 1, epochs=epochs,
            objective=total_loss / total, error_percent=100 * total_errors / total,
            kl_nats=model.compute_kl().detach().item() if is_posterior else 0.0,
            seconds=time.time() - begin)
        save_checkpoint(progress, phase=phase, epoch=epoch + 1, model=cpu_state(model),
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        scaler=scaler.state_dict(), rng=rng_state(device))
    save_checkpoint(args.out / f'{phase}.pt', model=cpu_state(model))
    progress.unlink(missing_ok=True)


@torch.no_grad()
def test_error(model, dataset, args, device, ids, stochastic=False):
    class Predict(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x, sample=stochastic) if isinstance(self.net, GaussianNetwork) else self.net(x)
    forward = parallel(Predict(model), ids).eval()
    errors, total = 0, 0
    for step, (x, y) in enumerate(loader(dataset, args, device, args.seed + 50000)):
        x, y = x.to(device), y.to(device)
        errors += (forward(x).argmax(1) != y).sum().item()
        total += len(y)
        if args.smoke_test and step == 1:
            break
    return errors / total


@torch.no_grad()
def certify(posterior, bound_data, args, device, ids, log):
    seed_all(args.seed + 200000)
    devices = max(1, len(ids))
    draws = math.ceil(args.mc_draws / devices) * devices
    observations = draws * args.mc_batch_size
    sampler = RandomSampler(bound_data, replacement=True, num_samples=observations,
                            generator=torch.Generator().manual_seed(args.seed + 300000))
    batches = loader(bound_data, args, device, args.seed + 400000, sampler=sampler,
                     batch_size=devices * args.mc_batch_size)
    forward = parallel(IndependentBlockRisk(posterior), ids).eval()
    total, count = 0.0, 0
    for x, y in batches:
        # Full precision, no augmentation, fixed BN statistics, independent data
        # blocks and Gaussian draws on distinct GPU RNG streams.
        values = forward(x.to(device), y.to(device)).detach().cpu().double()
        total += values.sum().item()
        count += values.numel()
        if count % (devices * 250) == 0 or count == draws:
            log(event='certification_progress', mc_draws=count, total_draws=draws,
                empirical_risk_mc=total / count)
    assert count == draws
    kl = posterior.compute_kl(cpu_double=True).item()
    result = certificate(total / count, kl, len(bound_data), count, args.delta_pb, args.delta_mc)
    result.update(mc_images=observations, mc_images_per_draw=args.mc_batch_size,
                  estimator='independent Gaussian network + uniform-with-replacement B block',
                  is_smoke_test=args.smoke_test)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.out = args.out.expanduser().resolve()
    if args.certify_only:
        saved = json.loads((args.out / 'config.json').read_text())
        if saved.get('architecture_version') != 2:
            raise ValueError('Old custom-head checkpoint: train the standard backbone in a new --out directory')
        for key, value in saved.items():
            if key not in {'out', 'device', 'data_parallel', 'gpu_ids', 'num_workers', 'cpu_threads',
                           'resume', 'certify_only', 'train_only', 'data_root', 'amp'}:
                setattr(args, key, value)
        args.data_root = args.data_root or saved['data_root']
    if args.dataset is None or args.data_root is None:
        raise ValueError('--dataset and --data-root are required for a new run')
    for key, value in PRESETS[args.dataset].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.smoke_test:
        args.prior_epochs = args.posterior_epochs = 1
        args.batch_size = min(args.batch_size, 16)
        args.mc_draws, args.mc_batch_size, args.test_draws = 16, 4, 1
    for name in ('prior_epochs', 'posterior_epochs', 'batch_size', 'mc_draws', 'mc_batch_size', 'test_draws', 'cpu_threads'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    for name in ('sigma_prior', 'prior_lr', 'posterior_lr'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not 0 < args.pmin < 1 or not 0 <= args.posterior_momentum < 1 or args.num_workers < 0:
        raise ValueError('Invalid pmin, momentum or worker count')
    certificate(0.0, 0.0, 2, 1, args.delta_pb, args.delta_mc)  # validate confidence before training
    if args.certify_only and args.train_only:
        raise ValueError('--certify-only and --train-only cannot be combined')
    torch.set_num_threads(args.cpu_threads)
    device, ids = select_device(args)
    if args.batch_size < max(1, len(ids)):
        raise ValueError('Training batch size must be at least the GPU count')
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = {**vars(args), 'out': str(args.out), 'architecture_version': 2}
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / 'config.json'
    if config_path.exists() and not (args.resume or args.certify_only):
        raise FileExistsError(f'{args.out} already contains a run; use --resume or a new --out')
    if args.resume:
        saved = json.loads(config_path.read_text())
        if saved.get('architecture_version') != 2:
            raise ValueError('Old custom-head checkpoint: train the standard backbone in a new --out directory')
        runtime = {'out', 'data_root', 'device', 'data_parallel', 'gpu_ids', 'num_workers', 'cpu_threads',
                   'resume', 'certify_only', 'train_only', 'amp'}
        changes = [key for key in saved if key not in runtime and saved[key] != config[key]]
        if changes:
            raise ValueError(f'Resume configuration differs: {changes}; repeat the original training arguments')
    elif not args.certify_only:
        write_json(config_path, config)

    def log(**row):
        row['time'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        line = json.dumps(row, allow_nan=False)
        print(line, flush=True)
        with (args.out / 'run.jsonl').open('a') as handle:
            handle.write(line + '\n')

    log(event='start', dataset=args.dataset, device=str(device), gpu_ids=ids, smoke_test=args.smoke_test)
    x, y, tx, ty, resolved_root = load_arrays(args.dataset, args.data_root)
    a, b = split_indices(len(x), args.split_seed)
    fingerprint = split_hash(a, b)
    split_path = args.out / 'split.npz'
    if split_path.exists():
        with np.load(split_path) as previous:
            if not np.array_equal(previous['prior_indices'], a.numpy()) or not np.array_equal(previous['bound_indices'], b.numpy()):
                raise ValueError('Saved split does not match the requested split')
    else:
        np.savez(split_path, prior_indices=a.numpy(), bound_indices=b.numpy())
    log(event='data', prior_examples=len(a), bound_examples=len(b), test_examples=len(tx),
        split_sha256=fingerprint, resolved_data_root=resolved_root)
    prior_data = Images(x, y, a, args.dataset, augment=args.dataset != 'mnist')
    bound_data = Images(x, y, b, args.dataset)
    test_data = Images(tx, ty, torch.arange(len(tx)), args.dataset)
    prior = make_model(args.dataset).to(device)
    if (args.resume or args.certify_only) and (args.out / 'prior.pt').exists():
        prior.load_state_dict(torch.load(args.out / 'prior.pt', map_location='cpu', weights_only=True)['model'])
    elif args.certify_only:
        raise FileNotFoundError('Missing prior.pt')
    else:
        fit(prior, prior_data, 'prior', args, device, ids, log)
    prior.eval()
    posterior = GaussianNetwork(prior, args.sigma_prior).to(device)
    if (args.resume or args.certify_only) and (args.out / 'posterior.pt').exists():
        posterior.load_state_dict(torch.load(args.out / 'posterior.pt', map_location='cpu', weights_only=True)['model'])
    elif args.certify_only:
        raise FileNotFoundError('Missing posterior.pt')
    else:
        seed_all(args.seed + 10000)
        fit(posterior, bound_data, 'posterior', args, device, ids, log)
    if args.train_only:
        log(event='training_complete', posterior_checkpoint=str(args.out / 'posterior.pt'))
        return
    result = certify(posterior, bound_data, args, device, ids, log)
    # Save the certificate before optional diagnostic passes, so interruption of
    # test evaluation never discards a completed certificate.
    result.update(dataset=args.dataset, split_sha256=fingerprint, n_prior=len(a),
                  architecture_version=2,
                  backbone='PBB CNNet4l' if args.dataset == 'mnist' else 'standard WRN-28-4',
                  adaptation='all conv, linear and BN affine parameters Gaussian; BN statistics frozen from A',
                  posterior_selection='final epoch of fixed schedule', normalization=NORMALIZATION[args.dataset],
                  torch_version=str(torch.__version__), python_version=platform.python_version(),
                  device=str(device), gpu_ids=ids, resolved_data_root=resolved_root)
    try:
        result['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                                       cwd=Path(__file__).resolve().parents[1]).strip()
    except (OSError, subprocess.CalledProcessError):
        result['git_commit'] = None
    write_json(args.out / 'metrics.json', result)
    result['prior_test_error_percent'] = 100 * test_error(prior, test_data, args, device, ids)
    result['posterior_mean_test_error_percent'] = 100 * test_error(posterior, test_data, args, device, ids)
    result['gibbs_test_error_percent'] = 100 * sum(test_error(posterior, test_data, args, device, ids, True)
                                                  for _ in range(args.test_draws)) / args.test_draws
    write_json(args.out / 'metrics.json', result)
    log(event='complete', **result)


if __name__ == '__main__':
    main()
