import gzip
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from pbb import benchmark
from pbb.benchmark import parallel, seed_all, TrainingPrecision
from pbb.benchmark_bounds import binary_kl, certificate, IndependentBlockRisk, inverse_kl_upper
from pbb.benchmark_data import IMAGENET_NORMALIZATION, Images, idx_names, locate, read_idx, split_hash, split_indices
from pbb.benchmark_models import GaussianLayer, GaussianNetwork, GaussianParameter, imagenet_resnet18, make_model, WideBlock


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gaussian_kl_matches_torch_and_updates_without_forward(self):
        q = GaussianParameter(torch.tensor([0.1, -0.3, 0.7]), 0.03)
        self.assertEqual(q.kl().item(), 0)
        with torch.no_grad():
            q.mu.add_(torch.tensor([0.01, -0.04, 0.02]))
            q.rho.add_(torch.tensor([0.1, -0.5, 0.0]))
        sigma = nn.functional.softplus(q.rho).detach().double()
        posterior = torch.distributions.Normal(q.mu.detach().double(), sigma)
        prior = torch.distributions.Normal(q.prior_mu.double(), q.prior_sigma.double())
        expected = torch.distributions.kl_divergence(posterior, prior).sum().item()
        self.assertAlmostEqual(q.kl(cpu_double=True).item(), expected, places=10)
        q.kl().backward()
        self.assertTrue(torch.isfinite(q.mu.grad).all())
        self.assertTrue(torch.isfinite(q.rho.grad).all())

    def test_matched_architectures_and_gaussian_mean_equivalence(self):
        for dataset, classes in [('mnist', 10), ('cifar10', 10), ('cifar100', 100)]:
            with self.subTest(dataset=dataset):
                prior = make_model(dataset).eval()
                x = torch.randn(2, 1, 28, 28) if dataset == 'mnist' else torch.randn(2, 3, 32, 32)
                self.assertFalse(hasattr(prior, 'projection'))
                self.assertFalse(any(isinstance(m, nn.LayerNorm) for m in prior.modules()))
                if dataset != 'mnist':
                    self.assertEqual([(m.in_features, m.out_features) for m in prior.modules()
                                      if isinstance(m, nn.Linear)], [(256, classes)])
                    self.assertEqual(sum(isinstance(m, WideBlock) for m in prior.modules()), 12)
                    self.assertEqual(prior.conv.out_channels, 16)
                    self.assertEqual([prior.blocks[i].conv1.out_channels for i in (0, 4, 8)], [64, 128, 256])
                    self.assertEqual(prior.blocks[4].conv1.stride, (2, 2))
                else:
                    from pbb.models import CNNet4l
                    reference = CNNet4l(dropout_prob=0).eval()
                    reference.load_state_dict(prior.state_dict())
                    with torch.no_grad():
                        torch.testing.assert_close(prior(x).log_softmax(1), reference(x))
                posterior = GaussianNetwork(prior, 0.005).eval()
                with torch.no_grad():
                    actual = posterior(x, sample=False)
                    torch.testing.assert_close(actual, prior(x), atol=1e-6, rtol=1e-5)
                    self.assertEqual(actual.shape, (2, classes))
                    self.assertEqual(posterior.compute_kl().item(), 0.0)
                    self.assertFalse(torch.equal(posterior(x), posterior(x)))

    def test_imagenet_transfer_is_standard_resnet18_with_fresh_cifar_head(self):
        torch.manual_seed(912)
        c10 = imagenet_resnet18(10, pretrained=False)
        torch.manual_seed(912)
        c100 = imagenet_resnet18(100, pretrained=False)
        self.assertEqual(c10.conv1.kernel_size, (7, 7))
        self.assertEqual(c10.conv1.stride, (2, 2))
        self.assertEqual(c10.fc.in_features, 512)
        self.assertEqual(c10.fc.out_features, 10)
        self.assertEqual(c100.fc.out_features, 100)
        torch.testing.assert_close(c10.fc.weight, c100.fc.weight[:10])
        self.assertFalse(any(isinstance(m, nn.LayerNorm) for m in c10.modules()))
        x = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
        y = torch.tensor([2, 7])
        training = Images(x, y, [0], 'cifar10', augment=True, imagenet_resnet=True)
        evaluation = Images(x, y, [0], 'cifar10', imagenet_resnet=True)
        self.assertEqual(training[0][0].shape, (3, 224, 224))
        self.assertEqual(evaluation[0][0].shape, (3, 224, 224))
        self.assertEqual(tuple(evaluation.normalize.mean), IMAGENET_NORMALIZATION[0])

    def test_imagenet_transfer_uses_full_cifar_set_without_reading_it_for_prior(self):
        x = torch.arange(32 * 3 * 32 * 32).remainder(256).byte().reshape(32, 3, 32, 32)
        y = torch.arange(32).remainder(10)
        arrays = (x, y, x[:8], y[:8], 'synthetic')
        events = []

        def transfer_model(classes, weights_path=None, pretrained=True):
            events.append(('prior', classes, pretrained))
            return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, classes))

        def transfer_arrays(*_):
            events.append(('data',))
            return arrays

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(benchmark, 'imagenet_resnet18', side_effect=transfer_model), \
             patch.object(benchmark, 'load_arrays', side_effect=transfer_arrays), \
             patch('builtins.print'):
            benchmark.main(['--dataset', 'cifar10', '--prior-source', 'imagenet',
                            '--data-root', directory, '--out', str(Path(directory) / 'transfer'),
                            '--device', 'cpu', '--num-workers', '0', '--cpu-threads', '2', '--smoke-test'])
            metrics = json.loads((Path(directory) / 'transfer' / 'metrics.json').read_text())
        self.assertEqual(events[:2], [('prior', 10, True), ('data',)])
        self.assertEqual(metrics['prior_source'], 'imagenet')
        self.assertEqual(metrics['n_prior'], 0)
        self.assertEqual(metrics['n_bound'], 32)
        self.assertEqual(metrics['normalization'], [list(v) for v in IMAGENET_NORMALIZATION])

    def test_bn_buffers_frozen_but_affine_and_conv_receive_gradients(self):
        prior = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4),
                              nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3))
        prior.train()
        prior(torch.randn(8, 3, 8, 8))  # A updates running statistics.
        posterior = GaussianNetwork(prior.eval(), 0.01).train()
        original = {key: val.clone() for key, val in posterior.named_buffers()}
        opt = torch.optim.SGD(posterior.parameters(), lr=0.01)
        loss = posterior(torch.randn(8, 3, 8, 8)).square().mean() + posterior.compute_kl() / 1000
        loss.backward()
        for name, value in posterior.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
        opt.step()
        for name, value in posterior.named_buffers():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self.assertGreater(posterior.compute_kl().item(), 0)
        # Prediction on an example must not depend on other examples in its batch.
        x = torch.randn(5, 3, 8, 8)
        torch.testing.assert_close(posterior(x, sample=False)[:1], posterior(x[:1], sample=False))

    def test_layer_norm_affine_is_in_kl(self):
        posterior = GaussianNetwork(nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 2)), 0.01)
        layer = posterior.net[0]
        self.assertIsInstance(layer, GaussianLayer)
        with torch.no_grad():
            layer.weight.mu[0] += 0.01
        self.assertAlmostEqual(posterior.compute_kl().item(), 0.5, places=4)

    def test_split_is_disjoint_complete_and_rng_independent(self):
        for n in (60000, 50000):
            a, b = split_indices(n, 42)
            torch.randn(15)
            aa, bb = split_indices(n, 42)
            self.assertEqual(len(a), n // 2)
            self.assertEqual(len(b), n // 2)
            self.assertEqual(len(torch.unique(torch.cat((a, b)))), n)
            self.assertEqual(split_hash(a, b), split_hash(aa, bb))

    def test_idx_and_readonly_nested_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'MNIST' / 'raw'
            root.mkdir(parents=True)
            image_file = root / 'train-images.idx3-ubyte.gz'
            with gzip.open(image_file, 'wb') as f:
                f.write(struct.pack('>IIII', 2051, 2, 28, 28) + bytes([100]) * (2 * 28 * 28))
            label_file = root / 'train-labels.idx1-ubyte'
            label_file.write_bytes(struct.pack('>II', 2049, 2) + bytes([2, 7]))
            self.assertEqual(locate(directory, idx_names('train', True)), root.resolve())
            x, y = read_idx(image_file, True), read_idx(label_file, False)
            self.assertEqual(x.shape, (2, 1, 28, 28))
            self.assertEqual(y.tolist(), [2, 7])
            dataset = Images(x, y, [1], 'mnist')
            self.assertEqual(dataset[0][1].item(), 7)
            torch.testing.assert_close(dataset[0][0], dataset[0][0])

    def test_inverse_kl_boundaries_and_monotonicity(self):
        self.assertEqual(inverse_kl_upper(1, 0.3), 1)
        self.assertEqual(inverse_kl_upper(0.3, 0), 0.3)
        self.assertAlmostEqual(inverse_kl_upper(0, 0.2), 1 - math.exp(-0.2))
        for q in (0.001, 0.1, 0.5, 0.999):
            upper = inverse_kl_upper(q, 0.01)
            self.assertGreater(upper, q)
            self.assertAlmostEqual(binary_kl(q, upper), 0.01, places=9)
        small = certificate(0.02, 10, 25000, 20000, 0.04, 0.01)
        fewer_data = certificate(0.02, 10, 12500, 20000, 0.04, 0.01)
        fewer_draws = certificate(0.02, 10, 25000, 1000, 0.04, 0.01)
        self.assertLess(small['certificate'], fewer_data['certificate'])
        self.assertLess(small['certificate'], fewer_draws['certificate'])
        self.assertAlmostEqual(small['confidence'], 0.95)

    def test_mc_reports_blocks_not_images(self):
        posterior = GaussianNetwork(nn.Linear(3, 2), 0.1)
        block = IndependentBlockRisk(posterior)
        values = block(torch.randn(17, 3), torch.zeros(17, dtype=torch.long))
        self.assertEqual(values.shape, (1,))
        self.assertTrue(0 <= values.item() <= 1)

    def test_interrupted_epoch_resume_and_certify_saved_posterior(self):
        # Small synthetic integration fixture; production loaders separately
        # enforce official dataset sizes. Verify real checkpoint/optimizer/RNG
        # recovery, not merely the existence of checkpoint files.
        x = torch.arange(32 * 784).remainder(256).byte().reshape(32, 1, 28, 28)
        y = torch.arange(32).remainder(10)
        arrays = (x, y, x[:8], y[:8], 'synthetic')
        with tempfile.TemporaryDirectory() as directory:
            interrupted, complete = Path(directory) / 'interrupted', Path(directory) / 'complete'
            common = ['--dataset', 'mnist', '--data-root', directory, '--device', 'cpu',
                      '--num-workers', '0', '--cpu-threads', '2', '--prior-epochs', '2',
                      '--posterior-epochs', '2', '--batch-size', '8', '--mc-draws', '8',
                      '--mc-batch-size', '4', '--test-draws', '1', '--train-only']
            original_save = benchmark.save_checkpoint

            def interrupt_after_save(path, **state):
                original_save(path, **state)
                if state.get('phase') == 'posterior' and state.get('epoch') == 1:
                    raise RuntimeError('Simulated process interruption')

            with patch.object(benchmark, 'load_arrays', return_value=arrays), \
                 patch.object(benchmark, 'make_model', side_effect=lambda _: nn.Sequential(nn.Flatten(), nn.Linear(784, 10))), \
                 patch('builtins.print'):
                with patch.object(benchmark, 'save_checkpoint', side_effect=interrupt_after_save):
                    with self.assertRaisesRegex(RuntimeError, 'Simulated'):
                        benchmark.main(common + ['--out', str(interrupted)])
                self.assertTrue((interrupted / 'training.pt').exists())
                benchmark.main(common + ['--out', str(interrupted), '--resume'])
                self.assertFalse((interrupted / 'training.pt').exists())
                benchmark.main(common + ['--out', str(complete)])
                resumed = torch.load(interrupted / 'posterior.pt', weights_only=True)['model']
                uninterrupted = torch.load(complete / 'posterior.pt', weights_only=True)['model']
                for name in resumed:
                    torch.testing.assert_close(resumed[name], uninterrupted[name], rtol=0, atol=0)
                benchmark.main(['--certify-only', '--out', str(interrupted), '--device', 'cpu', '--num-workers', '0'])
            metrics = json.loads((interrupted / 'metrics.json').read_text())
            self.assertEqual(metrics['n_bound'], 16)
            self.assertEqual(metrics['mc_draws'], 8)
            self.assertEqual(metrics['mc_images'], 32)
            self.assertGreaterEqual(metrics['certificate'], metrics['empirical_gibbs_risk_mc'])

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'Requires two CUDA GPUs')
    def test_dataparallel_sampling_and_gradient(self):
        seed_all(17)
        posterior = GaussianNetwork(make_model('mnist'), 0.03).cuda(0)
        x = torch.randn(2, 1, 28, 28, device='cuda:0').repeat(2, 1, 1, 1)
        for amp in (False, True):
            posterior.zero_grad(set_to_none=True)
            forward = parallel(TrainingPrecision(posterior, amp), [0, 1])
            logits = forward(x)
            self.assertFalse(torch.equal(logits[:2], logits[2:]))
            loss = logits.float().square().mean() + posterior.compute_kl() / 30000
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in posterior.parameters()))
        block = parallel(IndependentBlockRisk(posterior), [0, 1])
        self.assertEqual(block(x, torch.zeros(4, device='cuda:0', dtype=torch.long)).shape, (2,))


if __name__ == '__main__':
    unittest.main()
