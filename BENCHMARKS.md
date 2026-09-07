# Parameter-space PBB: matched split-prior benchmarks

This runner adapts [Pérez-Ortiz et al., Tighter Risk Certificates for Neural
Networks](https://jmlr.org/papers/volume22/20-879/20-879.pdf) to the specified
original PBB MNIST CNN and standard CIFAR WRN-28-4. It reuses the original `PBBobj.bound` quadratic
objective (`fquad`), with a bounded cross-entropy surrogate and a diagonal
Gaussian distribution over weights. The new model and evaluation paths avoid
the legacy examples' cached KL, split-size, and batch-averaging issues.

## Kaggle setup

Enable **GPU T4 x2** and Internet (for cloning/installing), and add the dataset
for the run:

- MNIST: [`hojjatk/mnist-dataset`](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
- CIFAR-10: [`pankrzysiu/cifar10-python`](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python).
- CIFAR-100: [`fedesoriano/cifar100`](https://www.kaggle.com/datasets/fedesoriano/cifar100).

Run this notebook cell once. Keep Kaggle's installed CUDA torch/torchvision;
do not install the historical conda requirements file.

```python
!git clone --branch codex/pbb-mnist-cifar-benchmarks https://github.com/phan-tho/PBB.git /kaggle/working/PBB
%cd /kaggle/working/PBB
!python -m pip install -r requirements-benchmark.txt
!python -c "import torch; print(torch.__version__); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

The runner requires PyTorch >=2.3 with a matching torchvision. It does not
download datasets or modify `/kaggle/input`. MNIST accepts gzip/uncompressed
IDX files, hyphen/dot filenames, and separate per-file subdirectories. CIFAR
accepts the original Python batches, either directly in the supplied root or
inside a nested directory. CIFAR-100 uses the 100 **fine labels**.

The paths below use the supplied Kaggle mount convention. If your notebook
uses a shorter mount such as `/kaggle/input/cifar100`, change only `--data-root`.
Passing `/kaggle/input` also works when it contains a unique matching dataset.

## Run one setting per notebook

MNIST:

```python
!python -u -m pbb.benchmark --dataset mnist \
    --data-root /kaggle/input/datasets/hojjatk/mnist-dataset \
    --out /kaggle/working/pbb-mnist \
    --device cuda --data-parallel --gpu-ids 0,1 --amp
```

CIFAR-10:

```python
!python -u -m pbb.benchmark --dataset cifar10 \
    --data-root /kaggle/input/datasets/pankrzysiu/cifar10-python \
    --out /kaggle/working/pbb-cifar10 \
    --device cuda --data-parallel --gpu-ids 0,1 --amp
```

CIFAR-100:

```python
!python -u -m pbb.benchmark --dataset cifar100 \
    --data-root /kaggle/input/datasets/fedesoriano/cifar100 \
    --out /kaggle/working/pbb-cifar100 \
    --device cuda --data-parallel --gpu-ids 0,1 --amp
```

## ImageNet ResNet-18 transfer on CIFAR

This separate setting uses the official Torchvision ImageNet-1K ResNet-18 as
the PBB prior and transfers it to CIFAR. The complete standard ResNet-18 is
adapted: after ImageNet weights are loaded, its 1000-way classifier is replaced
with one freshly seeded `Linear(512, K)` CIFAR classifier. The seed is fixed
before any CIFAR image or label is read, so that classifier is also part of the
data-independent prior. There is no projection, custom head, or LayerNorm.

Because this prior has not used downstream CIFAR data, all 50,000 CIFAR
training examples are available to train and certify the posterior (`n=50,000`);
the 50/50 A/B split applies only to the learned split-prior settings above.
BatchNorm running statistics remain the ImageNet values while all convolution,
linear, and BatchNorm affine parameters have Gaussian posterior distributions.

Torchvision downloads/caches the official `ResNet18_Weights.IMAGENET1K_V1`
weights by default. If Internet is disabled, attach the official state-dict as
a Kaggle dataset and pass its mounted path with `--imagenet-weights`.

CIFAR-10 transfer:

```python
!python -u -m pbb.benchmark --dataset cifar10 --prior-source imagenet \
    --data-root /kaggle/input/datasets/pankrzysiu/cifar10-python \
    --out /kaggle/working/pbb-imagenet-cifar10 \
    --device cuda --data-parallel --gpu-ids 0,1 --amp
```

CIFAR-100 transfer:

```python
!python -u -m pbb.benchmark --dataset cifar100 --prior-source imagenet \
    --data-root /kaggle/input/datasets/fedesoriano/cifar100 \
    --out /kaggle/working/pbb-imagenet-cifar100 \
    --device cuda --data-parallel --gpu-ids 0,1 --amp
```

For an attached state-dict, append
`--imagenet-weights /kaggle/input/<your-resnet18-weights>/resnet18-f37072fd.pth`
to either command.

Training resizes CIFAR images to 256, applies a random 224 crop and horizontal
flip, and uses ImageNet normalization. Certification and test diagnostics use
resize-to-256, center-crop-to-224, and the same normalization, matching the
official ResNet-18 evaluation transform. The full parameter-space KL for
ResNet-18 is much larger than for WRN-28-4, so its certificate may be loose;
run this fixed one-shot setting as a baseline rather than tuning it against
test results.

Each command trains the prior, trains the posterior, computes the certificate,
and evaluates diagnostic test errors. There is no hyperparameter sweep and no
test-based checkpoint selection. Logs stream once per epoch and periodically
during certification. GPU time has not been measured on T4 hardware here;
run the CIFAR settings in separate notebooks rather than assuming all three
fit in one session.

## Fixed presets

| Setting | MNIST | CIFAR-10 | CIFAR-100 |
|---|---|---|---|
| Backbone | Original PBB CNNet4l | Standard WRN-28-4 | Standard WRN-28-4 |
| A / B size | 30,000 / 30,000 | 25,000 / 25,000 | 25,000 / 25,000 |
| Prior epochs | 30 | 200 | 200 |
| Prior optimizer | Adam | Nesterov SGD, momentum 0.9 | Nesterov SGD, momentum 0.9 |
| Prior initial learning rate | 0.001 | 0.1 | 0.1 |
| Prior weight decay | 0.0001 | 0.0005 | 0.0005 |
| Gaussian prior standard deviation | 0.03 | 0.005 | 0.005 |
| Posterior epochs | 100 | 100 | 100 |
| Posterior optimizer | SGD, momentum 0.95 | same | same |
| Posterior initial learning rate | 0.005 | 0.005 | 0.005 |
| Total training batch size | 256 | 256 | 256 |

Both stages use cosine learning-rate decay to 1% of the initial rate. The
posterior uses no weight decay, `fquad` with the true KL coefficient 1, and
`pmin=1e-5`. Gradients are clipped to norm 10. Seed is 7; the independently
generated split uses seed 42. There is no dropout. CIFAR prior training uses
random crop/flip; posterior training and certification use no augmentation.
Normalization constants are recorded in `metrics.json` and are fixed before
training. CUDA AMP applies only to training; certification uses FP32 and its
final parameter KL is accumulated on CPU in float64.

The PBB objective, posterior learning rate/momentum, probability floor, and
prior scales follow choices considered in the original paper. The longer
SGD prior schedule for CIFAR is an architecture adaptation for WRN. These are
one-shot presets, not a claim of optimal hyperparameters or reproduced
published numerical results.

## Exact architecture and adaptation scope

MNIST uses the original PBB `CNNet4l` topology: Conv(1,32,3) → ReLU →
Conv(32,64,3) → ReLU → pool(2) → flatten(9216) → Linear(9216,128) → ReLU →
Linear(128,10). Dropout is disabled; the adapter returns logits.

CIFAR: initial 3×3 convolution with 16 channels, then three groups of four
pre-activation residual blocks with widths 64/128/256. Groups two and three
downsample by stride 2. Dimension-changing shortcuts apply a 1×1 convolution
to the pre-activated input; equal-dimension shortcuts are identities. Final
BN → ReLU → 8×8 average pooling → Linear(256,K). There is zero dropout,
no feature projection, no LayerNorm, and no tanh. This follows the WRN
authors' [wide-resnet.lua](https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua)
and [initialization utilities](https://github.com/szagoruyko/wide-residual-networks/blob/master/models/utils.lua):
convolution biases disabled, fan-in He normal initialization, classifier bias zero.

The prior is trained only on A. After A, a Gaussian prior is centered at its
weights and the posterior starts at that same distribution. Posterior means
and standard deviations for **all convolution, linear, and BatchNorm affine
parameters** may adapt on B. The parameter KL includes
all of them, including biases where present. BatchNorm running means and
variances are copied from A and never updated on B, even in training mode.
The prior tensors are registered buffers and are never optimized.

This is the parameter-space baseline with network-wide adaptation. It differs
from the paper's output-space method, which freezes its backbone and score
head before B. Report that distinction; do not describe this run as a
frozen-backbone Gaussian-head control. PBB's original examples did not implement
WRN/BatchNorm: freezing running statistics is the explicit certification policy
of this extension, not a claim about a WRN experiment in the original PBB paper.
The model's sampled top-one predictor is the
certified object, not its posterior-mean classifier or ensemble vote.

## Certificate and Monte Carlo accounting

The final endpoint is a PAC-Bayes-kl upper bound on population top-one Gibbs
risk, using **n = |B|**, the full unscaled parameter KL, and no test examples.
Default failure probabilities are `delta_pb=0.04` and `delta_mc=0.01`, giving
at least **95% confidence for each reported setting**. This is not a joint
95% guarantee across the three settings.

The posterior is fixed at the final epoch before certification starts. For
each of M=20,000 independent draws, draw one Gaussian network and a block of
32 indices uniformly **with replacement** from B. Evaluate the average 0–1
loss on that block without augmentation. This gives an observation Z in
[0,1]. Conditional on B and the selected posterior Q:

```
E[Z] = (1 / |B|) sum_{i in B} E_{w ~ Q}[1{argmax f_w(x_i) != y_i}]
```

Independent network/data blocks therefore estimate exactly the empirical
Gibbs risk needed by the PBB theorem. A bounded-variable Chernoff inequality
gives its upper confidence limit by solving:

```
r_upper = kl_inverse_upper(mean(Z), log(1 / delta_mc) / M)
certificate = kl_inverse_upper(
    r_upper, (KL(Q || P) + log(2 sqrt(|B|) / delta_pb)) / |B|
)
```

The MC sample size is **20,000 blocks, not 640,000 images**. On two GPUs each
forward contributes two independent blocks, with separately seeded CUDA RNG
streams. The draw count is rounded upward to a multiple of the GPU count.
BatchNorm is fixed so that a prediction depends only on that image and the
sampled weights. This evaluation adapts the original full-dataset MC loops
to reduce WRN cost while explicitly charging for the MC approximation. It is
more conservative than treating every image in a shared-weight batch as an
independent weight draw, which would be incorrect.

Do not select among repeated certificate evaluations by taking the minimum
without appropriate confidence accounting. A single run uses one fixed
posterior and one final certificate. Prior scales are fixed before B; they
are not optimized using B or test results.

## Outputs and recovery

Architecture version 3 adds the ImageNet ResNet-18 transfer protocol. Use a
new output directory for these runs. Old checkpoints cannot be resumed or
certified with this version.

Each output directory contains:

- `metrics.json`: use **`certificate_percent`** for the table. Also includes
  empirical MC risk, its upper limit, parameter KL, KL/|B|, sample counts,
  confidence allocation, split hash, runtime versions, and diagnostic errors.
- `config.json`, `run.jsonl`, and (for split-prior runs) `split.npz`: effective
  hyperparameters, exact A/B indices where applicable, and epoch/evaluation logs.
- `prior.pt`, `posterior.pt`: portable state-dict checkpoints.
- `training.pt`: last completed epoch including optimizer/scheduler/scaler
  and torch RNG state, retained only while a training stage is unfinished.

Repeat the original command with `--resume` to recover from the last completed
epoch or skip completed training stages. Training settings and the split-prior
indices, where applicable, must match. Changing hardware/worker counts can
change numerical trajectories.
Keep the output directory available between Kaggle sessions to resume.

Use `--train-only` to stop after training. Later certify the saved posterior:

```python
!python -u -m pbb.benchmark --certify-only \
    --out /kaggle/working/pbb-cifar10 \
    --data-root /kaggle/input/datasets/pankrzysiu/cifar10-python \
    --device cuda --data-parallel --gpu-ids 0,1
```

`--certify-only` restores training and certification settings from the saved
configuration. It permits runtime/data-root changes, not silent changes to
the prior, split, or MC budget.

## Validation

```bash
python -m unittest discover -s tests -v
```

The tests cover Gaussian KL against PyTorch distributions, gradient flow,
mean-model equivalence for all three architectures, frozen normalization
buffers, LayerNorm KL, split accounting, IDX loading, inverse-KL boundaries,
and MC block counting. A two-CUDA-GPU test exercises DataParallel sampling,
gradient reduction, and AMP, and is skipped when that hardware is unavailable.

Short local MPS integration check using the user's existing MNIST files:

```bash
python -m pbb.benchmark --dataset mnist \
  --data-root '/Users/mac/Documents/PAC Bayes KD/experiments/data/MNIST' \
  --out runs/mnist-mps-smoke --device mps --num-workers 0 --smoke-test
```

`--smoke-test` uses two tiny batches per training stage and tiny certification
and test budgets. Its metrics explicitly say `is_smoke_test=true`; it is not
a paper result. A shortened full-data training run likewise validates the
pipeline but does not replace the fixed benchmark presets.
