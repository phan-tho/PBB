"""Read-only loaders for Kaggle CIFAR Python batches and raw MNIST IDX files."""
import gzip
import hashlib
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


NORMALIZATION = {
    'mnist': ((0.1307,), (0.3081,)),
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610)),
    'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

IMAGENET_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def locate(root, names):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Data root does not exist: {root}')
    direct = [root / name for name in names if (root / name).is_file()]
    matches = direct or sorted({p for name in names for p in root.rglob(name) if p.is_file()})
    # A directory can contain both the gzip archive and its decompressed file.
    parents = {p.parent for p in matches}
    if len(parents) != 1:
        raise FileNotFoundError(f'Expected one directory containing {names} under {root}; '
                                f'found {sorted(map(str, parents))}. Pass a more specific --data-root.')
    return next(iter(parents))


def read_idx(path, images):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as handle:
        raw = handle.read()
    magic, count = struct.unpack('>II', raw[:8])
    if images:
        if magic != 2051:
            raise ValueError(f'Invalid MNIST image header: {path}')
        rows, cols = struct.unpack('>II', raw[8:16])
        return torch.from_numpy(np.frombuffer(raw[16:], np.uint8).copy()).reshape(count, 1, rows, cols)
    if magic != 2049:
        raise ValueError(f'Invalid MNIST label header: {path}')
    return torch.from_numpy(np.frombuffer(raw[8:], np.uint8).copy()).long().reshape(count)


def idx_names(prefix, images):
    suffix = 'images' if images else 'labels'
    kind = 'idx3-ubyte' if images else 'idx1-ubyte'
    return [f'{prefix}-{suffix}{sep}{kind}{gz}' for sep in ('-', '.') for gz in ('', '.gz')]


def load_arrays(dataset, root):
    if dataset == 'mnist':
        directory = Path(root).expanduser().resolve()
        arrays = []
        for prefix in ('train', 't10k'):
            for images in (True, False):
                # Some Kaggle mirrors put each IDX file in its own subdirectory.
                file_directory = locate(root, idx_names(prefix, images))
                matches = [file_directory / name for name in idx_names(prefix, images)
                           if (file_directory / name).is_file()]
                arrays.append(read_idx(matches[0], images))
    else:
        directory = locate(root, ['data_batch_1'] if dataset == 'cifar10' else ['train'])
        groups = ([f'data_batch_{i}' for i in range(1, 6)], ['test_batch']) if dataset == 'cifar10' else (['train'], ['test'])
        arrays = []
        for names in groups:
            images, labels = [], []
            for name in names:
                # These are the official, user-provided CIFAR Python batch files.
                with open(directory / name, 'rb') as handle:
                    batch = pickle.load(handle, encoding='bytes')
                images.append(np.asarray(batch[b'data'], dtype=np.uint8).reshape(-1, 3, 32, 32))
                labels.extend(batch[b'labels' if dataset == 'cifar10' else b'fine_labels'])
            arrays.extend((torch.from_numpy(np.concatenate(images)), torch.tensor(labels, dtype=torch.long)))
    x, y, tx, ty = arrays
    expected = (60000, 10000) if dataset == 'mnist' else (50000, 10000)
    if (len(x), len(tx)) != expected or len(y) != len(x) or len(ty) != len(tx):
        raise ValueError(f'Unexpected official dataset sizes: {(len(x), len(y), len(tx), len(ty))}')
    classes = 100 if dataset == 'cifar100' else 10
    for target in (y, ty):
        if target.min() < 0 or target.max() >= classes:
            raise ValueError('Invalid labels (CIFAR-100 must use fine_labels)')
    return x, y, tx, ty, str(directory)


class Images(Dataset):
    def __init__(self, x, y, indices, dataset, augment=False, imagenet_resnet=False):
        self.x, self.y = x, y
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        if imagenet_resnet:
            # ResNet18_Weights.IMAGENET1K_V1 evaluation preprocessing is
            # Resize(256), CenterCrop(224), and ImageNet normalization.
            self.augment = (transforms.Compose([transforms.Resize(256, antialias=True),
                                                 transforms.RandomCrop(224),
                                                 transforms.RandomHorizontalFlip()]) if augment
                            else transforms.Compose([transforms.Resize(256, antialias=True),
                                                     transforms.CenterCrop(224)]))
            self.normalize = transforms.Normalize(*IMAGENET_NORMALIZATION)
        else:
            self.augment = (transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip()]) if augment else None)
            self.normalize = transforms.Normalize(*NORMALIZATION[dataset])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        x = self.x[i].float().div(255)
        if self.augment is not None:
            x = self.augment(x)
        return self.normalize(x), self.y[i]


def split_indices(n, seed):
    # Independent of model initialization and augmentation RNG; shared by all stages.
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return permutation[:n // 2], permutation[n // 2:]


def split_hash(a, b):
    return hashlib.sha256(torch.cat((a, b)).numpy().astype('<i8').tobytes()).hexdigest()
