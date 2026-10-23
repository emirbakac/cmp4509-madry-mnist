"""Sabit MNIST train/validation ayrımı; test seti model seçimi için kullanılmaz."""

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split


def _seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _positive_int(name, value, allow_none=False):
    if allow_none and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} pozitif bir tamsayı olmalı.")


def _limit(dataset, count, name):
    if count is None:
        return dataset
    if count > len(dataset):
        raise ValueError(f"{name}={count}, mevcut örnek sayısını ({len(dataset)}) aşıyor.")
    return Subset(dataset, range(count))


def _synthetic_split(size, seed):
    # Sadece çevrimdışı yazılım smoke testi. MNIST verisi ya da bilimsel sonuç değil.
    # Ayrı RNG'ler, limit küçültüldüğünde örnek/etiket eşleşmesini sabit tutar.
    images = torch.rand(size, 1, 28, 28, generator=torch.Generator().manual_seed(seed))
    labels = torch.randint(10, (size,), generator=torch.Generator().manual_seed(seed + 1000))
    return TensorDataset(images, labels)


def make_loaders(
    data_dir,
    batch_size=128,
    seed=0,
    split_seed=2026,
    val_size=5000,
    train_limit=None,
    val_limit=None,
    test_limit=None,
    num_workers=0,
    synthetic=False,
):
    """{'train', 'val', 'test'} DataLoader sözlüğü döndürür.

    split_seed veriyi ayırır; seed yalnızca batch sırasını/worker RNG'lerini
    yönetir. Dolayısıyla farklı eğitim seed'leri aynı validation verisini görür.
    Piksel ölçeği [0, 1]'dir: epsilon=0.3 doğrudan bu ölçekte uygulanır.
    synthetic=True rastgele görüntü/etiket üretir ve yalnızca smoke test içindir.
    """
    _positive_int("batch_size", batch_size)
    _positive_int("val_size", val_size)
    for name, limit in (("train_limit", train_limit), ("val_limit", val_limit), ("test_limit", test_limit)):
        _positive_int(name, limit, allow_none=True)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers negatif olmayan bir tamsayı olmalı.")
    for name, value in (("seed", seed), ("split_seed", split_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
            raise ValueError(f"{name}, [0, 2**32) aralığında bir tamsayı olmalı.")

    if synthetic:
        datasets = {
            "train": _synthetic_split(train_limit or 256, split_seed),
            "val": _synthetic_split(val_limit or 64, split_seed + 1),
            "test": _synthetic_split(test_limit or 64, split_seed + 2),
        }
    else:
        # Ortamda torchvision bozuk olsa bile offline attack testleri çalışabilir.
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor

        data_dir = Path(data_dir).expanduser()
        full_train = MNIST(str(data_dir), train=True, download=True, transform=ToTensor())
        if val_size >= len(full_train):
            raise ValueError("val_size, toplam eğitim örnek sayısından küçük olmalı.")
        train, val = random_split(
            full_train,
            [len(full_train) - val_size, val_size],
            generator=torch.Generator().manual_seed(split_seed),
        )
        test = MNIST(str(data_dir), train=False, download=True, transform=ToTensor())
        datasets = {
            "train": _limit(train, train_limit, "train_limit"),
            "val": _limit(val, val_limit, "val_limit"),
            "test": _limit(test, test_limit, "test_limit"),
        }

    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(seed + index),
            persistent_workers=False,
        )
        for index, (name, dataset) in enumerate(datasets.items())
    }
