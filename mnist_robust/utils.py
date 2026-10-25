"""Seed ve ortam kaydı: sonuçlar kullanılan koşullarla birlikte saklanır."""

from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch


def source_fingerprint():
    """Hash the executed package sources, including standalone notebook exports.

    This records future runs only; it cannot recover historical source versions.
    """
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(Path(__file__).parent.glob('*.py'))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'files': files}


def seed_everything(seed: int):
    """Aynı ortamda tekrar çalıştırılabilirlik; farklı donanımlar eşitlik garantisi vermez."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed, [0, 2**32) aralığında bir tamsayı olmalı.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Bazı cihazlarda deterministik implementasyonu olmayan operatörleri bildir.
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested="auto"):
    """İstenen cihaz yoksa sessizce fallback yapmak yerine anlaşılır hata verir."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device auto, cpu, cuda veya mps olmalı.")
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA bulunamadı. Colab/Kaggle GPU'yu etkinleştirin veya --device cpu kullanın.")
    if requested == "mps" and not has_mps:
        raise RuntimeError("Bu ortamda MPS kullanılamıyor; --device cpu kullanın.")
    return torch.device(requested)


def environment_info():
    """JSON'a yazılabilir temel sürüm/donanım bilgileri; torchvision import edilmez."""
    try:
        torchvision_version = version("torchvision")
    except PackageNotFoundError:
        torchvision_version = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torchvision": torchvision_version,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }
