"""Untargeted FGSM ve L-infinity PGD; tehdit modeli ham [0, 1] piksellerde.

Değerlendirmede her örnek için bulunan herhangi bir yanlış sınıflandırmayı
koruruz. Böylece son iterasyonun tesadüfen tekrar doğru sınıflandırılması,
ölçülen robust accuracy değerini yapay olarak yükseltmez.
"""

from contextlib import contextmanager
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def _attack_mode(model: nn.Module):
    # Mixed train/eval alt modül ayarlarını da geri yükle. BN istatistikleri
    # saldırı üretilirken güncellenmez; dış eğitim adımı bağımsız kalır.
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def _nonnegative_finite(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} sonlu ve negatif olmayan bir sayı olmalı.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} sonlu ve negatif olmayan bir sayı olmalı.")


def _validate_inputs(x, y, eps):
    _nonnegative_finite("eps", eps)
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise ValueError("x kayan noktalı bir Tensor olmalı.")
    if x.ndim < 2 or x.shape[0] == 0:
        raise ValueError("x boş olmayan bir batch içermeli.")
    if not isinstance(y, torch.Tensor) or y.ndim != 1 or len(y) != len(x):
        raise ValueError("y her giriş için bir sınıf etiketi içermeli.")
    if y.dtype != torch.long or y.device != x.device:
        raise ValueError("y torch.long olmalı ve x ile aynı cihazda bulunmalı.")
    if not torch.isfinite(x).all() or torch.any(x < 0) or torch.any(x > 1):
        raise ValueError("Saldırı ham, sonlu [0, 1] pikseller bekliyor.")


def _project(candidate, clean, eps):
    return torch.maximum(torch.minimum(candidate, clean + eps), clean - eps).clamp(0, 1)


def _select_candidate(best, best_loss, best_wrong, candidate, logits, losses, y):
    """Önce başarılı saldırı, sonra cross-entropy: örnek bazında seçim."""
    with torch.no_grad():
        wrong = logits.argmax(dim=1).ne(y)
        replace = (wrong & ~best_wrong) | ((wrong == best_wrong) & (losses > best_loss))
        broadcast = replace.reshape((-1,) + (1,) * (candidate.ndim - 1))
        best = torch.where(broadcast, candidate.detach(), best)
        best_loss = torch.where(replace, losses.detach(), best_loss)
        best_wrong = torch.where(replace, wrong, best_wrong)
    return best, best_loss, best_wrong


def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float):
    """Bir gradient-sign adımı; modelin modu ve parametre .grad'ları korunur."""
    _validate_inputs(x, y, eps)
    clean = x.detach()
    if eps == 0:
        return clean.clone()
    with _attack_mode(model), torch.enable_grad():
        candidate = clean.clone().requires_grad_(True)
        loss = F.cross_entropy(model(candidate), y, reduction="sum")
        gradient, = torch.autograd.grad(loss, candidate, only_inputs=True)
        return _project(candidate.detach() + eps * gradient.sign(), clean, eps).detach()


def pgd_linf(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.3,
    alpha: float = 0.01,
    steps: int = 40,
    restarts: int = 1,
    random_start: bool = True,
    track_best: bool = True,
):
    """Untargeted L-infinity PGD.

    track_best=True: temiz girişi, her iterasyonu ve restart'ı karşılaştırır;
    bulunan yanlış sınıflandırmayı her örnek için korur. Değerlendirme içindir.
    track_best=False: tek restart'ın son iterasyonunu döndürür. Adversarial
    training için kullanılır ve restarts=1 gerektirir.

    Dönen Tensor graph'tan ayrıdır. Giriş ya da model parametrelerinin .grad
    alanlarına yazılmaz. torch.no_grad() içinden de çağrılabilir; inference_mode
    içinde çağırmayın, çünkü saldırı giriş türevi gerektirir.
    """
    _validate_inputs(x, y, eps)
    _nonnegative_finite("alpha", alpha)
    for name, value, minimum in (("steps", steps, 0), ("restarts", restarts, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} en az {minimum} olan bir tamsayı olmalı.")
    if not isinstance(random_start, bool) or not isinstance(track_best, bool):
        raise ValueError("random_start ve track_best bool olmalı.")
    if not track_best and restarts != 1:
        raise ValueError("track_best=False son iterasyon içindir; restarts=1 kullanın.")
    clean = x.detach()
    if eps == 0:
        return clean.clone()

    with _attack_mode(model):
        if track_best:
            with torch.no_grad():
                logits = model(clean)
                best = clean.clone()
                best_loss = F.cross_entropy(logits, y, reduction="none")
                best_wrong = logits.argmax(dim=1).ne(y)

        for _ in range(restarts):
            if random_start:
                candidate = _project(clean + torch.empty_like(clean).uniform_(-eps, eps), clean, eps)
            else:
                candidate = clean.clone()

            for _ in range(steps):
                with torch.enable_grad():
                    candidate = candidate.detach().requires_grad_(True)
                    logits = model(candidate)
                    losses = F.cross_entropy(logits, y, reduction="none")
                    if track_best:
                        best, best_loss, best_wrong = _select_candidate(
                            best, best_loss, best_wrong, candidate, logits, losses, y
                        )
                    gradient, = torch.autograd.grad(losses.sum(), candidate, only_inputs=True)
                candidate = _project(candidate.detach() + alpha * gradient.sign(), clean, eps).detach()

            if track_best:
                with torch.no_grad():
                    logits = model(candidate)
                    losses = F.cross_entropy(logits, y, reduction="none")
                    best, best_loss, best_wrong = _select_candidate(
                        best, best_loss, best_wrong, candidate, logits, losses, y
                    )

        return (best if track_best else candidate).detach()
