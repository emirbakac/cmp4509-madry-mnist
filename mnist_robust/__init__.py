"""MNIST üzerinde küçük ve tekrar çalıştırılabilir adversarial training projesi."""

from .attacks import fgsm, pgd_linf
from .model import SmallCNN

__all__ = ["SmallCNN", "fgsm", "pgd_linf"]
__version__ = "0.1.0"
