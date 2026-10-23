# Measured outputs

Accuracies are percentages below; CSV stores fractions. ± is sample standard deviation across training seeds (ddof=1), not a confidence interval. One seed has undefined SD. SMOKE_ONLY is a pipeline check, never evidence of reproduction.

| Dataset/config/protocol | Mode | Seeds | Clean | FGSM | PGD | Combined | Status |
|---|---|---|---|---|---|---|---|
| MNIST/smoke/0992c611394368c0 | adversarial | 0 | 7.81 (SD undefined) | 7.81 (SD undefined) | 7.81 (SD undefined) | 7.81 (SD undefined) | SMOKE_ONLY |
| MNIST/smoke/0992c611394368c0 | baseline | 0 | 7.81 (SD undefined) | 7.81 (SD undefined) | 7.81 (SD undefined) | 7.81 (SD undefined) | SMOKE_ONLY |

Combined accuracy survives clean + FGSM + every retained PGD candidate. PGD accuracy is an empirical result for the stated attack; it does not certify robustness.
