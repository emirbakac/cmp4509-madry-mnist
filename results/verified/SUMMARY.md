# Independently recomputed statistics

Final CSV values are preserved. Percentages below use sample SD (ddof=1) across three training seeds.

| Method | Clean (%) | FGSM (%) | PGD100 x 5 (%) |
|---|---:|---:|---:|
| Baseline | 99.03 ± 0.05 | 11.49 ± 1.25 | 0.00 ± 0.00 |
| PGD40 | 38.12 ± 46.37 | 34.41 ± 39.95 | 31.28 ± 34.53 |
| Curriculum | 97.85 ± 0.30 | 93.91 ± 0.62 | 88.13 ± 0.87 |

Separate validation summaries use rounded author-reported percentages, not raw counts:

- PGD100 x 20: 84.96 ± 1.37% (reported N=512).
- Full standard AutoAttack: 83.34 ± 1.59% (reported N=512).

The AutoAttack mean is 83.34% when calculated from the provided rounded percentages. Exact raw counts could change the last displayed digit; no counts were reconstructed as observations.

No full-model checkpoints or raw validation attack logs were available for independent replay.
