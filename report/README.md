# Audited report sources

This folder is a working copy of the original `overleaf/` report. The original files have been preserved. The supplied article layout, section structure and bibliography style are retained; instructional guide boxes are disabled. `main.tex` is the main document and `references.bib` supplies its references.

## Evidence and limits

- Main-result tables use the preserved nine-row `../results/FINAL_TEST_ALL_MODELS.csv`. All three seeds and both low-accuracy PGD40 runs are retained. Means and sample standard deviations (`ddof=1`) were recomputed, including FGSM.
- The historical test protocol is documented as full MNIST test, epsilon 0.3, alpha 0.01, PGD100 with **five** random starts. Attack settings are not fields in that original CSV. Config defaults alone cannot independently establish which overrides produced it.
- Recorded checkpoint SHA-256 strings are retained; absent historical checkpoint bytes prevent hash verification. Full histories, per-example predictions, runtime manifests and original curriculum-transfer artifacts were not available in the audit snapshot.
- The PGD100×20 and full-standard-AutoAttack figures for a stated common 512-image validation subset are **author-reported supplementary evidence**. Raw logs, subset indices, exact counts, package revision and evaluation seeds were not available. They are not newly executed or independently verified measurements.
- Validation summaries are calculated from the supplied rounded percentages. The AutoAttack mean is 83.34% using those inputs; integer counts compatible with the displayed values could instead round the mean to 83.33%. No counts have been fabricated to resolve this rounding distinction.
- The historical PGD10 checkpoint transfer and the integrated 3+7 epoch scheduler are distinguished. Their intended schedules agree, but historical checkpoint identity and exact replay equivalence have not been established.
- The Square Attack warning and approximately 82.42% higher-budget seed-0 result remain author-reported because their logs are absent. Gradient masking is not claimed to have been ruled out.
- Contributions distinguish author-reported completed work from assigned reviews; the latter must be completed and confirmed by the named students before submission.

## Build

The report references generated PDF figures in `figures/`. From the repository root, `python tools/verify_results.py` regenerates those figures from the preserved inputs. From this report directory, compile with a current pdfLaTeX + BibTeX environment, for example:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively run `pdflatex main`, `bibtex main`, then `pdflatex main` twice, or upload this folder to Overleaf and select `main.tex`. A PDF is verified only after successful compilation and visual inspection; the existence of an older PDF is not such verification.

The checked `report.pdf` was built from these sources with **Tectonic 0.17.0** on 25 October 2026. All ten pages were rendered and inspected; the final log has no overfull boxes, unresolved references or bibliography warnings. The engine's benign message that `inputenc` is ignored by a UTF-8 engine does not affect this build. `build_manifest.json` records source/figure/PDF hashes. Official-template provenance remains unverified despite successful compilation. Full details are in [the audit](../docs/FINAL_AUDIT.md).

## Submission blockers to resolve

The original instructor-distributed LaTeX template or its provenance must be checked against this source before declaring template compliance. The source comment alone does not establish that it is the official course template. Preserve this layout until the supplied template is located. Confirm the two assigned student reviews and author approval before submission. Recover and archive the historical experiment artifacts where possible; otherwise retain the explicit evidence limitations in the report.

The user-supplied report deadline is **13 December 2026, 23:59 Europe/Istanbul**. No slides are created by this audit.

## Primary sources checked

- Madry et al., ICLR 2018: https://arxiv.org/html/1706.06083v4 — robust objective, MNIST architecture and Table 1 comparison.
- MadryLab: https://github.com/MadryLab/mnist_challenge — `config.json` and `train.py`: PGD40/0.01/0.3, Adam 1e-4, batch 50, 100,000 optimizer steps.
- Goodfellow et al., ICLR 2015: https://research.google/pubs/explaining-and-harnessing-adversarial-examples/ and https://arxiv.org/abs/1412.6572 — FGSM and corrected conference-year bibliography metadata.
- Kingma and Ba, ICLR 2015: https://arxiv.org/abs/1412.6980 — Adam citation.
- Croce and Hein, ICML 2020: https://proceedings.mlr.press/v119/croce20b.html — title, authors, venue, pages and ensemble evaluation.
- AutoAttack official implementation: https://github.com/fra31/auto-attack — current standard components; `autoattack/checks.py` explains the Square warning. This source check does not identify the historical installed revision.
