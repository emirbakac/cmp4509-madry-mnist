# Final audit — 25 October 2026

## Readiness and scope

The preserved final CSV supports the reported three-seed arithmetic. The implementation, standalone notebooks, figures and manuscript have been audited and corrected. This is **not an independent replay of the nine historical GPU experiments**. Official course-template provenance and the students' final contribution/review confirmation remain submission blockers.

The user-supplied Desktop folder contained an original PDF, report sources/PNG figures, the final CSV and the course announcement, but was not a Git checkout. Those files were inventoried and backed up with SHA-256 digests before changes. A separate `repository/` clone was created alongside them from public `emirbakac/cmp4509-madry-mnist`, starting at `fd842b9`, on `codex/final-audit-2026-10-25`. Original report files and original measurements were not overwritten. No reset, force-push, merge or expensive retraining is part of this audit.

The inventory and source review covered every package module, five configs, both notebook generators, tests, requirements, ignore rules, documentation, existing results, report sources, figures and announcement. No original nine-model checkpoints, complete histories, validation attack logs, subset indices or original course-template archive were located in the available project files and targeted local search.

## Numerical verification

`results/FINAL_TEST_ALL_MODELS.csv` is byte-identical to the supplied `all_models.csv` and the starting Git revision. Its SHA-256 is:

```
654e9696f9a11c86e514b44ea9bd9d38fcfb81dabb281d7482b25f8e7de5d345
```

The audit checked nine rows; all three methods and seeds 0/1/2; selected epochs (baseline 10/8/10, direct PGD40 10/1/1, curriculum 10/10/10); 10,000 test examples per row; finite fractions in [0,1]; integer-compatible hit counts at that denominator; clean-correctness bounds; and 64-character hexadecimal checkpoint hashes. Hash syntax is verified; matching those hashes against absent checkpoint bytes is not possible.

Percentages below are mean ± **sample standard deviation, ddof=1**, across all three training seeds:

| Method | Clean | FGSM | PGD100 × 5 |
|---|---:|---:|---:|
| Baseline | 99.03 ± 0.05 | 11.49 ± 1.25 | 0.00 ± 0.00 |
| PGD40 from scratch | 38.12 ± 46.37 | 34.41 ± 39.95 | 31.28 ± 34.53 |
| PGD10 → PGD40 | 97.85 ± 0.30 | 93.91 ± 0.62 | 88.13 ± 0.87 |

Neither failed PGD40 seed was excluded. The two 11.35% endpoints and selected epoch 1 agree with the original records. Predictions of a single class, later-epoch behavior and gradient diagnostics are author reports; aggregate accuracies alone cannot establish those details or prove a dying-ReLU mechanism. A nearly constant classifier's equal clean/attacked accuracy is not evidence of useful robustness.

The compact CSV does not encode its attack settings. `FINAL_TEST_PROTOCOL.json` explicitly distinguishes the author's frozen test description (epsilon .3, alpha .01, 100 steps, **5** starts, 10,000 test images) from directly observable CSV fields. Historical code revision, evaluation RNG seed and environment remain unavailable.

Separate validation values are retained in `validation_reported.json` as **rounded author-reported evidence**, not independently rerun observations. On the stated common 512-image subset they yield PGD100 × **20**: 84.96 ± 1.37%, and full standard AutoAttack: 83.34 ± 1.59%. The latter mean uses the rounded supplied percentages; unavailable exact counts could change its final displayed digit. The reported higher-budget seed-0 Square result (about 82.42%) and warning lack original logs/settings. Gradient masking has not been conclusively excluded. Validation and test protocols are not pooled or treated as interchangeable.

## Scientific and implementation review

- Actual cached MNIST verifies 55,000 train / 5,000 validation / 10,000 test, zero train/validation index intersection, union 60,000, fixed split seed 2026 across training seeds, `float32` inputs in [0,1] and `int64` labels. Only `ToTensor()` is applied.
- The 1→16→32 convolutional network, two pools and 1568→128→10 head have **215,370** trainable parameters. The main optimizer is Adam, learning rate .001, batch 128, ten epochs.
- FGSM ascends input cross-entropy. PGD projects to the epsilon ball around the **original** clean input and clips to [0,1]. Parameter `.grad` values are not accumulated by attack generation; adversarial inputs are detached before the outer training update. Model modes are restored. Evaluation retains successful attacks per example across initial/intermediate/final iterates and random starts and requires clean correctness. Training uses one randomized start's final iterate.
- Baseline selection maximizes clean validation accuracy; adversarial selection maximizes PGD40 × 1 validation accuracy. Strict `>` retains the first checkpoint on exact ties. Test scores do not enter the inspected checkpoint-selection code. The records cannot independently prove every historical decision or the chronology of protocol freezing.
- Integrated curriculum uses PGD10 for epochs 1–3 and PGD40 for epochs 4–10. This matches the intended schedule, while the historical experiment transferred three-epoch probe weights, Adam/RNG state and epoch progress into separate continuation runs. Exact equivalence to those unavailable original artifacts is unverified.
- The original paper and official code use a larger model, different learning rate/batch/update budget and stronger 20-start paper evaluation. Our five-start test and exploratory curriculum are not exact numerical reproductions or certificates.

## Repairs made

1. New evaluations now use unique directories and a `latest_evaluation.json` pointer. Prior outputs remain intact; `metrics.csv` is written atomically only after the other outputs succeed. Future evaluations record source/environment and checkpoint provenance. An interrupted CSV write cannot publish a partial completion marker.
2. Resume fails explicitly when the requested last checkpoint is absent. Best checkpoints are saved before last checkpoints so interruption cannot lose a newly selected winner. Existing best-only/history-only artifacts prevent fresh training from overwriting them. Resume repairs missing/stale `history.csv` from checkpoint state even after the last epoch. Historical CSV histories with fewer columns remain readable when new `train_pgd_steps` entries are added.
3. Curriculum stage two now uses its declared configuration value instead of silently ignoring it. Actual training attack steps are recorded in history.
4. Summaries count identical reevaluations of the same checkpoint once per training seed and reject conflicting/unverifiable duplicates or invalid accuracy relationships. Distinct protocols/configs/source fingerprints remain separate.
5. Notebook final-mode guards enforce explicit PGD100 × 5 for main/curriculum. Existing config defaults (ten starts) remain unchanged for checkpoint compatibility. Repeated settings do not nest folders; outputs and archives have distinct locations. Every embedded source/config is regenerated deterministically.
6. README and supporting documentation describe completed experiments, historical execution provenance and actual evidence limits. A second standalone notebook recomputes recorded results/figures without training or attacks.
7. Figures use the source records, sample SD and individual seed points. Failed runs remain visible; independent seeds are not connected as a time trajectory. Validation uses an explicitly expanded-scale dot plot. Descriptive SD may extend beyond [0,100]; it is not a confidence interval.
8. Report figure references previously required missing PDF assets. Vector PDF and PNG assets are now generated from source data. Equations distinguish empirical and population risk, bibliography metadata and the unnumbered Code link are corrected, provenance/limitations are explicit, guide boxes are disabled, and contributions distinguish assigned reviews from completed work.

## Executed quality gates

Local verification environment: Python 3.13.0, torch 2.14.1, torchvision 0.29.1, NumPy 2.4.4, pandas 3.0.2, Matplotlib 3.10.9. CUDA was unavailable; Apple MPS was available. The environment was reused without replacing a working CUDA installation.

| Executed check | Actual outcome |
|---|---|
| `python -m unittest discover -s tests -v` | **37/37 passed**, 1.027 seconds on the final root run; tiny CPU models/synthetic data and mocked expensive notebook commands |
| `python -m compileall -q mnist_robust tests tools` | Passed |
| `python tools/build_notebook.py`, then `--check` | 39-cell notebook regenerated and synchronized; deterministic generation and every embedded package/config byte checked by tests |
| `python tools/build_results_notebook.py --check` | 15-cell analysis notebook synchronized; embedded original CSV/metadata/script match |
| Unmodified notebook 01 through NBClient, isolated directory, real cached MNIST | **All 39 cells passed**, default smoke on Apple MPS; both clean and adversarial training, checkpoint saving, validation, FGSM/PGD evaluation, summaries, figures and ZIP export executed |
| Entire notebook 01 rerun in a fresh kernel with the same output directory | **All 39 cells passed**; checkpoint/history/original metric SHA-256 values unchanged; no additional training; four separate evaluations and two unique ZIPs after two invocations; repeated evaluation counted once per training seed |
| Unmodified notebook 02 through NBClient in a separate empty directory | **All 15 cells passed**; its regenerated CSV/JSON/Markdown statistics equal the independently generated repository outputs byte-for-byte |
| Real MNIST split/dtype/forward checks | 55,000/5,000/10,000, disjoint fixed training/validation indices across seeds 0 and 2, raw [0,1] float32 images, int64 labels, 215,370 parameters, output shape (2,10) |
| `python tools/verify_results.py` | Preserved digest, nine records, all aggregates and three PNG/PDF figures verified/generated; original data untouched |
| Tectonic 0.17.0 build of `report/main.tex` | **10-page PDF compiled**; six bibliography entries resolve; no overfull boxes, undefined references or BibTeX warnings; only benign UTF-8-engine `inputenc` notice |
| PDF rendering and visual review | All pages and all three standalone figures inspected; misleading seed connections/truncated validation bars removed; captions, labels, tables, hyperlinks and title/author metadata checked |
| JSON, Markdown/local LaTeX references, privacy/file inventory, `git diff --check` | Passed; no dataset/checkpoint/private local path included in the public changes |
| Original-file preservation | All **11** inventoried Desktop files retain their initial SHA-256; final CSV remains identical to Desktop and Git `fd842b9` copies |

Regression coverage includes attack direction/projection/clean correctness, model-mode and RNG preservation, exact 3+7 curriculum boundary, first-on-tie checkpoint choice, uninterrupted versus resumed CPU state, completed-run behavior, old history columns, missing-last protection, best-only preservation, final-epoch CSV recovery, atomic-write interruption, repeat evaluation preservation/deduplication and invalid summaries. Seven notebook contract tests cover all five config choices, seed/mode selection, same-kernel settings reentry, frozen final100×5 commands and incomplete/unfrozen test guards.

Smoke used 256 training, 64 validation and 64 test images, one epoch, a reduced smoke model and PGD5 × 2 evaluation. Its accuracies are **not scientific results** and are not added to the final comparison. Full main/probe/curriculum GPU training, full-test PGD100 × 5 replay and AutoAttack were not run. Full config branches were exercised by lightweight contract tests, not represented as fully execution-verified CUDA experiments.

The first manuscript build exposed missing figure-format assets, two overfull paragraphs and bibliography warnings; these were repaired before the final build. A notebook test initially needed macOS symlink-path normalization; the corrected test passed. The committed-range whitespace check caught CRLF line endings in the newly derived summary CSV; the generator now uses LF, and notebook 02 was regenerated and executed again successfully. The original measurement CSV was not altered. No failing final quality gate remains. Successful compilation does not resolve missing official-template provenance.

## Remaining human actions and limits

1. Obtain the original Teams `CMP4509_report_template.zip` and compare this report's layout/preamble against it. The supplied source comment and visual resemblance do **not** establish official-template compliance. The course announcement requires that template and a PDF by **13 December 2026, 23:59 Türkiye time**.
2. The named students must complete their assigned theory/results reviews, confirm actual contributions and approve the manuscript. Assigned work is not represented as already completed.
3. If available, archive the nine historical checkpoints, full histories, Colab environment/runtime logs, exact validation indices and AutoAttack/Square settings. These would permit stronger provenance and replay checks. Until then, retain the report's limitations rather than inventing evidence.
4. Review the audit branch/PR before merging. This audit does not merge changes, submit the report, prepare slides, or rerun the expensive nine-model/AutoAttack experiments. Presentation remains 30 November or 7 December, 10 minutes including Q&A.

Google Drive authorization, Colab/Kaggle persistence and full CUDA experiments require the actual service/device and were not exercised by local checks. CPU epoch-boundary tests do not establish cross-device bitwise replay. Attacks provide empirical evidence, not a guarantee against every allowed perturbation.

## Changed and added paths

Paths below are relative to the repository; the original final CSV and all five configs are unchanged.

```text
.gitignore
README.md
docs/COURSE_DELIVERABLES.md
docs/EXPERIMENT_PLAN.md
docs/FINAL_AUDIT.md
docs/THEORY.md
docs/VALIDATION.md
docs/references.bib
mnist_robust/experiment.py
mnist_robust/summarize.py
mnist_robust/utils.py
notebooks/01_mnist_reproduction.ipynb
notebooks/02_verify_recorded_results.ipynb
report/README.md
report/build_manifest.json
report/figures/final_accuracy_comparison.pdf
report/figures/final_accuracy_comparison.png
report/figures/seed_robustness.pdf
report/figures/seed_robustness.png
report/figures/validation_attack_comparison.pdf
report/figures/validation_attack_comparison.png
report/main.tex
report/references.bib
report/report.pdf
results/FINAL_TEST_PROTOCOL.json
results/README.md
results/validation_reported.json
results/verified/SUMMARY.md
results/verified/final_summary.csv
results/verified/statistics.json
tests/test_audit_regressions.py
tests/test_notebook.py
tests/test_pipeline.py
tools/build_notebook.py
tools/build_results_notebook.py
tools/verify_results.py
```
