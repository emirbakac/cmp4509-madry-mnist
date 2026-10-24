# Results and provenance

`FINAL_TEST_ALL_MODELS.csv` is the original nine-row measurement record for Baseline, PGD40, and Curriculum, each with seeds 0, 1, and 2. Preserve its original values. Accuracy columns contain fractions in `[0,1]`; each row records 10,000 test images, the selected epoch, and a checkpoint SHA-256 identifier.

The experimenters identify the frozen final protocol as MNIST test, L∞ ε=0.3, α=0.01, PGD100 with **five restarts**. Attack settings are not columns in the original CSV; raw evaluation logs/checkpoints are needed to independently verify the historical execution, not merely its arithmetic. The audit recomputes all method means and sample standard deviations (ddof=1), including FGSM, without retraining or replacing measurements.

Reported 512-image validation PGD100×20/AutoAttack summaries are a separate evidence category. In the absence of their raw logs, per-example records, and subset indices, they remain author-reported checks rather than independently replayed results. Never pool them with the final test.

`smoke_check/` contains historical software-check outputs from a tiny dataset. Its figures are not the final experiment or full training histories. Do not use them as main-study learning curves.

New training produces `history.csv` and checkpoints under its run directory. New evaluation produces unique subdirectories under `evaluations/`, with metric, per-example, environment/source, and checkpoint provenance; `latest_evaluation.json` points to the latest output. The summarizer treats repeat measurements as one seed only when checkpoint and numerical results agree; conflicts require investigation. Generated summary figures/tables are derived artifacts, not replacement raw records.

CSV locale warning: `0.975` means 97.5%, not 975. Use a parser with decimal `.` (for example `pandas.read_csv`) and never resave distorted spreadsheet displays over the original CSV. All final plotted uncertainty is sample SD across training seeds, not standard error or confidence intervals.
