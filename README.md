# CMP4509 — Adversarial training, instability, and a PGD curriculum on MNIST

[Public repository](https://github.com/emirbakac/cmp4509-madry-mnist) · [Open standalone notebook in Colab](https://colab.research.google.com/github/emirbakac/cmp4509-madry-mnist/blob/main/notebooks/01_mnist_reproduction.ipynb) · [Original final-test CSV](results/FINAL_TEST_ALL_MODELS.csv) · [Recorded-results analysis notebook](notebooks/02_verify_recorded_results.ipynb) · [Audit and verification](docs/FINAL_AUDIT.md)

This project investigates the central methodology of [Madry et al. (ICLR 2018)](https://arxiv.org/abs/1706.06083): training against projected gradient descent (PGD) adversarial examples can improve resistance to norm-bounded attacks. A smaller PyTorch CNN and a limited training budget make this a **scaled-down methodological reproduction**, not an exact reproduction of the paper's numerical results.

We compare clean training, PGD40 adversarial training from random initialization, and a PGD10→PGD40 curriculum across seeds **0, 1, and 2**. The recorded measurements show strong attack vulnerability after clean training, severe variability after PGD40 training, and consistent learning in the three curriculum runs. Failed runs remain in every aggregate.

## Recorded final-test results

The original [nine-row CSV](results/FINAL_TEST_ALL_MODELS.csv) records **10,000 MNIST test images per model**, selected checkpoint epochs, accuracy fractions, and checkpoint SHA-256 identifiers. The experimenters identify its protocol as **L∞ ε=0.3, α=0.01, PGD100 with five random restarts**. The CSV does not itself store the attack settings or per-example predictions; those historical details require the original run logs for independent replay verification.

The following values are recomputed directly from that CSV. Units are percent; uncertainty is **sample standard deviation across three training seeds, ddof=1**, not standard error or a confidence interval.

| Training method | Clean | FGSM | PGD100×5 |
|---|---:|---:|---:|
| Standard baseline | 99.03 ± 0.05 | 11.49 ± 1.25 | 0.00 ± 0.00 |
| PGD40 from scratch | 38.12 ± 46.37 | 34.41 ± 39.95 | 31.28 ± 34.53 |
| PGD10→PGD40 curriculum | 97.85 ± 0.30 | 93.91 ± 0.62 | 88.13 ± 0.87 |

PGD40 seeds 1 and 2 each record **11.35% clean, FGSM, and PGD accuracy**, with selected epoch 1; seed 0 records 71.15% PGD accuracy at selected epoch 10. The experimenters report near-single-class prediction collapse and completion of ten training epochs. The CSV independently establishes these scores and selected epochs, but does not establish prediction histograms or the full training trajectory. Such low clean performance is not a successful robustness result. The cause of collapse, including any proposed dying-ReLU explanation, remains unproven.

The curriculum was introduced after observing PGD40 instability. The result supports this intervention under the studied settings; three seeds do not establish a universal ranking of training methods. No recorded measurement was changed during the audit. See [results provenance](results/README.md), [protocol metadata](results/FINAL_TEST_PROTOCOL.json), and [recomputed summaries](results/verified/SUMMARY.md).

## Separate stronger validation checks

The experimenters additionally report evaluations on the same **512 validation images**, using curriculum checkpoints:

| Reported validation attack | Seed 0 | Seed 1 | Seed 2 | Reported mean ± sample SD (%) |
|---|---:|---:|---:|---:|
| PGD100×20 | 84.38 | 86.52 | 83.98 | 84.96 ± 1.37 |
| Full standard AutoAttack | 82.62 | 85.16 | 82.23 | 83.34 ± 1.59 |

These rounded values are **author-reported secondary evidence**: raw attack logs, subset indices, exact counts, and original checkpoint files were not available for independent replay during this audit. They must not be relabeled as independently verified measurements. The reported AutoAttack components are APGD-CE, targeted APGD, targeted FAB, and Square; an increased Square budget reportedly reduced seed 0 to approximately 82.42%. Preserve the reported warning that Square found additional adversarial examples: **gradient masking has not been conclusively ruled out**. [AutoAttack reference and official code](https://github.com/fra31/auto-attack)

The 512-image validation checks and 10,000-image final test use different examples and budgets. Their scores are not interchangeable. None of these attack measurements is a mathematical robustness certificate. The repository's standard experiment command runs FGSM/PGD; it does not reproduce the historical AutoAttack session.

## Experimental setup

| Component | Configuration |
|---|---|
| Data | MNIST: fixed 55,000 train / 5,000 validation / 10,000 test; split seed 2026 |
| Inputs | `ToTensor()`, pixels in `[0,1]`, no mean/std normalization |
| Network | 5×5 convolutions 1→16→32, ReLU and 2×2 pooling after each; 1568→128→10 classifier |
| Trainable parameters | **215,370** |
| Optimizer | Adam, learning rate `0.001`, batch 128, ten epochs |
| Training attack | ε=0.3, α=0.01, one randomized PGD start; 40 steps or the curriculum below |
| Routine validation | PGD40×1, attack seed 12345, fixed validation split |
| Checkpoint selection | Baseline: highest clean validation accuracy; adversarial: highest PGD validation accuracy; exact ties keep the earlier checkpoint |
| Frozen final test | Clean, FGSM, PGD100×5 on selected `best.pt`; all 10,000 test images |

The code separates training/validation from test evaluation when `--train-only` is used. It does not use test scores to select checkpoints. The available CSV alone cannot reconstruct every historical model-selection decision.

**Curriculum provenance matters.** The reported historical runs first trained separate `main_pgd10_probe` models for three epochs, then transferred model, Adam, RNG, and epoch state into curriculum run directories and continued under PGD40. The current integrated `main_curriculum_pgd10_to_pgd40` configuration directly schedules **epochs 1–3: PGD10; epochs 4–10: PGD40**. It reproduces the intended schedule without requiring a separate probe run. The original transfer procedure and integrated execution are distinct provenance paths; checkpoint identity or bitwise equivalence has not been established. Do not copy a probe checkpoint into a new config directory as an undocumented resume workaround.

The [authors' MNIST implementation](https://github.com/MadryLab/mnist_challenge) uses 32/64 convolution channels, 1024 hidden units, Adam at `1e-4`, batch 50, and 100,000 updates. Our model capacity, optimization budget, initialization, validation selection, and final attack strength differ. The curriculum is our experimental intervention. The paper's PGD100×20 evaluation and our PGD100×5 test are not an equal-budget numerical comparison.

## Reproduce the workflow

### Standalone Colab/Kaggle notebook

Open [the notebook](notebooks/01_mnist_reproduction.ipynb), select a GPU, and run top to bottom. It writes its embedded source/config files and needs no repository clone. First use the default **`smoke`** (256 train / 64 validation / 64 test examples, one epoch). This verifies software operation, not scientific performance.

For the final three strategies, use **separate notebook runs**:

| `CONFIG_NAME` | Modes selected by default | Seeds | Purpose |
|---|---|---|---|
| `smoke` / `pilot` | baseline, adversarial | 0 | Software check / exploratory budget |
| `main` | baseline, adversarial | 0, 1, 2 | Clean training and PGD40 from scratch |
| `main_pgd10_probe` | adversarial | 0, 1, 2 | Optional three-epoch probe; not required by integrated curriculum |
| `main_curriculum_pgd10_to_pgd40` | adversarial | 0, 1, 2 | Direct ten-epoch 3+7 curriculum |

Use `USE_GOOGLE_DRIVE=True` in Colab for persistent checkpoints; Kaggle uses `/kaggle/working` and requires **Save Version** to retain outputs. `RESUME=True` resumes existing `last.pt` files. `EPOCHS_PER_CALL=1` limits each invocation to one additional epoch; a partially completed epoch is rerun after interruption. Existing completed training is not restarted. Save the environment/config screen and result figures, as well as the underlying CSV files.

Keep `PROTOCOL_FROZEN=False` while making validation-based decisions. For the final `main` or curriculum test, set **both `PROTOCOL_FROZEN=True` and `FINAL_STRONG_EVAL=True`**. The notebook then explicitly passes **100 steps and five restarts**. Config defaults retain the historical ten-restart setting to preserve compatibility with saved checkpoints; the explicit final override is required. Twenty restarts refer to the separate validation study.

### Command line

In a fresh Python environment, install `requirements.txt`. In Colab/Kaggle, preserve the platform's compatible CUDA `torch`/`torchvision` pair. The notebook installs only missing dependencies.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# First smoke check; add --synthetic only for an offline software test.
python -m mnist_robust.experiment --config configs/smoke.json --mode baseline --seed 0 --output-dir runs --device auto

# Repeat each of these three commands for seeds 0, 1, and 2.
python -m mnist_robust.experiment --config configs/main.json --mode baseline --seed 0 --output-dir runs --device auto --train-only
python -m mnist_robust.experiment --config configs/main.json --mode adversarial --seed 0 --output-dir runs --device auto --train-only
python -m mnist_robust.experiment --config configs/main_curriculum_pgd10_to_pgd40.json --mode adversarial --seed 0 --output-dir runs --device auto --train-only
```

Resume with the same config and `--resume`; optionally add `--epochs-per-call 1`. Config changes require a separate run location. Full training and strong evaluation require substantial GPU time and are not launched by the audit tests.

After freezing decisions, evaluate every selected checkpoint with the same final protocol. For example:

```bash
python -m mnist_robust.experiment \
  --config configs/main_curriculum_pgd10_to_pgd40.json \
  --mode adversarial --seed 0 --output-dir runs --device auto --eval-only \
  --checkpoint runs/main_curriculum_pgd10_to_pgd40/adversarial/seed_0/best.pt \
  --eval-steps 100 --eval-restarts 5

python -m mnist_robust.summarize --runs runs/main --output results/main
python -m mnist_robust.summarize --runs runs/main_curriculum_pgd10_to_pgd40 --output results/curriculum
```

`last.pt` stores resume state; `best.pt` stores the validation-selected model. New evaluations use unique directories under `evaluations/`, retaining previous results. `latest_evaluation.json` identifies the latest files. Identical repeated measurements from the same checkpoint are counted once by the summarizer; conflicting duplicate seeds are rejected. Distinct configs, datasets, source/protocol fingerprints, and attack budgets are not pooled.

### Notebook generation and checks

The separate [recorded-results notebook](notebooks/02_verify_recorded_results.ipynb) independently recomputes the archived statistics and figures without model training. It verifies arithmetic and supplied data provenance, not the historical attack execution. The equivalent local command is `python tools/verify_results.py` (`--no-figures` for numerical checks only).

```bash
python tools/build_notebook.py
python tools/build_notebook.py --check
python tools/build_results_notebook.py
python tools/build_results_notebook.py --check
python -m unittest discover -s tests -v
```

The generator embeds current package files, all configs, documentation, and archived result tables. Regenerate it after changing those inputs. Generation is deterministic; runtime checkpoints and archives naturally contain execution-specific data. [Validation record](docs/VALIDATION.md) distinguishes executed checks, static inspection, historical claims, and work not replayed.

## Files and submission

- `mnist_robust/`: model, data, attacks, training/evaluation, seed controls, aggregation.
- `configs/`: smoke, pilot, main, PGD10 probe, and integrated curriculum.
- `notebooks/` and `tools/build_notebook.py`: standalone notebook and its source generator.
- `results/`: original final measurements, derived summaries/figures, and clearly labeled software smoke outputs.
- `docs/`: [theory](docs/THEORY.md), [experiment/provenance guide](docs/EXPERIMENT_PLAN.md), [verification](docs/VALIDATION.md), [course deliverables](docs/COURSE_DELIVERABLES.md), and [bibliography](docs/references.bib).

Large datasets/checkpoints and personal storage paths are not committed. Recorded checkpoint hashes can only be independently verified against the original files when those files are available. The course report must use the supplied LaTeX template and be submitted as PDF by **13 December 2026, 23:59 Türkiye time**. The supplied report sources are in [report/](report/README.md). The original course template archive has not yet been located, so template provenance remains a submission blocker; resemblance to the course layout is insufficient. The repository's audit records compilation, PDF inspection and remaining requirements. Slides are a separate stage.
