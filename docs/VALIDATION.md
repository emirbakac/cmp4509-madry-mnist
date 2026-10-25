# Verification record — 25 October 2026 audit

Verification has several distinct levels. The original measurements are preserved; checking their arithmetic does not rerun the historical training or attack jobs.

## Recorded measurements

- `results/FINAL_TEST_ALL_MODELS.csv` contains nine rows, three methods and seeds 0/1/2 for each, with 10,000 test images per row. Clean/FGSM/PGD values are fractions in `[0,1]` and the method summaries use sample SD (`ddof=1`).
- `results/FINAL_TEST_PROTOCOL.json` records the unchanged CSV digest and clearly labels the PGD100×5 attack settings as historical author-supplied provenance. Those settings are not encoded in the original compact CSV.
- `tools/verify_results.py` checks the preserved bytes, seed coverage, selected epochs, denominators, accuracy ranges and hash syntax; it recomputes statistics and derived figures. `results/verified/` contains derived arithmetic, not new model measurements.
- Recorded checkpoint hash strings cannot be matched to original full-model bytes until those checkpoint files are available.
- `results/validation_reported.json` preserves the separate rounded, author-reported 512-image PGD100×20/AutoAttack values. Raw logs, subset indices and per-example predictions are unavailable. Rounded percentages were not converted into invented observed counts.

## Notebook contract tests executed

`python -m unittest discover -s tests -p test_notebook.py -v`: **7 tests passed** during this audit. They cover deterministic generation, exact embedded package/config contents, all five config selections, seeds/modes, repeated settings in one kernel, frozen final100×5 command construction, blocked test calls before training completion/freezing, and read-only stale-notebook detection. These tests generate into temporary files and mock the expensive experiment commands; they do not train full models or run full AutoAttack.

The first execution exposed a test assertion's macOS `/var` versus `/private/var` symlink representation; resolving both paths fixed the assertion. The generated settings behavior itself was unchanged by that test correction.

`python -m py_compile tools/build_notebook.py` also completed successfully. After any further edits, regenerate `notebooks/01_mnist_reproduction.ipynb`, run `python tools/build_notebook.py --check`, and rerun the full suite. Current final full-suite and notebook execution outcomes are recorded in [FINAL_AUDIT.md](FINAL_AUDIT.md), rather than inferred from this narrower check.

## Historical software verification

The prior 24 October record reports 17 passing tests and an end-to-end real-MNIST smoke execution on CPU/Apple MPS, including checkpoint resume and artifact generation. Those checks concerned the earlier code/notebook version. They do not prove that the current notebook, CUDA full training, or historical nine-model experiment was independently replayed. Historical files in `results/smoke_check/` remain labeled as software checks.

## Limits of this audit

No nine-model retraining, full 10,000-image attack replay, or full AutoAttack run is performed for code auditing. Integrated curriculum scheduling and epoch-boundary resume can be tested cheaply, but that does not establish equivalence to the original probe-checkpoint transfer process. Cross-device bitwise equivalence is not assumed. Exact current test counts, actual notebook runs, report compilation/inspection and remaining template-provenance issues belong in [the final audit](FINAL_AUDIT.md).
