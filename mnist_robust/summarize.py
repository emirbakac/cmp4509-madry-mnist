"""Aggregate identical protocols across independent training seeds (sample SD)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ['clean_accuracy', 'fgsm_accuracy', 'pgd_accuracy', 'combined_accuracy']
KEYS = ['dataset', 'config_name', 'protocol_id', 'mode', 'n_test', 'eval_steps', 'eval_restarts', 'epsilon']


def summarize(runs, output, allow_single_seed=False):
    paths = sorted(Path(runs).glob('**/evaluations/*/metrics.csv'))
    if not paths:
        raise FileNotFoundError(f'No evaluations found in {runs}; run experiment first.')
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame['source'] = str(path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    required = KEYS + ['seed'] + METRICS
    if set(required) - set(data.columns) or data[required].isna().any().any():
        raise ValueError('Evaluation rows have missing required fields.')
    if not np.isfinite(data[METRICS].to_numpy(dtype=float)).all() or not data[METRICS].ge(0).all().all() or not data[METRICS].le(1).all().all():
        raise ValueError('Accuracy measurements must be finite fractions in [0, 1].')
    if ((data.fgsm_accuracy > data.clean_accuracy) | (data.pgd_accuracy > data.clean_accuracy)
            | (data.combined_accuracy > data[['fgsm_accuracy', 'pgd_accuracy']].min(axis=1))).any():
        raise ValueError('Robust/combined accuracy violates clean-correctness inclusion.')
    duplicate_keys = KEYS + ['seed']
    repeated = data[data.duplicated(duplicate_keys, keep=False)]
    for _, group in repeated.groupby(duplicate_keys, dropna=False):
        identical = METRICS + ['checkpoint_sha256', 'selected_epoch']
        if (set(identical) - set(group.columns) or group[identical].isna().any().any()
                or any(group[column].nunique() != 1 for column in identical)):
            raise ValueError('Duplicate seed for an identical protocol has conflicting or unverifiable evaluations. Use separate run roots.')
    if len(repeated):
        print('Identical repeat evaluations of the same checkpoint are counted once per training seed.')
        data = data.drop_duplicates(duplicate_keys, keep='first')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / 'per_seed.csv', index=False)
    rows = []
    for key, group in data.groupby(KEYS, dropna=False, sort=True):
        row = dict(zip(KEYS, key))
        row['n_seeds'] = len(group)
        row['seeds'] = ','.join(map(str, sorted(group.seed.tolist())))
        row['status'] = ('SMOKE_ONLY' if row['config_name'] == 'smoke' or row['dataset'] == 'synthetic'
                         else ('INCOMPLETE_SINGLE_SEED' if len(group) < 2 else 'MEASURED'))
        for metric in METRICS:
            row[metric + '_mean'] = group[metric].mean()
            row[metric + '_std'] = group[metric].std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'summary.csv', index=False)
    lines = ['# Measured outputs', '', 'Accuracies are percentages below; CSV stores fractions. '
             '± is sample standard deviation across training seeds (ddof=1), not a confidence interval. '
             'One seed has undefined SD. SMOKE_ONLY is a pipeline check, never evidence of reproduction.', '',
             '| Dataset/config/protocol | Mode | Seeds | Clean | FGSM | PGD | Combined | Status |',
             '|---|---|---|---|---|---|---|---|']
    for row in rows:
        def fmt(metric):
            mean, sd = row[metric+'_mean'], row[metric+'_std']
            return f'{100*mean:.2f} ± {100*sd:.2f}' if pd.notna(sd) else f'{100*mean:.2f} (SD undefined)'
        values = [fmt(metric) for metric in METRICS]
        lines.append(f"| {row['dataset']}/{row['config_name']}/{row['protocol_id']} | {row['mode']} | {row['seeds']} | " + ' | '.join(values) + f" | {row['status']} |")
    lines += ['', 'Combined accuracy survives clean + FGSM + every retained PGD candidate. '
              'PGD accuracy is an empirical result for the stated attack; it does not certify robustness.',
              'Error bars show descriptive sample SD, not bounded confidence intervals; mean ± SD may extend outside 0–100%.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    groups = list(summary.groupby(['dataset', 'config_name', 'protocol_id'], sort=True))
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 4.2*len(groups)), squeeze=False)
    for ax, ((dataset, config, protocol), group) in zip(axes[:, 0], groups):
        x = np.arange(len(METRICS))
        width = 0.75 / len(group)
        lower, upper = 0.0, 100.0
        for i, (_, row) in enumerate(group.iterrows()):
            means = [100*row[m+'_mean'] for m in METRICS]
            sd = np.array([100*row[m+'_std'] for m in METRICS])
            finite = np.isfinite(sd)
            if finite.any():
                mean_array = np.asarray(means)
                lower = min(lower, float(np.min(mean_array[finite] - sd[finite])))
                upper = max(upper, float(np.max(mean_array[finite] + sd[finite])))
            # Do not draw a zero-width error bar for an undefined single-seed SD.
            ax.bar(x + (i - (len(group)-1)/2)*width, means, width,
                   yerr=sd if np.isfinite(sd).all() else None, capsize=4,
                   label=f"{row['mode']} (n={row['n_seeds']})")
        ax.set_xticks(x, ['Clean', 'FGSM', 'PGD', 'Combined'])
        pad = 0.02 * (upper - lower)
        ax.set_ylim(lower - pad if lower < 0 else 0, upper + pad if upper > 100 else 100)
        ax.set_ylabel('Accuracy (%)')
        suffix = ' — PIPELINE CHECK ONLY' if config == 'smoke' or dataset == 'synthetic' else ''
        ax.set_title(f'{dataset} / {config}: PGD-{int(group.iloc[0].eval_steps)} × {int(group.iloc[0].eval_restarts)}{suffix}')
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'accuracy.png', dpi=180)
    plt.close(fig)
    histories = sorted(Path(runs).rglob('history.csv'))
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for path in histories:
            h = pd.read_csv(path)
            name = '/'.join(path.parts[-4:-1])
            axes[0].plot(h.epoch, h.val_clean_accuracy*100, label=name)
            axes[1].plot(h.epoch, h.val_pgd_accuracy*100, label=name)
        for ax, title in zip(axes, ['Validation clean', 'Validation PGD (configuration-specific attack)']):
            ax.set(title=title, xlabel='Epoch', ylabel='Accuracy (%)', ylim=(0, 100))
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / 'training_curves.png', dpi=180)
        plt.close(fig)
    print('\n'.join(lines))
    if not allow_single_seed and (summary.n_seeds < 2).any():
        raise SystemExit('Outputs saved; at least one group has one seed. Add seeds, or use --allow-single-seed for smoke/pilot inspection.')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--output', default='results')
    parser.add_argument('--allow-single-seed', action='store_true')
    args = parser.parse_args()
    summarize(args.runs, args.output, args.allow_single_seed)


if __name__ == '__main__':
    main()
