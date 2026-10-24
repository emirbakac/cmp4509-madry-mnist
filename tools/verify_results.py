"""Validate preserved measurements and regenerate report figures without training.

Historical final values are never edited. Validation figures use explicitly
labelled author-reported rounded percentages, not invented raw attack logs.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('Baseline', 'PGD40', 'Curriculum')
METRICS = ('clean_accuracy', 'fgsm_accuracy', 'pgd_accuracy')


def verify(root=ROOT, make_figures=True):
    root = Path(root)
    source = root / 'results/FINAL_TEST_ALL_MODELS.csv'
    protocol = json.loads((root / 'results/FINAL_TEST_PROTOCOL.json').read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != protocol['measurement_sha256']:
        raise ValueError('Original final measurement bytes changed; investigate before updating provenance.')
    records = list(csv.DictReader(source.open(newline='')))
    if len(records) != 9 or set(r['method'] for r in records) != set(METHODS):
        raise ValueError('Expected exactly nine records and all three methods.')
    expected_epochs = {'Baseline': [10, 8, 10], 'PGD40': [10, 1, 1], 'Curriculum': [10, 10, 10]}
    summary = []
    for method in METHODS:
        group = sorted((r for r in records if r['method'] == method), key=lambda r: int(r['seed']))
        if [int(r['seed']) for r in group] != [0, 1, 2]:
            raise ValueError(f'Duplicate/missing seeds: {method}')
        if [int(r['best_epoch']) for r in group] != expected_epochs[method]:
            raise ValueError(f'Selected epochs differ from the preserved historical record: {method}')
        for row in group:
            if int(row['test_images']) != 10000 or not re.fullmatch(r'[0-9a-f]{64}', row['checkpoint_sha256']):
                raise ValueError('Invalid denominator or SHA-256 syntax.')
            for metric in METRICS:
                value = float(row[metric])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'Invalid accuracy: {row}')
                if abs(value * 10000 - round(value * 10000)) > 1e-8:
                    raise ValueError('Accuracy is not an integer count over 10000.')
            if float(row['pgd_accuracy']) > float(row['clean_accuracy']) or float(row['fgsm_accuracy']) > float(row['clean_accuracy']):
                raise ValueError('Attack-survival accuracy exceeds clean accuracy.')
        for metric in METRICS:
            values = [float(row[metric]) * 100 for row in group]
            summary.append({'method': method, 'metric': metric, 'n_seeds': 3,
                            'mean_percent': statistics.mean(values),
                            'sample_sd_percent': statistics.stdev(values)})
    validation = json.loads((root / 'results/validation_reported.json').read_text())
    val_summary = []
    for attack in validation['attacks']:
        values = attack['accuracy_percent']
        if len(values) != 3 or any(not 0 <= x <= 100 for x in values):
            raise ValueError('Invalid reported validation percentages.')
        val_summary.append({'attack': attack['name'], 'mean_percent': statistics.mean(values),
                            'sample_sd_percent': statistics.stdev(values),
                            'source_precision': 'author-reported rounded percentages'})
    out = root / 'results/verified'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'final_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    result = {'final_csv_sha256': source_hash, 'n_records': 9, 'n_seeds_each': 3,
              'final_test': summary, 'reported_validation': val_summary,
              'checkpoint_hashes': 'syntax verified only; checkpoint files unavailable',
              'validation_provenance': validation['status']}
    (out / 'statistics.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Independently recomputed statistics', '',
             'Final CSV values are preserved. Percentages below use sample SD (ddof=1) across three training seeds.', '',
             '| Method | Clean (%) | FGSM (%) | PGD100 x 5 (%) |', '|---|---:|---:|---:|']
    for method in METHODS:
        cells = [f"{r['mean_percent']:.2f} ± {r['sample_sd_percent']:.2f}" for r in summary if r['method'] == method]
        lines.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Separate validation summaries use rounded author-reported percentages, not raw counts:', '']
    for row in val_summary:
        lines.append(f"- {row['attack']}: {row['mean_percent']:.2f} ± {row['sample_sd_percent']:.2f}% (reported N=512).")
    lines += ['', 'The AutoAttack mean is 83.34% when calculated from the provided rounded percentages. '
              'Exact raw counts could change the last displayed digit; no counts were reconstructed as observations.',
              '', 'No full-model checkpoints or raw validation attack logs were available for independent replay.']
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    if make_figures:
        plot(root, records, summary, validation, val_summary)
    print('\n'.join(lines))
    return result


def plot(root, records, summary, validation, val_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 220})
    folder = root / 'report/figures'
    folder.mkdir(parents=True, exist_ok=True)
    colors = ['#315f8c', '#bb5b37', '#25856b']
    labels = ['Standard baseline', 'PGD40 from scratch', 'PGD10 → PGD40']
    def save(fig, name):
        fig.savefig(folder / (name + '.png'), bbox_inches='tight')
        fig.savefig(folder / (name + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(3)
    for i, method in enumerate(METHODS):
        selected = [r for r in summary if r['method'] == method]
        means = [r['mean_percent'] for r in selected]
        sds = [r['sample_sd_percent'] for r in selected]
        xs = x + (i - 1) * 0.20
        ax.errorbar(xs, means, yerr=sds, fmt='o', capsize=5, color=colors[i], label=labels[i], markersize=6, lw=1.4)
        for j, metric in enumerate(METRICS):
            raw = [100 * float(r[metric]) for r in records if r['method'] == method]
            ax.scatter([xs[j] - .035, xs[j], xs[j] + .035], raw, s=18, facecolors='none', edgecolors=colors[i], alpha=.7)
    ax.axhline(0, color='#999999', lw=.7)
    ax.set(xticks=x, xticklabels=['Clean', 'FGSM', 'PGD100 × 5'], ylim=(-13, 106), ylabel='Test accuracy (%)')
    fig.suptitle('Full MNIST test: 10,000 images per checkpoint', y=.99, fontsize=12)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, fontsize=9, frameon=False)
    fig.text(.10, .01, 'Solid points: mean ± sample SD, n=3 seeds. Open points: individual seeds.\nSD is descriptive, not a confidence interval; a symmetric SD range can extend below zero.', fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .92))
    save(fig, 'final_accuracy_comparison')
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for i, method in enumerate(METHODS):
        group = sorted((r for r in records if r['method'] == method), key=lambda r: int(r['seed']))
        ys = [100 * float(r['pgd_accuracy']) for r in group]
        xs = np.arange(3) + (i - 1) * .12
        ax.scatter(xs, ys, color=colors[i], marker=['o', 's', '^'][i], s=65, label=labels[i])
        for px, py in zip(xs, ys):
            ax.annotate(f'{py:.2f}', (px, py), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
    ax.set(xticks=range(3), xticklabels=['Seed 0', 'Seed 1', 'Seed 2'], ylim=(-5, 104), ylabel='PGD100 × 5 test accuracy (%)')
    ax.set_title('Independent training seeds; no failed runs excluded')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.06), ncol=3, fontsize=8, frameon=False)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    save(fig, 'seed_robustness')
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for i, attack in enumerate(validation['attacks']):
        values = attack['accuracy_percent']
        ax.scatter(values, [1-i-.10, 1-i, 1-i+.10], s=28, color=colors[i], alpha=.7)
        row = val_summary[i]
        ax.errorbar(row['mean_percent'], 1-i, xerr=row['sample_sd_percent'], fmt='D', capsize=5, markersize=7, color=colors[i])
        ax.text(89.4, 1-i, f"{row['mean_percent']:.2f} ± {row['sample_sd_percent']:.2f}", va='center', ha='right', fontsize=9)
    ax.set(yticks=[1,0], yticklabels=['PGD100 × 20', 'Full AutoAttack'], xlim=(79,90), ylim=(-.5,1.5), xlabel='Reported validation accuracy (%) (expanded scale)')
    ax.set_title('Curriculum: reported 512-image validation subset', pad=12)
    ax.grid(axis='x', alpha=.2)
    fig.text(.04,.01,'Diamonds: mean ± sample SD; small points: seeds 0–2. Rounded author-reported values.\nRaw logs and subset identities were unavailable for independent verification.',fontsize=8)
    fig.tight_layout(rect=(0,.12,1,1))
    save(fig,'validation_attack_comparison')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    verify(make_figures=not args.no_figures)
