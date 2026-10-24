"""Train on train/validation only; evaluate frozen checkpoints on the test split."""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .attacks import fgsm, pgd_linf
from .data import make_loaders
from .model import SmallCNN
from .utils import environment_info, resolve_device, seed_everything


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rng_state(loader=None):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'mps': torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            'loader': loader.generator.get_state() if loader is not None else None}


def restore_rng(state, loader=None):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    if state.get('mps') is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'].cpu())
    if loader is not None and state['loader'] is not None:
        loader.generator.set_state(state['loader'].cpu())


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def validate(model, loader, device, cfg, seed):
    """Fixed validation attack RNG; validation does not consume training RNG."""
    previous = rng_state()
    seed_everything(seed)
    was_training = model.training
    model.eval()
    total = clean_correct = robust_correct = 0
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                clean = model(x).argmax(1).eq(y)
            adv = pgd_linf(model, x, y, eps=cfg['epsilon'], alpha=cfg['eval_alpha'],
                           steps=cfg['val_steps'], restarts=cfg['val_restarts'])
            with torch.no_grad():
                robust = clean & model(adv).argmax(1).eq(y)
            total += len(y)
            clean_correct += clean.sum().item()
            robust_correct += robust.sum().item()
    finally:
        model.train(was_training)
        restore_rng(previous)
    return clean_correct / total, robust_correct / total


def train(model, loaders, device, cfg, args, run_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    start, best_score, history = 0, -1.0, []
    last = run_dir / 'last.pt'
    if last.exists():
        if not args.resume:
            raise FileExistsError(f'{last} exists. Use --resume or a new --output-dir.')
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['config'] != cfg or state['mode'] != args.mode or state['seed'] != args.seed or state['synthetic'] != args.synthetic:
            raise ValueError('Resume config/mode/seed/dataset differs from saved checkpoint.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        start, best_score, history = state['epoch'], state['best_score'], state['history']
        restore_rng(state['rng'], loaders['train'])
    end = cfg['epochs']
    if args.epochs_per_call:
        end = min(end, start + args.epochs_per_call)
    for epoch in range(start, end):
        begin = time.monotonic()
        model.train()
        loss_sum, correct, total = 0., 0, 0
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            if args.mode == 'adversarial':
                steps = cfg['train_steps']
            
                curriculum = cfg.get('curriculum')
                if curriculum is not None and epoch < 3:
                    steps = curriculum['epochs_1_to_3']['train_steps']
            
                x = pgd_linf(
                    model, x, y,
                    eps=cfg['epsilon'],
                    alpha=cfg['train_alpha'],
                    steps=steps,
                    restarts=1,
                    track_best=False
                )
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += logits.argmax(1).eq(y).sum().item()
            total += len(y)
        val_clean, val_pgd = validate(model, loaders['val'], device, cfg, cfg['eval_seed'])
        score = val_clean if args.mode == 'baseline' else val_pgd
        improved = score > best_score
        best_score = max(best_score, score)
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / total,
               'train_accuracy': correct / total, 'val_clean_accuracy': val_clean,
               'val_pgd_accuracy': val_pgd, 'seconds': time.monotonic() - begin}
        history.append(row)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'epoch': epoch + 1, 'best_score': best_score, 'history': history,
                 'config': cfg, 'seed': args.seed, 'mode': args.mode,
                 'synthetic': args.synthetic, 'rng': rng_state(loaders['train'])}
        save_checkpoint(last, state)
        if improved:
            save_checkpoint(run_dir / 'best.pt', state)
        write_csv(run_dir / 'history.csv', history)
        print(json.dumps(row), flush=True)
    return end >= cfg['epochs']


def plot_examples(path, examples, epsilon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x, xf, xp, y, pc, pf, pp = examples
    n = min(len(x), 8)
    fig, axes = plt.subplots(4, n, figsize=(1.7 * n, 7), squeeze=False)
    for i in range(n):
        for row, (images, predictions, label) in enumerate([(x, pc, 'Clean'), (xf, pf, 'FGSM'), (xp, pp, 'PGD')]):
            axes[row, i].imshow(images[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[row, i].set_title(f'{label}: {predictions[i].item()} / y={y[i].item()}', fontsize=8)
        axes[3, i].imshow((xp[i, 0] - x[i, 0]).abs(), cmap='magma', vmin=0, vmax=epsilon)
        axes[3, i].set_title('|PGD − clean|', fontsize=8)
        for row in range(4):
            axes[row, i].axis('off')
    fig.suptitle('First test examples (not cherry-picked); perturbation color range [0, ε]')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate(model, loader, device, cfg, args, run_dir, checkpoint):
    model.eval()
    seed_everything(cfg['eval_seed'])
    steps = args.eval_steps if args.eval_steps is not None else cfg['eval_steps']
    restarts = args.eval_restarts if args.eval_restarts is not None else cfg['eval_restarts']
    if steps <= cfg['train_steps'] or restarts <= 1:
        raise ValueError('Final PGD evaluation requires more steps than training and >1 restart.')
    protocol = {'epsilon': cfg['epsilon'], 'alpha': cfg['eval_alpha'], 'steps': steps,
                'restarts': restarts, 'random_start': True, 'loss': 'cross_entropy',
                'candidate_selection': 'success_first_all_iterates_including_clean',
                'eval_seed': cfg['eval_seed'], 'batch_size': cfg['batch_size']}
    label = f'pgd{steps}_r{restarts}'
    eval_dir = run_dir / 'evaluations' / label
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows, examples = [], None
    begin = time.monotonic()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pc = model(x).argmax(1)
        xf = fgsm(model, x, y, cfg['epsilon'])
        xp = pgd_linf(model, x, y, eps=cfg['epsilon'], alpha=cfg['eval_alpha'], steps=steps, restarts=restarts)
        with torch.no_grad():
            pf, pp = model(xf).argmax(1), model(xp).argmax(1)
        for j in range(len(y)):
            clean = int(pc[j].eq(y[j]).item())
            f = clean * int(pf[j].eq(y[j]).item())
            p = clean * int(pp[j].eq(y[j]).item())
            rows.append({'index': len(rows), 'label': y[j].item(), 'clean_correct': clean,
                         'fgsm_correct': f, 'pgd_correct': p, 'combined_correct': f * p})
        if examples is None:
            examples = tuple(t[:8].detach().cpu() for t in (x, xf, xp, y, pc, pf, pp))
        print(f'Evaluation {label}: {len(rows)}/{len(loader.dataset)}', flush=True)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    group = hashlib.sha256(json.dumps({'config': cfg, 'protocol': protocol, 'synthetic': args.synthetic}, sort_keys=True).encode()).hexdigest()[:16]
    result = {'config_name': cfg['name'], 'protocol_id': group, 'mode': args.mode,
              'seed': args.seed, 'dataset': 'synthetic' if args.synthetic else 'MNIST',
              'n_test': len(rows), 'clean_accuracy': sum(r['clean_correct'] for r in rows) / len(rows),
              'fgsm_accuracy': sum(r['fgsm_correct'] for r in rows) / len(rows),
              'pgd_accuracy': sum(r['pgd_correct'] for r in rows) / len(rows),
              'combined_accuracy': sum(r['combined_correct'] for r in rows) / len(rows),
              'eval_steps': steps, 'eval_restarts': restarts, 'epsilon': cfg['epsilon'],
              'checkpoint_sha256': digest, 'eval_seconds': time.monotonic() - begin}
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    result['selected_epoch'] = state['epoch']
    detail = {'metrics': result, 'protocol': protocol, 'config': cfg,
              'checkpoint': str(checkpoint), 'interpretation': 'Smoke/synthetic is plumbing only; empirical attack accuracy is not certified robustness.'}
    for folder in (eval_dir, run_dir):
        write_json(folder / 'evaluation.json', detail)
        write_csv(folder / 'metrics.csv', [result])
    write_csv(eval_dir / 'per_example.csv', rows)
    plot_examples(eval_dir / 'examples.png', examples, cfg['epsilon'])
    # Latest view for notebook convenience; all protocols remain under evaluations/.
    (run_dir / 'examples.png').write_bytes((eval_dir / 'examples.png').read_bytes())
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['baseline', 'adversarial'], default='baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', default='runs')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--synthetic', action='store_true', help='Offline plumbing test only; never a scientific result.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train-only', action='store_true')
    group.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--eval-steps', type=int)
    parser.add_argument('--eval-restarts', type=int)
    parser.add_argument('--epochs-per-call', type=int, default=0, help='0=all remaining; checkpointed chunks for free sessions.')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.epochs_per_call < 0:
        parser.error('--epochs-per-call must be nonnegative')
    if args.synthetic and cfg['name'] != 'smoke':
        parser.error('--synthetic is restricted to the smoke configuration')
    if args.eval_only and not args.checkpoint:
        parser.error('--eval-only requires --checkpoint')
    if args.checkpoint and not args.eval_only:
        parser.error('--checkpoint is only for --eval-only; use --resume to continue training')
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    run_dir = Path(args.output_dir) / (cfg['name'] + ('_synthetic' if args.synthetic else '')) / args.mode / f'seed_{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device={device}; run={run_dir}; synthetic={args.synthetic}', flush=True)
    model = SmallCNN(width=cfg['width'], hidden=cfg['hidden']).to(device)
    loaders = make_loaders(args.data_dir, batch_size=cfg['batch_size'], seed=args.seed,
                           split_seed=cfg['split_seed'], val_size=cfg['val_size'],
                           train_limit=cfg['train_limit'], val_limit=cfg['val_limit'], test_limit=cfg['test_limit'],
                           num_workers=cfg['num_workers'], synthetic=args.synthetic)
    if not args.eval_only:
        # Validate an existing run before changing its provenance files.
        config_path = run_dir / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != cfg:
            raise ValueError('Existing run has a different config; choose a new output directory.')
        write_json(config_path, cfg)
        env = environment_info()
        env.update({'device_selected': str(device), 'seed': args.seed,
                    'split_seed': cfg['split_seed'], 'synthetic': args.synthetic,
                    'session_started_utc': datetime.now(timezone.utc).isoformat()})
        if not (run_dir / 'environment.json').exists():
            write_json(run_dir / 'environment.json', env)
        with (run_dir / 'sessions.jsonl').open('a') as f:
            f.write(json.dumps(env) + '\n')
        complete = train(model, loaders, device, cfg, args, run_dir)
        if not complete:
            print('Chunk complete. Resume with --resume; test evaluation waits until all epochs finish.')
            return
        if args.train_only:
            print('Training complete. Freeze choices using validation, then run --eval-only on best.pt.')
            return
    checkpoint = Path(args.checkpoint) if args.eval_only else run_dir / 'best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['config'] != cfg or state['mode'] != args.mode or state['seed'] != args.seed or state['synthetic'] != args.synthetic:
        raise ValueError('Evaluation checkpoint does not match config/mode/seed/dataset.')
    model.load_state_dict(state['model'])
    evaluate(model, loaders['test'], device, cfg, args, run_dir, checkpoint)


if __name__ == '__main__':
    main()
