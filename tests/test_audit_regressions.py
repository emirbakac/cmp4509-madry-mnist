"""Final-audit regressions; all data are tiny synthetic fixtures, never results."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from mnist_robust.data import make_loaders
from mnist_robust.experiment import evaluate, main, train, training_steps, write_csv
from mnist_robust.model import SmallCNN
from mnist_robust.summarize import METRICS, summarize
from mnist_robust.utils import source_fingerprint


def config():
    return {'name': 'smoke', 'epochs': 2, 'batch_size': 4, 'lr': 0.001,
            'width': 2, 'hidden': 4, 'split_seed': 2026, 'val_size': 5000,
            'train_limit': 8, 'val_limit': 4, 'test_limit': 4, 'num_workers': 0,
            'epsilon': 0.3, 'train_alpha': 0.01, 'train_steps': 1,
            'val_steps': 2, 'val_restarts': 1, 'eval_alpha': 0.01,
            'eval_steps': 2, 'eval_restarts': 2, 'eval_seed': 12345}


def arguments(**overrides):
    values = dict(resume=False, mode='adversarial', seed=7, synthetic=True,
                  epochs_per_call=0, eval_steps=None, eval_restarts=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def loaders():
    return make_loaders('', batch_size=4, seed=7, train_limit=8,
                        val_limit=4, test_limit=4, synthetic=True)


class FinalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_main_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in SmallCNN().parameters()), 215370)

    def test_actual_curriculum_config_exact_boundary(self):
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root / 'configs/main_curriculum_pgd10_to_pgd40.json').read_text())
        self.assertEqual([training_steps(cfg, epoch) for epoch in range(10)], [10]*3 + [40]*7)

    def test_training_uses_schedule_and_keeps_first_checkpoint_on_ties(self):
        cfg = config()
        cfg.update(epochs=10, train_steps=40,
                   curriculum={'epochs_1_to_3': {'train_steps': 10},
                               'epochs_4_to_10': {'train_steps': 40}})
        observed = []

        def attack(_model, x, _y, **kwargs):
            observed.append(kwargs['steps'])
            self.assertEqual(kwargs['restarts'], 1)
            self.assertFalse(kwargs['track_best'])
            return x

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run = Path(tmp)
            with patch('mnist_robust.experiment.pgd_linf', side_effect=attack), \
                    patch('mnist_robust.experiment.validate', return_value=(0.5, 0.25)):
                train(SmallCNN(2, 4), loaders(), torch.device('cpu'), cfg, arguments(), run)
            self.assertEqual(observed, [10]*6 + [40]*14)  # Two batches per epoch.
            best = torch.load(run / 'best.pt', weights_only=False)
            last = torch.load(run / 'last.pt', weights_only=False)
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(last['epoch'], 10)
            self.assertEqual([r['train_pgd_steps'] for r in last['history']], [10]*3 + [40]*7)
            before = {p.name: p.read_bytes() for p in run.glob('*.pt')}
            with patch('torch.optim.Adam.step', side_effect=AssertionError('completed run retrained')):
                self.assertTrue(train(SmallCNN(2, 4), loaders(), torch.device('cpu'), cfg,
                                      arguments(resume=True), run))
            self.assertEqual(before, {p.name: p.read_bytes() for p in run.glob('*.pt')})

    def test_missing_resume_checkpoint_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, 'Cannot resume'):
                train(SmallCNN(2, 4), loaders(), torch.device('cpu'), config(),
                      arguments(resume=True), Path(tmp))
            self.assertFalse(list(Path(tmp).iterdir()))

    def test_completed_resume_repairs_missing_or_stale_history_without_training(self):
        cfg = config()
        cfg['epochs'] = 1
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run = Path(tmp)
            with patch('mnist_robust.experiment.write_csv', side_effect=OSError('interrupted CSV write')):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    train(SmallCNN(2, 4), loaders(), torch.device('cpu'), cfg, arguments(), run)
            self.assertTrue((run / 'last.pt').is_file())
            self.assertFalse((run / 'history.csv').exists())
            checkpoints = {p.name: p.read_bytes() for p in run.glob('*.pt')}
            for stale in (False, True):
                if stale:
                    (run / 'history.csv').write_text('epoch\n0\n')
                with patch('torch.optim.Adam.step', side_effect=AssertionError('resume retrained')):
                    self.assertTrue(train(SmallCNN(2, 4), loaders(), torch.device('cpu'), cfg,
                                          arguments(resume=True), run))
                self.assertEqual(pd.read_csv(run / 'history.csv').epoch.tolist(), [1])
            self.assertEqual(checkpoints, {p.name: p.read_bytes() for p in run.glob('*.pt')})

    def test_partial_or_best_only_run_cannot_be_overwritten_by_fresh_training(self):
        for name in ('best.pt', 'history.csv'):
            with self.subTest(artifact=name), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                (run / name).write_bytes(b'original artifact')
                with patch('torch.optim.Adam.step', side_effect=AssertionError('fresh training started')):
                    with self.assertRaisesRegex(FileExistsError, 'partial or archived'):
                        train(SmallCNN(2, 4), loaders(), torch.device('cpu'), config(), arguments(), run)
                self.assertEqual((run / name).read_bytes(), b'original artifact')
                self.assertFalse((run / 'last.pt').exists())

    def test_interrupted_csv_is_not_a_complete_evaluation_and_retry_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            folder = root / 'seed_0/evaluations/pgd2_r2'
            folder.mkdir(parents=True)
            target = folder / 'metrics.csv'
            row = dict(dataset='synthetic', config_name='smoke', protocol_id='same', mode='baseline',
                       n_test=4, eval_steps=2, eval_restarts=2, epsilon=0.3, seed=0)
            row.update({metric: 0.5 for metric in METRICS})
            with patch('mnist_robust.experiment.csv.DictWriter.writerows', side_effect=OSError('interrupted CSV write')):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    write_csv(target, [row])
            self.assertFalse(target.exists())
            self.assertFalse(list(folder.iterdir()))
            write_csv(target, [row])
            original = target.read_bytes()
            with patch('mnist_robust.experiment.csv.DictWriter.writerows', side_effect=OSError('interrupted CSV write')):
                with self.assertRaises(OSError):
                    write_csv(target, [row])
            self.assertEqual(target.read_bytes(), original)
            summary = summarize(root, root / 'summary', allow_single_seed=True)
            self.assertEqual(summary.iloc[0].n_seeds, 1)

    def test_checkpoint_selection_uses_clean_for_baseline_and_pgd_for_adversarial(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for mode, expected_epoch in [('baseline', 1), ('adversarial', 2)]:
                run = Path(tmp) / mode
                run.mkdir()
                with patch('mnist_robust.experiment.validate', side_effect=[(0.9, 0.1), (0.8, 0.2)]):
                    train(SmallCNN(2, 4), loaders(), torch.device('cpu'), config(), arguments(mode=mode), run)
                self.assertEqual(torch.load(run / 'best.pt', weights_only=False)['epoch'], expected_epoch)

    def test_evaluation_must_exceed_actual_curriculum_attack_steps(self):
        cfg = config()
        cfg.update(epochs=4, curriculum={'epochs_1_to_3': {'train_steps': 1},
                                        'epochs_4_to_10': {'train_steps': 3}})
        with self.assertRaisesRegex(ValueError, 'more steps than training'):
            evaluate(SmallCNN(2, 4), loaders()['test'], torch.device('cpu'), cfg,
                     arguments(), Path('unused'), Path('unused.pt'))

    def test_resume_accepts_old_history_without_new_step_column(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run = Path(tmp)
            train(SmallCNN(2, 4), loaders(), torch.device('cpu'), config(),
                  arguments(epochs_per_call=1), run)
            state = torch.load(run / 'last.pt', weights_only=False)
            del state['history'][0]['train_pgd_steps']
            torch.save(state, run / 'last.pt')
            train(SmallCNN(2, 4), loaders(), torch.device('cpu'), config(), arguments(resume=True), run)
            history = pd.read_csv(run / 'history.csv')
            self.assertEqual(len(history), 2)
            self.assertTrue(pd.isna(history.loc[0, 'train_pgd_steps']))
            self.assertEqual(history.loc[1, 'train_pgd_steps'], 1)

    def test_evaluation_rerun_is_immutable_and_has_provenance(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            run = Path(tmp)
            checkpoint = run / 'best.pt'
            torch.save({'epoch': 1}, checkpoint)
            for name in ('metrics.csv', 'evaluation.json', 'examples.png'):
                (run / name).write_bytes(b'preserve historical output')
            old = run / 'evaluations' / 'pgd2_r2'
            old.mkdir(parents=True)
            (old / 'metrics.csv').write_bytes(b'preserve historical evaluation')
            before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
            model = SmallCNN(2, 4)

            def plot(path, *_args):
                path.write_bytes(b'test image placeholder')

            with patch('mnist_robust.experiment.plot_examples', side_effect=plot):
                evaluate(model, loaders()['test'], torch.device('cpu'), config(), arguments(), run, checkpoint)
                first = json.loads((run / 'latest_evaluation.json').read_text())
                first_files = {p: p.read_bytes() for p in (run / first['evaluation_dir']).iterdir()}
                evaluate(model, loaders()['test'], torch.device('cpu'), config(), arguments(), run, checkpoint)
                second = json.loads((run / 'latest_evaluation.json').read_text())
            self.assertNotEqual(first['evaluation_dir'], second['evaluation_dir'])
            for path, data in {**before, **first_files}.items():
                self.assertEqual(path.read_bytes(), data)
            detail = json.loads((run / second['evaluation_dir'] / 'evaluation.json').read_text())
            self.assertEqual(detail['metrics']['checkpoint_sha256'], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(detail['source'], source_fingerprint())
            self.assertIsNone(detail['checkpoint_source'])  # Never invent old source provenance.
            self.assertIn('torch', detail['environment'])
            self.assertTrue((run / second['metrics']).is_file())

    def test_eval_only_loads_selected_checkpoint_without_training(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root, cfg = Path(tmp), config()
            cfg_path, checkpoint = root / 'config.json', root / 'best.pt'
            cfg_path.write_text(json.dumps(cfg))
            model = SmallCNN(2, 4)
            state = {'config': cfg, 'mode': 'adversarial', 'seed': 7, 'synthetic': True,
                     'epoch': 1, 'model': model.state_dict()}
            torch.save(state, checkpoint)
            before = checkpoint.read_bytes()
            argv = ['experiment', '--config', str(cfg_path), '--mode', 'adversarial',
                    '--seed', '7', '--synthetic', '--device', 'cpu', '--eval-only',
                    '--checkpoint', str(checkpoint), '--output-dir', str(root / 'runs'),
                    '--eval-steps', '100', '--eval-restarts', '5']
            with patch.object(sys, 'argv', argv), \
                    patch('mnist_robust.experiment.train', side_effect=AssertionError('eval-only trained')), \
                    patch('mnist_robust.experiment.evaluate') as evaluation:
                main()
                self.assertEqual(evaluation.call_count, 1)
                args = evaluation.call_args.args[4]
                self.assertEqual((args.eval_steps, args.eval_restarts), (100, 5))
                for key, value in evaluation.call_args.args[0].state_dict().items():
                    self.assertTrue(torch.equal(value, state['model'][key]))
            self.assertEqual(checkpoint.read_bytes(), before)
            state['seed'] = 8
            torch.save(state, checkpoint)
            with patch.object(sys, 'argv', argv), patch('mnist_robust.experiment.evaluate') as evaluation:
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    main()
                evaluation.assert_not_called()

    def test_identical_repeat_evaluations_do_not_inflate_seed_count(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            for seed in (0, 1):
                for repeat in (0, 1):
                    folder = root / f'seed_{seed}' / 'evaluations' / f'repeat_{repeat}'
                    folder.mkdir(parents=True)
                    row = dict(dataset='MNIST', config_name='main', protocol_id='same', mode='baseline',
                               n_test=10000, eval_steps=100, eval_restarts=5, epsilon=0.3, seed=seed,
                               checkpoint_sha256=f'{seed:064x}', selected_epoch=10)
                    row.update({metric: 0.5 for metric in METRICS})
                    pd.DataFrame([row]).to_csv(folder / 'metrics.csv', index=False)
            summary = summarize(root, root / 'summary')
            self.assertEqual(summary.iloc[0].n_seeds, 2)
            altered = root / 'seed_1/evaluations/repeat_1/metrics.csv'
            data = pd.read_csv(altered)
            data['pgd_accuracy'] = 0.4
            data['combined_accuracy'] = 0.4
            data.to_csv(altered, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate seed'):
                summarize(root, root / 'summary')


if __name__ == '__main__':
    unittest.main()
