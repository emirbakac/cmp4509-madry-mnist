"""Pipeline invariants using synthetic data; no MNIST downloads or research claims."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from mnist_robust.data import make_loaders
from mnist_robust.experiment import rng_state, train, validate
from mnist_robust.model import SmallCNN
from mnist_robust.summarize import METRICS, summarize
from mnist_robust.utils import seed_everything


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return {
            'name': 'smoke', 'epochs': 2, 'batch_size': 4, 'lr': 0.001,
            'width': 2, 'hidden': 4, 'epsilon': 0.3, 'train_alpha': 0.01,
            'train_steps': 1, 'eval_alpha': 0.01, 'val_steps': 2,
            'val_restarts': 1, 'eval_seed': 12345,
        }

    def loaders(self, workers=0):
        return make_loaders('', batch_size=4, seed=7, split_seed=2026,
                            train_limit=8, val_limit=4, test_limit=4,
                            num_workers=workers, synthetic=True)

    def test_validation_preserves_training_rng_and_model_mode(self):
        seed_everything(7)
        model = SmallCNN(width=2, hidden=4).train()
        loaders = self.loaders()
        before = rng_state(loaders['train'])
        validate(model, loaders['val'], torch.device('cpu'), self.config(), 12345)
        after = rng_state(loaders['train'])
        self.assertTrue(model.training)
        self.assertEqual(before['python'], after['python'])
        self.assertTrue(np.array_equal(before['numpy'][1], after['numpy'][1]))
        self.assertEqual(before['numpy'][2:], after['numpy'][2:])
        self.assertTrue(torch.equal(before['torch'], after['torch']))
        self.assertTrue(torch.equal(before['loader'], after['loader']))

    def test_epoch_boundary_resume_matches_uninterrupted_training(self):
        cfg = self.config()
        args = SimpleNamespace(resume=False, mode='adversarial', seed=7,
                               synthetic=True, epochs_per_call=0)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            full, chunked = root / 'full', root / 'chunked'
            full.mkdir()
            chunked.mkdir()
            seed_everything(7)
            model = SmallCNN(width=2, hidden=4)
            self.assertTrue(train(model, self.loaders(), torch.device('cpu'), cfg, args, full))
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            seed_everything(7)
            model = SmallCNN(width=2, hidden=4)
            args.epochs_per_call = 1
            self.assertFalse(train(model, self.loaders(), torch.device('cpu'), cfg, args, chunked))
            seed_everything(7)
            model = SmallCNN(width=2, hidden=4)
            args.resume = True
            self.assertTrue(train(model, self.loaders(), torch.device('cpu'), cfg, args, chunked))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(expected[name], value), name)
            state = torch.load(chunked / 'last.pt', weights_only=False)
            self.assertEqual(state['epoch'], 2)
            self.assertEqual(len(state['history']), 2)
            self.assertTrue((chunked / 'best.pt').is_file())

    def test_aggregation_separates_protocols_and_uses_sample_sd(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            for protocol, values in [('a', [0.4, 0.6]), ('b', [0.1, 0.3])]:
                for seed, value in enumerate(values):
                    folder = root / 'runs' / 'pilot' / 'baseline' / f'seed_{seed}' / 'evaluations' / protocol
                    folder.mkdir(parents=True)
                    row = dict(dataset='MNIST', config_name='pilot', protocol_id=protocol,
                               mode='baseline', n_test=100, eval_steps=40 if protocol == 'a' else 100,
                               eval_restarts=3, epsilon=0.3, seed=seed)
                    row.update({metric: value for metric in METRICS})
                    pd.DataFrame([row]).to_csv(folder / 'metrics.csv', index=False)
            summary = summarize(root / 'runs', root / 'results')
            self.assertEqual(len(summary), 2)
            by_protocol = summary.set_index('protocol_id')
            self.assertAlmostEqual(by_protocol.loc['a', 'clean_accuracy_mean'], 0.5)
            self.assertAlmostEqual(by_protocol.loc['b', 'clean_accuracy_mean'], 0.2)
            self.assertAlmostEqual(by_protocol.loc['a', 'clean_accuracy_std'], np.std([0.4, 0.6], ddof=1))
            self.assertEqual(set(summary.n_seeds), {2})

    def test_duplicate_seed_same_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['one', 'two']:
                folder = root / name / 'evaluations' / 'pgd'
                folder.mkdir(parents=True)
                row = dict(dataset='MNIST', config_name='pilot', protocol_id='same',
                           mode='baseline', n_test=100, eval_steps=40,
                           eval_restarts=3, epsilon=0.3, seed=0)
                row.update({metric: 0.5 for metric in METRICS})
                pd.DataFrame([row]).to_csv(folder / 'metrics.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate seed'):
                summarize(root, root / 'results')


if __name__ == '__main__':
    unittest.main()
