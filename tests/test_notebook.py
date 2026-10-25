"""Lightweight standalone-notebook contract tests; no ML training or downloads."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / 'tools/build_notebook.py'


class NotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.generated = Path(cls.workspace.name) / 'first.ipynb'
        cls.second = Path(cls.workspace.name) / 'second.ipynb'
        for target in (cls.generated, cls.second):
            subprocess.run([sys.executable, str(BUILDER), '--output', str(target)],
                           cwd=ROOT, check=True, capture_output=True, text=True)
        cls.notebook = json.loads(cls.generated.read_text())
        cls.control = [c['source'] for c in cls.notebook['cells']
                       if c['cell_type'] == 'code' and not c['source'].startswith('%%writefile ')]
        cls.settings = next(s for s in cls.control if 'CONFIG_NAME = "smoke"' in s)
        cls.evaluation = next(s for s in cls.control if s.startswith('EVALUATION_COMPLETED = False'))

    def test_generation_is_deterministic_and_control_cells_parse(self):
        self.assertEqual(self.generated.read_bytes(), self.second.read_bytes())
        self.assertEqual(self.notebook['nbformat'], 4)
        ids = [cell['id'] for cell in self.notebook['cells']]
        self.assertEqual(len(ids), len(set(ids)))
        for source in self.control:
            ast.parse(source)

    def test_every_embedded_file_matches_repository_and_all_configs_exist(self):
        embedded = {}
        for cell in self.notebook['cells']:
            source = cell['source']
            if cell['cell_type'] == 'code' and source.startswith('%%writefile '):
                first, content = source.split('\n', 1)
                relative = first[len('%%writefile '):]
                self.assertNotIn(relative, embedded)
                self.assertEqual(content, (ROOT / relative).read_text(encoding='utf-8'), relative)
                embedded[relative] = content
        expected = list((ROOT / 'mnist_robust').glob('*.py')) + list((ROOT / 'configs').glob('*.json'))
        for file in expected:
            self.assertIn(file.relative_to(ROOT).as_posix(), embedded)
        for name in ('main', 'main_pgd10_probe', 'main_curriculum_pgd10_to_pgd40'):
            self.assertIn(f'configs/{name}.json', embedded)

    def run_settings(self, config, *, frozen=False, final=False):
        source = self.settings.replace('CONFIG_NAME = "smoke"', f'CONFIG_NAME = "{config}"', 1)
        source = source.replace('PROTOCOL_FROZEN = False', f'PROTOCOL_FROZEN = {frozen}', 1)
        source = source.replace('FINAL_STRONG_EVAL = False', f'FINAL_STRONG_EVAL = {final}', 1)
        # Keep the settings test isolated even when the suite itself runs on Colab/Kaggle.
        source = source.replace('IS_KAGGLE = Path("/kaggle/working").exists()', 'IS_KAGGLE = False')
        namespace = {'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
                     'importlib': SimpleNamespace(util=SimpleNamespace(find_spec=lambda _: None))}
        with tempfile.TemporaryDirectory() as directory:
            before = Path.cwd()
            try:
                os.chdir(directory)
                with redirect_stdout(io.StringIO()):
                    exec(source, namespace)
                self.assertEqual(namespace['PROJECT_DIR'], Path(directory).resolve() / 'madry_mnist_project')
                self.assertEqual(namespace['RESULTS_DIR'].name, config)
                # Rerunning settings in one kernel must not nest project directories.
                with redirect_stdout(io.StringIO()):
                    exec(source, namespace)
                self.assertEqual(namespace['PROJECT_DIR'], Path(directory).resolve() / 'madry_mnist_project')
                return namespace
            finally:
                os.chdir(before)

    def test_config_selection_modes_seeds_and_test_gate(self):
        for config in ('smoke', 'pilot', 'main', 'main_pgd10_probe', 'main_curriculum_pgd10_to_pgd40'):
            with self.subTest(config=config):
                state = self.run_settings(config)
                self.assertEqual(state['SEEDS'], [0] if config in {'smoke', 'pilot'} else [0, 1, 2])
                self.assertEqual(state['MODES'], ['baseline', 'adversarial']
                                 if config in {'smoke', 'pilot', 'main'} else ['adversarial'])
                self.assertEqual(state['RUN_TEST_EVALUATION'], config == 'smoke')

    def test_final_configs_reject_unintended_ten_restart_fallback(self):
        for config in ('main', 'main_curriculum_pgd10_to_pgd40'):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, 'PGD100 x 5'):
                    self.run_settings(config, frozen=True, final=False)
                state = self.run_settings(config, frozen=True, final=True)
                self.assertTrue(state['RUN_TEST_EVALUATION'])

    def evaluation_state(self, directory, config='main', complete=True, frozen=True):
        commands = []
        modes = ['baseline', 'adversarial'] if config == 'main' else ['adversarial']
        root = Path(directory)
        for mode in modes:
            for seed in [0, 1, 2]:
                checkpoint = root / config / mode / f'seed_{seed}' / 'best.pt'
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b'placeholder: mock command does not load checkpoint')
        return {'TRAINING_COMPLETE': complete, 'RUN_TEST_EVALUATION': frozen,
                'MODES': modes, 'SEEDS': [0, 1, 2], 'RUN_ROOT': root,
                'RUN_CONFIG_NAME': config, 'CONFIG_NAME': config,
                'SYNTHETIC': False, 'FINAL_STRONG_EVAL': True, 'sys': sys,
                'run_command': lambda command: commands.append(command)}, commands

    def test_frozen_final_builds_five_restart_commands_for_every_seed(self):
        for config in ('main', 'main_curriculum_pgd10_to_pgd40'):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as directory:
                state, commands = self.evaluation_state(directory, config)
                with redirect_stdout(io.StringIO()):
                    exec(self.evaluation, state)
                self.assertEqual(len(commands), 6 if config == 'main' else 3)
                self.assertTrue(state['EVALUATION_COMPLETED'])
                for command in commands:
                    self.assertIn('--eval-only', command)
                    self.assertEqual(command[command.index('--eval-steps') + 1], '100')
                    self.assertEqual(command[command.index('--eval-restarts') + 1], '5')

    def test_incomplete_or_unfrozen_experiments_do_not_evaluate(self):
        for complete, frozen in [(False, True), (True, False)]:
            with self.subTest(complete=complete, frozen=frozen), tempfile.TemporaryDirectory() as directory:
                state, commands = self.evaluation_state(directory, complete=complete, frozen=frozen)
                with redirect_stdout(io.StringIO()):
                    exec(self.evaluation, state)
                self.assertEqual(commands, [])
                self.assertFalse(state['EVALUATION_COMPLETED'])

    def test_check_mode_detects_stale_file_without_replacing_it(self):
        subprocess.run([sys.executable, str(BUILDER), '--output', str(self.generated), '--check'],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        stale = Path(self.workspace.name) / 'stale.ipynb'
        stale.write_text('{}\n')
        result = subprocess.run([sys.executable, str(BUILDER), '--output', str(stale), '--check'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(stale.read_text(), '{}\n')


if __name__ == '__main__':
    unittest.main()
