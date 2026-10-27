"""Compatibility and I/O contracts for the behavior-preserving follow-up package."""
import csv
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
MODULES = {
    'phase1_transfer_cross_corpus': 'cross_corpus',
    'phase1_transfer_recovery': 'recovery',
    'phase1_transfer_gradient_check': 'gradients',
    'phase1_transfer_support': 'support',
    'phase1_transfer_native_fixed': 'native_fixed',
}


class Compatibility(unittest.TestCase):
    def test_legacy_imports_are_the_actual_implementation_modules(self):
        for legacy, name in MODULES.items():
            with self.subTest(legacy=legacy):
                old = importlib.import_module(legacy)
                new = importlib.import_module('transfer_followup.' + name)
                self.assertIs(old, new)
                self.assertIs(old.run, new.run)
                marker = object()
                old._compatibility_probe = marker
                try:
                    self.assertIs(new._compatibility_probe, marker)
                finally:
                    del old._compatibility_probe

    def test_private_legacy_helpers_remain_accessible(self):
        names = {'cross_corpus': ['rows_from_csv'], 'recovery': ['_predict', '_sha'],
                 'gradients': ['_value', '_groups', '_row', '_sha'],
                 'support': ['_num', '_same', '_quartiles', '_sha'],
                 'native_fixed': ['_dirs', '_check_planned']}
        for legacy, name in MODULES.items():
            mod = importlib.import_module(legacy)
            for attr in names[name]:
                self.assertTrue(callable(getattr(mod, attr)))

    def test_direct_scripts_and_module_commands_work_from_another_directory(self):
        env = dict(os.environ, PYTHONPATH=str(SCRIPTS))
        with tempfile.TemporaryDirectory() as cwd:
            for legacy, name in MODULES.items():
                for command in ([str(SCRIPTS / (legacy + '.py'))], ['-m', 'transfer_followup.' + name]):
                    with self.subTest(command=command):
                        result = subprocess.run([sys.executable, '-B', *command, '--help'],
                                                cwd=cwd, env=env, text=True, capture_output=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn('--out', result.stdout)
                        self.assertIn('usage:', result.stdout)
                        self.assertEqual(result.stderr, '')

    def test_canonical_first_import_also_resolves_to_same_module(self):
        env = dict(os.environ, PYTHONPATH=str(SCRIPTS))
        code = '; '.join(f'import transfer_followup.{name} as impl; import {legacy} as old; assert old is impl'
                         for legacy, name in MODULES.items())
        result = subprocess.run([sys.executable, '-B', '-c', code], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class Tables(unittest.TestCase):
    def tables(self):
        return importlib.import_module('transfer_followup.tables')

    def test_per_context_types_missing_values_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'items.csv'
            header = 'feature,sentence,cluster,language_index,kind,split,eligible,fp_norm,cosine\n'
            row = '2,3,4,5,real,heldout,True,1.5,\n'
            path.write_text(header + row)
            rows = self.tables().rows_from_csv(path)
            self.assertEqual(rows[(2, 3, 'real')], dict(feature=2, sentence=3, cluster=4,
                language_index=5, kind='real', split='heldout', eligible=True, fp_norm=1.5, cosine=None))
            path.write_text(header + row + row)
            with self.assertRaisesRegex(ValueError, 'occurs twice') as cm:
                self.tables().rows_from_csv(path)
            self.assertIn(str(path), str(cm.exception))

    def test_csv_columns_follow_first_appearance_and_missing_cells_stay_blank(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'rows.csv'
            self.tables().write_csv(path, [{'z': 1, 'a': None}, {'a': 2, 'b': 3}])
            self.assertEqual(path.read_bytes(), b'z,a,b\r\n1,,\r\n,2,3\r\n')
            self.tables().write_csv(path, [])
            self.assertEqual(path.read_bytes(), b'\r\n')

    def test_explicit_column_order_and_unknown_key_failure_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'rows.csv'
            self.tables().write_csv(path, [{'a': 2, 'z': 1}], columns=['z', 'a'])
            self.assertEqual(path.read_bytes(), b'z,a\r\n1,2\r\n')
            with self.assertRaises(ValueError):
                self.tables().write_csv(path, [{'unexpected': 1}], columns=['z'])

    def test_hash_uses_the_original_file_bytes_and_missing_files_raise(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'data'
            path.write_bytes(b'abc')
            self.assertEqual(self.tables().sha256_file(path),
                'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
            with self.assertRaises(FileNotFoundError):
                self.tables().sha256_file(path.with_name('absent'))


if __name__ == '__main__':
    unittest.main()
