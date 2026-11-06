"""Directory checks for starting a named run."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prelib import common


class InputProtectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(common, 'ROOT', self.root).start()
        self.out = self.root / 'outputs/example'
        self.out.mkdir(parents=True)
        self.sentinel = self.out / 'keep.txt'
        self.sentinel.write_text('input')
        self.cfg = dict(run_name='example', source_root=str(self.root / 'raw'),
                        edge_source='all', cpc_match_mode='primary',
                        embedding_index='index.parquet', embedding_vectors='vectors.npy')

    def protected(self):
        with self.assertRaises(ValueError):
            common.initialize_run(self.cfg)
        self.assertEqual(self.sentinel.read_text(), 'input')

    def test_source_inside_output(self):
        self.cfg['source_root'] = str(self.out / 'raw')
        self.protected()

    def test_output_inside_source(self):
        self.cfg['source_root'] = str(self.root)
        retained = self.root / 'vectors.npy'
        retained.write_text('source')
        common.initialize_run(self.cfg)
        self.assertEqual(retained.read_text(), 'source')

    def test_absolute_embedding_inside_output(self):
        self.cfg['embedding_vectors'] = str(self.sentinel)
        self.protected()

    def test_input_symlink_into_output(self):
        raw = Path(self.cfg['source_root'])
        raw.mkdir()
        (raw / 'vectors.npy').symlink_to(self.sentinel)
        self.protected()

    def test_prepared_input_inside_output(self):
        self.cfg['prepared_data_root'] = str(self.out / 'cache')
        self.protected()

    def test_prepared_provenance_symlink_into_output(self):
        cache = self.root / 'prepared'
        (cache / 'data').mkdir(parents=True)
        (cache / 'data/prepared_scope.json').symlink_to(self.sentinel)
        self.cfg['prepared_data_root'] = str(cache)
        self.protected()

    def test_legacy_provenance_symlink_into_output(self):
        cache = self.root / 'prepared'
        (cache / 'tables').mkdir(parents=True)
        (cache / 'tables/sample_overview.csv').symlink_to(self.sentinel)
        self.cfg['prepared_data_root'] = str(cache)
        self.protected()

    def test_separate_input_directory(self):
        raw = Path(self.cfg['source_root'])
        raw.mkdir()
        retained = raw / 'vectors.npy'
        retained.write_text('source')
        common.initialize_run(self.cfg)
        self.assertEqual(retained.read_text(), 'source')
        self.assertFalse(self.sentinel.exists())
        self.assertTrue((self.out / 'data').is_dir())


if __name__ == '__main__':
    unittest.main()
