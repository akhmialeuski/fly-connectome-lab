"""Validate official annotation formats and local-only image handling."""

from pathlib import Path

import numpy as np
import pytest

from flystate.datasets.celeba import CelebAAdapter, sample_id_for
from flystate.datasets.errors import DatasetError
from tests.synthetic import make_synthetic_celeba


class TestCelebA:
    """Exercise cardinality, annotation integrity, layouts, and RGB decoding."""

    def test_complete_fixture(self, synthetic_celeba_dir: Path) -> None:
        """Read all fixture records, decode RGB, and isolate cached list ownership.

        :param synthetic_celeba_dir: Generated 200-image dataset.
        :type synthetic_celeba_dir: Path
        """
        adapter = CelebAAdapter(root=synthetic_celeba_dir, expected=None)
        assert adapter.validate(full=True)['ok'] is True
        records = adapter.records()
        assert len(records) == 200
        assert {record.identity for record in records} == set(range(1, 9))
        assert [record.partition for record in records[:5]] == [0, 0, 0, 1, 2]
        assert len(records[0].landmarks) == 10
        assert sample_id_for(filename=records[0].filename) == 'celeba-000001'
        image = adapter.load_image(filename=records[0].filename)
        assert image.shape == (218, 178, 3)
        assert image.dtype == np.uint8
        records.clear()
        assert len(adapter.records()) == 200
        assert CelebAAdapter(root=synthetic_celeba_dir).validate()['ok'] is False
        assert len(adapter.annotation_fingerprint()) == 64

    def test_original_layout(self, tmp_path: Path) -> None:
        """Find official Img/Anno/Eval resources recursively.

        :param tmp_path: Generated layout directory.
        :type tmp_path: Path
        """
        make_synthetic_celeba(root=tmp_path, identities=2, images_per_identity=5)
        source = tmp_path / 'celeba'
        for name, parent in (
            ('img_align_celeba', 'Img'),
            ('identity_CelebA.txt', 'Anno'),
            ('list_landmarks_align_celeba.txt', 'Anno'),
            ('list_eval_partition.txt', 'Eval'),
        ):
            destination = tmp_path / parent
            destination.mkdir(exist_ok=True)
            (source / name).rename(target=destination / name)
        source.rmdir()
        adapter = CelebAAdapter(root=tmp_path, expected=(10, 2))
        assert adapter.validate(full=True)['ok'] is True
        assert adapter.locate().images_dir == tmp_path / 'Img' / 'img_align_celeba'

    def test_image_inventory(self, tmp_path: Path) -> None:
        """Require missing and extra JPEGs only during a full inventory check.

        :param tmp_path: Generated image directory.
        :type tmp_path: Path
        """
        make_synthetic_celeba(root=tmp_path, identities=2, images_per_identity=5)
        adapter = CelebAAdapter(root=tmp_path, expected=None)
        images = adapter.locate().images_dir
        (images / '000001.jpg').unlink()
        (images / '999999.jpg').write_bytes(data=b'not an image')
        assert adapter.validate()['ok'] is True
        full = adapter.validate(full=True)
        assert full['ok'] is False
        assert 'Missing 1' in str(full['problems']) and 'Unexpected 1' in str(full['problems'])
        with pytest.raises(expected_exception=DatasetError, match='Cannot decode'):
            adapter.load_image(filename='000001.jpg')
        with pytest.raises(expected_exception=DatasetError, match='Invalid CelebA filename'):
            adapter.load_image(filename='../000001.jpg')

    @pytest.mark.parametrize(
        'resource,contents,message',
        [
            ('identity_CelebA.txt', '', 'empty annotation'),
            ('identity_CelebA.txt', '000001.jpg 1\n000001.jpg 1\n', 'duplicate filename'),
            ('identity_CelebA.txt', '../000001.jpg 1\n', 'Invalid CelebA filename'),
            ('list_landmarks_align_celeba.txt', '1\nbad header\n', 'landmark header'),
            ('list_landmarks_align_celeba.txt', 'bad count', 'landmark row count'),
            ('list_eval_partition.txt', 'remove-last', 'different filename sets'),
            ('identity_CelebA.txt', 'wrong-fields', 'Wrong number'),
            ('identity_CelebA.txt', 'negative', 'Invalid identity'),
            ('identity_CelebA.txt', 'noninteger', 'Invalid annotation'),
            ('list_eval_partition.txt', 'bad-partition', 'Invalid identity'),
            ('list_landmarks_align_celeba.txt', 'nonfinite', 'nonfinite landmarks'),
        ],
    )
    def test_malformed_annotations(
        self, tmp_path: Path, resource: str, contents: str, message: str
    ) -> None:
        """Reject malformed source annotations with a useful validation report.

        :param tmp_path: Small generated source dataset.
        :type tmp_path: Path
        :param resource: Annotation basename to corrupt.
        :type resource: str
        :param contents: Corruption case or replacement text.
        :type contents: str
        :param message: Expected diagnostic fragment.
        :type message: str
        """
        make_synthetic_celeba(root=tmp_path, identities=2, images_per_identity=5)
        adapter = CelebAAdapter(root=tmp_path, expected=None)
        assert adapter.validate()['ok'] is True
        source = tmp_path / 'celeba' / resource
        lines = source.read_text().splitlines()
        if contents == 'bad count':
            lines[0] = 'not-an-integer'
        elif contents == 'remove-last':
            lines.pop()
        elif contents == 'wrong-fields':
            lines[0] += ' extra'
        elif contents in {'negative', 'noninteger', 'bad-partition'}:
            value = {'negative': '-1', 'noninteger': 'invalid', 'bad-partition': '9'}[contents]
            lines[0] = f'000001.jpg {value}'
        elif contents == 'nonfinite':
            tokens = lines[2].split()
            tokens[1] = 'nan'
            lines[2] = ' '.join(tokens)
        else:
            lines = contents.splitlines()
        source.write_text(data='\n'.join(lines) + '\n')
        report = adapter.validate()
        assert report['ok'] is False
        assert message in str(report['problems'])

    def test_missing_official_files(self, tmp_path: Path) -> None:
        """Explain unsupported mirror layouts and invalid fixture dimensions.

        :param tmp_path: Empty source directory.
        :type tmp_path: Path
        """
        report = CelebAAdapter(root=tmp_path).validate()
        assert report['ok'] is False
        assert 'Kaggle CSV' in str(report['problems'])
        with pytest.raises(expected_exception=ValueError):
            make_synthetic_celeba(root=tmp_path, identities=0)
