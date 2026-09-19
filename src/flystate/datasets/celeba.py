"""Strict local reader for official CelebA images, identities, and landmarks."""

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from flystate.datasets.errors import DatasetError
from flystate.hashing import sha256_file, sha256_obj

FILENAME_PATTERN: re.Pattern[str] = re.compile(pattern=r'[0-9]{6}\.jpg')
LANDMARK_HEADER: tuple[str, ...] = (
    'lefteye_x',
    'lefteye_y',
    'righteye_x',
    'righteye_y',
    'nose_x',
    'nose_y',
    'leftmouth_x',
    'leftmouth_y',
    'rightmouth_x',
    'rightmouth_y',
)
OFFICIAL_COUNTS: tuple[int, int] = (202_599, 10_177)


@dataclass(frozen=True)
class CelebAFiles:
    """Resolved official image directory and three annotation files."""

    images_dir: Path
    identity_file: Path
    landmarks_file: Path
    partition_file: Path


@dataclass(frozen=True)
class CelebARecord:
    """Image identity, ten pixel coordinates, and the official partition code."""

    filename: str
    identity: int
    landmarks: tuple[float, ...]
    partition: int


def sample_id_for(filename: str) -> str:
    """Convert a safe official JPEG filename into a stable sample identifier.

    :param filename: Six-digit JPEG basename.
    :type filename: str
    :returns: Identifier formatted as celeba-NNNNNN.
    :rtype: str
    :raises DatasetError: If the filename is malformed or contains a path.
    """
    if not FILENAME_PATTERN.fullmatch(string=filename):
        raise DatasetError(f'Invalid CelebA filename: {filename!r}.')
    return f'celeba-{filename[:-4]}'


def _rows(path: Path, header: bool = False) -> dict[str, tuple[str, ...]]:
    """Read unique annotation rows and verify the optional landmark header.

    :param path: UTF-8 annotation file.
    :type path: Path
    :param header: Require landmark count and column headers.
    :type header: bool
    :returns: Safe filename to annotation tokens.
    :rtype: dict[str, tuple[str, ...]]
    :raises DatasetError: If names, headers, duplicates or row counts are invalid.
    """
    lines = path.read_text(encoding='utf-8').splitlines()
    expected: int | None = None
    if header:
        if len(lines) < 2 or tuple(lines[1].split()) != LANDMARK_HEADER:
            raise DatasetError('Invalid CelebA landmark header.')
        try:
            expected = int(lines[0])
        except ValueError as error:
            raise DatasetError('Invalid CelebA landmark row count.') from error
        lines = lines[2:]
    result: dict[str, tuple[str, ...]] = {}
    for line_number, line in enumerate(lines, start=3 if header else 1):
        tokens = line.split()
        if not tokens:
            continue
        name = tokens[0]
        sample_id_for(filename=name)
        if name in result:
            raise DatasetError(f'{path.name}:{line_number}: duplicate filename {name}.')
        result[name] = tuple(tokens[1:])
    if not result or (expected is not None and expected != len(result)):
        raise DatasetError(f'{path.name}: empty annotation or incorrect declared row count.')
    return result


class CelebAAdapter:
    """Read official on-disk CelebA layouts without network access or image copies."""

    def __init__(self, root: Path, expected: tuple[int, int] | None = OFFICIAL_COUNTS) -> None:
        """Configure a local adapter with optional official cardinality checks.

        :param root: Root containing official files, possibly under nested directories.
        :type root: Path
        :param expected: Expected image and identity counts, or null for fixtures.
        :type expected: Optional[tuple[int, int]]
        """
        self.root: Path = root.expanduser().resolve()
        self.expected: tuple[int, int] | None = expected
        self._files: CelebAFiles | None = None
        self._records: tuple[CelebARecord, ...] | None = None

    def locate(self) -> CelebAFiles:
        """Find the first sorted official path for each required resource.

        :returns: Resolved resource locations.
        :rtype: CelebAFiles
        :raises DatasetError: If official files or the image directory are missing.
        """
        if self._files is not None:
            return self._files
        resources = (
            'img_align_celeba',
            'identity_CelebA.txt',
            'list_landmarks_align_celeba.txt',
            'list_eval_partition.txt',
        )
        found: list[Path] = []
        for name in resources:
            matches = sorted(
                path
                for path in self.root.rglob(pattern=name)
                if (path.is_dir() if name == 'img_align_celeba' else path.is_file())
            )
            if not matches:
                raise DatasetError(
                    f'Missing {name} under {self.root}. Use official CelebA files; '
                    'Kaggle CSV mirrors without identities are not supported.'
                )
            found.append(matches[0])
        self._files = CelebAFiles(
            images_dir=found[0],
            identity_file=found[1],
            landmarks_file=found[2],
            partition_file=found[3],
        )
        return self._files

    def records(self) -> list[CelebARecord]:
        """Parse and cache validated annotation records sorted by filename.

        :returns: Independent list of immutable image records.
        :rtype: list[CelebARecord]
        :raises DatasetError: If annotations disagree or have invalid field values.
        """
        if self._records is not None:
            return list(self._records)
        files = self.locate()
        identities = _rows(path=files.identity_file)
        landmarks = _rows(path=files.landmarks_file, header=True)
        partitions = _rows(path=files.partition_file)
        if set(identities) != set(landmarks) or set(identities) != set(partitions):
            raise DatasetError('The three CelebA annotation files list different filename sets.')
        result: list[CelebARecord] = []
        for name in sorted(identities):
            try:
                if (
                    len(identities[name]) != 1
                    or len(partitions[name]) != 1
                    or len(landmarks[name]) != 10
                ):
                    raise ValueError('Wrong number of annotation fields.')
                identity = int(identities[name][0])
                partition = int(partitions[name][0])
                coordinates = tuple(float(value) for value in landmarks[name])
                if (
                    identity < 1
                    or partition not in (0, 1, 2)
                    or not all(map(math.isfinite, coordinates))
                ):
                    raise ValueError('Invalid identity, partition, or nonfinite landmarks.')
            except ValueError as error:
                raise DatasetError(f'Invalid annotation for {name}: {error}') from error
            result.append(
                CelebARecord(
                    filename=name, identity=identity, landmarks=coordinates, partition=partition
                )
            )
        self._records = tuple(result)
        return list(self._records)

    def validate(self, full: bool = False) -> dict[str, object]:
        """Check annotations, cardinality, and optionally all JPEG directory entries.

        :param full: Check the complete image filename set in one scandir pass.
        :type full: bool
        :returns: Status, counts, problems, and annotation content fingerprint.
        :rtype: dict[str, object]
        """
        self._records = None
        problems: list[str] = []
        images = 0
        identities = 0
        fingerprint: str | None = None
        try:
            files = self.locate()
            records = self.records()
            images, identities = len(records), len({record.identity for record in records})
            fingerprint = self.annotation_fingerprint()
            if self.expected is not None and (images, identities) != self.expected:
                problems.append(
                    f'Expected {self.expected[0]} images and {self.expected[1]} identities; '
                    f'found {images} and {identities}.'
                )
            if full:
                with os.scandir(path=files.images_dir) as entries:
                    actual = {
                        entry.name
                        for entry in entries
                        if entry.is_file() and entry.name.endswith('.jpg')
                    }
                expected = {record.filename for record in records}
                if missing := expected - actual:
                    problems.append(
                        f'Missing {len(missing)} JPEGs; examples: {sorted(missing)[:5]}.'
                    )
                if extra := actual - expected:
                    problems.append(
                        f'Unexpected {len(extra)} JPEGs; examples: {sorted(extra)[:5]}.'
                    )
        except (DatasetError, OSError, UnicodeError) as error:
            problems.append(str(error))
        return {
            'ok': not problems,
            'images': images,
            'identities': identities,
            'problems': problems,
            'annotation_fingerprint': fingerprint,
        }

    def load_image(self, filename: str) -> NDArray[np.uint8]:
        """Decode one official JPEG as an independent uint8 RGB image.

        :param filename: Safe six-digit JPEG basename.
        :type filename: str
        :returns: RGB pixels of shape (height,width,3), normally (218,178,3).
        :rtype: NDArray[np.uint8]
        :raises DatasetError: If the name is invalid or the image cannot be decoded.
        """
        sample_id_for(filename=filename)
        try:
            with Image.open(fp=self.locate().images_dir / filename) as image:
                return np.array(object=image.convert(mode='RGB'), dtype=np.uint8)
        except (OSError, ValueError) as error:
            raise DatasetError(f'Cannot decode {filename}: {error}') from error

    def annotation_fingerprint(self) -> str:
        """Hash all three original annotation files by content.

        :returns: Canonical SHA-256 identity of the annotation set.
        :rtype: str
        """
        files = self.locate()
        return sha256_obj(
            obj={
                path.name: sha256_file(path=path)
                for path in (files.identity_file, files.landmarks_file, files.partition_file)
            }
        )
