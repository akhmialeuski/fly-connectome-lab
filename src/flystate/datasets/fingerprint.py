"""Content identities for selected source images and prepared dataset semantics."""

from collections.abc import Sequence

from flystate.datasets.celeba import CelebAAdapter
from flystate.datasets.subset import Sample
from flystate.experiments.config import PreprocessConfig
from flystate.hashing import sha256_file, sha256_obj, sha256_text


def selected_images_hash(adapter: CelebAAdapter, samples: Sequence[Sample]) -> str:
    """Hash ordered source JPEG digests before consulting any preprocessing cache.

    :param adapter: Located source dataset.
    :type adapter: CelebAAdapter
    :param samples: Stable selected sample order.
    :type samples: Sequence[Sample]
    :returns: SHA-256 of concatenated per-file hexadecimal SHA-256 values.
    :rtype: str
    """
    directory = adapter.locate().images_dir
    return sha256_text(
        text=''.join(sha256_file(path=directory / sample.filename) for sample in samples)
    )


def dataset_fingerprint(
    annotation: str, samples: Sequence[Sample], images_sha256: str, preprocess: PreprocessConfig
) -> str:
    """Hash dataset content, split assignments, labels and alignment parameters.

    :param annotation: Combined annotation fingerprint.
    :type annotation: str
    :param samples: Ordered selected samples.
    :type samples: Sequence[Sample]
    :param images_sha256: Hash of ordered source JPEG digests.
    :type images_sha256: str
    :param preprocess: Alignment parameters.
    :type preprocess: PreprocessConfig
    :returns: Canonical SHA-256 dataset identity.
    :rtype: str
    """
    return sha256_obj(
        obj={
            'annotation': annotation,
            'samples': [[item.sample_id, item.split, item.label] for item in samples],
            'images_sha256': images_sha256,
            'preprocess': preprocess.model_dump(mode='json'),
        }
    )
