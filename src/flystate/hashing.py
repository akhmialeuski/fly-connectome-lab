"""Portable content hashes and deterministic random-stream identifiers."""

import hashlib
import json
from pathlib import Path

HASH_CHUNK_SIZE: int = 1 << 20
SEED_BYTES: int = 8


def canonical_json(obj: object) -> str:
    """Encode a finite JSON value with canonical key ordering.

    :param obj: JSON-compatible value.
    :type obj: object
    :returns: Compact UTF-8-compatible JSON text.
    :rtype: str
    :raises ValueError: If a number is not finite.
    :raises TypeError: If the value cannot be serialized.
    """
    return json.dumps(
        obj=obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False
    )


def sha256_text(text: str) -> str:
    """Hash UTF-8 text.

    :param text: Input text.
    :type text: str
    :returns: Lowercase hexadecimal digest.
    :rtype: str
    """
    return hashlib.sha256(string=text.encode(encoding='utf-8')).hexdigest()


def sha256_obj(obj: object) -> str:
    """Hash a canonical JSON representation.

    :param obj: JSON-compatible value.
    :type obj: object
    :returns: SHA-256 digest.
    :rtype: str
    """
    return sha256_text(text=canonical_json(obj=obj))


def sha256_file(path: Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """Hash a file without loading it into memory.

    :param path: File to read.
    :type path: Path
    :param chunk_size: Positive read-buffer size in bytes.
    :type chunk_size: int
    :returns: SHA-256 digest.
    :rtype: str
    :raises ValueError: If the buffer size is not positive.
    :raises OSError: If reading fails.
    """
    if chunk_size < 1:
        raise ValueError('Hash chunk size must be positive.')
    digest = hashlib.sha256()
    with path.open(mode='rb') as source:
        while block := source.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def stable_int(key: str) -> int:
    """Derive an unsigned 64-bit stream identifier.

    :param key: Stable stream name.
    :type key: str
    :returns: First eight SHA-256 bytes interpreted in big-endian order.
    :rtype: int
    """
    digest = hashlib.sha256(string=key.encode(encoding='utf-8')).digest()
    return int.from_bytes(bytes=digest[:SEED_BYTES], byteorder='big')
