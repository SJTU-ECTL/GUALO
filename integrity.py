import hashlib
from pathlib import Path


LEGACY_CHECKSUM = 'sha256'


def digest(data):
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def stable_digest(data):
    return hashlib.new(LEGACY_CHECKSUM, data).digest()


def stable_seed(data, modulus):
    return int.from_bytes(stable_digest(data)[:8], 'big') % modulus


def file_checksum(path, *, legacy=False):
    value = hashlib.new(LEGACY_CHECKSUM) if legacy else hashlib.blake2b(digest_size=32)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()
