"""The hash chain that names a full block of KV by everything that produced it.

A block's digest depends on its tokens and on the digest of the block before it, so equal digests
mean equal prefixes, not only equal blocks. The chain starts at a **root** that folds in what makes
the stored values differ for the same tokens: the model, the compute dtype, the codec, the block
size, the cache format and the salt that isolates one tenant from another::

    root = blake2b(len(model_id) || model_id || dtype || codec || block_size ||
                   CACHE_FORMAT_VERSION || lookahead || len(salt) || salt)
    d_i  = blake2b(d_{i-1} || tokens of block i as little-endian uint32 || aux)

Every integer is little-endian and every variable-length field carries its length, so two
different inputs never concatenate to the same bytes. Digests are 16 bytes of BLAKE2b from the
standard library. Tokens become bytes in one NumPy call per prompt, never a Python loop over
tokens; the per-block loop only hashes slices of that buffer.

Bump :data:`CACHE_FORMAT_VERSION` whenever a change makes the KV for the same tokens differ (a
kernel, a codec, the weights layout): cached blocks are never migrated, a new version simply
stops matching the old ones.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CACHE_FORMAT_VERSION",
    "DIGEST_SIZE",
    "ChainRoot",
    "chain_digests",
    "root_digest",
]

#: Version of the cached KV format; part of every root.
CACHE_FORMAT_VERSION: Final[int] = 1

#: Bytes of one digest.
DIGEST_SIZE: Final[int] = 16

_U32: Final = struct.Struct("<I")


@dataclass(frozen=True, slots=True)
class ChainRoot:
    """What a root digest is made of, apart from the salt.

    Attributes:
        model_id: Identity of the loaded weights (source and architecture fingerprint).
        dtype: Label of the compute dtype the values are stored in, such as ``"fp16"``.
        codec: The KV codec, such as ``"raw"``.
        block_size: Tokens per block.
        lookahead: Tokens beyond the block a digest depends on; 0 for plain text.
    """

    model_id: str
    dtype: str
    codec: str
    block_size: int
    lookahead: int = 0

    def __post_init__(self) -> None:
        require_int(self.block_size, "block_size", minimum=1)
        require_int(self.lookahead, "lookahead")


def _field(data: bytes) -> bytes:
    return _U32.pack(len(data)) + data


def root_digest(root: ChainRoot, salt: str | None) -> bytes:
    """The digest every chain of this model, format and salt starts from.

    ``None`` and the empty salt are the same tenant.
    """
    hasher = hashlib.blake2b(digest_size=DIGEST_SIZE)
    hasher.update(_field(root.model_id.encode()))
    hasher.update(_field(root.dtype.encode()))
    hasher.update(_field(root.codec.encode()))
    hasher.update(_U32.pack(root.block_size))
    hasher.update(_U32.pack(CACHE_FORMAT_VERSION))
    hasher.update(_U32.pack(root.lookahead))
    hasher.update(_field((salt or "").encode()))
    return hasher.digest()


def chain_digests(
    parent: bytes, token_ids: Sequence[int], block_size: int, *, aux: bytes = b""
) -> list[bytes]:
    """Digests of the full blocks of ``token_ids``, chained from ``parent``.

    Args:
        parent: The root, or the digest of the block just before ``token_ids``.
        token_ids: Tokens that start on a block boundary; a partial last block is ignored.
        block_size: Tokens per block.
        aux: Extra bytes folded into every block (a media hash later); empty for text.

    Returns:
        One digest per full block, in order.
    """
    full = len(token_ids) // block_size
    if full == 0:
        return []
    data = np.asarray(token_ids[: full * block_size], dtype="<u4").tobytes()
    stride = 4 * block_size
    view = memoryview(data)
    digests: list[bytes] = []
    previous = parent
    for index in range(full):
        hasher = hashlib.blake2b(previous, digest_size=DIGEST_SIZE)
        hasher.update(view[index * stride : (index + 1) * stride])
        if aux:
            hasher.update(aux)
        previous = hasher.digest()
        digests.append(previous)
    return digests
