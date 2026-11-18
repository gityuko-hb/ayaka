"""The hash chain: what the root folds in, how blocks chain, and fixed vectors."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import replace

import pytest

from ayaka.kvcache.prefix import hashing
from ayaka.kvcache.prefix.hashing import (
    CACHE_FORMAT_VERSION,
    DIGEST_SIZE,
    ChainRoot,
    chain_digests,
    root_digest,
)

pytestmark = pytest.mark.unit

_IDENTITY = ChainRoot(model_id="tiny/abc", dtype="fp16", codec="raw", block_size=4)


def _manual_root(identity: ChainRoot, salt: str) -> bytes:
    """The root written out from the documented layout, sharing no code with the module."""

    def field(data: bytes) -> bytes:
        return struct.pack("<I", len(data)) + data

    payload = (
        field(identity.model_id.encode())
        + field(identity.dtype.encode())
        + field(identity.codec.encode())
        + struct.pack("<I", identity.block_size)
        + struct.pack("<I", CACHE_FORMAT_VERSION)
        + struct.pack("<I", identity.lookahead)
        + field(salt.encode())
    )
    return hashlib.blake2b(payload, digest_size=16).digest()


def test_the_root_follows_the_documented_layout() -> None:
    assert root_digest(_IDENTITY, "tenant") == _manual_root(_IDENTITY, "tenant")
    assert len(root_digest(_IDENTITY, None)) == DIGEST_SIZE


def test_fixed_vectors_catch_a_silent_format_change() -> None:
    # Recomputed from the layout in the module docstring; change only with a version bump.
    root = root_digest(_IDENTITY, None)
    chain = chain_digests(root, [1, 2, 3, 4, 5, 6, 7, 8], 4)

    assert root.hex() == "e499ceb10cddaf9c9f88fe08f2c238e3"
    assert [d.hex() for d in chain] == [
        "257c3f2ae96f095040cb36829f09ddd8",
        "f730581d6ee5e1dfd1a551222619e215",
    ]


@pytest.mark.parametrize(
    "changed",
    [
        replace(_IDENTITY, model_id="tiny/abd"),
        replace(_IDENTITY, dtype="bf16"),
        replace(_IDENTITY, codec="fp8"),
        replace(_IDENTITY, block_size=8),
        replace(_IDENTITY, lookahead=1),
    ],
)
def test_every_identity_field_changes_the_root(changed: ChainRoot) -> None:
    assert root_digest(changed, None) != root_digest(_IDENTITY, None)


def test_the_salt_changes_the_root_and_none_equals_empty() -> None:
    assert root_digest(_IDENTITY, "a") != root_digest(_IDENTITY, "b")
    assert root_digest(_IDENTITY, "a") != root_digest(_IDENTITY, None)
    assert root_digest(_IDENTITY, "") == root_digest(_IDENTITY, None)


def test_the_format_version_changes_the_root(monkeypatch: pytest.MonkeyPatch) -> None:
    before = root_digest(_IDENTITY, None)
    monkeypatch.setattr(hashing, "CACHE_FORMAT_VERSION", CACHE_FORMAT_VERSION + 1)

    assert root_digest(_IDENTITY, None) != before


def test_length_prefixes_keep_fields_apart() -> None:
    joined = replace(_IDENTITY, model_id="ab", dtype="c")
    split = replace(_IDENTITY, model_id="a", dtype="bc")

    assert root_digest(joined, None) != root_digest(split, None)


def test_only_full_blocks_get_a_digest() -> None:
    root = root_digest(_IDENTITY, None)

    assert chain_digests(root, [1, 2, 3], 4) == []
    assert len(chain_digests(root, list(range(11)), 4)) == 2


def test_a_digest_depends_on_every_block_before_it() -> None:
    root = root_digest(_IDENTITY, None)
    a = chain_digests(root, [1, 2, 3, 4, 9, 9, 9, 9], 4)
    b = chain_digests(root, [1, 2, 3, 5, 9, 9, 9, 9], 4)

    assert a[0] != b[0]
    assert a[1] != b[1]  # the same second block under a different first block


def test_chaining_in_pieces_equals_chaining_at_once() -> None:
    root = root_digest(_IDENTITY, "s")
    tokens = list(range(100, 124))
    whole = chain_digests(root, tokens, 4)
    first = chain_digests(root, tokens[:8], 4)
    rest = chain_digests(first[-1], tokens[8:], 4)

    assert whole == first + rest


def test_large_token_ids_are_hashed_as_uint32() -> None:
    root = root_digest(_IDENTITY, None)

    assert chain_digests(root, [2**32 - 1] * 4, 4) != chain_digests(root, [0] * 4, 4)


def test_aux_bytes_change_the_digest() -> None:
    root = root_digest(_IDENTITY, None)

    assert chain_digests(root, [1, 2, 3, 4], 4, aux=b"img") != chain_digests(root, [1, 2, 3, 4], 4)
