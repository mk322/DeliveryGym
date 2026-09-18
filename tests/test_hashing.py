"""Contract tests for content hashing.

The properties tested here are the ones every later milestone leans on:
digests must be relocation-stable (M2 copies a WorldBundle to another directory
and revalidates), order-independent, and sensitive to real changes.
"""

from __future__ import annotations

import json
import shutil

import pytest

from embodiedbench.artifacts.hashing import (
    canonical_json,
    digest_tree,
    sha256_bytes,
    sha256_file,
    sha256_json,
)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "bundle"
    (root / "nav").mkdir(parents=True)
    (root / "semantics").mkdir()
    (root / "world.json").write_text('{"id": "paris"}')
    (root / "nav" / "graph.json").write_text('{"nodes": 144}')
    (root / "semantics" / "entities.jsonl").write_text('{"a": 1}\n{"b": 2}\n')
    return root


def test_content_digest_survives_relocation(tree, tmp_path):
    """the design plan M2: checksums must validate after copying to another directory."""
    before = digest_tree(tree, level="content")
    moved = tmp_path / "elsewhere" / "deeper" / "bundle"
    moved.parent.mkdir(parents=True)
    shutil.copytree(tree, moved)
    assert digest_tree(moved, level="content").digest == before.digest


def test_structure_digest_ignores_content_but_not_size(tree):
    before = digest_tree(tree, level="structure").digest
    # Same length: structure digest is deliberately blind to this.
    (tree / "world.json").write_text('{"id": "PARIS"}')
    assert digest_tree(tree, level="structure").digest == before
    # Different length: must be visible.
    (tree / "world.json").write_text('{"id": "paris-extended"}')
    assert digest_tree(tree, level="structure").digest != before


def test_content_digest_catches_same_size_edit(tree):
    before = digest_tree(tree, level="content").digest
    (tree / "world.json").write_text('{"id": "PARIS"}')
    assert digest_tree(tree, level="content").digest != before


def test_added_and_removed_files_change_both_levels(tree):
    for level in ("structure", "content"):
        before = digest_tree(tree, level=level).digest
        (tree / "nav" / "components.json").write_text("{}")
        after = digest_tree(tree, level=level).digest
        assert after != before
        (tree / "nav" / "components.json").unlink()
        assert digest_tree(tree, level=level).digest == before


def test_excluded_dirs_do_not_contribute(tree):
    before = digest_tree(tree, level="content", exclude_dirs=(".git",)).digest
    (tree / ".git").mkdir()
    (tree / ".git" / "HEAD").write_text("ref: refs/heads/main")
    assert digest_tree(tree, level="content", exclude_dirs=(".git",)).digest == before


def test_symlink_recorded_by_target_not_followed(tree, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    link = tree / "link.json"
    link.symlink_to(outside)
    before = digest_tree(tree, level="content").digest
    link.unlink()
    link.symlink_to(tmp_path / "other.json")
    assert digest_tree(tree, level="content").digest != before


def test_missing_root_raises_rather_than_returning_empty(tmp_path):
    """A missing baseline must fail loudly, not match another missing baseline."""
    with pytest.raises(FileNotFoundError):
        digest_tree(tmp_path / "nope", level="structure")


def test_file_counts_and_sizes_reported(tree):
    result = digest_tree(tree, level="content")
    assert result.file_count == 3
    assert result.total_bytes == sum(
        p.stat().st_size for p in tree.rglob("*") if p.is_file()
    )


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})


def test_canonical_json_rejects_non_finite():
    """NaN/Infinity are not JSON and would not round-trip through consumers."""
    with pytest.raises(ValueError):
        canonical_json({"deadline": float("nan")})


def test_canonical_json_is_compact_and_utf8():
    assert canonical_json({"road": "Rue de Rivoli"}) == b'{"road":"Rue de Rivoli"}'


def test_sha256_file_matches_bytes(tmp_path):
    path = tmp_path / "x.bin"
    payload = b"\x00\x01paris"
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)


def test_digest_of_single_file_root(tmp_path):
    path = tmp_path / "world.json"
    path.write_text(json.dumps({"a": 1}))
    result = digest_tree(path, level="content")
    assert result.file_count == 1
    assert result.digest == sha256_file(path)
