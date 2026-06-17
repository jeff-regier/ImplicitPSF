"""Provenance: stamp git/command/checkpoint identity into every artifact.

Every result parquet self-describes its origin (the producing commit, the checkpoint
and its sha256, the exact command) so a stale or mismatched result is detectable rather
than silently believed. `write_result` writes the stamped parquet and appends a line to
`results/INDEX.jsonl`; `read_result_provenance` reads the stamp back; `checkpoint_provenance`
supplies the same identity for checkpoints. See CLAUDE.md "Provenance".
"""

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = REPO_ROOT / "results" / "INDEX.jsonl"
METADATA_KEY = b"implicitpsf_provenance"


def _git(*args):
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def git_sha():
    return _git("rev-parse", "HEAD")


def git_dirty():
    return _git("status", "--porcelain") != ""


def git_diff_sha():
    """16-hex digest of the uncommitted diff, so a dirty-tree run is still pinned."""
    return hashlib.sha256(_git("diff", "HEAD").encode()).hexdigest()[:16]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uv_lock_sha():
    return sha256_file(REPO_ROOT / "uv.lock")[:16]


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _base_provenance():
    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "git_diff_sha": git_diff_sha(),
        "uv_lock_sha": uv_lock_sha(),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "run_dir": os.environ.get("IMPLICITPSF_RUN_DIR"),
        "created_at": _now(),
    }


def checkpoint_provenance(manifest=None):
    """Identity to embed in a saved checkpoint dict (alongside hyperparameters)."""
    provenance = _base_provenance()
    provenance["argv"] = sys.argv
    if manifest is not None:
        provenance["data_manifest"] = str(manifest)
        provenance["data_manifest_sha256"] = sha256_file(manifest)
    return provenance


def result_provenance(checkpoint=None, source=None, purpose=None):
    provenance = _base_provenance()
    provenance["source"] = source
    provenance["purpose"] = purpose
    if checkpoint is not None:
        provenance["checkpoint_path"] = str(checkpoint)
        provenance["checkpoint_sha256"] = sha256_file(checkpoint)
    return provenance


def _relpath(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if REPO_ROOT in path.parents else str(path)


def _append_index(record):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def write_result(df, path, *, checkpoint=None, source=None, purpose=None):
    """Write a result DataFrame to parquet with embedded provenance + index it.

    Args:
        df: tidy results DataFrame
        path: output .parquet path
        checkpoint: the checkpoint that produced these results (its sha256 is recorded)
        source: producing module, e.g. "run_eval"
        purpose: free-text description for the index
    """
    provenance = result_provenance(checkpoint=checkpoint, source=source, purpose=purpose)
    table = pa.Table.from_pandas(df)
    metadata = dict(table.schema.metadata or {})
    metadata[METADATA_KEY] = json.dumps(provenance).encode()
    table = table.replace_schema_metadata(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    _append_index(
        {
            "kind": "result",
            "path": _relpath(path),
            "sha256": sha256_file(path),
            "git_sha": provenance["git_sha"],
            "git_dirty": provenance["git_dirty"],
            "checkpoint": provenance.get("checkpoint_path"),
            "checkpoint_sha256": provenance.get("checkpoint_sha256"),
            "run_dir": provenance["run_dir"],
            "source": source,
            "purpose": purpose,
            "created_at": provenance["created_at"],
        }
    )
    return provenance


def read_result_provenance(path):
    """Return the provenance dict stamped into a result parquet (fails loudly if absent)."""
    metadata = pq.read_schema(path).metadata or {}
    blob = metadata.get(METADATA_KEY)
    if blob is None:
        raise ValueError(f"{path} has no embedded provenance (not written via write_result)")
    return json.loads(blob)


def append_index_run(run_dir, provenance):
    """Index a wrapped run (from implicitpsf.record) as one INDEX.jsonl line."""
    _append_index(
        {
            "kind": "run",
            "run_dir": _relpath(run_dir),
            "git_sha": provenance["git_sha"],
            "git_dirty": provenance["git_dirty"],
            "command": provenance["command"],
            "purpose": provenance["purpose"],
            "exit_code": provenance.get("exit_code"),
            "created_at": provenance["created_at"],
        }
    )
