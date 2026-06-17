"""Run wrapper: mint a runs/ dir, stamp provenance, tee output, index the run.

Replaces bare `nohup python -m ... > log` launches so every compute job is a
self-contained, timestamped record. Usage:

    uv run python -m implicitpsf.record --kind eval --purpose "psctx converged eval" -- \
        python -m implicitpsf.evaluation.run_eval --checkpoint ... --out ...

The wrapped command inherits IMPLICITPSF_RUN_DIR; train_psf/run_eval read it so their
checkpoints and result parquets back-reference this run. See CLAUDE.md "Provenance".
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time

from implicitpsf.provenance import (
    REPO_ROOT,
    append_index_run,
    git_diff_sha,
    git_dirty,
    git_sha,
    uv_lock_sha,
)


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["train", "eval", "probe", "figure"])
    parser.add_argument("--purpose", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command")
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("no command given after --")
    return args.kind, args.purpose, command


def tee(command, env, log_path):
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def main():
    kind, purpose, command = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / f"{timestamp}_{kind}_{slugify(purpose)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "invocation.sh").write_text("#!/bin/sh\n" + " ".join(command) + "\n")
    provenance = {
        "kind": kind,
        "purpose": purpose,
        "command": " ".join(command),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "git_diff_sha": git_diff_sha(),
        "uv_lock_sha": uv_lock_sha(),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    env = dict(os.environ, IMPLICITPSF_RUN_DIR=str(run_dir))
    print(f"[record] run dir: {run_dir}", flush=True)
    exit_code = tee(command, env, run_dir / "run.log")

    provenance["exit_code"] = exit_code
    append_index_run(run_dir, provenance)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
