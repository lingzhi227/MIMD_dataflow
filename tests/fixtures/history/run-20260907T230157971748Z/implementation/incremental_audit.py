"""Fail-fast audits of atomic completed-call snapshots, separate from qualification."""

import hashlib, inspect, json, os
from pathlib import Path


def make_checker(root):
    root = Path(root)
    read = lambda name: json.loads((root / name).read_text())
    schedule = read("schedule.json")
    if schedule.get("profile") == "mesh_batched_fanout.v1":
        from mesh_batched_fanout_sdk import audit_cases
    elif schedule.get("profile") == "mesh_batched_feed_forward.v1":
        from batched_ffn_reference import audit_cases
    elif schedule.get("profile") == "mesh_cache_attention.v1":
        from cache_attention_reference import audit_cases
    else:
        return None

    if "require_complete" not in inspect.signature(audit_cases).parameters:
        return None
    semantic, batches = read("semantic.json"), read("batches.json")
    seen = None
    seen_identity = None
    completed_count = 0

    def identity(stat):
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def check():
        nonlocal seen, seen_identity, completed_count
        path = root / "results.json"
        if not path.exists():
            return
        # The transport atomically replaces snapshots. Stat is only a fast
        # unchanged-file test, never the integrity evidence for a new snapshot.
        if identity(path.stat()) == seen_identity:
            return
        with path.open("rb") as stream:
            before = identity(os.fstat(stream.fileno()))
            raw = stream.read()
            after = identity(os.fstat(stream.fileno()))
        if before != after:
            raise ValueError("completed snapshot was modified in place during read")
        digest = hashlib.sha256(raw).hexdigest()
        if digest == seen:
            seen_identity = after
            return
        result = json.loads(raw)
        # Zero-call initialization is not a completed-call observation.
        if len(result.get("cases", [])) < completed_count:
            raise ValueError("completed-call snapshot regressed")
        if not result.get("cases"):
            seen, seen_identity = digest, after
            return
        report = dict(
            passed=False,
            results_sha256=digest,
            completed_calls=len(result["cases"]),
            scope="Frozen audit of saved completed calls; full qualification still requires process exit and final audit.",
        )
        try:
            report["audit"] = audit_cases(
                schedule, semantic, batches, result, require_complete=False
            )
            report["passed"] = True
        except Exception as error:
            report["error"] = repr(error)
            raise
        finally:
            tmp = root / "completed-audit.json.tmp"
            tmp.write_text(json.dumps(report, indent=2) + "\n")
            tmp.replace(root / "completed-audit.json")
        seen, seen_identity = digest, after
        completed_count = len(result["cases"])

    return check
