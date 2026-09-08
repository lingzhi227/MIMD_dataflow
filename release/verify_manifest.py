"""Verify the curated release payload, without modifying files."""

import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "release/MANIFEST.json").read_text())
errors = []
for item in manifest["files"]:
    path = root / item["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        errors.append(item["path"])
if errors:
    raise SystemExit("Missing or changed files: " + ", ".join(errors))
print(f"Verified {len(manifest['files'])} release files")
