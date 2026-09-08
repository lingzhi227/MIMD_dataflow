"""Retrospective, frozen native/target preflight failures; never SDK qualification."""

import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parent = ROOT / "projects/waferllm/cache_attention_5x256x512_8x8"
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
first = parent / "run-20260907T214858391453Z"
second = parent / "run-20260907T215130589417Z"
third = parent / "run-20260907T220401939966Z"
assert not read(first / "application-gate.json")["passed"]
assert read(first / "application-gate.json")["failure"]["epoch"] == 2
code = """import sys,json,importlib.util
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'implementation'))
from integrity import verify_bundle
from mesh_cache_attention import values
from cache_attention_reference import reference
verify_bundle(root)
s,m,b=[json.loads((root/n).read_text()) for n in ('schedule.json','semantic.json','batches.json')]
spec=importlib.util.spec_from_file_location('original',root/'application-reference.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
_,stages=reference(s,values(m,b[6]));out={'result':stages.pop('result').ravel().tolist()}
try:ref.check(s['B'],s['N'],s['S'],b[6],out,stages)
except AssertionError as error:print(json.dumps(dict(epoch=6,passed=False,error=repr(error))))
else:raise AssertionError('Expected preserved target numerical failure')
"""
target = json.loads(
    subprocess.check_output([sys.executable, "-c", code, str(second)], text=True)
)
assert read(second / "application-gate.json")["passed"]
assert all(
    read(third / n)["passed"]
    for n in ("application-gate.json", "target-application-gate.json")
)
assert all(
    read(p / "batches.json") == read(third / "batches.json") for p in (first, second)
)
report = ROOT / "evidence/cache-attention-numerical-boundaries.json"
assert not report.exists()
report.write_text(
    json.dumps(
        dict(
            passed=True,
            scope="Same original inputs and fixed accuracy gates; native failure and separately predicted target failure retained. No SDK result inferred.",
            native_unblocked_failure=read(first / "application-gate.json")["failure"],
            target_shard64_failure=target,
            blocked32_native_and_target_passed=True,
            artifacts={
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    first / "manifest.json",
                    first / "application-gate.json",
                    second / "manifest.json",
                    second / "application-gate.json",
                    third / "manifest.json",
                    third / "application-gate.json",
                    third / "target-application-gate.json",
                    Path(__file__),
                )
            },
        ),
        indent=2,
    )
    + "\n"
)
print(report)
