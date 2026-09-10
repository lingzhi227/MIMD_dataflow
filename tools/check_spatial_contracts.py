"""Reproduce the survey's finite protocol and exact GEMM mapping checks."""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import struct

from bootstrap import configure
ROOT = configure(Path(__file__))
from frontend import parse
from ir import verify
from planner import plan
from protocol_safety import explore
from summa_protocol import from_plan, mismatched_stream_order


def mapping_checks():
    """Exact integer checks. Logical slots do not measure device cycles."""
    cases = 0
    macs = 0
    for m, n, kmax, latency in itertools.product(range(1, 5), range(1, 5), range(1, 5), (1, 2, 4)):
        a = [[(i * 3 + k * 5) % 7 - 3 for k in range(kmax)] for i in range(m)]
        b = [[(k * 7 + j * 3) % 11 - 5 for j in range(n)] for k in range(kmax)]
        acc = [[0] * n for _ in range(m)]
        instances = list(itertools.product(range(m), range(n), range(kmax)))
        time = lambda v: v[0] + v[1] + latency * v[2]
        occupied = set()
        for i, j, k in sorted(instances, key=time):
            t = time((i, j, k))
            # A injection: i + L*k; B injection: j + L*k; each hop takes one slot.
            assert t == i + latency*k + j == j + latency*k + i
            if k:
                assert t - time((i, j, k-1)) >= latency
            if j:
                assert t - time((i, j-1, k)) >= 1
            if i:
                assert t - time((i-1, j, k)) >= 1
            assert (i, j, t) not in occupied
            occupied.add((i, j, t))
            acc[i][j] += a[i][k] * b[k][j]
        ref = [[sum(a[i][k]*b[k][j] for k in range(kmax)) for j in range(n)] for i in range(m)]
        assert acc == ref
        cases += 1
        macs += len(instances)
    f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
    large = float(2**24)
    left = f32(f32(large + 1) - large)
    right = f32(large + f32(1 - large))
    assert (left, right) == (0.0, 1.0)
    return {"status": "PASS", "profiles": cases, "mac_instances": macs,
            "arithmetic": "exact Python integers; values small enough for exact int32 arithmetic",
            "grid": "unfolded M by N", "schedule": "theta(i,j,k)=i+j+L*k",
            "assumptions": "unit-hop links; independent operand forwarding; MAC recurrence latency L; logical slots only",
            "folding_counterexample": {"map": "all instances to PE (0,0)",
                "instances": [[0,1,0], [1,0,0]], "common_time": 1,
                "failure": "two MACs require the same single-issue PE in one slot"},
            "float32_counterexample": {"a": large, "b": 1, "c": -large,
                "(a+b)+c": left, "a+(b+c)": right, "exact_sum": 1}}


def run():
    source = ROOT / 'benchmarks/linear_algebra/sdk_examples/mesh_gemm_64x64x64_4x4_vector/hls.cpp'
    existing_plan = plan(verify(parse(source), 4, 64))
    cases = {"summa_p4": from_plan(existing_plan),
             "summa_missing_y_wait": from_plan(existing_plan, omit_y_wait=True),
             "summa_early_panel_reuse": from_plan(existing_plan, early_reuse=True),
             "dag_fifo_depth1": mismatched_stream_order(1),
             "dag_fifo_depth2": mismatched_stream_order(2)}
    outcomes = {name: explore(case) for name, case in cases.items()}
    expected = dict(summa_p4="PASS", summa_missing_y_wait="UNSAFE",
                    summa_early_panel_reuse="UNSAFE", dag_fifo_depth1="DEADLOCK", dag_fifo_depth2="PASS")
    for name, result in outcomes.items():
        if result['status'] != expected[name]:
            raise RuntimeError(f'Unexpected result for {name}: {result}')
    paths = ['lib/IR/event_protocol.py', 'lib/Analysis/protocol_safety.py',
             'lib/Transforms/summa_protocol.py', 'tools/check_spatial_contracts.py',
             'runtime/csl/mesh_gemm_pe.csl', 'runtime/csl/mesh_gemm_vector.csl',
             'lib/Conversion/mesh_gemm.py']
    return {"schema": "spatial-contract-checks.v1", "mapping": mapping_checks(),
            "protocol_results": outcomes,
            "plan_input": {"path": str(source.relative_to(ROOT)), "text": source.read_text(),
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "epochs": 4, "bound": 64},
            "serialized_plan": existing_plan,
            "plan_canonical_json_sha256": hashlib.sha256(json.dumps(existing_plan, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest(),
            "frozen_protocol_inputs": {name: asdict(case) for name, case in cases.items()},
            "source_sha256": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths},
            "limitations": ["SUMMA protocol is a manually reviewed abstraction of the existing template, not extracted CSL semantics",
                "No numerical values, collective routing, task bits or clock timing in protocol exploration",
                "Every modeled step advances an actor; unmodeled environment delays require eventual-completion/fairness assumptions",
                "No SDK or hardware execution is claimed by this report",
                "The systolic mapping check and blocked SUMMA backend are distinct realizations of GEMM"]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({name: {k: v for k, v in value.items() if k in ('status','states','transitions')}
                      for name, value in result['protocol_results'].items()}, indent=2))
