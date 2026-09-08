"""Development typed lowering of the resident mixed-width attention/MLP chain.

Public admission remains closed until target arithmetic range, transport audit,
and fresh frozen SDK qualification are complete. No catalog entry is implied.
"""

import copy
from frontend import check
from input_attention_precision_structure import canonical, shadow
from mesh_input_attention import plan as half_plan, generate as half_generate
from input_attention_mixed_lifetimes import plan as lifetime_plan
from input_attention_mixed_csl import generate as mixed_generate
from input_attention_mixed_reference import stages
from rms_f32_bounds import half_output, contraction, upper_f32, U, ETA
from input_contracts import effective_bound


def verify(module, epochs, bound):
    m, _ = canonical(module, epochs, bound)
    check(
        m["instrumentation"] == "counters", "mixed resident profile requires counters"
    )
    check(
        m["nodes"][0]["shape"] == [64, 64] and m["nodes"][8]["shape"] == [64, 256],
        "mixed resident profile currently requires M=N=64 F=256",
    )
    check(
        all(
            n.get("dataflow", {}).get("rows", 8) == 8
            and n.get("dataflow", {}).get("cols", 8) == 8
            for n in m["nodes"]
        ),
        "mixed resident profile currently requires P=8",
    )
    plan(m)
    return m


def plan(m, partitions=1):
    s = half_plan(shadow(m), partitions)
    ns = m["nodes"]
    # Preserve compatible half prefix certificates only. All mixed-boundary
    # certificates are recomputed, never inferred from the half shadow.
    prefix = copy.deepcopy(s["input_prefix_bounds"])
    norm = prefix["v"]["normalized_row"]
    v = contraction(norm["row_l1_upper"] * effective_bound(ns[4], m["input_bound"]), 64)
    prefix["v"] = dict(normalized_row=norm, projection_absolute=v, accumulation="f32")
    mass = upper_f32((1 + U) ** 2 / (1 - U) ** 72 + 128 * ETA)
    a = contraction(v * mass, 64)
    o = contraction(64 * a * effective_bound(ns[7], m["input_bound"]), 64)
    z = upper_f32((o + effective_bound(ns[0], m["input_bound"])) * (1 + U) + ETA)
    rms = half_output(
        effective_bound(ns[1], m["input_bound"]), 64, ns[23]["epsilon"], 72
    )
    # Old MLP certificate is only reusable if its operand bound covers the new
    # independently derived normalized-half range. It is not an RMS proof.
    check(
        rms["output"] <= s["numerical_bounds"]["inputs"][0],
        "mixed normalized range fits checked half-block MLP",
    )
    final = upper_f32((z + s["numerical_bounds"]["output"]) * (1 + U) + ETA)
    check(final <= 65504, "mixed final narrowing stays finite")
    for key in (
        "normalization_bounds",
        "prelude_numerical_bounds",
        "attention_output_bound",
    ):
        s.pop(key, None)
    # Nested half attention plan is not consumed by the CSL continuation and
    # would expose incorrect precision/range facts to plan readers.
    s.pop("attention_schedule", None)
    s.update(
        profile="mesh_input_attention_mixed.v1",
        input_prefix_bounds=prefix,
        precision_boundaries=copy.deepcopy(m["precision_boundaries"]),
        mixed_numerical_bounds=dict(
            value=v,
            probability_mass=mass,
            attention=a,
            projection=o,
            residual=z,
            normalized=rms,
            final_pre_half=final,
            status="development_conditional",
            pending=[
                "SDK sqrt normal-domain rounding derivation",
                "SDK exp finite nonnegative and zero-image domain certificate",
            ],
            scope="Range estimates conditional on documented SDK math properties; public compiler admission remains closed.",
        ),
        typed_transport=[
            dict(phase=8, left="f16", right="f16", accumulator="f32", result="f32"),
            dict(phase=4, left="f32", right="f32", accumulator="f32", result="f32"),
            dict(phase=3, left="f32", right="f16", accumulator="f32", result="f32"),
        ],
        qualification=dict(
            admitted=False, scope="development typed plan and code generation"
        ),
    )
    s["storage_lifetimes"] = lifetime_plan(s["storage_lifetimes"], s)
    s["memory_per_pe"]["mixed_f32_arrays"] = s["storage_lifetimes"][
        "mixed_allocation_bytes"
    ]
    s["memory_per_pe"]["mixed_code_descriptor_reserve"] = 4096
    check(sum(s["memory_per_pe"].values()) <= 49152, "mixed resident PE memory budget")
    s["composition"].update(
        borrowed="half score storage; separate f32 probability/V/A/Z buffers; half MLP begins after explicit RMS narrowing"
    )
    return s


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "mixed resident epoch count")
    return (
        [
            {m["nodes"][-1]["host"]: stages(m, b)["result"].ravel().tolist()}
            for b in batches
        ],
        {},
    )


def generate(s, dest):
    check(s["profile"] == "mesh_input_attention_mixed.v1", "mixed schedule identity")
    check(
        s["typed_transport"]
        == [
            dict(phase=8, left="f16", right="f16", accumulator="f32", result="f32"),
            dict(phase=4, left="f32", right="f32", accumulator="f32", result="f32"),
            dict(phase=3, left="f32", right="f16", accumulator="f32", result="f32"),
        ],
        "mixed transport widths agree with lowering",
    )
    from input_attention_mixed_lifetimes import validate as verify_lifetimes

    life = s["storage_lifetimes"]
    verify_lifetimes(life)
    half_generate(s, dest)
    mixed_generate(dest, s)
    from pathlib import Path

    notice = Path(dest) / "SOURCE-NOTICE.txt"
    notice.write_text(
        notice.read_text()
        + "Explicit mixed precision continuation: f32 V/probability/PV/O/Z and post-residual RMS, followed by half-block/f32-merge MLP. Development profile; no qualified compiler, source-relative performance or hardware claim.\n"
    )
