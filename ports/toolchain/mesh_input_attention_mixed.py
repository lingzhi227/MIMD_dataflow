"""Development typed lowering of the resident mixed-width attention/MLP chain.

The scoped backend accepts this explicit policy after typed SDK validation.
Catalog qualification and performance review remain separate requirements.
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
    from input_contracts import verify_declarations

    verify_declarations(module)
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
    from precision_math_contracts import sqrt_bounds, exp_bounds, SDK_MATH_SHA256
    from input_contracts import half_dot_bound

    math_contract = dict(
        sqrt=sqrt_bounds(),
        exp=exp_bounds(),
        sdk_math_sha256=SDK_MATH_SHA256,
        assumptions="Binary32 RN operations (fusion may omit rounding), truncating float-to-i16, exact normal power-of-two scaling; pinned SDK source expressions",
    )
    score_partial = half_dot_bound(
        prefix["q"]["pair"]["output_absolute"],
        prefix["k"]["pair"]["output_absolute"],
        s["Nt"],
    )
    score_upper = half_dot_bound(score_partial, 1, s["P"])
    shifted_upper = upper_f32(2 * score_upper * s["scale"] * (1 + U) ** 2)
    check(
        shifted_upper <= 12,
        "mixed softmax must remain inside the scoped SDK exp validation domain [-12,0]",
    )
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
            score_absolute=score_upper,
            softmax_shift_absolute=shifted_upper,
            sdk_math_contract=math_contract,
            square_sum_upper=contraction(64 * z * z, 72),
            status="pinned_source_range_contract",
            pending=[
                "complete standard-driver frozen qualification and performance review",
            ],
            scope="Finite-range contract under explicit pinned SDK source/arithmetic assumptions; empirical accuracy and catalog qualification are separate gates.",
        ),
        typed_transport=[
            dict(phase=8, left="f16", right="f16", accumulator="f32", result="f32"),
            dict(phase=4, left="f32", right="f32", accumulator="f32", result="f32"),
            dict(phase=3, left="f32", right="f16", accumulator="f32", result="f32"),
        ],
        qualification=dict(
            admitted=True, catalog_qualified=False, scope="scoped typed backend; catalog qualification remains separate"
        ),
    )
    s["input_prefix_stages"][2].update(
        accumulation="f32", left_dtype="f16", right_dtype="f16", result_dtype="f32"
    )
    s["projection_prelude"].update(
        accumulation="f32", left_dtype="f32", right_dtype="f16", result_dtype="f32"
    )
    s["mlp_numerical_bounds"] = s.pop("numerical_bounds")
    s["storage_lifetimes"] = lifetime_plan(s["storage_lifetimes"], s)
    s["stages"] = [p["name"] for p in s["storage_lifetimes"]["phases"]]
    s["ownership"].update(
        inputs="eleven immutable public half tensors; original X is the first residual",
        x_work="half input normalization owner; later half post-Z normalization and MLP output",
        gate="half score partial then MLP activated gate",
        mixed_work="separate f32 V/A routes, widened residual/delta and squared-Z scratch; phase-joined leases",
        normalization="RMS(Z) runs in f32; narrow once before half-block MLP",
    )
    s["descriptor_entry_states"][
        "mixed_compute"
    ] = "V half column converts before loading DSR1; PV has f32 left and strided f32 right; O has f32 left and contiguous f16 right"
    s["resources"]["mixed_transport"] = copy.deepcopy(s["typed_transport"])

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
        + "Explicit mixed precision continuation: f32 V/probability/PV/O/Z and post-residual RMS, followed by half-block/f32-merge MLP. Scoped backend; no catalog qualification, source-relative performance or hardware claim.\n"
    )
