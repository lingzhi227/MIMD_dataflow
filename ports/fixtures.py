"""Independent application-level inputs/checks; not imported by compiler/codegen."""

import numpy as np
import json
from pathlib import Path
from import_stencil import reference as mlir_reference

ROOT = Path(__file__).resolve().parent


def batches(kind, count=4):
    if kind.startswith("composed_ffn:"):
        from composed_ffn_fixtures import batches as make_batches

        return make_batches(*map(int, kind.split(":")[1:]))
    if kind.startswith("projected_cache:"):
        from projected_cache_fixtures import batches as make_batches

        return make_batches(*map(int, kind.split(":")[1:]))
    if kind.startswith("cache_attention:"):
        from cache_attention_fixtures import batches as make_batches

        return make_batches(*map(int, kind.split(":")[1:]))
    if kind.startswith("batched_ffn:"):
        from batched_ffn_fixtures import batches as make_batches

        return make_batches(*map(int, kind.split(":")[1:]))
    if kind.startswith("batched_fanout:"):
        from batched_fanout_fixtures import batches as make_batches

        return make_batches(*map(int, kind.split(":")[1:]))
    if kind.startswith("batched_rms:"):
        from batched_rms_fixtures import batches as make_batches

        return make_batches(*map(int, kind.split(":")[1:]))[:count]
    if kind.startswith("input_attention_mixed:"):
        from input_attention_fixtures import batches as input_batches

        return input_batches(*map(int, kind.split(":")[1:]))[:count]
    if kind.startswith("pair_rotation:"):
        from pair_rotation_fixtures import batches as pair_batches

        _, m, n, broadcast, order = kind.split(":")
        return pair_batches(int(m), int(n), bool(int(broadcast)), order)[:count]
    if kind.startswith("gated_silu:"):
        from swiglu_fixtures import batches as gating_batches

        return gating_batches(*map(int, kind.split(":")[1:]))[:count]
    if kind.startswith("attention_tail:"):
        from attention_tail_fixtures import batches as attention_batches

        return attention_batches(*map(int, kind.split(":")[1:]))[:count]
    if kind.startswith("prefill_tail:"):
        from prefill_tail_fixtures import batches as tail_batches

        return tail_batches(*map(int, kind.split(":")[1:]))[:count]
    if kind.startswith("feed_forward:"):
        from feed_forward_fixtures import batches as ff_batches

        return ff_batches(*map(int, kind.split(":")[1:]))[:count]
    if kind.startswith("projection_residual_rms:"):
        from projection_residual_rms_fixtures import batches as composed_batches

        _, m, n = kind.split(":")
        return composed_batches(int(m), int(n))[:count]
    if kind.startswith("mlp_blocked:"):
        from mlp_fixtures import batches as mlp_batches

        _, m, n, f, p = kind.split(":")
        return mlp_batches(int(m), int(n), int(f), int(p))[:count]
    if kind.startswith("mlp:"):
        from mlp_fixtures import batches as mlp_batches

        _, m, n, f = kind.split(":")
        return mlp_batches(int(m), int(n), int(f))[:count]
    if kind.startswith("attention:"):
        from attention_fixtures import batches as attention_batches

        _, m, n, scale = kind.split(":")
        return attention_batches(int(m), int(n))[:count]
    if kind.startswith("score_softmax:"):
        from score_softmax_fixtures import batches as resident_batches

        _, m, n, scale = kind.split(":")
        return resident_batches(int(m), int(n))[:count]
    if kind.startswith("device_matmul:"):
        from device_matmul_fixtures import batches as device_batches

        return device_batches(*map(int, kind.split(":")[1:]))[:count]
    if kind.startswith("score:"):
        from score_fixtures import batches as score_batches

        return score_batches(*map(int, kind.split(":")[1:]))[:count]
    if kind.startswith("normalized_fanout:"):
        from normalized_fanout_fixtures import batches as fanout_batches

        return fanout_batches(*map(int, kind.split(":")[1:]))[:count]
    if kind.startswith("normalized_matmul:"):
        from normalized_matmul_fixtures import batches as resident_batches

        return resident_batches(*map(int, kind.split(":")[1:]))[:count]
    if kind.startswith("distributed_softmax:"):
        from softmax_fixtures import batches as softmax_batches

        return softmax_batches(*map(int, kind.split(":")[1:]))[:count]
    if kind.startswith("distributed_rms:"):
        from rms_fixtures import batches as rms_batches

        return rms_batches(*map(int, kind.split(":")[1:]))[:count]
    if kind.startswith("distributed_fft:"):
        from fft_fixtures import batches as fft_batches

        return fft_batches(int(kind.split(":")[1]))[:count]
    if kind.startswith("grouped_half:"):
        from grouped_half_fixtures import batches as grouped_batches

        return grouped_batches(*map(int, kind.split(":")[1:]), count=count)
    if kind.startswith("half_gemm:"):
        from half_fixtures import batches as half_batches

        return half_batches(*map(int, kind.split(":")[1:]), count=count)
    if kind.startswith("mesh_power:"):
        from power_fixtures import batches as power_batches

        return power_batches(count)
    if kind.startswith("mesh_bicgstab:"):
        from bicgstab_fixtures import batches as bicg_batches

        return bicg_batches(count=count)
    if kind.startswith("mesh_pcg:"):
        from pcg_fixtures import batches as pcg_batches

        return pcg_batches(count=count)
    if kind.startswith("mesh_cg:"):
        from solver_fixtures import batches as cg_batches

        return cg_batches(int(kind.split(":")[1]), count)
    result = []
    for epoch in range(count):
        rng = np.random.default_rng(911 + epoch)

        def vec(n):
            return rng.uniform(-1, 1, n).astype(np.float32).tolist()

        a = rng.uniform(-0.5, 0.5, (4, 4))
        a = a @ a.T + np.diag(np.arange(4) + 2 + epoch * 0.25)
        if kind.startswith("mesh_reduction:"):
            _, op, nn = kind.split(":")
            nn = int(nn)
            xx = rng.uniform(-1, 1, nn).astype(np.float32)
            yy = xx.copy()
            if epoch % 4 == 1:
                xx[:] = 0
                yy[:] = 1
            elif epoch % 4 == 2:
                xx = rng.integers(-16, 17, nn).astype(np.float32) / 16
                yy = rng.integers(-16, 17, nn).astype(np.float32) / 16
            elif epoch % 4 == 3:
                if op == "nrm2":
                    # Squaring these f32 values directly rounds every term to zero.
                    xx *= np.float32(1.0e-30)
                else:
                    yy = rng.uniform(-1, 1, nn).astype(np.float32)
            data = {"x": xx.tolist()}
            if op == "dot":
                data["y"] = yy.tolist()
        elif kind.startswith("mesh_spmv:"):
            m, n, nnz = map(int, kind.split(":")[1:])
            keys = {(m - 1, n - 1)}
            while len(keys) < nnz:
                row, col = int(rng.integers(m)), int(rng.integers(n))
                # Deliberately empty row/column and a whole leading partition.
                if row != 2 and col != 1 and not (row < m // 4 and col < n // 4):
                    keys.add((row, col))
            keys = sorted(keys, key=lambda p: (p[1], p[0]))
            counts = [0] * n
            for row, col in keys:
                counts[col] += 1
            ptr = [0]
            for count in counts:
                ptr.append(ptr[-1] + count)
            vals = rng.uniform(-1, 1, nnz).astype(np.float32)
            vals[::17] = 0
            if epoch % 4 == 1:
                vals[:] = 0
            xx = rng.uniform(-1, 1, n).astype(np.float32)
            data = dict(
                values=vals.tolist(),
                row_indices=[r for r, c in keys],
                column_offsets=ptr,
                x=xx.tolist(),
            )
        elif kind.startswith("mesh_qr:"):
            m, n = map(int, kind.split(":")[1:])
            a = rng.uniform(-0.2, 0.2, (m, n)).astype(np.float32)
            a[np.arange(n), np.arange(n)] += 3
            if epoch % 4 == 1:
                a = np.zeros((m, n), np.float32)
                a[np.arange(n), np.arange(n)] = (1 + np.arange(n) % 4) * np.where(
                    np.arange(n) % 2, -1, 1
                )
            elif epoch % 4 == 2:
                a *= np.float32(0.125)
            elif epoch % 4 == 3:
                a = a[:, ::-1].copy()
            data = {"a": a.ravel().tolist()}
        elif kind.startswith("mesh_lu:"):
            n = int(kind.split(":")[1])
            a = rng.uniform(-0.2, 0.2, (n, n)).astype(np.float32)
            np.fill_diagonal(a, 0)
            np.fill_diagonal(a, np.sum(np.abs(a), axis=1) + 2 + epoch * 0.25)
            if epoch % 4 == 1:
                a = np.diag((1 + np.arange(n) % 4).astype(np.float32))
            elif epoch % 4 == 2:
                a *= np.float32(0.125)
            elif epoch % 4 == 3:
                a = np.eye(n, dtype=np.float32) * 2
                a += np.diag(np.full(n - 1, -0.25, np.float32), 1)
                a += np.diag(np.full(n - 1, 0.5, np.float32), -1)
            data = {"a": a.ravel().tolist()}
        elif kind.startswith("mesh_cholesky:"):
            n = int(kind.split(":")[1])
            q = rng.uniform(-0.2, 0.2, (n, n))
            a = (q @ q.T + np.diag(np.linspace(2, 4, n))).astype(np.float32)
            if epoch % 4 == 1:
                a = np.diag((1 + np.arange(n) % 4).astype(np.float32) ** 2)
            elif epoch % 4 == 2:
                a *= np.float32(0.25)
            elif epoch % 4 == 3:
                a = np.eye(n, dtype=np.float32) * 2
                a += np.diag(np.full(n - 1, -0.5, np.float32), 1)
                a += np.diag(np.full(n - 1, -0.5, np.float32), -1)
            data = {"a": a.ravel().tolist()}
        elif kind.startswith("mesh_gemm:"):
            rows, k, cols = map(int, kind.split(":")[1:])
            data = {"a": vec(rows * k), "b": vec(k * cols)}
            if epoch % 4 == 1:
                matrix = np.zeros((rows, k), np.float32)
                matrix[np.arange(rows), np.arange(rows) % k] = 1
                coded = np.arange(k * cols, dtype=np.float32).reshape(k, cols) / (
                    2.0 ** int(np.ceil(np.log2(k * cols)))
                )
                data = {"a": matrix.ravel().tolist(), "b": coded.ravel().tolist()}
            elif epoch % 4 == 2:
                data["a"] = [0.0] * (rows * k)
            elif epoch % 4 == 3:
                matrix = np.zeros((rows, k), np.float32)
                matrix[:, 0] = 1.0 + (np.arange(rows) % 2) * 2.0**-23
                matrix[:, 1] = 2.0**-24
                rhs = np.zeros((k, cols), np.float32)
                rhs[:2, :] = 1
                data = {"a": matrix.ravel().tolist(), "b": rhs.ravel().tolist()}
        elif kind.startswith("mesh_gemv:"):
            rows, cols = map(int, kind.split(":")[1:])
            data = {"a": vec(rows * cols), "x": vec(cols)}
        elif kind.startswith("grid:"):
            x, y, z, steps = map(int, kind.split(":")[1:])
            data = {
                "field": vec(x * y * z),
                "coeff": [0.04, 0.07, 0.03, 0.05, 0.02, 0.06, 0.7],
            }
        elif kind == "stencil_grid":
            data = {"field": vec(16), "coeff": vec(7)}
        elif kind.startswith("fft:"):
            dim = int(kind.split(":")[1])
            data = {"x": vec(2 * 4**dim), "twiddle": [1.0, 0.0, 0.0, -1.0]}
        elif kind.startswith("mlir:"):
            block = json.loads(
                (
                    ROOT
                    / "projects/wse_stencil"
                    / kind.split(":")[1]
                    / "source-expression.json"
                ).read_text()
            )
            data = {"samples": vec(len(block["slots"]))}
        elif kind in ("cg", "power", "bicgstab"):
            a = (
                np.diag(np.arange(4) + 2 + epoch * 0.25)
                + np.diag([-0.2, -0.3, -0.1], 1)
                + np.diag(
                    [-0.2, -0.3, -0.1] if kind != "bicgstab" else [-0.1, -0.4, -0.2], -1
                )
            )
            data = {"a": a.astype(np.float32).ravel().tolist(), "b": vec(4)}
        elif kind == "mc_xs":
            table = [0.0, 1.0, 2.0, 3.0, 0.0, 0.7, 1.7, 3.0] + vec(40) + [0.3, 0.7]
            particles = np.array(vec(24)).reshape(4, 6)
            particles[:, 0] = [0.25, 1.2, 1.9, 2.8]
            data = {"table": table, "particles": particles.ravel().tolist()}
        elif kind == "flip":
            q = rng.uniform(-1, 1, (4, 4))
            q = (q + q.T) / 2
            data = {
                "q": q.astype(np.float32).ravel().tolist(),
                "spins": [float((epoch >> i) & 1) for i in range(4)],
            }
        elif kind == "accept":
            data = {
                "values": [-1.0, 1.0, 0.9, 0.0, 1.0, 0.3, 1.0, 1.0, 0.1, 1.0, 1.0, 0.9]
            }
        elif kind in ("cholesky", "lu", "qr"):
            data = {"a": a.astype(np.float32).ravel().tolist()}
        elif kind in ("gemm", "gemv"):
            data = {"a": vec(16), "b": vec(16 if kind == "gemm" else 4)}
        elif kind == "axpy":
            data = {"x": vec(4), "y": vec(4)}
        elif kind == "residual":
            data = {"a": vec(16), "x": vec(4), "b": vec(4)}
        elif kind in ("collective", "allreduce"):
            data = {"p" + str(i): vec(4) for i in range(4)}
        elif kind == "broadcast":
            data = {"x": vec(4)}
        elif kind == "rmsnorm":
            data = {"x": vec(8), "w": vec(4)}
        elif kind == "softmax":
            data = {"x": vec(8)}
        elif kind == "silu":
            data = {"up": vec(8), "gate": vec(8)}
        elif kind == "rope":
            data = {"x": vec(8), "coeff": [0.8, 0.6, 0.0, 1.0]}
        else:
            raise ValueError(kind)
        result.append(data)
    return result


def check_application(kind, batch, output):
    if kind.startswith("composed_ffn:"):
        from composed_ffn_fixtures import check

        return check(*map(int, kind.split(":")[1:]), batch, output)
    if kind.startswith("projected_cache:"):
        from projected_cache_fixtures import original, metric

        expected = original(*map(int, kind.split(":")[1:]), batch)
        assert set(output) == {"result", "new_key", "new_value"}
        return dict(
            contract="projected-cache-half-normwise-v1",
            fixed_accuracy_passed=True,
            all_stage_gates=False,
            metrics={
                name: metric(output[port], expected[name])
                for port, name in (
                    ("result", "result"),
                    ("new_key", "rotated_key"),
                    ("new_value", "value_projection"),
                )
            },
            scope="Only public outputs; mandatory native/device stage observations follow before qualification",
        )
    if kind.startswith("cache_attention:"):
        from cache_attention_fixtures import check as cache_check

        return cache_check(*map(int, kind.split(":")[1:]), batch, output)
    if kind.startswith("batched_ffn:"):
        from batched_ffn_fixtures import check as ffn_check

        return ffn_check(*map(int, kind.split(":")[1:]), batch, output)
    if kind.startswith("batched_fanout:"):
        from batched_fanout_fixtures import check as fanout_check

        return fanout_check(*map(int, kind.split(":")[1:]), batch, output)
    if kind.startswith("batched_rms:"):
        from batched_rms_fixtures import check as rms_check

        return rms_check(*map(int, kind.split(":")[1:]), batch, output)
    if kind.startswith("input_attention_mixed:"):
        import math
        from input_attention_fixtures import child_inputs
        from attention_tail_fixtures import check as tail_check

        _, m, n, f, p = kind.split(":")
        _, child = child_inputs(int(m), int(n), 1e-6, batch)
        result = tail_check(
            int(m), int(n), int(f), 1e-6, 1 / math.sqrt(int(n)), child, output
        )
        result["contract"] = "shared-input-attention-final-only-v1"
        result["scope"] = (
            "Final output only; full qualification requires separately executed branch observations."
        )
        return result
    if kind.startswith("pair_rotation:"):
        from pair_rotation_fixtures import check as pair_check

        _, m, n, broadcast, order = kind.split(":")
        return pair_check(int(m), int(n), bool(int(broadcast)), order, batch, output)
    if kind.startswith("gated_silu:"):
        from swiglu_fixtures import check as gating_check

        return gating_check(*map(int, kind.split(":")[1:]), batch, output)
    if kind.startswith("attention_tail:"):
        import math
        from attention_tail_fixtures import check as attention_check

        _, m, n, f, p = kind.split(":")
        return attention_check(
            int(m), int(n), int(f), 1e-6, 1 / math.sqrt(int(n)), batch, output
        )
    if kind.startswith("prefill_tail:"):
        from prefill_tail_fixtures import check as tail_check

        _, m, n, f, p = kind.split(":")
        return tail_check(int(m), int(n), int(f), 1e-6, batch, output)
    if kind.startswith("feed_forward:"):
        from feed_forward_fixtures import check as ff_check

        _, m, n, f, p = kind.split(":")
        return ff_check(int(m), int(n), int(f), 1e-6, batch, output)
    if kind.startswith("projection_residual_rms:"):
        from projection_residual_rms_fixtures import check as composed_check

        _, m, n = kind.split(":")
        return composed_check(int(m), int(n), 1e-6, batch, output)
    if kind.startswith("mlp_blocked:"):
        from mlp_fixtures import check as mlp_check

        _, m, n, f, p = kind.split(":")
        return mlp_check(int(m), int(n), int(f), batch, output)
    if kind.startswith("mlp:"):
        from mlp_fixtures import check as mlp_check

        _, m, n, f = kind.split(":")
        return mlp_check(int(m), int(n), int(f), batch, output)
    if kind.startswith("attention:"):
        from attention_fixtures import check as attention_check

        _, m, n, scale = kind.split(":")
        return attention_check(int(m), int(n), float(scale), batch, output)
    if kind.startswith("score_softmax:"):
        from score_softmax_fixtures import check as resident_check

        _, m, n, scale = kind.split(":")
        return resident_check(int(m), int(n), float(scale), batch, output)
    if kind.startswith("device_matmul:"):
        from device_matmul_fixtures import check as device_check

        return device_check(*map(int, kind.split(":")[1:]), batch, output)
    if kind.startswith("score:"):
        from score_fixtures import check as score_check

        return score_check(*map(int, kind.split(":")[1:]), batch, output)
    if kind.startswith("normalized_fanout:"):
        from normalized_fanout_fixtures import check as fanout_check

        return fanout_check(*map(int, kind.split(":")[1:]), batch, output)
    if kind.startswith("normalized_matmul:"):
        from normalized_matmul_fixtures import check as resident_check

        return resident_check(*map(int, kind.split(":")[1:]), batch, output)
    if kind.startswith("distributed_softmax:"):
        from softmax_fixtures import check as softmax_check

        return softmax_check(*map(int, kind.split(":")[1:]), batch, output)
    if kind.startswith("distributed_rms:"):
        from rms_fixtures import check as rms_check

        return rms_check(*map(int, kind.split(":")[1:]), batch, output)
    if kind.startswith("distributed_fft:"):
        from fft_fixtures import check as fft_check

        _, n, direction, norm = kind.split(":")
        return fft_check(int(n), direction, norm, batch, output)
    if kind.startswith("grouped_half:"):
        from grouped_half_fixtures import check as grouped_check

        return grouped_check(batch, output, *map(int, kind.split(":")[1:]))
    if kind.startswith("half_gemm:"):
        from half_fixtures import check as half_check

        return half_check(batch, output, *map(int, kind.split(":")[1:]))
    if kind.startswith("mesh_power:"):
        from power_fixtures import check as power_check

        return power_check(batch, output)
    if kind.startswith("mesh_bicgstab:"):
        from bicgstab_fixtures import check as bicg_check

        return bicg_check(batch, output)
    if kind.startswith("mesh_pcg:"):
        from pcg_fixtures import check as pcg_check

        return pcg_check(batch, output)
    if kind.startswith("mesh_cg:"):
        from solver_fixtures import check as cg_check

        return cg_check(batch, output)

    def v(key):
        return np.asarray(batch[key], dtype=np.float64)

    def assert_close(actual, expected):
        if not np.allclose(actual, expected, rtol=3e-5, atol=3e-6):
            raise ValueError(
                "independent "
                + kind
                + " mismatch: "
                + str(np.max(np.abs(actual - expected)))
            )

    r = np.asarray(output.get("result", output.get("l", [])), dtype=np.float64)
    if kind.startswith("mesh_reduction:"):
        from toolchain.mesh_reduction_sdk import check_result

        op = kind.split(":")[1]
        return check_result(
            op,
            [batch[k] for k in (("x", "y") if op == "dot" else ("x",))],
            output["result"],
        )
    elif kind.startswith("mesh_spmv:"):
        from toolchain.mesh_spmv_sdk import check_result

        m, n, _ = map(int, kind.split(":")[1:])
        return check_result(
            m,
            n,
            batch["column_offsets"],
            batch["row_indices"],
            batch["values"],
            batch["x"],
            r,
        )
    elif kind.startswith("mesh_qr:"):
        from toolchain.mesh_qr_sdk import check_factor

        m, n = map(int, kind.split(":")[1:])
        return check_factor(v("a").reshape(m, n), r.reshape(m, n))
    elif kind.startswith("mesh_lu:"):
        from toolchain.mesh_lu_sdk import check_factor

        n = int(kind.split(":")[1])
        return check_factor(v("a").reshape(n, n), r.reshape(n, n))
    elif kind.startswith("mesh_cholesky:"):
        from toolchain.mesh_cholesky_sdk import check_factor

        n = int(kind.split(":")[1])
        return check_factor(v("a").reshape(n, n), r.reshape(n, n))
    elif kind.startswith("mesh_gemm:"):
        rows, k, cols = map(int, kind.split(":")[1:])
        from toolchain.roundoff import check_matrix_roundoff

        a, b = v("a").reshape(rows, k), v("b").reshape(k, cols)
        reference = a @ b
        actual = r.reshape(rows, cols)
        proof = check_matrix_roundoff(a, b, actual, reference)
        witnesses = []
        # Exact reset and round-to-nearest-even witnesses, in addition to the envelope.
        if not np.any(a):
            np.testing.assert_array_equal(actual, np.zeros_like(actual))
            witnesses.append("exact_zero_reset")
        if not np.any(a[:, 2:]) and np.all(b[:2, :] == 1) and not np.any(b[2:, :]):
            np.testing.assert_array_equal(actual, reference.astype(np.float32))
            witnesses.append("nearest_even_halfway")
        return {
            "contract": "componentwise-f32-dot-v1",
            "roundoff_passed": True,
            "fixed_accuracy_passed": proof["old_fixed_tolerance_passed"],
            "roundoff": proof,
            "exact_witnesses": witnesses,
        }
    elif kind.startswith("mesh_gemv:"):
        rows, cols = map(int, kind.split(":")[1:])
        assert_close(r, v("a").reshape(rows, cols) @ v("x"))
    elif kind.startswith("grid:"):
        x, y, z, steps = map(int, kind.split(":")[1:])
        field = v("field").reshape(x, y, z)
        c = v("coeff")
        for step in range(steps):
            p = np.pad(field, 1)
            field = (
                c[6] * field
                + c[0] * p[:-2, 1:-1, 1:-1]
                + c[1] * p[2:, 1:-1, 1:-1]
                + c[2] * p[1:-1, :-2, 1:-1]
                + c[3] * p[1:-1, 2:, 1:-1]
                + c[4] * p[1:-1, 1:-1, :-2]
                + c[5] * p[1:-1, 1:-1, 2:]
            )
        assert_close(r, field.ravel())
    elif kind == "stencil_grid":
        field = v("field").reshape(2, 2, 4)
        coeff = v("coeff")
        padded = np.pad(field, 1)
        expected = coeff[6] * field
        for weight, axis, shift in (
            (0, 0, -1),
            (1, 0, 1),
            (2, 1, -1),
            (3, 1, 1),
            (4, 2, -1),
            (5, 2, 1),
        ):
            slices = [slice(1, 3), slice(1, 3), slice(1, 5)]
            slices[axis] = slice(1 + shift, 1 + shift + field.shape[axis])
            expected = expected + coeff[weight] * padded[tuple(slices)]
        for t in range(4):
            assert_close(output["tile" + str(t)], expected.reshape(4, 4)[t])
    elif kind.startswith("fft:"):
        dim = int(kind.split(":")[1])
        pairs = v("x").reshape(-1, 2)
        z = (pairs[:, 0] + 1j * pairs[:, 1]).reshape((4,) * dim)
        fft = np.fft.fftn(z).ravel()
        expected = np.column_stack([fft.real, fft.imag]).ravel()
        assert_close(r, expected)
    elif kind.startswith("mlir:"):
        block = json.loads(
            (
                ROOT
                / "projects/wse_stencil"
                / kind.split(":")[1]
                / "source-expression.json"
            ).read_text()
        )
        assert_close(r, [mlir_reference(block, batch["samples"])])
    elif kind == "cholesky":
        assert_close(r.reshape(4, 4), np.linalg.cholesky(v("a").reshape(4, 4)))
    elif kind == "power":
        x = v("b")
        a = v("a").reshape(4, 4)
        for _ in range(8):
            x = a @ x
            x = x / np.linalg.norm(x)
        assert_close(r, x)
    elif kind in ("cg", "bicgstab"):
        assert_close(r, np.linalg.solve(v("a").reshape(4, 4), v("b")))
    elif kind == "mc_xs":
        original = v("particles").reshape(4, 6)
        out = original.copy()
        table = v("table")
        for n in range(2):
            for xs in range(5):
                out[:, xs + 1] += table[48 + n] * np.interp(
                    original[:, 0],
                    table[n * 4 : n * 4 + 4],
                    table[8 + n * 20 + xs : 8 + n * 20 + 20 : 5],
                )
        assert_close(r, out.ravel())
    elif kind == "flip":
        q = v("q").reshape(4, 4)
        spins = v("spins")

        def energy(s):
            return np.diag(q) @ s + sum(
                q[i, j] * s[i] * s[j] for i in range(4) for j in range(i + 1, 4)
            )

        expected = []
        for i in range(4):
            flipped = spins.copy()
            flipped[i] = 1 - flipped[i]
            expected.append(energy(flipped) - energy(spins))
        assert_close(r, expected)
    elif kind == "accept":
        data = v("values").reshape(4, 3)
        assert_close(
            r,
            ((data[:, 0] < 0) | (data[:, 2] < np.exp(-data[:, 0] / data[:, 1]))).astype(
                float
            ),
        )
    elif kind == "lu":
        lu = r.reshape(4, 4)
        l = np.tril(lu, -1) + np.eye(4)
        u = np.triu(lu)
        assert_close(l @ u, v("a").reshape(4, 4))
    elif kind == "qr":
        rr = r[:16].reshape(4, 4)
        qt = r[16:].reshape(4, 4)
        assert_close(qt @ v("a").reshape(4, 4), rr)
        assert_close(qt @ qt.T, np.eye(4))
        assert_close(np.tril(rr, -1), np.zeros((4, 4)))
    elif kind == "gemm":
        assert_close(r, (v("a").reshape(4, 4) @ v("b").reshape(4, 4)).ravel())
    elif kind == "gemv":
        assert_close(r, v("a").reshape(4, 4) @ v("b"))
    elif kind == "axpy":
        assert_close(r, 2.5 * v("x") + v("y"))
    elif kind == "residual":
        assert_close(r, v("b") - v("a").reshape(4, 4) @ v("x"))
    elif kind == "collective":
        assert_close(r, sum(v("p" + str(i)) for i in range(4)))
    elif kind == "allreduce":
        for value in output.values():
            assert_close(value, sum(v("p" + str(i)) for i in range(4)))
    elif kind == "broadcast":
        for value in output.values():
            assert_close(value, v("x"))
    elif kind == "rmsnorm":
        x = v("x").reshape(2, 4)
        assert_close(
            r,
            (
                x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6) * v("w")
            ).ravel(),
        )
    elif kind == "softmax":
        x = v("x").reshape(2, 4)
        e = np.exp(x - x.max(axis=1, keepdims=True))
        assert_close(r, (e / e.sum(axis=1, keepdims=True)).ravel())
    elif kind == "silu":
        assert_close(r, v("up") * v("gate") / (1 + np.exp(-v("gate"))))
    elif kind == "rope":
        x = v("x").reshape(2, 2, 2)
        c = v("coeff").reshape(2, 2)
        o = np.empty_like(x)
        o[:, :, 0] = x[:, :, 0] * c[:, 0] - x[:, :, 1] * c[:, 1]
        o[:, :, 1] = x[:, :, 1] * c[:, 0] + x[:, :, 0] * c[:, 1]
        assert_close(r, o.ravel())
    else:
        raise ValueError(kind)
    return True
