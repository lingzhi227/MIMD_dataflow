"""Clang AST adapter for the explicit middleware tensor profile; fail closed."""

import hashlib, json, re, subprocess
from pathlib import Path

if __package__:
    from .pragma_contracts import parse as parse_pragma, INTEGER_ATTRIBUTES
else:
    from pragma_contracts import parse as parse_pragma, INTEGER_ATTRIBUTES

ROOT = Path(__file__).resolve().parent


class Error(ValueError):
    pass


def check(ok, msg):
    if not ok:
        raise Error(msg)


def unwrap(n):
    while n["kind"] in (
        "ImplicitCastExpr",
        "ParenExpr",
        "ExprWithCleanups",
        "MaterializeTemporaryExpr",
    ):
        check(len(n.get("inner", [])) == 1, "invalid wrapper")
        n = n["inner"][0]
    return n


def shape(n, kind="tensor"):
    t = n.get("type", {})
    v = t.get("desugaredQualType", t.get("qualType", ""))
    scalar = (
        r"(?:,\s*(?:float|(?:spatial::)?f16|_Float16))?" if kind == "tensor" else ""
    )
    m = re.fullmatch(r"(?:spatial::)?" + kind + r"<(\d+),\s*(\d+)" + scalar + r">", v)
    check(m is not None, "unsupported type " + v)
    return list(map(int, m.groups()))


def tensor_dtype(n):
    t = n.get("type", {})
    v = t.get("desugaredQualType", t.get("qualType", ""))
    return "f16" if re.search(r",\s*(?:(?:spatial::)?f16|_Float16)>$", v) else "f32"


def ref(n):
    n = unwrap(n)
    if n["kind"] == "MemberExpr":
        check(
            len(n.get("inner", [])) == 1 and not n.get("isArrow", False),
            "direct record field required",
        )
        return ref(n["inner"][0]) + "." + n["name"]
    check(n["kind"] == "DeclRefExpr", "expected named object")
    return n["referencedDecl"]["name"]


def string(n):
    n = unwrap(n)
    check(n["kind"] == "StringLiteral", "expected literal port name")
    s = json.loads(n["value"])
    check(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", s), "invalid port")
    return s


def expression(n, param):
    check(
        n.get("type", {}).get("qualType") in ("float", "int"),
        "map arithmetic requires float",
    )
    n = unwrap(n)
    k = n["kind"]
    if k in ("IntegerLiteral", "FloatingLiteral"):
        return ["const", float(n["value"])]
    if k == "DeclRefExpr":
        check(ref(n) == param, "lambda capture unsupported")
        return ["var", "x"]
    if k == "UnaryOperator":
        check(n["opcode"] == "-", "unsupported unary")
        return ["neg", expression(n["inner"][0], param)]
    if k == "BinaryOperator":
        check(n["opcode"] in ("+", "-", "*"), "unsupported arithmetic")
        return [n["opcode"]] + [expression(c, param) for c in n["inner"]]
    raise Error("unsupported expression " + k)


def parse(path, artifact_dir=None):
    path = Path(path).resolve()
    source = path.read_text()
    command = [
        "clang++",
        "-std=c++17",
        "-Werror",
        "-Wno-unknown-pragmas",
        "-I",
        str(ROOT / "include"),
        "-Xclang",
        "-ast-dump=json",
        "-Xclang",
        "-ast-dump-filter=design",
        "-fsyntax-only",
        str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if artifact_dir is not None:
        dest = Path(artifact_dir)
        (dest / "00_clang_ast.json").write_text(result.stdout)
        (dest / "00_clang_diagnostics.txt").write_text(result.stderr)
        (dest / "00_frontend_command.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
    check(result.returncode == 0, "C++ frontend failed:\n" + result.stderr)
    try:
        ast = json.loads(result.stdout)
    except (ValueError, TypeError) as e:
        raise Error("exactly one design() definition required") from e
    check(
        ast["kind"] == "FunctionDecl"
        and ast.get("name") == "design"
        and ast["type"]["qualType"] == "void ()",
        "expected void design()",
    )
    # No helpers, macros, namespaces or extra entrypoints can escape the profile.
    begin = ast["range"]["begin"]["offset"]
    end = ast["range"]["end"]["offset"] + ast["range"]["end"]["tokLen"]
    exterior = source[:begin] + source[end:]
    exterior = re.sub(r"//[^\n]*|/\*.*?\*/", "", exterior, flags=re.S)
    check(
        re.fullmatch(r'\s*#include\s+"spatial.hpp"\s*', exterior),
        "only spatial.hpp include and design allowed",
    )
    body = [n for n in ast["inner"] if n["kind"] == "CompoundStmt"]
    check(len(body) == 1, "missing body")
    pragmas = {}
    for number, line in enumerate(source.splitlines(), 1):
        if re.fullmatch(
            r'\s*#include\s+"spatial.hpp"\s*', line
        ) and number <= source.count("\n", 0, begin):
            continue
        if line.lstrip().startswith("#"):
            try:
                pragmas[number] = parse_pragma(line)
            except ValueError as e:
                raise Error(f"unsupported pragma at line {number}: {e}") from e
    used = set()
    nodes = []
    states = {}
    names = set()

    def lineof(n):
        return source.count("\n", 0, n["range"]["begin"]["offset"]) + 1

    for stmt in body[0]["inner"]:
        line = lineof(stmt)
        prev = line - 1
        while prev > 0 and not source.splitlines()[prev - 1].strip():
            prev -= 1
        pragma = pragmas.get(prev)
        if pragma:
            used.add(prev)
        if stmt["kind"] == "DeclStmt":
            check(len(stmt["inner"]) == 1, "one declaration per statement required")
            decl = stmt["inner"][0]
            name = decl["name"]
            check(
                not name.startswith("__mw_") and name != "comm_runtime",
                "reserved identifier",
            )
            check(name not in names, "duplicate object")
            names.add(name)
            if "state<" in decl["type"]["qualType"]:
                check(
                    pragma == "resident" and decl.get("storageClass") == "static",
                    "state requires static and resident annotation",
                )
                check(
                    len(decl.get("inner", [])) == 1
                    and decl["inner"][0]["kind"] == "CXXConstructExpr"
                    and not decl["inner"][0].get("inner"),
                    "state initialization must be zero",
                )
                states[name] = shape(decl, "state")
                continue
            check(
                pragma != "resident" and "storageClass" not in decl,
                "only state may be resident/static",
            )
            check(len(decl.get("inner", [])) == 1, "initializer required")
            call = unwrap(decl["inner"][0])
            outshape = shape(
                decl,
                (
                    "power_result"
                    if "power_result<" in decl["type"]["qualType"]
                    else (
                        "solver_result"
                        if "solver_result<" in decl["type"]["qualType"]
                        else (
                            "index_tensor"
                            if "index_tensor<" in decl["type"]["qualType"]
                            else "tensor"
                        )
                    )
                ),
            )
        else:
            check(
                stmt["kind"] == "CallExpr" and pragma is None,
                "unsupported statement " + stmt["kind"],
            )
            call = stmt
            name = "sink" + str(len(nodes))
            outshape = None
        check(call["kind"] == "CallExpr", "expected library call")
        callee = unwrap(call["inner"][0])
        op = ref(callee)
        # Qualified source call is required; exterior restriction excludes user overrides.
        offset = call["range"]["begin"]["offset"]
        check(
            source[offset:].startswith("spatial::" + op),
            "qualified spatial call required",
        )
        args = call["inner"][1:]
        node = {"id": name, "op": op, "shape": outshape, "inputs": [], "line": line}
        if outshape is not None and tensor_dtype(decl) == "f16":
            node["dtype"] = "f16"
        expected = {
            "input": 1,
            "index_input": 1,
            "spmv_csc": 4,
            "cg_csc": 7,
            "pcg_csc": 7,
            "bicgstab_csc": 7,
            "power_csc": 5,
            "dot": 2,
            "nrm2": 1,
            "output": 2,
            "map": 2,
            "add": 2,
            "matmul": 2,
            "cholesky": 1,
            "lu_no_pivot": 1,
            "qr_r": 1,
            "fft3d": 1,
            "rmsnorm": 3,
            "softmax": 2,
            "rotate_pairs": 3,
            "silu": 1,
            "multiply": 2,
            "transpose": 1,
            "row_sum": 1,
            "accumulate": 2,
            "kernel": len(args),
            "grid_iterate": 3,
        }
        check(op in expected and len(args) == expected[op], "unsupported call " + op)
        if op in ("cg_csc", "pcg_csc", "bicgstab_csc"):
            from solver_ir import result_type

            node["result_type"] = result_type(*outshape)
            node["shape"] = None
        if op == "power_csc":
            from power_ir import result_type

            node["result_type"] = result_type(*outshape)
            node["shape"] = None
        if op in ("input", "index_input"):
            node["host"] = string(args[0])
        elif op == "output":
            check(outshape is None, "output cannot initialize a tensor")
            node["host"] = string(args[0])
            node["inputs"] = [ref(args[1])]
        elif op == "softmax":
            scale = unwrap(args[1])
            check(
                scale["kind"] == "FloatingLiteral",
                "softmax scale requires a floating literal",
            )
            node["scale"] = float(scale["value"])
            node["inputs"] = [ref(args[0])]
        elif op == "rmsnorm":
            epsilon = unwrap(args[2])
            check(
                epsilon["kind"] == "FloatingLiteral",
                "RMS epsilon requires a floating literal",
            )
            node["epsilon"] = float(epsilon["value"])
            node["inputs"] = [ref(x) for x in args[:2]]
        elif op == "rotate_pairs":
            match = re.match(
                r"spatial::rotate_pairs<\s*spatial::pair_order::(even_odd|odd_even)\s*>",
                source[offset:],
            )
            check(match, "pair rotation explicit input-order template")
            node["pair_order"] = match[1]
            node["inputs"] = [ref(x) for x in args]
        elif op == "fft3d":
            match = re.match(
                r"spatial::fft3d<\s*(\d+)\s*,\s*spatial::fft_direction::(forward|inverse)\s*,\s*spatial::fft_norm::(backward|ortho|forward)\s*>",
                source[offset:],
            )
            check(match, "FFT explicit dimension/direction/normalization template")
            node["fft"] = dict(N=int(match[1]), direction=match[2], norm=match[3])
            node["inputs"] = [ref(args[0])]
        elif op == "grid_iterate":
            from local_kernel import parse_kernel

            match = re.match(
                r"spatial::grid_iterate<\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>",
                source[offset:],
            )
            check(match, "grid dimensions require literal template arguments")
            dims = list(map(int, match.groups()))
            node["grid"] = dict(zip(("x", "y", "z", "steps"), dims))
            node["inputs"] = [ref(x) for x in args[:2]]
            node["body"] = parse_kernel(args[2], source, [1, dims[2]])
        elif op == "kernel":
            from local_kernel import parse_kernel

            check(len(args) in (2, 3), "kernel arity")
            node["inputs"] = [ref(x) for x in args[:-1]]
            node["body"] = parse_kernel(args[-1], source, outshape)
        elif op == "map":
            node["inputs"] = [ref(args[0])]
            lam = unwrap(args[1])
            check(lam["kind"] == "LambdaExpr", "map requires lambda")
            offset = lam["range"]["begin"]["offset"]
            check(re.match(r"\[\s*\]", source[offset:]), "captures unsupported")
            methods = [
                x
                for x in lam["inner"][0]["inner"]
                if x["kind"] == "CXXMethodDecl" and x.get("name") == "operator()"
            ]
            check(len(methods) == 1, "lambda operator")
            params = [x for x in methods[0]["inner"] if x["kind"] == "ParmVarDecl"]
            check(
                len(params) == 1 and params[0]["type"]["qualType"] == "float",
                "one float parameter required",
            )
            blocks = [x for x in lam["inner"] if x["kind"] == "CompoundStmt"]
            check(
                len(blocks) == 1
                and len(blocks[0]["inner"]) == 1
                and blocks[0]["inner"][0]["kind"] == "ReturnStmt",
                "return-only lambda",
            )
            node["expr"] = expression(
                blocks[0]["inner"][0]["inner"][0], params[0]["name"]
            )
        elif op == "accumulate":
            node["state"] = ref(args[0])
            node["inputs"] = [ref(args[1])]
        else:
            node["inputs"] = [ref(x) for x in args]
        if pragma:
            if pragma.startswith("dataflow "):
                check(
                    op
                    in (
                        "matmul",
                        "cholesky",
                        "lu_no_pivot",
                        "qr_r",
                        "fft3d",
                        "rmsnorm",
                        "softmax",
                        "rotate_pairs",
                        "silu",
                        "multiply",
                        "spmv_csc",
                        "cg_csc",
                        "pcg_csc",
                        "bicgstab_csc",
                        "power_csc",
                        "dot",
                        "nrm2",
                    ),
                    "unsupported dataflow operation",
                )
                node["dataflow"] = dict(v.split("=") for v in pragma.split()[1:])
                for key in INTEGER_ATTRIBUTES:
                    if key in node["dataflow"]:
                        node["dataflow"][key] = int(node["dataflow"][key])
            elif pragma == "vectorize":
                check(
                    op == "grid_iterate", "vectorize currently requires grid iteration"
                )
                node["vectorize"] = True
            else:
                check(pragma.startswith("place"), "invalid annotation target")
                node["place"] = list(map(int, re.findall(r"\d+", pragma)))
        nodes.append(node)
    check(used == set(pragmas), "unattached pragma")
    return {
        "version": "hls.ports.f32.v1",
        "nodes": nodes,
        "states": states,
        "source": str(path),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "frontend_command": command,
        "clang_version": subprocess.check_output(
            ["clang++", "--version"], text=True
        ).splitlines()[0],
    }
