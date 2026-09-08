"""Generic typed lambda-body IR, interpreter and CSL lowering; no algorithm registry."""

import math, re
from float32 import f32
from frontend import check, Error


def parse_kernel(lam, source, outshape):
    check(lam["kind"] == "LambdaExpr", "kernel requires lambda")
    check(
        re.match(r"\[\s*\]", source[lam["range"]["begin"]["offset"] :]),
        "kernel captures unsupported",
    )
    method = next(
        x
        for x in lam["inner"][0]["inner"]
        if x["kind"] == "CXXMethodDecl" and x.get("name") == "operator()"
    )
    params = [x for x in method["inner"] if x["kind"] == "ParmVarDecl"]
    names = {}
    arrays = {}
    symbols = {}
    counter = 0

    def bind(n):
        nonlocal counter
        name = "v" + str(counter)
        counter += 1
        names[n["id"]] = name
        symbols[name] = {
            "name": n.get("name"),
            "line": source.count("\n", 0, n["range"]["begin"]["offset"]) + 1,
        }
        return name

    def tensor_shape(n):
        m = re.search(r"tensor<(\d+),\s*(\d+)>", n["type"]["qualType"])
        check(m, "tensor type")
        return [int(x) for x in m.groups()]

    for i, p in enumerate(params):
        check(
            "const " in p["type"]["qualType"] and "&" in p["type"]["qualType"],
            "kernel input must be const reference",
        )
        names[p["id"]] = "a" if i == 0 else "b"
        arrays[names[p["id"]]] = tensor_shape(p)

    def typ(n):
        t = n.get("type", {}).get("qualType", "").removeprefix("const ")
        if "tensor<" in t:
            return "tensor"
        check(
            t in ("float", "bool", "int", "void"), "unsupported kernel scalar type " + t
        )
        return t

    def e(n):
        kind = n["kind"]
        children = n.get("inner", [])
        if kind in (
            "ImplicitCastExpr",
            "ParenExpr",
            "CXXConstructExpr",
            "MaterializeTemporaryExpr",
        ):
            check(len(children) == 1, "unsupported expression wrapper")
            if kind == "ImplicitCastExpr":
                cast = n.get("castKind")
                check(
                    cast in ("LValueToRValue", "NoOp", "IntegralToFloating"),
                    "unsupported kernel cast " + str(cast),
                )
                typ(n)
                if cast == "IntegralToFloating":
                    return ["cast", "float", e(children[0])]
            return e(children[0])
        if kind == "DeclRefExpr":
            key = n["referencedDecl"]["id"]
            check(key in names, "free kernel variable")
            return ["var", names[key], typ(n)]
        if kind in ("IntegerLiteral", "FloatingLiteral", "CXXBoolLiteralExpr"):
            return ["const", n["value"], typ(n)]
        if kind == "ArraySubscriptExpr":
            base = children[0]
            while base["kind"] == "ImplicitCastExpr":
                base = base["inner"][0]
            check(
                base["kind"] == "MemberExpr" and base["name"] == "data",
                "only tensor.data indexing",
            )
            tensor = e(base["inner"][0])
            check(tensor[0] == "var" and tensor[1] in arrays, "array base")
            return ["index", tensor[1], e(children[1]), "float"]
        if kind in ("BinaryOperator", "CompoundAssignOperator"):
            op = n["opcode"]
            check(
                op
                in ("+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!=", "&&", "||"),
                "unsupported expression " + op,
            )
            return ["binary", op, e(children[0]), e(children[1]), typ(n)]
        if kind == "UnaryOperator":
            check(n["opcode"] in ("-", "!", "+"), "unsupported unary")
            return ["unary", n["opcode"], e(children[0]), typ(n)]
        if kind == "CallExpr":
            callee = children[0]
            while callee["kind"] == "ImplicitCastExpr":
                callee = callee["inner"][0]
            name = callee.get("referencedDecl", {}).get("name")
            check(
                name in ("sqrt", "exp", "abs", "require"), "unsupported kernel function"
            )
            check(
                source[n["range"]["begin"]["offset"] :].startswith("spatial::" + name),
                "qualified intrinsic required",
            )
            check(len(children) == 2, "intrinsic arity")
            return ["call", name, e(children[1]), typ(n)]
        raise Error("unsupported kernel expression " + kind)

    def stmt(n):
        k = n["kind"]
        ch = n.get("inner", [])
        if k == "CompoundStmt":
            return ["block", [stmt(x) for x in ch]]
        if k == "DeclStmt":
            check(len(ch) == 1, "one local declaration")
            v = ch[0]
            name = bind(v)
            init = v.get("inner", [])
            if "tensor<" in v["type"]["qualType"]:
                arrays[name] = tensor_shape(v)
                check(math.prod(arrays[name]) <= 256, "local storage extent")
                zero = not init or init[0]["kind"] == "InitListExpr"
                if zero:

                    def nonzero(n):
                        return n.get("kind") not in (
                            "InitListExpr",
                            "ImplicitValueInitExpr",
                            "CXXDefaultInitExpr",
                        ) or any(nonzero(x) for x in n.get("inner", []))

                    check(
                        not init or not nonzero(init[0]),
                        "tensor initializer must be empty",
                    )
                return ["array", name, None if zero else e(init[0])]
            check(
                v["type"]["qualType"] in ("int", "float", "bool") and len(init) == 1,
                "typed initialized scalar required",
            )
            return ["decl", name, typ(v), e(init[0])]
        if k == "ForStmt":
            parts = [x for x in ch if x.get("kind")]
            check(len(parts) == 4, "canonical for required")
            start, condition, increment, body = parts
            init = stmt(start)
            check(init[0] == "decl" and init[2] == "int", "int induction variable")
            check(
                increment["kind"] == "UnaryOperator" and increment["opcode"] == "++",
                "only incrementing for supported",
            )
            inc = e(increment["inner"][0])
            check(inc[1] == init[1], "loop induction mismatch")
            return ["for", init, e(condition), stmt(body)]
        if k == "IfStmt":
            check(len(ch) in (2, 3), "simple if")
            return [
                "if",
                e(ch[0]),
                stmt(ch[1]),
                stmt(ch[2]) if len(ch) == 3 else ["block", []],
            ]
        if k in ("BinaryOperator", "CompoundAssignOperator"):
            check(n["opcode"] in ("=", "+=", "-=", "*=", "/="), "assignment required")
            left = e(ch[0])
            check(left[0] in ("var", "index"), "assignment target")
            check(left[1] not in ("a", "b"), "input mutation")
            return ["assign", left, n["opcode"], e(ch[1])]
        if k == "CallExpr":
            return ["eval", e(n)]
        if k == "ReturnStmt":
            return ["return", e(ch[0])]
        raise Error("unsupported kernel statement " + k)

    body = stmt(next(x for x in lam["inner"] if x["kind"] == "CompoundStmt"))
    check(body[1] and body[1][-1][0] == "return", "final tensor return required")

    def count_returns(s):
        if not isinstance(s, list):
            return 0
        return int(bool(s) and s[0] == "return") + sum(
            count_returns(x) for x in s if isinstance(x, list)
        )

    check(count_returns(body) == 1, "early/nested returns unsupported")
    result = body[1][-1][1]
    check(result[0] == "var" and arrays.get(result[1]) == outshape, "return shape")
    return {
        "symbols": symbols,
        "arrays": arrays,
        "params": ["a", "b"][: len(params)],
        "body": body,
        "result": result[1],
        "output_shape": outshape,
    }


def evaluate_kernel(kernel, inputs, trace=None):
    env = {name: list(value) for name, value in zip(kernel["params"], inputs)}

    def ev(e):
        op = e[0]
        if op == "var":
            return env[e[1]]
        if op == "const":
            return f32(float(e[1])) if e[2] == "float" else int(e[1])
        if op == "cast":
            return f32(ev(e[2]))
        if op == "index":
            i = int(ev(e[2]))
            check(0 <= i < len(env[e[1]]), "kernel array bounds")
            return env[e[1]][i]
        if op == "unary":
            v = ev(e[2])
            return {"-": lambda: -v, "+": lambda: v, "!": lambda: not v}[e[1]]()
        if op == "call":
            v = ev(e[2])
            name = e[1]
            if name == "require":
                check(v, "kernel precondition")
                return 0
            return f32({"sqrt": math.sqrt, "exp": math.exp, "abs": abs}[name](v))
        if op == "binary":
            a = ev(e[2])
            operator = e[1]
            if operator == "&&":
                return bool(a) and bool(ev(e[3]))
            if operator == "||":
                return bool(a) or bool(ev(e[3]))
            b = ev(e[3])
            v = {
                "+": lambda: a + b,
                "-": lambda: a - b,
                "*": lambda: a * b,
                "/": lambda: a / b,
                "<": lambda: a < b,
                "<=": lambda: a <= b,
                ">": lambda: a > b,
                ">=": lambda: a >= b,
                "==": lambda: a == b,
                "!=": lambda: a != b,
            }[operator]()
            return f32(v) if e[4] == "float" else int(v)
        raise Error("bad body IR")

    def setvalue(left, v):
        if left[0] == "index":
            i = int(ev(left[2]))
            check(0 <= i < len(env[left[1]]), "kernel write bounds")
            env[left[1]][i] = f32(v)
        else:
            env[left[1]] = f32(v) if left[2] == "float" else int(v)

    def run(s):
        if trace is not None and len(trace) < 10000:
            trace.append(
                {
                    "statement": s,
                    "state": {
                        k: (list(v) if isinstance(v, list) else v)
                        for k, v in env.items()
                    },
                }
            )
        op = s[0]
        if op == "block":
            for x in s[1]:
                run(x)
        elif op == "array":
            env[s[1]] = (
                [0.0] * math.prod(kernel["arrays"][s[1]])
                if s[2] is None
                else list(ev(s[2]))
            )
        elif op == "decl":
            env[s[1]] = ev(s[3])
        elif op == "assign":
            value = ev(s[3])
            if s[2] != "=":
                a = ev(s[1])
                value = {
                    "+=": lambda: a + value,
                    "-=": lambda: a - value,
                    "*=": lambda: a * value,
                    "/=": lambda: a / value,
                }[s[2]]()
            setvalue(s[1], value)
        elif op == "for":
            run(s[1])
            count = 0
            while ev(s[2]):
                check(count < 256, "kernel loop bound")
                run(s[3])
                env[s[1][1]] += 1
                count += 1
        elif op == "if":
            run(s[2] if ev(s[1]) else s[3])
        elif op == "eval":
            ev(s[1])
        elif op == "return":
            pass

    run(kernel["body"])
    return list(env[kernel["result"]])


def emit_kernel(kernel):
    def name(n):
        return n if n in ("a", "b") else "lk_" + n

    def ex(e):
        op = e[0]
        if op == "var":
            return name(e[1])
        if op == "const":
            return (
                str(float(e[1]))
                if e[2] == "float"
                else ("true" if e[1] else "false") if e[2] == "bool" else str(int(e[1]))
            )
        if op == "cast":
            return "@as(f32," + ex(e[2]) + ")"
        if op == "index":
            return "get_" + name(e[1]) + "(" + ex(e[2]) + ")"
        if op == "unary":
            return "(" + {"+": ""}.get(e[1], e[1]) + ex(e[2]) + ")"
        if op == "binary":
            return (
                "("
                + ex(e[2])
                + {"&&": " and ", "||": " or "}.get(e[1], e[1])
                + ex(e[3])
                + ")"
            )
        if op == "call":
            return (
                ("@assert" if e[1] == "require" else "math." + e[1])
                + "("
                + ex(e[2])
                + ")"
            )
        raise Error("bad expression IR")

    def st(s):
        op = s[0]
        if op == "block":
            return "{\n" + "\n".join(st(x) for x in s[1]) + "\n}"
        if op == "array":
            size = math.prod(kernel["arrays"][s[1]])
            rhs = "0.0" if s[2] is None else ex(s[2]) + "[i]"
            return f"for (@range(u16,{size})) |i| {{ {name(s[1])}[i]={rhs}; }}"
        if op == "decl":
            return (
                "var "
                + name(s[1])
                + ":"
                + {"float": "f32", "int": "i32", "bool": "bool"}[s[2]]
                + "="
                + ex(s[3])
                + ";"
            )
        if op == "assign":
            l, assign, r = s[1:]
            value = ex(r) if assign == "=" else "(" + ex(l) + assign[0] + ex(r) + ")"
            if l[0] == "index":
                return "set_" + name(l[1]) + "(" + ex(l[2]) + "," + value + ");"
            return ex(l) + "=" + value + ";"
        if op == "if":
            return "if(" + ex(s[1]) + "){" + st(s[2]) + "}else{" + st(s[3]) + "}"
        if op == "for":
            v = name(s[1][1])
            return (
                "{"
                + st(s[1])
                + f"var guard_{v}:u16=0;while("
                + ex(s[2])
                + f"){{@assert(guard_{v}<256);guard_{v}+=1;"
                + st(s[3])
                + v
                + "+=1;}}"
            )
        if op == "eval":
            return ex(s[1]) + ";"
        if op == "return":
            return f'for (@range(u16,{math.prod(kernel["output_shape"])})) |i| {{ result[i]={ex(s[1])}[i]; @assert(result[i]==result[i] and math.abs(result[i])<3.4028234e38); }}'
        raise Error("bad statement IR")

    extra = []
    for n, shape in kernel["arrays"].items():
        size = math.prod(shape)
        v = name(n)
        if n not in ("a", "b"):
            extra.append(f"export var {v}=@zeros([{size}]f32);")
        extra.append(
            f"fn get_{v}(i:i32) f32 {{ @assert(i>=0 and i<{size});return {v}[@as(u16,i)]; }}"
        )
        if n not in ("a", "b"):
            extra.append(
                f"fn set_{v}(i:i32,value:f32) void {{ @assert(i>=0 and i<{size});{v}[@as(u16,i)]=value; }}"
            )
    return st(kernel["body"]), "\n".join(extra)
