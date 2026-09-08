"""Sequential f32 reference for the typed real BiCGStab operator."""

import math
from solver_reference import f, product, norm, apply


def solve(a, b, x, limit, tol, capacity):
    x = x.copy()
    r = [f(v - w) for v, w in zip(b, apply(a, x))]
    shadow = r.copy()
    p = r.copy()
    rho = product(r, r)
    threshold = max(f(tol[0] * norm(b)), tol[1])
    history = [0.0] * (capacity + 1)
    history[0] = rho
    k = 0
    reason = 1
    if not math.isfinite(rho) or not math.isfinite(threshold):
        reason = 3
    elif norm(r) <= threshold:
        reason = 0
    elif rho <= 0:
        reason = 3
    else:
        for step in range(limit):
            v = apply(a, p)
            denom = product(shadow, v)
            if not math.isfinite(denom) or denom == 0:
                reason = 3
                break
            alpha = f(rho / denom)
            if not math.isfinite(alpha):
                reason = 3
                break
            s = [f(-alpha * w + z) for w, z in zip(v, r)]
            ss = product(s, s)
            if not math.isfinite(ss):
                reason = 3
                break
            snorm = norm(s) if ss == 0 else math.sqrt(ss)
            if ss == 0 and snorm > threshold:
                reason = 3
                break
            if snorm <= threshold:
                x = [f(alpha * w + z) for w, z in zip(p, x)]
                r = s
                k = step + 1
                history[k] = ss
                reason = 0
                break
            t = apply(a, s)
            ts = product(t, s)
            tt = product(t, t)
            if not math.isfinite(ts) or not math.isfinite(tt) or tt <= 0:
                reason = 3
                break
            omega = f(ts / tt)
            if not math.isfinite(omega) or omega == 0:
                reason = 3
                break
            x = [f(omega * w + f(alpha * z + q)) for w, z, q in zip(s, p, x)]
            r = [f(-omega * w + z) for w, z in zip(t, s)]
            rr = product(r, r)
            k = step + 1
            history[k] = rr
            if not math.isfinite(rr):
                reason = 3
                break
            if rr == 0 and norm(r) > threshold:
                reason = 3
                break
            if math.sqrt(rr) <= threshold:
                reason = 0
                break
            if k == limit:
                break
            nxt = product(shadow, r)
            if not math.isfinite(nxt) or nxt == 0:
                reason = 3
                break
            beta = f(f(nxt / rho) * f(alpha / omega))
            if not math.isfinite(beta):
                reason = 3
                break
            p = [f(beta * f(-omega * w + z) + q) for w, z, q in zip(v, p, r)]
            rho = nxt
    true = norm([f(v - w) for v, w in zip(b, apply(a, x))])
    if not math.isfinite(true):
        reason = 3
    elif reason == 0 and true > threshold:
        reason = 4
    return dict(
        solution=x,
        reason=[reason],
        iterations=[k],
        residual_squared=history,
        true_residual_norm=[true],
    )
