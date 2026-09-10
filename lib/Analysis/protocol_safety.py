"""Exhaustive reachability for finite actor protocols, with shortest witnesses.

No clocks, stochastic delays, CSL task bits, network routing or numeric values
are modeled. All enabled actor interleavings are explored. A limit yields
INCONCLUSIVE, never PASS. Fairness is only needed if the environment can stutter
forever; every modeled step advances one finite actor program counter.
"""
from collections import deque
from dataclasses import dataclass

from event_protocol import Protocol


@dataclass(frozen=True)
class State:
    pc: tuple[int, ...]
    versions: tuple[int, ...]
    writers: tuple[tuple[str, str, int] | None, ...]
    readers: tuple[tuple[tuple[str, str, int], ...], ...]
    events: frozenset[str]
    occupancy: tuple[int, ...]


def explore(protocol: Protocol, max_states=100000):
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be positive")
    bindex = {name: i for i, name in enumerate(protocol.buffers)}
    qindex = {name: i for i, (name, _, _) in enumerate(protocol.fifos)}
    initial = State((0,) * len(protocol.actors), (-1,) * len(bindex),
                    (None,) * len(bindex), ((),) * len(bindex), frozenset(),
                    tuple(n for _, _, n in protocol.fifos))
    pending = deque([initial])
    previous = {initial: None}
    terminals = 0
    transitions = 0

    def witness(state, last=None):
        result = [] if last is None else [last]
        while previous[state] is not None:
            parent, event = previous[state]
            result.append(event)
            state = parent
        return list(reversed(result))

    def report(status, state, reason, event=None):
        return {"status": status, "states": len(previous), "transitions": transitions,
                "terminal_states": terminals, "reason": reason,
                "witness": witness(state, event),
                "scope": "all interleavings of this finite abstract protocol"}

    while pending:
        state = pending.popleft()
        if all(pc == len(ops) for pc, (_, ops) in zip(state.pc, protocol.actors)):
            if any(state.writers) or any(state.readers):
                return report("UNSAFE", state, "ownership leaked at termination")
            terminals += 1
            continue
        enabled = 0
        for actor_id, (name, ops) in enumerate(protocol.actors):
            pc = state.pc[actor_id]
            if pc == len(ops):
                continue
            op = ops[pc]
            step = {"actor": name, "pc": pc, "op": op.kind,
                    "resource": op.resource, "version": op.version, "token": op.token}
            versions, writers = list(state.versions), list(state.writers)
            readers = [set(r) for r in state.readers]
            events, occupancy = set(state.events), list(state.occupancy)
            failure = None
            identity = (name, op.token, op.version)
            if op.kind == "wait":
                if op.resource not in events:
                    continue
            elif op.kind == "signal":
                events.add(op.resource)
            elif op.kind in {"put", "get"}:
                q = qindex[op.resource]
                capacity = protocol.fifos[q][1]
                if (op.kind == "put" and occupancy[q] == capacity) or (
                        op.kind == "get" and occupancy[q] == 0):
                    continue
                occupancy[q] += 1 if op.kind == "put" else -1
            else:
                b = bindex[op.resource]
                if op.kind == "write_begin":
                    if writers[b] or readers[b]:
                        failure = "write overlaps an active writer or reader"
                    else:
                        writers[b] = identity
                elif op.kind == "write_end":
                    if writers[b] != identity:
                        failure = "write completion does not own buffer"
                    else:
                        writers[b] = None
                        versions[b] = op.version
                elif op.kind == "borrow":
                    if writers[b] or versions[b] != op.version or identity in readers[b]:
                        failure = "read before publication, stale version, or duplicate borrow"
                    else:
                        readers[b].add(identity)
                elif op.kind == "release":
                    if identity not in readers[b]:
                        failure = "release without owned borrow"
                    else:
                        readers[b].remove(identity)
            if failure:
                return report("UNSAFE", state, failure, step)
            enabled += 1
            transitions += 1
            counters = list(state.pc)
            counters[actor_id] += 1
            new = State(tuple(counters), tuple(versions), tuple(writers),
                        tuple(tuple(sorted(r)) for r in readers), frozenset(events), tuple(occupancy))
            if new not in previous:
                if len(previous) >= max_states:
                    return report("INCONCLUSIVE", state, "state limit reached")
                previous[new] = (state, step)
                pending.append(new)
        if not enabled:
            return report("DEADLOCK", state, "unfinished actors, no enabled operation")
    return {"status": "PASS", "states": len(previous), "transitions": transitions,
            "terminal_states": terminals, "reason": "all maximal modeled executions terminate safely",
            "witness": [], "scope": "finite abstract protocol; not arbitrary CSL or hardware"}
