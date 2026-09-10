"""Finite, straight-line actor protocols for the survey's bounded checks.

This is an executable research IR, not a semantics for arbitrary CSL. Events are
one-shot, versioned names. An operation is atomic at the abstraction boundary.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Op:
    kind: str
    resource: str
    version: int = 0
    token: str = ""


@dataclass(frozen=True)
class Protocol:
    actors: tuple[tuple[str, tuple[Op, ...]], ...]
    buffers: tuple[str, ...] = ()
    # FIFO capacity and initial occupancy. FIFO tokens are not buffer borrows.
    fifos: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self):
        names = [name for name, _ in self.actors]
        if len(names) != len(set(names)) or not names:
            raise ValueError("Actor names must be unique and nonempty")
        if len(self.buffers) != len(set(self.buffers)):
            raise ValueError("Duplicate buffer")
        queues = {name: (cap, initial) for name, cap, initial in self.fifos}
        if len(queues) != len(self.fifos):
            raise ValueError("Duplicate FIFO")
        if any(type(c) is not int or type(n) is not int or c < 1 or not 0 <= n <= c
               for c, n in queues.values()):
            raise ValueError("Invalid FIFO capacity or initial occupancy")
        signals = [op.resource for _, ops in self.actors for op in ops if op.kind == "signal"]
        if len(signals) != len(set(signals)):
            raise ValueError("Events must be signalled exactly once syntactically")
        kinds = {"wait", "signal", "write_begin", "write_end", "borrow", "release", "put", "get"}
        for _, ops in self.actors:
            for op in ops:
                if type(op.version) is not int or op.version < 0:
                    raise ValueError("Version must be a nonnegative integer")
                if op.kind not in kinds:
                    raise ValueError(f"Unknown operation {op.kind}")
                if op.kind in {"write_begin", "write_end", "borrow", "release"}:
                    if op.resource not in self.buffers or not op.token:
                        raise ValueError("Unknown buffer or missing ownership token")
                if op.kind in {"put", "get"} and op.resource not in queues:
                    raise ValueError("Unknown FIFO")
                if op.kind == "wait" and op.resource not in signals:
                    raise ValueError("Wait has no signal operation")
