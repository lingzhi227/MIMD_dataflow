"""Canonical CSC and SDK hypersparse structural lowering, without an oracle.

Indices are integer objects throughout. Values must already be finite f32;
explicit zeros remain entries. No sorting, duplicate summation, or randomization
is implicit. Device capacities are checked before conversion to u16.
"""

import math
import struct
from dataclasses import dataclass


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unsigned(value, bits, name):
    require(
        type(value) is int and 0 <= value < 1 << bits, name + " requires u" + str(bits)
    )
    return value


def finite_f32(value):
    require(
        type(value) in (float, int) and math.isfinite(value),
        "finite f32 value required",
    )
    try:
        rounded = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error) as e:
        raise ValueError("f32 overflow") from e
    require(rounded == value, "value must already be rounded to f32")
    return rounded


@dataclass(frozen=True)
class CSC:
    rows: int
    cols: int
    column_offsets: tuple
    row_indices: tuple
    values: tuple

    def __post_init__(self):
        require(unsigned(self.rows, 32, "rows") > 0, "positive rows required")
        require(unsigned(self.cols, 32, "cols") > 0, "positive columns required")
        # Defensive immutable copies: callers cannot invalidate a checked matrix.
        for field in ("column_offsets", "row_indices", "values"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        offsets, indices = self.column_offsets, self.row_indices
        require(len(offsets) == self.cols + 1, "CSC offset extent")
        require(len(indices) == len(self.values), "CSC entry extent")
        for p in offsets:
            unsigned(p, 32, "column offset")
        require(offsets[0] == 0 and offsets[-1] == len(indices), "CSC offset endpoints")
        require(
            all(a <= b for a, b in zip(offsets, offsets[1:])),
            "CSC offsets must be monotone",
        )
        for row in indices:
            require(
                unsigned(row, 32, "row index") < self.rows, "row index out of range"
            )
        for col in range(self.cols):
            start, end = offsets[col : col + 2]
            require(
                all(indices[i] < indices[i + 1] for i in range(start, end - 1)),
                "CSC rows must be sorted and duplicate-free within each column",
            )
        for value in self.values:
            finite_f32(value)

    def entries(self):
        for col in range(self.cols):
            for p in range(self.column_offsets[col], self.column_offsets[col + 1]):
                yield self.row_indices[p], col, self.values[p]


@dataclass(frozen=True)
class Capacity:
    nnz: int
    columns: int
    rows: int

    def __post_init__(self):
        for name in ("nnz", "columns", "rows"):
            value = getattr(self, name)
            # Matches the pinned SDK preprocess contract (< UINT16_MAX).
            require(
                type(value) is int and 1 <= value < 65535,
                "sparse capacity must be 1..65534: " + name,
            )


def geometry(rows, cols, pe_rows, pe_cols):
    for name, value in (
        ("rows", rows),
        ("cols", cols),
        ("pe_rows", pe_rows),
        ("pe_cols", pe_cols),
    ):
        require(type(value) is int and value > 0, "positive integer " + name)
    require(pe_rows >= 4, "SDK hypersparse requires at least four PE rows")
    bx, by = (cols + pe_cols - 1) // pe_cols, (rows + pe_rows - 1) // pe_rows
    lx, ly = (bx + pe_rows - 1) // pe_rows, (by + pe_cols - 1) // pe_cols
    require(
        max(bx, by, lx * pe_rows, ly * pe_cols) < 65535,
        "local coordinate or padded train extent exceeds u16 contract",
    )
    return dict(
        block_cols=bx,
        block_rows=by,
        local_vec_sz=lx,
        local_out_vec_sz=ly,
        y_pad_start_row_idx=by,
        prows=pe_rows,
        pcols=pe_cols,
    )


def partition(
    matrix,
    pe_rows,
    pe_cols,
    capacity=None,
    memory_limit=48 * 1024,
    control_reserve=8192,
):
    """Pack true sparse columns and compact row positions, preserving entry bits.

    The storage estimate follows SDK memory_usage.py plus an explicit control
    reserve; compiler allocation success remains a separate requirement.
    """
    require(isinstance(matrix, CSC), "validated CSC required")
    g = geometry(matrix.rows, matrix.cols, pe_rows, pe_cols)
    grouped = [[[] for _ in range(pe_cols)] for _ in range(pe_rows)]
    for row, col, value in matrix.entries():
        y, local_row = divmod(row, g["block_rows"])
        x, local_col = divmod(col, g["block_cols"])
        grouped[y][x].append((local_row, local_col, value))
    tiles = []
    maxima = dict(nnz=1, columns=1, rows=1)
    for row in grouped:
        packed_row = []
        for entries in row:
            ys = sorted({r for r, _, _ in entries})
            positions = {r: i for i, r in enumerate(ys)}
            tile = dict(
                mat_vals_buf=[],
                mat_rows_buf=[],
                mat_col_idx_buf=[],
                mat_col_loc_buf=[],
                mat_col_len_buf=[],
                y_rows_init_buf=ys,
            )
            last_col = None
            for r, c, value in entries:
                if c != last_col:
                    tile["mat_col_idx_buf"].append(c)
                    tile["mat_col_loc_buf"].append(len(tile["mat_vals_buf"]))
                    tile["mat_col_len_buf"].append(0)
                    last_col = c
                tile["mat_vals_buf"].append(value)
                tile["mat_rows_buf"].append(positions[r])
                tile["mat_col_len_buf"][-1] += 1
            tile.update(
                local_nnz=[len(entries)],
                local_nnz_cols=[len(tile["mat_col_idx_buf"])],
                local_nnz_rows=[len(ys)],
            )
            for key, field in (
                ("nnz", "local_nnz"),
                ("columns", "local_nnz_cols"),
                ("rows", "local_nnz_rows"),
            ):
                maxima[key] = max(maxima[key], tile[field][0])
            packed_row.append(tile)
        tiles.append(packed_row)
    inferred = Capacity(**maxima)
    if capacity is None:
        capacity = inferred
    require(isinstance(capacity, Capacity), "validated capacity required")
    for key, needed in maxima.items():
        require(getattr(capacity, key) >= needed, "sparse capacity exceeded: " + key)
    require(
        type(control_reserve) is int and control_reserve >= 8192,
        "control reserve must be at least 8192 bytes",
    )
    require(
        type(memory_limit) is int and 0 < memory_limit <= 48 * 1024,
        "PE memory limit 1..49152 bytes",
    )
    storage = (
        6 * capacity.nnz
        + 6 * capacity.columns
        + 26 * capacity.rows
        + 8 * ((capacity.rows + 1) // 2)  # two aligned metadata staging buffers
        + 20 * g["local_vec_sz"]
        + 4 * g["local_out_vec_sz"]
    )
    require(
        storage + control_reserve <= memory_limit,
        "sparse PE memory estimate exceeds budget",
    )
    extents = dict(
        mat_vals_buf=capacity.nnz,
        mat_rows_buf=capacity.nnz,
        mat_col_idx_buf=capacity.columns,
        mat_col_loc_buf=capacity.columns,
        mat_col_len_buf=capacity.columns,
        y_rows_init_buf=capacity.rows,
        local_nnz=1,
        local_nnz_cols=1,
        local_nnz_rows=1,
    )
    for row in tiles:
        for tile in row:
            for name, extent in extents.items():
                tile[name].extend([0] * (extent - len(tile[name])))
    return dict(
        geometry=g,
        capacity=capacity.__dict__,
        extents=extents,
        tiles=tiles,
        storage_bytes=storage,
        control_reserve=control_reserve,
        estimated_bytes=storage + control_reserve,
        storage_contract="SDK hypersparse compact-row-position CSC; f32 values, u16 device indices/counts; u32 host staging",
    )


def distribute_x(vector, rows, cols, pe_rows, pe_cols):
    g = geometry(rows, cols, pe_rows, pe_cols)
    require(len(vector) == cols, "input vector extent")
    values = [finite_f32(v) for v in vector]
    result = [
        [[1.0] * g["local_vec_sz"] for _ in range(pe_cols)] for _ in range(pe_rows)
    ]
    for col, value in enumerate(values):
        x, offset = divmod(col, g["block_cols"])
        y, z = divmod(offset, g["local_vec_sz"])
        result[y][x][z] = value
    return result


def gather_y(tiles, rows, cols, pe_rows, pe_cols):
    g = geometry(rows, cols, pe_rows, pe_cols)
    require(
        len(tiles) == pe_rows and all(len(row) == pe_cols for row in tiles),
        "output PE grid",
    )
    require(
        all(len(tile) == g["local_out_vec_sz"] for row in tiles for tile in row),
        "output local extent",
    )
    return [
        tiles[row // g["block_rows"]][(row % g["block_rows"]) // g["local_out_vec_sz"]][
            row % g["block_rows"] % g["local_out_vec_sz"]
        ]
        for row in range(rows)
    ]
