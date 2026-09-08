"""Reusable batch-major half local / SDK f32 collective placement policies."""

MATMUL_Y = dict(
    broadcast="resident_rows",
    axis="y",
    reduce="sdk_axis",
    result="feature_columns",
    replicas="rows",
    fusion="none",
    accumulation="f16",
    collective="f32",
    compute="dsr",
    fp="relaxed",
)
MATMUL_X = dict(
    MATMUL_Y,
    broadcast="resident_columns",
    axis="x",
    result="feature_rows",
    replicas="columns",
)
POINT_X = dict(layout="batch_major", axis="x", compute="dsr", fp="relaxed")
