"""One explicit Clang selection for AST extraction and native execution."""

import os


def executable():
    value = os.environ.get("HLS_CLANGXX", "clang++")
    if not value.strip():
        raise ValueError("HLS_CLANGXX must name a Clang C++ executable")
    # This is one argv element, never a shell command or a bag of flags.
    return value
