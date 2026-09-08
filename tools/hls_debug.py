"""Inspect a generated HLS execution bundle."""
from pathlib import Path
from bootstrap import configure
import runpy
configure(Path(__file__).resolve().parent)
runpy.run_module('debug', run_name='__main__')
