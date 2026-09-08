"""Compile a supported C++ HLS design into a fresh frozen CSL bundle."""
from pathlib import Path
from bootstrap import configure
import runpy
configure(Path(__file__).resolve().parent)
runpy.run_module('compile', run_name='__main__')
