import sys
from pathlib import Path
from probe_runtime import execute,mesh_half_worker
r=Path(sys.argv[2]).resolve()
if sys.argv[1]=="--worker":mesh_half_worker(r)
else:execute(r,2400)
