import sys

from pathlib import Path
from IPython import get_ipython


ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    ipython = get_ipython()

    if ipython is not None:
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")

except Exception:
    pass

'''
%load_ext autoreload
%autoreload 2
'''
