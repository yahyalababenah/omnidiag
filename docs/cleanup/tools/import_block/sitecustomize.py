"""Import blocker for dependency-removal gates (B7 areas 4-5).

Put this folder first on PYTHONPATH and name the packages:

    BLOCK_IMPORTS=lightgbm,flwr BLOCK_LOG=/path/attempts.log PYTHONPATH=docs/cleanup/tools/import_block ...

Importing a named package (or a submodule) raises ImportError, and every attempt is appended to
BLOCK_LOG with the importing frame -- so an import swallowed by a try/except still shows up.
Subprocesses inherit PYTHONPATH and are blocked too. The system sitecustomize is chained.
"""
import os
import sys
import traceback

_BLOCKED = tuple(p for p in os.environ.get("BLOCK_IMPORTS", "").split(",") if p)
_LOG = os.environ.get("BLOCK_LOG")


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in _BLOCKED:
            return None
        if _LOG:
            caller = next((f for f in reversed(traceback.extract_stack()[:-1])
                           if "importlib" not in f.filename and f.filename != __file__), None)
            where = f"{caller.filename}:{caller.lineno}" if caller else "?"
            with open(_LOG, "a") as fh:
                fh.write(f"{os.getpid()}\t{name}\t{where}\t{' '.join(sys.argv)[:200]}\n")
        raise ImportError(f"{name} is blocked by the import blocker (BLOCK_IMPORTS)")


if _BLOCKED:
    sys.meta_path.insert(0, _Blocker())

# chain the interpreter's own sitecustomize (Debian installs one), which this file shadows
_here = os.path.dirname(os.path.abspath(__file__))
for _d in sys.path:
    _f = os.path.join(_d, "sitecustomize.py")
    if _d and os.path.abspath(_d) != _here and os.path.isfile(_f):
        exec(compile(open(_f).read(), _f, "exec"), {"__name__": "sitecustomize_chained"})
        break
