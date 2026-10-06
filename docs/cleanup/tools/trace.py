"""Run a script with an audit hook that logs every file it opens inside the cwd.
usage (from a repo root): TRACE_OUT=<file> python trace.py <script.py> [args...]
Each line of TRACE_OUT: <relative path>\t<open mode>."""
import os, runpy, sys

root = os.path.realpath(os.getcwd())
log = open(os.environ["TRACE_OUT"], "a")
seen = set()


def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        p = os.path.realpath(p if os.path.isabs(p) else os.path.join(os.getcwd(), p))
        if p.startswith(root + os.sep) and p not in seen:
            seen.add(p)
            log.write(f"{os.path.relpath(p, root)}\t{args[1] if len(args) > 1 else 'r'}\n")
            log.flush()


sys.addaudithook(hook)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, root)
runpy.run_path(script, run_name="__main__")
