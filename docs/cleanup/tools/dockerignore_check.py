"""Is a path excluded from the Docker build context? Docker (moby/patternmatcher) rules:
patterns anchored at the context root; * and ? do not cross '/'; ** spans directories;
'!' re-includes; the LAST matching pattern wins; a pattern matching a parent directory
matches the path too.
usage: python dockerignore_check.py <.dockerignore> < paths   (one path per line)"""
import re, sys


def compile_pat(p):
    p = p.strip().lstrip("/").rstrip("/")
    out, i = "", 0
    while i < len(p):
        c = p[i]
        if c == "*":
            if p[i:i + 2] == "**":
                i += 2
                if p[i:i + 1] == "/":
                    i += 1
                    out += "(.*/)?"
                else:
                    out += ".*"
                continue
            out += "[^/]*"
        elif c == "?":
            out += "[^/]"
        else:
            out += re.escape(c)
        i += 1
    return re.compile("^" + out + "$")


def load(path):
    rules = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        neg = line.startswith("!")
        body = line[1:] if neg else line
        rules.append((neg, compile_pat(body)))
    return rules


def excluded(path, rules):
    parts = path.split("/")
    prefixes = ["/".join(parts[:k]) for k in range(1, len(parts) + 1)]
    state = False
    for neg, rx in rules:
        if any(rx.match(pre) for pre in prefixes):
            state = not neg
    return state


if __name__ == "__main__":
    rules = load(sys.argv[1])
    known = {"backend/.venv/bin/python": True, "scripts/retrain.py": True,
             "scripts/build_drift_reference.py": False, "scripts/train_heart_glm.py": False,
             "data/heart_disease/processed/uci_heart_by_site.csv": False,
             "README.md": True, "docs/guide.md": False, "docs/assets/research/x.png": True,
             "archive/post_expo_2026-10/INDEX.md": True, "logs/x.md": True, "backend/main.py": False}
    bad = {p: w for p, w in known.items() if excluded(p, rules) != w}
    print("matcher self-test:", "PASS" if not bad else f"FAIL {bad}")
    n_bad = 0
    for path in sys.stdin.read().split():
        ex = excluded(path, rules)
        n_bad += ex
        print(("EXCLUDED  " if ex else "included  ") + path)
    sys.exit(1 if (bad or n_bad) else 0)
