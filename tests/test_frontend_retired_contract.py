"""
The frontend keeps its own copy of the retired-module list
(frontend/src/constants/retiredDiseases.js) because it deploys before the
backend does. Two sources of truth must not drift: this pins them equal.

Remove both this test and the frontend constant once every backend the
frontend can reach has retired the module itself (backlog: after the HF release).
"""

import json
import os
import re
import shutil
import subprocess

import pytest

from backend.retired_diseases import RETIRED_DISEASES

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRONTEND_LIST = os.path.join(_ROOT, "frontend", "src", "constants", "retiredDiseases.js")


def _frontend_retired_by_regex():
    src = open(_FRONTEND_LIST, encoding="utf-8").read()
    m = re.search(r"export const RETIRED_DISEASES\s*=\s*\[([^\]]*)\]", src)
    assert m, "RETIRED_DISEASES not found in retiredDiseases.js"
    return re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))


def test_frontend_list_equals_the_backend_list():
    assert sorted(_frontend_retired_by_regex()) == sorted(RETIRED_DISEASES)


def test_frontend_list_as_node_evaluates_it(tmp_path):
    """The same check through the module itself, when node is available."""
    if shutil.which("node") is None:
        pytest.skip("node unavailable")
    copy = tmp_path / "retired.mjs"
    shutil.copy(_FRONTEND_LIST, copy)
    out = subprocess.run(
        ["node", "-e", "import(process.argv[1]).then(m => console.log(JSON.stringify(m.RETIRED_DISEASES)))", str(copy)],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout
    assert sorted(json.loads(out)) == sorted(RETIRED_DISEASES)


def test_the_disease_picker_applies_it():
    src = open(os.path.join(_ROOT, "frontend", "src", "context", "DiseaseContext.jsx"), encoding="utf-8").read()
    assert "withoutRetired(" in src
