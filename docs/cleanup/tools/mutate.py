"""In-memory mutation plugin: proves a test set is not hollow (B1b/B3 method, B7 area 6).

    MUT=<name> PYTHONPATH=docs/cleanup/tools backend/.venv/bin/python -m pytest -p mutate <tests>

At configure time (before any test module is imported) the named mutation rewrites one
snippet of a helper's source and re-executes the module into its own namespace, so every
`from module import helper` made during collection gets the broken helper. Nothing on
disk changes. A mutation whose snippet is not found exactly once aborts the run.
"""
import importlib
import inspect
import os

CF = "backend.counterfactual_generator"
RG = "backend.llm.report_generator"

MUTATIONS = {
    # What-If: the shared best-achievable helpers
    "every_lever_at_once": (CF, "    full = all_improvements(patient_data, policy)\n    levers =",
                            "    full = all_improvements(patient_data, policy)\n"
                            "    return full, float(score_fn(full))\n    levers ="),
    "no_helpful_only_candidate": (CF, "if len(helpful) > 1 and len(helpful) < len(levers):", "if False:"),
    "no_baseline_check": (CF, "    if not best_p < baseline:", "    if False:"),
    "decrease_lever_never_moves": (CF, "            if float(value) > bound:\n                improved[feat] = bound",
                                   "            if float(value) > bound:\n                improved[feat] = value"),
    # Report helpers
    "shap_lines_drop_the_name": (RG, "lines.append(f\"  - {item['feature']}: SHAP=", "lines.append(f\"  - SHAP="),
    "shap_ignores_top_n": (RG, "sorted_shap = sorted(shap_values, key=lambda x: abs(x.get(\"shap_value\", 0)), reverse=True)[:top_n]",
                           "sorted_shap = sorted(shap_values, key=lambda x: abs(x.get(\"shap_value\", 0)), reverse=True)"),
    "rule_based_drops_top_factors": (RG, "reverse=True)[:3]\n    top_names =", "reverse=True)[:0]\n    top_names ="),
    "prompt_forwards_the_features": (RG, "shap_summary=_format_shap(shap_values),",
                                     "shap_summary=_format_shap(shap_values) + '\\n' + repr(features),"),
}


def pytest_configure(config):
    name = os.environ.get("MUT")
    if not name:
        return
    module_name, old, new = MUTATIONS[name]
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    if source.count(old) != 1:
        raise SystemExit(f"mutation {name}: snippet found {source.count(old)} times in {module_name}")
    exec(compile(source.replace(old, new), module.__file__, "exec"), module.__dict__)
    print(f"\n[mutate] {name} applied to {module_name}")
