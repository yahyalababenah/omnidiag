# Adding a Model Family or a Disease

Design background: [MODEL_FAMILY_REGISTRY_DESIGN.md](MODEL_FAMILY_REGISTRY_DESIGN.md).
Proof: [`tests/test_model_family_registry.py`](../tests/test_model_family_registry.py).

A **model family** is a way of turning a weights file into predictions and SHAP values, for
example a single sklearn Pipeline or a stacking ensemble. A **disease** is one YAML config
that points at one family.

| Family (`model.family`) | Backend class | Explainer | `/batch` | `/counterfactuals` |
|---|---|---|---|---|
| `sklearn_pipeline` | `backend/model_backends/sklearn_pipeline.py` (wraps `ModelLoader`) | TreeExplainer | vectorised | yes (heart-specific) |
| `stacking_ensemble` | `backend/model_backends/stacking_ensemble.py` (wraps `EnsembleModelLoader`) | TreeExplainer per base model | per row | yes |
| `sklearn_generic` | `backend/model_backends/sklearn_generic.py` | generic `shap.Explainer` (slower) | per row | no (501) |

## Adding a disease within a registered family

Files, and no edits to `backend/`:

1. `configs/<disease>.yaml`
   ```yaml
   disease:
     name: my_disease
     display_name: "My Disease Risk"
     description: "..."
     version: "1.0.0"
   schema:                        # your pydantic input model
     module: my_package.schemas
     class: MyDiseaseInput
   model:
     family: sklearn_generic      # any name from registered_families()
     type: logistic_regression    # descriptive only
     weights_path: models/my_disease/bundle.joblib
     inference_threshold: 0.5
   ```
2. A Pydantic input schema class in any importable module, referenced by `schema:`.
3. A feature engineer, only if the family uses one (`stacking_ensemble` reads
   `features.module/class`; `sklearn_generic` does not).
4. Model weights in the format the family expects. For `sklearn_generic` that is a joblib dict
   `{"pipeline": fitted estimator, "features": [...], "background": DataFrame}`.

Measured for `demo_logreg` (`sklearn_generic`): a **13-line YAML** and a **7-line schema
class** (non-blank lines, docstring excluded). Changes needed in `backend/`: **0**.

**Limit (read before claiming this for the other families).** The config-only path is proven
for `sklearn_generic`. The two built-in families still hold disease-specific code:
`sklearn_pipeline` hard-codes heart's completeness-warning feature list
(`_HIGH_IMPACT_FEATURES`), the heart counterfactual ranges and the `prep`/`clf` step names.
`stacking_ensemble` hard-codes diabetes's engineered-feature names in
`generate_counterfactuals`. A second disease in either family would need that code moved to
YAML first.

## Adding a model family

One file: `backend/model_backends/<family>.py`. It is imported automatically (the package
imports every module in its folder), so no other file changes.

```python
from backend.model_backends.base import BackendCapabilities, ModelBackend, register_backend

@register_backend("my_family")
class MyFamilyBackend(ModelBackend):
    capabilities = BackendCapabilities(explainer="generic")   # or supports_tree_shap=True, ...

    @property
    def feature_names(self): ...           # raw input columns, in order
    def predict_proba(self, df): ...       # positive-class probability per row, model's own scale
    def shap_background(self): ...         # only if you use the generic explainer
```

What you get from `ModelBackend` for free:

- `predict()`: `predict_proba` compared with `model.inference_threshold` (default 0.5).
- `predict_batch()`: loops over `predict()`. Set `supports_vectorized_batch=True` only if you
  override it with a single vectorised call. `/batch` reads that flag.
- `explain()`: `shap_values()` passed through the same `generate_shap_explanation` the other
  families use.
- `shap_values()`: `shap.Explainer(predict_proba, Independent(background))`. It gives values in
  probability units, and `base_value` is the mean prediction on up to 50 background rows. For
  6 features it uses shap's exact algorithm, so `base_value + sum(shap) == confidence`, which
  the test asserts to 1e-6. It is slower than TreeExplainer. A tree family should override
  `shap_values()` and set `supports_tree_shap=True, explainer="tree"`.
- Counterfactuals: add a `generate_counterfactuals(patient)` method and set
  `supports_counterfactuals=True`. Without the method the router returns 501.

Measured for `sklearn_generic`: **1 file, 54 lines (29 code lines)**. Changes elsewhere in
`backend/`: **0**. The test checks this against git: the commit that added the file touched
nothing else under `backend/`.

If `model.family` is missing or unknown, startup fails with
`UnknownModelFamilyError: <file>.yaml: Unknown model family '...'. Set model.family ... to one of the registered families: [...]`.
