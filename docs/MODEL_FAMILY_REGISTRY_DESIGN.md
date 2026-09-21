# Model-Family Registry — Design

Status: implemented on `feat/model-family-registry` (2026-09-21). Not merged.

## 1. Where the code assumed a specific model family (before)

| Concern | Location | Assumption |
|---|---|---|
| Loader dispatch | `backend/router.py::_create_loader` | `model.ensemble` present → `EnsembleModelLoader`, otherwise `ModelLoader`. `model.type` is only echoed as `model_type` in `/diseases` and is not used for dispatch. |
| Counterfactual support flag | `backend/router.py::get_disease_info` | `supports_counterfactuals = ensemble present OR model.counterfactuals`. |
| Counterfactual availability | `backend/router.py::counterfactuals` | `hasattr(loader, "generate_counterfactuals")` → 501 otherwise (generic, duck-typed). |
| Batch path | `backend/main.py::batch_predict` | `hasattr(loader, "predict_batch")` → one vectorised call; otherwise one `router.predict` per row. Only heart's `ModelLoader` has it. The two paths isolate errors differently: one bad row fails the whole vectorised group but only itself in the per-row path. |
| Explainer (heart) | `ModelLoader.explainer` | `shap.TreeExplainer(bundle["pipeline"].named_steps["clf"])`, input = `named_steps["prep"].transform(df)`. Pipeline step names `prep`/`clf` are hard-coded. |
| Explainer (diabetes) | `EnsembleModelLoader.explain` | One `TreeExplainer` per base model, weighted by `|meta_learner.coef_|`, base value shifted by `log R`. |
| Predict (heart) | `ModelLoader.predict / predict_batch` | `bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1] >= bundle["threshold"]`. |
| Predict (diabetes) | `EnsembleModelLoader.predict` | Base models → meta-learner → Bayes prior-shift correction → `>= inference_threshold` (deployment scale). |
| Thresholds | heart: `bundle["threshold"]` (0.3695, config mirrors it for `/diseases`); diabetes: `model.inference_threshold` (0.108184, corrected scale) | |
| Prevalence correction | `EnsembleModelLoader.__init__/predict/explain/generate_counterfactuals`, `router.get_disease_info` (risk bands) | Applied only where both priors are configured. |
| Completeness warning | `backend/model_loader.py::_HIGH_IMPACT_FEATURES` | Heart's five feature names, hard-coded in the module. |
| Counterfactual predict calls | heart: `ModelLoader.generate_counterfactuals` → `self.predict` with a hard-coded heart `MUTABLE` table; diabetes: `EnsembleModelLoader.generate_counterfactuals` → `CounterfactualGenerator(predict_fn, pipeline_fn)` | The generator itself only sees callables. |
| Input schema | `backend/schemas.py::DISEASE_SCHEMA_REGISTRY` | A hard-coded dict. A new disease had to edit this file. |
| XGBoost `base_score` patch | both loaders | Duplicated per loader, internal to each. |

**`_LABELS_INVERTED`.** The brief lists it as the source of truth for SHAP sign handling. It no
longer exists: commit `447482a` (2026-09-20) removed it when heart moved to
`heart_full_tuned.pkl`, whose labels are not inverted (class 1 = disease). No code in
`backend/` inverts labels or SHAP signs, so this refactor has nothing to preserve there.
The golden master pins the SHAP signs as they are.

## 2. Interface

`backend/model_backends/base.py`

```python
class ModelBackend(ABC):
    family: str                        # set by @register_backend
    capabilities: BackendCapabilities  # supports_tree_shap, supports_vectorized_batch,
                                       # supports_counterfactuals, explainer ("tree"|"generic")
    def __init__(self, config)         # cheap; artifacts load lazily (as before)
    def load(self)                     # force artifacts into memory now
    # model-level
    @abstractmethod predict_proba(df) -> np.ndarray   # positive class, the model's OWN scale
    @property @abstractmethod feature_names -> list[str]
    def shap_values(df) -> ShapResult  # default: generic shap.Explainer + background sample
    def shap_background() -> DataFrame # required only by families using the generic explainer
    # service-level (what the router calls; dict shapes are the API contract)
    def predict(patient) -> dict       # default: predict_proba vs model.inference_threshold
    def predict_batch(patients) -> list[dict]   # default: loop over predict()
    def explain(patient) -> dict       # default: shap_values + generate_shap_explanation
    def invalidate()
```

Names were adapted to the codebase. The brief's `explain(df)` is `shap_values(df)`, because
`explain(patient)` is already the router-facing method and its dict is the `/explain` contract.
`load(config)` is the constructor plus an explicit `load()`, which keeps the existing lazy
loading.

Registry: `@register_backend("<family>")`, `get_backend(family)`. An unknown or missing family
raises `UnknownModelFamilyError`, and the message lists the registered families. The router
re-raises it, so the app fails at startup rather than registering a disease that 404s.
Every module in `backend/model_backends/` is imported automatically, so adding a family means
adding one file.

## 3. Where each concern lives after the change

| Concern | Lives in | Change |
|---|---|---|
| Dispatch | `router._create_loader` → `get_backend(model.family)` | `model.family` added to both YAMLs. The `ensemble:`-presence check is removed. |
| `sklearn_pipeline` (heart) | `SklearnPipelineBackend(ModelLoader, ModelBackend)` | Subclass. Every inherited method is unchanged. It only adds `predict_proba`, `feature_names`, `shap_values` and `capabilities`, which delegate to existing internals. |
| `stacking_ensemble` (diabetes) | `StackingEnsembleBackend(EnsembleModelLoader, ModelBackend)` | Same pattern. `predict_proba` returns the raw (training-prior) probability, which is the model's own scale. |
| Explainer selection | backend class | Tree families keep their existing `TreeExplainer` code untouched. The generic fallback in the base class uses `shap.Explainer(f, Independent(background))`, is flagged `explainer="generic"`, and is slower. |
| Batch path | `main.batch_predict` | `hasattr(loader, "predict_batch")` becomes `loader.capabilities.supports_vectorized_batch`. That keeps diabetes on the per-row path, because every backend now *has* `predict_batch`. |
| `supports_counterfactuals` | backend capability | Same values as before: heart comes from `model.counterfactuals`, the ensemble is always `True`. |
| Schema | `schemas.register_schema()` + optional YAML `schema: {module, class}` read by the router | Heart and diabetes stay in the hard-coded dict. A new disease can declare its schema in YAML. |
| Thresholds, prevalence correction, risk bands, completeness warning | unchanged, in the same place | Not moved and not duplicated. |

## 4. Known limits (kept on purpose, to limit how much existing code changes)

- `sklearn_pipeline` is still heart-shaped. It hard-codes the `prep`/`clf` step names, an
  XGBoost-only `TreeExplainer`, `_HIGH_IMPACT_FEATURES` and the counterfactual `MUTABLE` table.
  A second disease in *this* family would need those moved to YAML first. The "config only"
  claim is proven for the `sklearn_generic` family (see `ADDING_A_MODEL_FAMILY.md`).
- `stacking_ensemble` has a hard-coded list of engineered-feature names in
  `generate_counterfactuals` (diabetes-specific).
- Counterfactual scenario content for diabetes is not reproducible between two runs of the
  same code (see `scratch/golden_master.py::normalise`). This was found while building the
  golden master. It predates this refactor and is out of scope.
