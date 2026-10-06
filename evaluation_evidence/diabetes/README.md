# evaluation_evidence/diabetes — archived BRFSS module

Everything in this folder was produced for the **BRFSS 2015 diabetes stacking ensemble**
(XGBoost + LightGBM + random forest, logistic-regression meta-learner), which was
**retired in October 2026** (post-expo cleanup, gates B3–B7). It is not the module the
system serves: the diabetes slot is now NHANES dysglycaemia screening
(`configs/diabetes_nhanes.yaml`), with its own figures in the README and in
`models/diabetes_nhanes/model_card.json`.

The files are kept unchanged, as dated evidence. They are cited by the archived audit
(`archive/post_expo_2026-10/docs/DIABETES_AUDIT_REPORT.md`), and the scripts that
produced them are archived under `archive/post_expo_2026-10/scratch/` (see `MANIFEST.txt`;
it still names their old `scratch/` paths). The model files they describe are no longer
downloaded by the image or tracked in git; their JSON summaries are archived under
`archive/post_expo_2026-10/models/diabetes/`.

Thresholds, prevalence-correction figures and risk bands in these files (0.280854 raw,
0.108184 deployed, prevalence 0.237) describe that retired module only.
