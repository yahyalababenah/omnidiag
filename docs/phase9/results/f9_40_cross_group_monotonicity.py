"""NHANES cross-group decision non-monotonicity on the held-out 2017-2018 test cycle.

Data are rebuilt exactly as the shipped model's training script builds them (the head of
train_diabetes_nhanes_ebm_v2.py, as Gate 9.7 did). Scores come from the SHIPPED bundle
(sha256 fcceeb37...), and a random sample is cross-checked against the live backend code.
"""
import os, sys, io, json, contextlib, hashlib, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np, pandas as pd, joblib

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RES = [d for d in (os.path.join(os.path.expanduser("~/Desktop"), x) for x in os.listdir(os.path.expanduser("~/Desktop"))) if os.path.basename(d).startswith("بيانات")][0]
BUNDLE = os.path.join(REPO, "models/diabetes_nhanes/diabetes_nhanes_ebm.joblib")
assert hashlib.sha256(open(BUNDLE, "rb").read()).hexdigest().startswith("fcceeb37"), "not the shipped bundle"

os.chdir(RES)
head = open("train_diabetes_nhanes_ebm_v2.py", encoding="utf-8").read().split("# ============================================================================ 4. fits")[0]
head = head.replace('os.makedirs(OUT, exist_ok=True)', 'import tempfile; OUT=tempfile.mkdtemp()').replace('hashlib.sha256(open(__file__, "rb").read())', 'hashlib.sha256(b"x")')
sys.argv = ["x"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, "head", "exec"), globals())

sys.path.insert(0, REPO); _cwd = os.getcwd(); os.chdir(REPO)
from backend.router import OmniDiagRouter
router = OmniDiagRouter(os.path.join(REPO, "configs"))
backend = router._get_loader("diabetes_nhanes")
os.chdir(_cwd)
B = joblib.load(BUNDLE); FF = B["features"]; q = B["conformal"]["q_group"]
X = D.loc[te, FF]
raw = B["model"].predict_proba(X)[:, 1]
cal = B["calibrator"]
p = backend._platt(raw)                      # the production Platt map (on logit(raw))
age = D.loc[te, "RIDAGEYR"].values
band = np.where(age < 40, 0, np.where(age < 60, 1, 2))
q1 = np.array([q[f"{b}|1"] for b in band]); q0 = np.array([q[f"{b}|0"] for b in band])
in1 = (1 - raw) <= q1; in0 = raw <= q0
dec = np.where(in1 & in0, "uncertain", np.where(in1, "referral", np.where(in0, "no_referral", "uncertain")))
assert list(dec) == backend._decide(raw, band)[0], "decision rule differs from production"
RANK = {"no_referral": 0, "uncertain": 1, "referral": 2}       # conservativeness, 3 levels
r3 = np.array([RANK[d] for d in dec]); r2 = (r3 > 0).astype(int)  # tested (referral|uncertain) vs cleared
n = len(p)
out = {"n_test": int(n), "label": "HbA1c >= 5.7 (dysglycaemia)", "cycle": "2017-2018 held out",
       "bundle_sha256_prefix": "fcceeb37", "empty_sets": int((~in1 & ~in0).sum())}

# ── 1. cross-check against the live backend on a random sample ────────────────
MAND = B["mandatory_fields"]
complete = X[MAND].notna().all(axis=1).values
out["mandatory_fields"] = MAND
out["n_complete_mandatory"] = int(complete.sum())
rng = np.random.default_rng(0); idx = rng.choice(np.where(complete)[0], 300, replace=False)
names = {0: "normal", 1: "increased", 2: "high"}
rows = []
for i in idx:
    r = {f: (None if pd.isna(v) else float(v)) for f, v in X.iloc[i].items()}
    r["ADIPOSITY_BAND"] = names[int(r["ADIPOSITY_BAND"])]
    rows.append(r)
live = backend.predict_batch(rows)
out["backend_crosscheck"] = {
    "sample": len(idx),
    "decision_agree": int(sum(l["decision"] == dec[i] for l, i in zip(live, idx))),
    "max_abs_p_diff": float(max(abs(l["confidence"] - p[i]) for l, i in zip(live, idx))),
}

# ── 2. decision cut-points per age band, on raw score and on displayed p ─────
def platt(x):
    x = np.atleast_1d(np.asarray(x, float))
    return backend._platt(x)
bands = {0: "20-39", 1: "40-59", 2: "60+"}
cuts = {}
for b, label in bands.items():
    lo_ref = 1 - q[f"{b}|1"]; hi_clear = q[f"{b}|0"]
    m = band == b
    emp = {d: ([float(p[m & (dec == d)].min()), float(p[m & (dec == d)].max())] if (m & (dec == d)).any() else None, int((m & (dec == d)).sum()))
           for d in ("no_referral", "uncertain", "referral")}
    cuts[label] = {
        "n": int(m.sum()),
        "raw_cleared_below": round(lo_ref, 4), "raw_referral_above": round(hi_clear, 4),
        "p_cleared_below": round(float(platt(lo_ref)[0]), 4), "p_referral_above": round(float(platt(hi_clear)[0]), 4),
        "empirical_p_range_and_n": {d: {"p_range": v[0], "n": v[1]} for d, v in emp.items()},
    }
out["cutpoints_by_age_band"] = cuts

# ── 3. pairwise discordance: higher p, LESS conservative decision ────────────
POP = np.ones(n, bool)

def discord(rank, mask_i=None, mask_j=None):
    """Among ordered pairs (i, j) with p_i > p_j (i from mask_i, j from mask_j),
    the share where rank_i < rank_j."""
    I = np.where((mask_i if mask_i is not None else np.ones(n, bool)) & POP)[0]
    J = np.where((mask_j if mask_j is not None else np.ones(n, bool)) & POP)[0]
    tot = bad = 0
    for s in range(0, len(I), 512):
        ii = I[s:s + 512]
        gt = p[ii][:, None] > p[J][None, :]
        lt = rank[ii][:, None] < rank[J][None, :]
        tot += int(gt.sum()); bad += int((gt & lt).sum())
    return {"pairs_with_higher_p": tot, "discordant": bad, "fraction": (bad / tot) if tot else None}

for popname, popmask in (("all_test", np.ones(n, bool)), ("complete_mandatory", complete)):
  POP = popmask
  out.setdefault("discordance", {})[popname] = {"overall": {"3-level": discord(r3), "binary": discord(r2)},
      "by_band_pair": {f"higher-p in {la} vs lower-p in {lb}": {"3-level": discord(r3, band == a, band == b), "binary": discord(r2, band == a, band == b)}
                       for a, la in bands.items() for b, lb in bands.items()}}
print(json.dumps(out, indent=1, default=float))

# ── 4. clinical read-out: who is cleared, and how many of them are dysglycaemic ──
y = D.loc[te, "y"].values
ro = {}
for b, label in bands.items():
    m = band == b; cleared = m & (dec == "no_referral")
    ro[label] = {"prevalence": float(y[m].mean()), "cleared_share": float(cleared.sum() / m.sum()),
                 "dysglycaemic_among_cleared": float(y[cleared].mean()) if cleared.any() else None,
                 "dysglycaemic_cleared_share_of_band_positives": float((cleared & (y == 1)).sum() / max((m & (y == 1)).sum(), 1))}
print("CLINICAL", json.dumps(ro))
