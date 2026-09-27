"""
OmniDiag — input drift statistics (KS, chi-square, PSI), on scipy alone.

No new dependency: scipy, numpy and pandas are already pinned. Evidently, which
the old drift path imported, is installed on the Space but exposes an
incompatible API (docs/EVIDENTLY_COST.md); these functions replace what it was
being asked for.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
This is INPUT drift: the population that arrives has stopped looking like the
population the model was fitted on. That is not the same as the model getting
worse. A model can hold up on a drifted population, and it can degrade with no
input drift at all when the relationship between inputs and outcome changes
rather than the inputs' distribution. Measuring real degradation needs
follow-up labels, and not one patient in this system has come back with their
catheterisation result. So what these numbers support is "look at this", never
"the model is bad".

THE DECISION RULE
-----------------
A feature is flagged only when BOTH hold:

    (1) its p-value, Holm-Bonferroni corrected across the monitored features,
        is below alpha; and
    (2) its PSI is at or above `psi_major`.

Either one alone is misleading. With 920 reference rows a KS test reports
significance for differences that move no decision, and seven independent
features at p<0.05 give roughly a 30 % chance of at least one false alarm with
no correction at all. PSI alone swings wildly on twenty live rows. So the
p-value answers "is this more than sampling noise" and PSI answers "is it big
enough to care", and a flag needs both.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy import stats

# The conventional PSI reading, stated so it is not mistaken for a tuned value:
# under 0.1 no meaningful shift, 0.1-0.2 minor, 0.2 and over major.
PSI_MINOR = 0.1
PSI_MAJOR = 0.2
DEFAULT_ALPHA = 0.05
# Below this many current rows nothing is judged. A drift share of 0.0 computed
# on 12 rows reads as "measured, no drift", which is the same lie `count: 0` told
# about MLflow in Gate 8.6.
MIN_CURRENT_ROWS = 50


def quantile_edges(values: Sequence[float], n_bins: int = 10) -> List[float]:
    """Bin edges from the REFERENCE distribution's quantiles, outer edges infinite.

    The edges belong to the reference and are frozen with it. Re-binning against
    each incoming sample would let the reference move with the present, and PSI
    computed that way is not a number that means anything.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        raise ValueError("cannot derive bin edges from an empty reference")
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(v, qs))
    if edges.size < 2:            # a constant reference feature has one bin
        edges = np.array([edges[0], edges[0]], dtype=float)
    inner = edges[1:-1] if edges.size > 2 else np.array([], dtype=float)
    return [-np.inf, *inner.tolist(), np.inf]


def bin_counts(values: Sequence[float], edges: Sequence[float]) -> List[int]:
    """Count values into the given edges. NaN is counted as its own trailing bin."""
    v = np.asarray(list(values), dtype=float)
    finite = v[np.isfinite(v)]
    idx = np.searchsorted(np.asarray(edges[1:-1], dtype=float), finite, side="right")
    counts = np.bincount(idx, minlength=len(edges) - 1)
    return [*counts.tolist(), int(v.size - finite.size)]


def expand_ecdf(values: Sequence[float], counts: Sequence[int]) -> np.ndarray:
    """Rebuild the EXACT reference sample from its run-length encoding.

    Every numeric feature in both modules is low-cardinality (at most 216
    distinct values), so the reference sample compresses to value/count pairs
    with no loss. That matters: KS needs the reference SAMPLE, not its bins, and
    storing bins alone would have forced either an approximated CDF or a
    one-sample test that treats the reference as known. This keeps ks_2samp exact.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(counts, dtype=int)
    if v.shape != c.shape:
        raise ValueError(f"ecdf mismatch: {v.shape} values vs {c.shape} counts")
    return np.repeat(v, c)


def psi(reference_counts: Sequence[int], current_counts: Sequence[int]) -> float:
    """Population Stability Index between two binned distributions.

        PSI = sum over bins of (c_i - r_i) * ln(c_i / r_i)

    Empty bins are floored rather than dropped: a bin the reference never filled
    and the current sample does is exactly the drift worth seeing, and dropping
    it would hide it. The floor is half a count on the smaller sample, which
    bounds one empty bin's contribution instead of letting it run to infinity.
    """
    r = np.asarray(reference_counts, dtype=float)
    c = np.asarray(current_counts, dtype=float)
    if r.shape != c.shape:
        raise ValueError(f"bin count mismatch: {r.shape} vs {c.shape}")
    r_n, c_n = r.sum(), c.sum()
    if r_n == 0 or c_n == 0:
        raise ValueError("cannot compute PSI against an empty distribution")
    floor = 0.5 / max(r_n, c_n)
    r_p = np.clip(r / r_n, floor, None)
    c_p = np.clip(c / c_n, floor, None)
    return float(np.sum((c_p - r_p) * np.log(c_p / r_p)))


def ks_test(reference: Sequence[float], current: Sequence[float]) -> Dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov on the finite values of each side."""
    r = np.asarray([x for x in reference if x is not None and np.isfinite(x)], dtype=float)
    c = np.asarray([x for x in current if x is not None and np.isfinite(x)], dtype=float)
    if r.size < 2 or c.size < 2:
        return {"test": "ks_2samp", "statistic": None, "p_value": None,
                "detail": f"too few finite values (reference {r.size}, current {c.size})"}
    res = stats.ks_2samp(r, c)
    return {"test": "ks_2samp", "statistic": float(res.statistic),
            "p_value": float(res.pvalue), "detail": ""}


def chi2_test(reference: Sequence[Any], current: Sequence[Any]) -> Dict[str, Any]:
    """Chi-square on a 2 x k contingency table of category counts.

    A contingency table, not `scipy.stats.chisquare`: that one tests a sample
    against a KNOWN distribution, and here both sides are estimated from data.
    Using it would overstate significance. Missing values are their own
    category -- "stopped being recorded" is drift.
    """
    def counts(xs):
        out: Dict[str, int] = {}
        for x in xs:
            k = "__missing__" if x is None or (isinstance(x, float) and not np.isfinite(x)) else str(x)
            out[k] = out.get(k, 0) + 1
        return out

    r_c, c_c = counts(reference), counts(current)
    cats = sorted(set(r_c) | set(c_c))
    table = np.array([[r_c.get(k, 0) for k in cats], [c_c.get(k, 0) for k in cats]], dtype=float)
    # Columns empty on both sides cannot happen here, but a column empty in one
    # is kept: that is the signal.
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    if table.shape[1] < 2 or table.sum(axis=1).min() < 1:
        return {"test": "chi2_contingency", "statistic": None, "p_value": None,
                "categories": cats, "detail": "fewer than two non-empty categories"}
    chi2, p, dof, expected = stats.chi2_contingency(table)
    # Reported, not silently ignored: chi-square's approximation is unreliable
    # when expected cell counts fall below five, and this is where PSI carries
    # the judgement instead.
    low = int(np.sum(expected < 5))
    return {"test": "chi2_contingency", "statistic": float(chi2), "p_value": float(p),
            "dof": int(dof), "categories": [c for c, k in zip(cats, keep) if k],
            "low_expected_cells": low,
            "detail": (f"{low} expected cell(s) below 5 — the chi-square approximation is "
                       f"unreliable here; PSI carries the judgement") if low else ""}


def holm_bonferroni(p_values: Dict[str, Optional[float]], alpha: float = DEFAULT_ALPHA
                    ) -> Dict[str, Dict[str, Any]]:
    """Holm-Bonferroni step-down correction over the features that produced a p-value.

    Holm rather than plain Bonferroni: it controls the same family-wise error
    rate and rejects at least as much, so there is no reason to take the weaker
    one. Features whose test could not run are carried through as untested
    instead of counted in the family.
    """
    testable = {k: v for k, v in p_values.items() if v is not None}
    m = len(testable)
    out: Dict[str, Dict[str, Any]] = {k: {"p_value": None, "p_adjusted": None,
                                          "significant": False, "untested": True}
                                     for k, v in p_values.items() if v is None}
    running = 0.0
    for rank, (name, p) in enumerate(sorted(testable.items(), key=lambda kv: kv[1])):
        adj = min(1.0, max(running, (m - rank) * p))   # step-down: never decreasing
        running = adj
        out[name] = {"p_value": p, "p_adjusted": adj, "significant": adj < alpha,
                     "untested": False}
    return out


def missingness_test(reference_counts: Sequence[int], current_counts: Sequence[int]
                     ) -> Dict[str, Any]:
    """Chi-square on present vs missing, for a numeric feature.

    KS runs on finite values only, so a feature that simply STOPPED BEING
    MEASURED is invisible to it -- and when almost every value is missing, KS
    cannot run at all and the feature comes back untested. That is how the
    hardest drift in the seeded test (Zurich's cholesterol, PSI 5.0, 117 of 123
    rows missing) went unflagged on the first run.

    The last bin of a numeric profile is the missing count, by construction.
    """
    r = np.asarray(reference_counts, dtype=float)
    c = np.asarray(current_counts, dtype=float)
    table = np.array([[r[:-1].sum(), r[-1]], [c[:-1].sum(), c[-1]]], dtype=float)
    if table.sum(axis=1).min() < 1 or table.sum(axis=0).min() < 1:
        return {"test": "chi2_missingness", "statistic": None, "p_value": None,
                "reference_missing_share": None, "current_missing_share": None,
                "detail": "no variation in missingness on one side"}
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"test": "chi2_missingness", "statistic": float(chi2), "p_value": float(p),
            "reference_missing_share": float(r[-1] / r.sum()) if r.sum() else None,
            "current_missing_share": float(c[-1] / c.sum()) if c.sum() else None,
            "detail": ""}


def combine_p(p_values: List[Optional[float]]) -> Optional[float]:
    """Bonferroni-combine the tests applied WITHIN one feature.

    A numeric feature gets two tests -- KS on its finite values and chi-square on
    its missingness -- and taking the smaller p unadjusted would inflate the
    feature's own false-positive rate before the across-feature correction ever
    runs. Multiplying the minimum by the number of tests keeps the feature's rate
    at alpha, and Holm then handles the family across features.
    """
    live = [x for x in p_values if x is not None]
    if not live:
        return None
    return min(1.0, min(live) * len(live))


def compare_feature(
    name: str,
    kind: str,
    reference: Sequence[Any],
    current: Sequence[Any],
    edges: Optional[Sequence[float]] = None,
    reference_counts: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """One feature's test and effect size. Does not decide -- `flag_features` does."""
    if kind == "numeric":
        if edges is None:
            edges = quantile_edges(reference)
        ref_counts = list(reference_counts) if reference_counts is not None \
            else bin_counts(reference, edges)
        cur_counts = bin_counts(current, edges)
        ks = ks_test(reference, current)
        miss = missingness_test(ref_counts, cur_counts)
        test = {"test": "ks_2samp + chi2_missingness",
                "statistic": ks["statistic"], "p_value": combine_p([ks["p_value"],
                                                                    miss["p_value"]]),
                "ks": ks, "missingness": miss,
                "detail": "; ".join(d for d in (ks["detail"], miss["detail"]) if d)}
    else:
        test = chi2_test(reference, current)
        cats = test.get("categories") or []
        def cat_counts(xs):
            d: Dict[str, int] = {}
            for x in xs:
                k = "__missing__" if x is None or (isinstance(x, float) and not np.isfinite(x)) else str(x)
                d[k] = d.get(k, 0) + 1
            return [d.get(c, 0) for c in cats]
        ref_counts = list(reference_counts) if reference_counts is not None else cat_counts(reference)
        cur_counts = cat_counts(current)
    try:
        psi_value = psi(ref_counts, cur_counts)
    except ValueError as exc:
        psi_value, psi_detail = None, str(exc)
    else:
        psi_detail = ""
    return {"feature": name, "kind": kind, **test,
            "psi": psi_value, "psi_detail": psi_detail,
            "psi_band": (None if psi_value is None else
                         "major" if psi_value >= PSI_MAJOR else
                         "minor" if psi_value >= PSI_MINOR else "none"),
            "reference_counts": list(ref_counts), "current_counts": list(cur_counts)}


def flag_features(comparisons: Iterable[Dict[str, Any]], alpha: float = DEFAULT_ALPHA,
                  psi_major: float = PSI_MAJOR) -> Dict[str, Any]:
    """Apply the two-condition rule and return the share of monitored features flagged."""
    comps = list(comparisons)
    corrected = holm_bonferroni({c["feature"]: c.get("p_value") for c in comps}, alpha=alpha)
    flagged: List[str] = []
    for c in comps:
        adj = corrected[c["feature"]]
        c["p_adjusted"] = adj["p_adjusted"]
        c["significant"] = adj["significant"]
        c["drifted"] = bool(adj["significant"] and c.get("psi") is not None
                            and c["psi"] >= psi_major)
        if c["drifted"]:
            flagged.append(c["feature"])
    n = len(comps)
    return {"features": comps, "n_monitored": n, "n_drifted": len(flagged),
            "drifted_features": sorted(flagged),
            "drift_share": (len(flagged) / n) if n else 0.0,
            "rule": (f"flagged when Holm-corrected p < {alpha} AND PSI >= {psi_major}; "
                     f"the p-value answers whether it is more than sampling noise, "
                     f"PSI whether it is large enough to matter"),
            "alpha": alpha, "psi_major": psi_major}
