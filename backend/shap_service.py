"""
OmniDiag — SHAP Explanation Service
====================================
Pure logic for transforming raw SHAP values into structured JSON chart data
and human-readable textual explanations. Contains NO matplotlib, NO image
generation, and NO Base64 encoding — the React frontend renders the chart.

This service is called by ModelLoader.explain() after SHAP values are computed.
"""

from typing import List, Dict, Any


def generate_shap_explanation(
    shap_values_object: Any,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Transform raw SHAP Explanation object into frontend-ready chart data and
    a human-readable textual explanation.

    Args:
        shap_values_object: A SHAP Explanation object (e.g., from
                            shap.TreeExplainer(model)(df)). Only the first
                            row (single patient) is used.
        feature_names: List of feature names corresponding to the columns
                       of the input DataFrame.

    Returns:
        Dictionary with:
            - chart_data: List[Dict] of {"feature": str, "shap_value": float}
              sorted by |shap_value| descending. Ready for ShapBarChart.jsx.
            - text_explanation: Human-readable string identifying the top 3
              most impactful features with direction labels.
            - base_value: The base (expected) value from the explainer.

    Raises:
        ValueError: If the SHAP values object is empty or malformed.
    """
    # Extract values for the single patient (first row)
    patient_shap_values = shap_values_object.values[0]
    base_value = float(shap_values_object.base_values[0])

    # Pair features with values and sort by absolute impact
    feature_impacts: List[Dict[str, Any]] = [
        {"feature": str(f), "shap_value": float(v)}
        for f, v in zip(feature_names, patient_shap_values)
    ]
    feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # Generate textual explanation (top 3)
    top_features = feature_impacts[:3]
    details = []
    for item in top_features:
        effect = "increased risk" if item["shap_value"] > 0 else "decreased risk"
        details.append(f"{item['feature']} ({effect})")

    explanation_text = (
        "Top factors influencing this prediction: "
        + ", ".join(details)
        + "."
    )

    return {
        "chart_data": feature_impacts,
        "text_explanation": explanation_text,
        "base_value": base_value,
    }
