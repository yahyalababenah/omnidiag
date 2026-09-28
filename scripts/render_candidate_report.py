#!/usr/bin/env python3
"""
Render a static, single-file HTML report comparing the SHIPPED heart model
against one retraining CANDIDATE — for a room screen, not a dashboard.

    python scripts/render_candidate_report.py <candidate_dir> [--out reports/x.html]

Reads only `candidate_card.json` from the given candidate directory (never the
pkl, never the merged training CSV — no patient row is read or shown). Writes
one self-contained HTML file: no external CSS/JS, no network calls, so it opens
identically offline and on a projector.

Nothing here writes to the shipped bundle, `reference_scores.json`, or MLflow.
This is a read-only view of a `candidate_card.json` that a retraining run
already produced.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DECISIONS = ("referral", "no_referral", "uncertain")
DECISION_LABEL = {
    "referral": "Referral",
    "no_referral": "No referral",
    "uncertain": "Uncertain",
}
CELLS = ("F0", "F1", "M0", "M1")
CELL_LABEL = {
    "F0": "Female, no disease",
    "F1": "Female, disease",
    "M0": "Male, no disease",
    "M1": "Male, disease",
}


def esc(x) -> str:
    return html.escape(str(x))


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def render_transition_table(matrix: dict) -> str:
    counts = matrix["counts"]
    rows = []
    for before in DECISIONS:
        cells = []
        for after in DECISIONS:
            n = counts[f"{before}__to__{after}"]
            cls = "diag" if before == after else ("moved" if n else "zero")
            cells.append(f'<td class="num {cls}">{n}</td>')
        rows.append(
            f'<tr><th class="rowhead">{DECISION_LABEL[before]}</th>{"".join(cells)}</tr>'
        )
    header = "".join(f"<th>{DECISION_LABEL[d]}</th>" for d in DECISIONS)
    return f"""
    <table class="matrix">
      <caption>Decision transition — shipped model (rows) &rarr; candidate (columns), n = {matrix['n']}</caption>
      <thead><tr><th class="corner">shipped &darr; / candidate &rarr;</th>{header}</tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    """


def render_coverage_table(coverage: dict) -> str:
    target = coverage["declared_target"]

    def row(label: str, kind_key: str, badge: str, badge_cls: str) -> str:
        c = coverage[kind_key]
        cells = "".join(
            f'<td class="num">{pct(c[cell])}<span class="n">n={c[f"n_{cell}"]}</span></td>'
            for cell in CELLS
        )
        below = " below-target" if c["marginal"] < target else ""
        return (
            f'<tr><th class="rowhead">{label} '
            f'<span class="badge {badge_cls}">{badge}</span></th>'
            f'<td class="num marginal{below}">{pct(c["marginal"])}</td>{cells}</tr>'
        )

    header = "".join(f"<th>{CELL_LABEL[c]}</th>" for c in CELLS)
    return f"""
    <table class="coverage">
      <caption>Conformal coverage vs. declared target 1&minus;&alpha; = {pct(target)}
        &mdash; <span class="badge by-construction">by construction</span> = measured on the same
        rows the quantile was taken from (an identity check, not an estimate);
        <span class="badge cross-conformal">cross-conformal</span> = measured on rows whose
        Mondrian cell was computed WITHOUT them (5-fold)</caption>
      <thead><tr><th class="corner"></th><th>Marginal</th>{header}</tr></thead>
      <tbody>
        {row("Shipped", "shipped_in_sample", "by construction", "by-construction")}
        {row("Shipped", "shipped_cross_conformal", "cross-conformal", "cross-conformal")}
        {row("Candidate", "candidate_in_sample", "by construction", "by-construction")}
        {row("Candidate", "candidate_cross_conformal", "cross-conformal", "cross-conformal")}
      </tbody>
    </table>
    """


def render_provenance_table(card: dict) -> str:
    def sha_row(label: str, value: str) -> str:
        # label is a static string we author, already valid HTML (may carry an
        # entity like &mdash;); only the value, which comes off the card, is
        # escaped. Escaping the label too turned "&mdash;" into the literal
        # text "&mdash;" on the page instead of an em dash.
        return f'<tr><th class="rowhead">{label}</th><td class="mono">{esc(value)}</td></tr>'

    def val_row(label: str, value) -> str:
        return f'<tr><th class="rowhead">{label}</th><td>{esc(value)}</td></tr>'

    return f"""
    <table class="provenance">
      <caption>Identifiers, row counts, and sha256 hashes for this run</caption>
      <tbody>
        {val_row("MLflow run id", card["mlflow_run_id"])}
        {val_row("Created at (UTC)", card["created_at"])}
        {val_row("Config version", card["config_version"])}
        {val_row("Conformal alpha", card["conformal_alpha"])}
        {val_row("Rows &mdash; base UCI / from review queue / rejected / total",
                 f'{card["rows_base_uci"]} / {card["rows_from_review_queue"]} / '
                 f'{card["rows_rejected"]} / {card["rows_total"]}')}
        {sha_row("Candidate bundle sha256", card["bundle_sha256"])}
        {sha_row("Merged training data sha256", card["training_data_sha256"])}
        {sha_row("Base training CSV sha256", card["base_training_csv_sha256"])}
        {sha_row("Shipped bundle sha256 (unchanged by this run)", card["shipped_bundle_sha256"])}
        {sha_row("reference_scores.json sha256 (unchanged by this run)", card["reference_scores_sha256"])}
      </tbody>
    </table>
    """


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Heart candidate vs. shipped — {run_id}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --bg: #0b0f14; --panel: #121822; --border: #263041; --text: #eef2f7;
    --muted: #9fb0c3; --accent: #4fb0ff; --good: #35c07a; --warn: #f0a63a;
    --diag: #16324d; --moved: #5a2a1f; --zero: #0f1520;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: "Segoe UI", -apple-system, Helvetica, Arial, sans-serif;
    font-size: 22px; line-height: 1.45; margin: 0; padding: 48px 64px 96px;
  }}
  h1 {{ font-size: 40px; margin: 0 0 6px; }}
  h2 {{ font-size: 28px; margin: 56px 0 18px; border-bottom: 2px solid var(--border); padding-bottom: 10px; }}
  .subtitle {{ color: var(--muted); font-size: 24px; margin: 0 0 8px; }}
  .badges {{ margin: 20px 0 0; display: flex; gap: 14px; flex-wrap: wrap; }}
  .pill {{
    display: inline-block; padding: 8px 18px; border-radius: 999px;
    font-size: 20px; font-weight: 600; border: 2px solid var(--border);
  }}
  .pill.candidate {{ background: #12240f; border-color: var(--good); color: var(--good); }}
  .pill.not-promoted {{ background: #241512; border-color: var(--warn); color: var(--warn); }}
  .pill.readonly {{ background: #0f1a24; border-color: var(--accent); color: var(--accent); }}
  table {{ border-collapse: collapse; width: 100%; margin: 0 0 12px; }}
  caption {{
    text-align: left; caption-side: top; color: var(--muted);
    font-size: 20px; padding: 0 0 14px; font-weight: 500;
  }}
  th, td {{ border: 1px solid var(--border); padding: 12px 16px; text-align: center; }}
  th.rowhead, th.corner {{ text-align: left; background: var(--panel); font-weight: 600; }}
  thead th {{ background: var(--panel); font-weight: 700; }}
  td.num {{ font-variant-numeric: tabular-nums; font-size: 24px; }}
  td.mono {{ font-family: "JetBrains Mono", Consolas, monospace; font-size: 16px; text-align: left; word-break: break-all; color: var(--muted); }}
  .matrix td.diag {{ background: var(--diag); font-weight: 700; }}
  .matrix td.moved {{ background: var(--moved); font-weight: 700; color: #ffb199; }}
  .matrix td.zero {{ color: #4a5568; }}
  .coverage td.marginal {{ font-weight: 700; }}
  .coverage td.below-target {{ color: var(--warn); }}
  .coverage .n {{ display: block; font-size: 14px; color: var(--muted); font-weight: 400; }}
  .badge {{ font-size: 14px; padding: 3px 10px; border-radius: 6px; font-weight: 600; margin-left: 8px; }}
  .badge.by-construction {{ background: #241a0c; color: var(--warn); border: 1px solid var(--warn); }}
  .badge.cross-conformal {{ background: #0c2418; color: var(--good); border: 1px solid var(--good); }}
  .movement {{ display: flex; gap: 24px; margin: 18px 0 40px; flex-wrap: wrap; }}
  .stat {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px 28px; min-width: 220px; }}
  .stat .n {{ font-size: 44px; font-weight: 800; color: var(--accent); }}
  .stat .label {{ color: var(--muted); font-size: 18px; margin-top: 4px; }}
  .limit {{
    background: #1a1206; border: 2px solid var(--warn); border-radius: 12px;
    padding: 24px 28px; font-size: 19px; line-height: 1.6; color: #f4d9ab;
  }}
  .limit strong {{ color: var(--warn); }}
  footer {{ margin-top: 64px; color: var(--muted); font-size: 15px; border-top: 1px solid var(--border); padding-top: 20px; }}
</style>
</head>
<body>
  <h1>Heart CAD retraining &mdash; candidate vs. shipped</h1>
  <p class="subtitle">Spline-GLM + IVAP + Mondrian conformal &middot; comparison over the 920-patient UCI cohort</p>
  <div class="badges">
    <span class="pill candidate">CANDIDATE, NOT a replacement</span>
    <span class="pill not-promoted">promoted: no</span>
    <span class="pill readonly">shipped bundle &amp; reference_scores.json: read-only, unchanged</span>
  </div>

  <h2>Decision movement, shipped &rarr; candidate</h2>
  <div class="movement">
    <div class="stat"><div class="n">{decisions_changed}</div><div class="label">decisions changed / {n_total}</div></div>
    <div class="stat"><div class="n">{moved_out}</div><div class="label">moved OUT of uncertain<br>(candidate committed)</div></div>
    <div class="stat"><div class="n">{moved_in}</div><div class="label">moved INTO uncertain<br>(candidate withdrew)</div></div>
  </div>
  {transition_table}

  <h2>Conformal coverage</h2>
  {coverage_table}

  <h2>Provenance</h2>
  {provenance_table}

  <h2>Scientific limit</h2>
  <div class="limit"><strong>Not a validated model &mdash; evidence for a human decision.</strong><br><br>{limit_text}</div>

  <footer>
    Generated {generated_at} by <code>scripts/render_candidate_report.py</code> from
    <code>{card_path}</code> only. No patient row, no pkl, and no training CSV was read to build this page.
  </footer>
</body>
</html>
"""


def render(card: dict, card_path: Path) -> str:
    import datetime

    matrix = card["transition_matrix"]
    return TEMPLATE.format(
        run_id=esc(card["mlflow_run_id"]),
        decisions_changed=matrix["decisions_changed_total"],
        n_total=matrix["n"],
        moved_out=matrix["moved_out_of_uncertain"],
        moved_in=matrix["moved_into_uncertain"],
        transition_table=render_transition_table(matrix),
        coverage_table=render_coverage_table(card["conformal_coverage"]),
        provenance_table=render_provenance_table(card),
        limit_text=esc(card["limit"]),
        generated_at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        card_path=esc(card_path.relative_to(PROJECT_ROOT)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path, help="models/heart_disease/candidates/<run_id>/")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    card_path = (args.candidate_dir / "candidate_card.json").resolve()
    card = json.loads(card_path.read_text())

    out = args.out or (PROJECT_ROOT / "reports" / f"heart_candidate_{card['mlflow_run_id'][:12]}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(card, card_path))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
