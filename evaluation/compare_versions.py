"""Reproducible V1/V2 comparison.

V1 is the shared ``out/main`` CSV dataset (the versioned V1 CSV is verified as
an identical copy). V2 uses the final tables designated by the author. All
comparisons join exact incident IDs. Paired V1/V2 improvements use the 53/57
common-decided incidents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DEFAULT = ROOT / "evaluation/results"
SAMPLE_DEFAULT = ROOT / "data/sample_work.csv"
CATEGORIES = ["L", "I", "Nr", "D", "DD", "U", "Nc"]
PRIMARY = ["L", "I", "DD"]
N_EXPECTED = 100
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261020

DEFAULT_PATHS = {
    "expert": ROOT / "data/sample_experts.csv",
    "claude_v1": ROOT / "data/llm_run/out/main/claude/labels.csv",
    "deepseek_v1": ROOT / "data/llm_run/out/main/deepseek/labels.csv",
    "claude_v2": ROOT / "data/llm_run/runs/v2_two_step/main/claude/labels.csv",
    "deepseek_v2": ROOT / "data/llm_run/runs/v2_two_step/main/deepseek/labels.csv",
}
SHARED_V1 = {
    "claude_v1": ROOT / "data/llm_run/runs/v1_single_step/main/claude/labels.csv",
    "deepseek_v1": ROOT / "data/llm_run/runs/v1_single_step/main/deepseek/labels.csv",
}
FINAL_V2 = {
    "claude_v2": ROOT / "data/llm_run/runs/v2_two_step/main/claude/labels.csv",
    "deepseek_v2": ROOT / "data/llm_run/runs/v2_two_step/main/deepseek/labels.csv",
}


def read_semicolon(path: Path, name: str):
    """Read a semicolon CSV and fail on missing/duplicate exact IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if not reader.fieldnames:
            raise ValueError(f"{name}: no header in {path}")
        rows = {}
        duplicates = []
        for line_no, row in enumerate(reader, start=2):
            incident_id = row.get("Incident ID")
            if incident_id is None or not incident_id.strip():
                raise ValueError(f"{name}: missing Incident ID at line {line_no}")
            if incident_id in rows:
                duplicates.append(incident_id)
            rows[incident_id] = row
        if duplicates:
            raise ValueError(f"{name}: duplicate IDs {sorted(set(duplicates))[:10]}")
    return rows, list(reader.fieldnames)


def normalize(row: dict, condition: str) -> dict:
    """Normalize category X markers; keep V1 empty set distinct from explicit NONE."""
    labels = set()
    errors = []
    for category in CATEGORIES:
        value = row.get(category)
        marker = "" if value is None else str(value).strip()
        if marker.casefold() == "x":
            labels.add(category)
        elif marker and marker.casefold() != "nan":
            errors.append(f"{category}={marker!r}")

    is_v2 = condition.endswith("_v2")
    has_outcome = "Outcome" in row
    if is_v2 and not has_outcome:
        outcome = ""
        errors.append("missing Outcome column")
    elif not is_v2 and has_outcome:
        outcome = ""
        errors.append("unexpected Outcome field on V1")
    elif not is_v2:
        outcome = "LABELS" if labels else "EMPTY_LABELSET_UNINTERPRETED"
    else:
        outcome = (row.get("Outcome") or "").strip().upper()
        if outcome not in {"LABELS", "NONE", "UNCERTAIN"}:
            errors.append(f"invalid Outcome={outcome!r}")
        elif outcome == "LABELS" and not labels:
            errors.append("LABELS with empty label set")
        elif outcome in {"NONE", "UNCERTAIN"} and labels:
            errors.append(f"{outcome} co-occurs with categories")
    return {"labels": frozenset(labels), "outcome": outcome,
            "is_uncertain": outcome == "UNCERTAIN", "is_explicit_none": outcome == "NONE",
            "valid": not errors, "invalid_detail": "; ".join(errors)}


def load_data(paths: dict[str, Path], sample_path: Path):
    """Load expert/sample/model CSVs and verify they match on all 100 IDs."""
    data = {}
    for condition, path in paths.items():
        rows, headers = read_semicolon(path, condition)
        required = ["Incident ID", *CATEGORIES]
        missing = [column for column in required if column not in headers]
        if missing:
            raise ValueError(f"{condition}: required columns missing: {missing}")
        if condition.endswith("_v2") and "Outcome" not in headers:
            raise ValueError(f"{condition}: Outcome column is required for V2")
        data[condition] = {i: normalize(row, condition) for i, row in rows.items()}
    sample_rows, _ = read_semicolon(sample_path, "sample")
    expert_ids = set(data["expert"])
    if len(expert_ids) != N_EXPECTED or set(sample_rows) != expert_ids:
        raise ValueError("Sample and expert reference must contain the same 100 unique IDs")
    for condition in ["claude_v1", "deepseek_v1", "claude_v2", "deepseek_v2"]:
        ids = set(data[condition])
        if ids != expert_ids:
            raise ValueError(f"{condition}: ID set mismatch: missing={len(expert_ids-ids)}, extra={len(ids-expert_ids)}")
        invalid = [(i, record["invalid_detail"]) for i, record in data[condition].items() if not record["valid"]]
        if invalid:
            raise ValueError(f"{condition}: malformed labels: {invalid[:5]}")
    return data, sorted(expert_ids)


def verify_sources(paths: dict[str, Path]) -> list[str]:
    """Verify duplicate V1 copies and confirm V2 source paths are author-designated."""
    notes = []
    for condition, duplicate in SHARED_V1.items():
        base_rows, _ = read_semicolon(paths[condition], condition)
        copy_rows, _ = read_semicolon(duplicate, f"{condition}_duplicate")
        base_labels = {i: normalize(row, condition)["labels"] for i, row in base_rows.items()}
        copy_labels = {i: normalize(row, condition)["labels"] for i, row in copy_rows.items()}
        if base_labels != copy_labels:
            raise ValueError(f"The two V1 copies differ for {condition}")
        notes.append(f"{condition}: out/main and runs/v1_single_step/main are identical copies; counted once")
    for condition, expected in FINAL_V2.items():
        if paths[condition].resolve() != expected.resolve():
            raise ValueError(f"{condition}: expected author-designated final source {expected}")
    notes.append("V2 source: author-designated final labels.csv; archived manifest/error mismatch does not invalidate these files")
    return notes


def verify_final_v2(paths: dict[str, Path]) -> None:
    """Require the author-designated final V2 prediction files."""
    expected = {
        "claude_v2": ROOT / "data/llm_run/runs/v2_two_step/main/claude/labels.csv",
        "deepseek_v2": ROOT / "data/llm_run/runs/v2_two_step/main/deepseek/labels.csv",
    }
    for condition, expected_path in expected.items():
        if paths[condition].resolve() != expected_path.resolve():
            raise ValueError(f"{condition} must use final V2 source {expected_path}")


def verify_shared_v1(paths: dict[str, Path]) -> list[str]:
    """Confirm out/main and versioned V1 CSVs are identical, then count them once."""
    notes = []
    for condition, copy_path in SHARED_V1.items():
        main_rows, _ = read_semicolon(paths[condition], condition)
        copy_rows, _ = read_semicolon(copy_path, f"{condition}_copy")
        main_labels = {i: normalize(row, condition)["labels"] for i, row in main_rows.items()}
        copy_labels = {i: normalize(row, condition)["labels"] for i, row in copy_rows.items()}
        if main_labels != copy_labels:
            raise ValueError(f"V1 CSV copies differ for {condition}")
        notes.append(f"{condition}: out/main and versioned runs V1 label CSVs are byte-data-equivalent by incident labels; counted once")
    return notes


def audit_historical_expert_counts() -> list[str]:
    """Identify prior metric support values that differ from the current expert CSV."""
    path = ROOT / "data/llm_run/evaluation/metrics.json"
    if not path.is_file():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    support = obj.get("preannotation", {}).get("per_category", {})
    old_l = support.get("L", {}).get("support")
    old_i = support.get("I", {}).get("support")
    if old_l not in (None, 22) or old_i not in (None, 39):
        return [f"Older metrics.json records L={old_l}, I={old_i}; current expert CSV verifies L=22, I=39."]
    return []


def expert_support(reference: dict, ids: list[str]) -> tuple[dict, int]:
    """Verify source-file expert frequencies, NONE count, and declared primary set."""
    support = {c: sum(c in reference[i]["labels"] for i in ids) for c in CATEGORIES}
    none_count = sum(not reference[i]["labels"] for i in ids)
    expected = {"L": 22, "I": 39, "Nr": 2, "D": 2, "DD": 45, "U": 1, "Nc": 0}
    if len(ids) != 100 or none_count != 50 or support != expected:
        raise ValueError(f"Expert reference verification failed: n={len(ids)}, NONE={none_count}, support={support}")
    if [c for c in CATEGORIES if support[c] >= 5] != PRIMARY:
        raise ValueError("Primary categories do not match the five-positive threshold")
    return support, none_count


def score(reference: dict[str, frozenset], prediction: dict[str, frozenset], ids: list[str]) -> dict:
    """Compute category counts/F1, primary/all macro-F1, micro-F1, Jaccard, and exact match."""
    per_category = {}
    category_f1 = []
    primary_f1 = []
    total_tp = total_fp = total_fn = 0
    jaccard_values = []
    exact = 0
    for i in ids:
        gold, pred = reference[i], prediction[i]
        exact += gold == pred
        union = gold | pred
        jaccard_values.append(len(gold & pred) / len(union) if union else 1.0)
    for c in CATEGORIES:
        tp = sum(c in reference[i] and c in prediction[i] for i in ids)
        fp = sum(c not in reference[i] and c in prediction[i] for i in ids)
        fn = sum(c in reference[i] and c not in prediction[i] for i in ids)
        support = sum(c in reference[i] for i in ids)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_category[c] = {"support": support, "TP": tp, "FP": fp, "FN": fn,
                           "precision": precision, "recall": recall, "F1": f1}
        category_f1.append(f1)
        if c in PRIMARY:
            primary_f1.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro_denom = 2 * total_tp + total_fp + total_fn
    return {"n": len(ids), "macro_f1_primary": float(np.mean(primary_f1)) if primary_f1 else np.nan,
            "macro_f1_all_seven": float(np.mean(category_f1)) if category_f1 else np.nan,
            "micro_f1": 2 * total_tp / micro_denom if micro_denom else 0.0,
            "jaccard": float(np.mean(jaccard_values)) if jaccard_values else np.nan,
            "exact_match": exact / len(ids) if ids else np.nan,
            "exact_match_count": exact, "per_category": per_category}


def none_confusion(reference: dict, prediction: dict, ids: list[str]) -> dict:
    """Use expert NONE as positive class and require explicit V2 Outcome NONE."""
    tp = fp = fn = tn = 0
    for i in ids:
        actual_none = not reference[i]["labels"]
        pred_none = prediction[i]["is_explicit_none"]
        if actual_none and pred_none:
            tp += 1
        elif not actual_none and pred_none:
            fp += 1
        elif actual_none:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"positive_class": "expert NONE", "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall, "F1": f1, "n": len(ids)}


def selective_counts(reference: dict, prediction: dict, ids: list[str]) -> dict:
    """Separate exact-set correct, incorrect, and UNCERTAIN outcomes."""
    correct = incorrect = abstentions = 0
    for i in ids:
        if prediction[i]["is_uncertain"]:
            abstentions += 1
        elif prediction[i]["labels"] == reference[i]["labels"]:
            correct += 1
        else:
            incorrect += 1
    decided = correct + incorrect
    return {"eligible": len(ids), "correct": correct, "incorrect": incorrect,
            "abstentions": abstentions, "decided": decided,
            "coverage": decided / len(ids) if ids else np.nan,
            "abstention_rate": abstentions / len(ids) if ids else np.nan,
            "selective_risk": incorrect / decided if decided else np.nan,
            "incorrect_fraction": incorrect / len(ids) if ids else np.nan}


def paired_bootstrap(reference, v1, v2, ids, draws, seed):
    """Paired percentile bootstrap changes by resampling incident records."""
    names = ["macro_f1_primary", "macro_f1_all_seven", "micro_f1", "jaccard", "exact_match"]
    first = {i: (v1[i]["labels"] if isinstance(v1[i], dict) else v1[i]) for i in ids}
    second = {i: (v2[i]["labels"] if isinstance(v2[i], dict) else v2[i]) for i in ids}
    ref = {i: reference[i]["labels"] for i in ids}
    v1_scores = score(ref, first, ids)
    v2_scores = score(ref, second, ids)
    rng = np.random.default_rng(seed)
    sampled = {name: [] for name in names}
    for _ in range(draws):
        sample = rng.choice(ids, size=len(ids), replace=True).tolist()
        a, b = score(ref, first, sample), score(ref, second, sample)
        for name in names:
            sampled[name].append(b[name] - a[name])
    ci = {name: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
          for name, values in sampled.items()}
    return v1_scores, v2_scores, ci


def format_value(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def escape_latex(value):
    return str(value).replace("&", r"\&").replace("_", r"\_")


def write_tables(output_dir, version_rows, paired_rows, category_rows, expert_counts):
    """Write two compact IEEE-style LaTeX booktabs tables."""
    version_table = [r for r in version_rows if r["view"] in {"full_sample", "decided_subset"}
                     and not (r["version"] == "v2" and r["view"] == "full_sample")]
    lines = ["% Generated by evaluation/compare_versions.py", r"\begin{table*}[t]", r"\centering",
             r"\caption{V1 and V2 classification results. V2 metrics are conditional on non-abstention; coverage and abstention rates are shown.}",
             r"\label{tab:v1-v2-results}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrrrrrrr}",
             r"\toprule", r"Model & Version / view & Eligible/scored $n$ & Coverage & Abstention & Primary Macro-$F_1$ & All-seven Macro-$F_1$ & Micro-$F_1$ & Jaccard & Exact match \\", r"\midrule"]
    for r in version_table:
        lines.append(f"{r['model']} & {r['version']} / {escape_latex(r['view'])} & {r['n_expected']}/{r['n_classification']} & {format_value(r['coverage'])} & {format_value(r['abstention_rate'])} & {format_value(r['macro_f1_primary'])} & {format_value(r['macro_f1_all_seven'])} & {format_value(r['micro_f1'])} & {format_value(r['jaccard'])} & {format_value(r['exact_match'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}%", r"\vspace{1mm}",
              r"\footnotesize V1 uses 100 available labels. V2 values use its decided cases and are conditional on shown coverage; -- denotes not applicable.", r"\end{table*}", "",
              r"\begin{table}[t]", r"\centering", r"\caption{Paired V1--V2 changes on the common-decided subset.}", r"\label{tab:v1-v2-paired}",
              r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llrrrrr}", r"\toprule", r"Model & Metric & V1 & V2 & Change & Paired $n$ & V2 coverage \\", r"\midrule"]
    for r in paired_rows:
        lines.append(f"{r['model']} & {escape_latex(r['metric'])} & {format_value(r['v1'])} & {format_value(r['v2'])} & {format_value(r['absolute_change'])} & {r['paired_n']} & {format_value(r['v2_coverage'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}%", r"\vspace{1mm}",
              r"\footnotesize Paired percentile bootstrap intervals are reported in the CSV; resampling unit is the incident (10,000 draws, seed 20261020). Descriptive only.", r"\end{table}", ""]
    (output_dir / "v1_v2_compact_tables.tex").write_text("\n".join(lines), encoding="utf-8")


def write_summary(output_dir, support, none_n, version_rows, paired_rows, coverage_rows, selection_rows,
                  caveats):
    """Write concise Markdown tables and methodological caveats."""
    lines = ["# V1/V2 results", "", "## Verified reference", "",
             f"Expert CSV: n=100, NONE={none_n}, non-empty={100-none_n}; category supports: " +
             ", ".join(f"{c}={support[c]}" for c in CATEGORIES) + ".",
             "Primary Macro-F1 categories: L, I, DD (threshold: five expert positives from the existing evaluator/methodology).",
             "", "## Version-specific results", "",
             "| Model | Version | View | Eligible | Scored n | UNCERTAIN | Coverage | Abstention | Primary Macro-F1 | All-seven Macro-F1 | Micro-F1 | Jaccard | Exact | Selective risk |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in version_rows:
        lines.append(f"| {r['model']} | {r['version']} | {r['view']} | {r['n_expected']} | {r['n_classification']} | {format_value(r['n_uncertain'],0)} | {format_value(r['coverage'])} | {format_value(r['abstention_rate'])} | {format_value(r['macro_f1_primary'])} | {format_value(r['macro_f1_all_seven'])} | {format_value(r['micro_f1'])} | {format_value(r['jaccard'])} | {format_value(r['exact_match'])} | {format_value(r['selective_risk'])} |")
    lines += ["", "## Paired common-decided changes", "",
              "| Model | Metric | V1 | V2 | Absolute change | 95% paired CI | Paired n | V2 coverage |",
              "|---|---|---:|---:|---:|---|---:|---:|"]
    for r in paired_rows:
        lines.append(f"| {r['model']} | {r['metric']} | {format_value(r['v1'])} | {format_value(r['v2'])} | {format_value(r['absolute_change'])} | [{format_value(r['change_ci95_low'])}, {format_value(r['change_ci95_high'])}] | {r['paired_n']} | {format_value(r['v2_coverage'])} |")
    lines += ["", "## Abstention selection diagnostic", "",
              "V1 quality stratified by V2 deciding versus abstaining is descriptive and helps identify whether V2 abstains on cases V1 found harder.",
              "", "| Model | Stratum | n | V1 primary Macro-F1 | V1 exact match | V2 coverage |", "|---|---|---:|---:|---:|---:|"]
    for r in selection_rows:
        if r["view"] == "v1_full_sample":
            continue
        label = {"v1_on_v2_decided":"V1 on V2-decided IDs", "v1_on_v2_abstained":"V1 on V2-abstained IDs", "v2_on_decided":"V2 on decided IDs"}[r["view"]]
        lines.append(f"| {r['model']} | {label} | {r['n']} | {format_value(r['macro_f1_primary'])} | {format_value(r['exact_match'])} | {format_value(r['V2_full_sample_coverage'])} |")
    lines += ["", "## Definitions and caveats", "",
              "- Paired common-decided subset: identical Incident IDs in V1 and V2, retaining only cases where V2 outcome is not UNCERTAIN; each comparison has its own paired n.",
              "- V1 blank labelsets are not NONE. V1 has no explicit case outcome; V1 NONE metrics are not applicable.",
              "- V2 selective risk = incorrect exact-set decisions / V2 decided cases. Coverage = decided / eligible; abstention rate = UNCERTAIN / eligible. Abstentions are separate from errors.",
              "- Full-sample secondary sensitivity maps UNCERTAIN to an empty label set; it does not count abstentions as ordinary errors and is explicitly separate from selective metrics.",
              "- Bootstrap confidence intervals are paired percentile intervals over incident-level resampling (10,000 draws; seed 20261020). No statistical significance test is performed.",
              "- V1 sources are the identical `out/main` and `runs/v1_single_step/main` label CSVs, counted once. V2 files are the author-designated final valid tables; archived V2 run metadata does not override them.",
              "- Older `data/llm_run/evaluation/metrics.json` used L=23 and I=38; the current expert file gives L=22 and I=39.", ""]
    lines += [f"- {caveat}" for caveat in caveats]
    lines += ["", "Reproduce from repository root: `python3 evaluation/compare_versions.py`."]
    (output_dir / "v1_v2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--sample", type=Path, default=SAMPLE_DEFAULT)
    parser.add_argument("--expert", type=Path, default=DEFAULT_PATHS["expert"])
    parser.add_argument("--v1-claude", type=Path, default=DEFAULT_PATHS["claude_v1"])
    parser.add_argument("--v1-deepseek", type=Path, default=DEFAULT_PATHS["deepseek_v1"])
    parser.add_argument("--v2-claude", type=Path, default=DEFAULT_PATHS["claude_v2"])
    parser.add_argument("--v2-deepseek", type=Path, default=DEFAULT_PATHS["deepseek_v2"])
    args = parser.parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    selected_paths = {
        "expert": args.expert,
        "claude_v1": args.v1_claude,
        "deepseek_v1": args.v1_deepseek,
        "claude_v2": args.v2_claude,
        "deepseek_v2": args.v2_deepseek,
    }
    verify_final_v2({k: selected_paths[k] for k in ["claude_v2", "deepseek_v2"]})
    data, ids = load_data(selected_paths, args.sample)
    expert_supports, expert_none_n = expert_support(data["expert"], ids)
    expert_counts = expert_supports
    expert_records = data["expert"]
    expert_labels = {i: data["expert"][i]["labels"] for i in ids}
    source_notes = verify_shared_v1(selected_paths)
    source_notes += audit_historical_expert_counts()

    invalids = []
    for condition in ["claude_v1", "deepseek_v1"]:
        invalids.extend({"condition": condition, "Incident ID": i, "detail": r["invalid_detail"]}
                        for i, r in data[condition].items() if not r["valid"])
    if invalids:
        print(f"WARNING: {len(invalids)} invalid V1 prediction rows will be excluded:", file=sys.stderr)
        for row in invalids:
            print(f"  {row['condition']} {row['Incident ID']}: {row['detail']}", file=sys.stderr)

    version_rows, paired_rows, coverage_rows, per_category_rows = [], [], [], []
    paired_case_rows, selection_rows, none_rows = [], [], []
    primary_metric_names = ["macro_f1_primary", "macro_f1_all_seven", "micro_f1", "jaccard", "exact_match"]
    for model, v1_name, v2_name in [("claude", "claude_v1", "claude_v2"), ("deepseek", "deepseek_v1", "deepseek_v2")]:
        v1, v2 = data[v1_name], data[v2_name]
        v1_ids = sorted(i for i in ids if v1[i]["valid"])
        v2_ids = sorted(i for i in ids if v2[i]["valid"])
        v1_valid = set(v1_ids)
        v2_valid = set(v2_ids)
        common_available = sorted(set(v1_ids) & set(v2_ids))
        common_decided = [i for i in common_available if not v2[i]["is_uncertain"]]
        v1_labels = {i: v1[i]["labels"] for i in v1_ids}
        v2_labels = {i: v2[i]["labels"] for i in v2_ids}
        v2_decided = [i for i in v2_ids if not v2[i]["is_uncertain"]]
        v2_abstained_ids = [i for i in v2_ids if v2[i]["is_uncertain"]]

        # A. Each version's full available sample; V2 category metrics are on its decided subset.
        v1_full = score(expert_labels, v1_labels, v1_ids)
        v2_selective = score(expert_labels, v2_labels, v2_decided)
        v2_decisions = selective_counts(data["expert"], v2, v2_ids)
        version_rows.append({
            "model": model, "version": "v1", "view": "version_full_sample",
            "n_expected": len(ids), "n_available": len(v1), "n_valid": len(v1_ids),
            "n_invalid": len(v1) - len(v1_ids), "n_missing": len(set(ids)-set(v1)),
            "n_uncertain": np.nan, "n_decided": len(v1_ids), "n_classification": len(v1_ids),
            "coverage": len(v1_ids)/len(ids), "abstention_rate": np.nan,
            **{m: v1_full[m] for m in primary_metric_names}, "selective_risk": np.nan,
            "n_correct_exact_set": v1_full["exact_match_count"],
            "n_incorrect_exact_set": len(v1_ids)-v1_full["exact_match_count"],
            "note": "V1 has no abstention outcome or explicit NONE; exact-set error is descriptive only",
        })
        version_rows.append({
            "model": model, "version": "v2", "view": "version_full_sample_abstention_aware",
            "n_expected": len(ids), "n_available": len(v2), "n_valid": len(v2_ids),
            "n_invalid": len(v2)-len(v2_ids), "n_missing": len(set(ids)-set(v2)),
            "n_uncertain": v2_decisions["abstentions"], "n_decided": v2_decisions["decided"],
            "n_classification": v2_decisions["decided"], "coverage": v2_decisions["coverage"],
            "abstention_rate": v2_decisions["abstention_rate"],
            **{m: np.nan for m in primary_metric_names}, "selective_risk": v2_decisions["selective_risk"],
            "n_correct_exact_set": v2_decisions["correct"],
            "n_incorrect_exact_set": v2_decisions["incorrect"],
            "incorrect_decision_fraction": v2_decisions["incorrect_fraction"],
            "note": "Primary category/set metrics appear on separate decided_subset row; UNCERTAIN remains separate",
        })
        version_rows.append({
            "model": model, "version": "v2", "view": "decided_subset",
            "n_expected": len(ids), "n_available": len(v2), "n_valid": len(v2_ids),
            "n_invalid": len(v2)-len(v2_ids), "n_missing": len(set(ids)-set(v2)),
            "n_uncertain": 0, "n_decided": len(v2_decided), "n_classification": len(v2_decided),
            "coverage": len(v2_decided)/len(ids), "abstention_rate": v2_decisions["abstention_rate"],
            **{m: v2_selective[m] for m in primary_metric_names}, "selective_risk": v2_decisions["selective_risk"],
            "n_correct_exact_set": v2_decisions["correct"],
            "n_incorrect_exact_set": v2_decisions["incorrect"],
            "note": "Category and set metrics conditional on V2 non-abstention; coverage shown",
        })

        # Secondary full-sample sensitivity: map UNCERTAIN to empty set, while retaining coverage.
        v2_all_labels = {i: v2[i]["labels"] for i in v2_ids}
        v2_all_scores = score(expert_labels, v2_all_labels, v2_ids)
        version_rows.append({
            "model": model, "version": "v2", "view": "full_sample_uncertain_as_empty_secondary",
            "n_expected": len(ids), "n_available": len(v2), "n_valid": len(v2_ids),
            "n_invalid": len(v2)-len(v2_ids), "n_missing": len(set(ids)-set(v2)),
            "n_uncertain": v2_decisions["abstentions"], "n_decided": len(v2_decided),
            "n_classification": len(v2_ids), "coverage": v2_decisions["coverage"],
            "abstention_rate": v2_decisions["abstention_rate"],
            **{m: v2_all_scores[m] for m in primary_metric_names}, "selective_risk": np.nan,
            "n_correct_exact_set": v2_all_scores["exact_match_count"],
            "n_incorrect_exact_set": len(v2_ids)-v2_all_scores["exact_match_count"],
            "note": "Secondary sensitivity only; UNCERTAIN maps to empty category set, not to an ordinary incorrect decision",
        })

        # B. Common available preserves V2 abstentions for coverage. Common decided uses exactly shared IDs.
        for version, records in [("v1", v1), ("v2", v2)]:
            eligible = common_available
            abstained = sum(v2[i]["is_uncertain"] for i in eligible)
            scored = common_decided if version == "v2" else common_available
            scored_labels = {i: records[i]["labels"] for i in scored}
            scores = score(expert_labels, scored_labels, scored) if scored else {}
            version_rows.append({
                "model": model, "version": version, "view": "common_available",
                "n_expected": len(ids), "n_available": len(common_available), "n_valid": len(common_available),
                "n_invalid": 0, "n_missing": len(ids)-len(common_available),
                "n_uncertain": abstained if version == "v2" else np.nan,
                "n_decided": len(common_decided) if version == "v2" else len(common_available),
                "n_classification": len(scored),
                "coverage": len(common_decided)/len(ids) if version == "v2" else len(common_available)/len(ids),
                "abstention_rate": abstained/len(ids) if version == "v2" else np.nan,
                **{m: scores.get(m, np.nan) for m in primary_metric_names},
                "selective_risk": np.nan,
                "note": "Common-available ID universe; V2 abstentions retained as separate outcomes",
            })
        v1_common_scores = score(expert_labels, v1_labels, common_decided)
        v2_common_scores = score(expert_labels, v2_labels, common_decided)
        version_rows.extend([
            {"model": model, "version": "v1", "view": "common_decided", "n_expected": len(ids),
             "n_available": len(common_available), "n_valid": len(common_available), "n_invalid": 0,
             "n_missing": len(ids)-len(common_available), "n_uncertain": np.nan, "n_decided": len(common_decided),
             "n_classification": len(common_decided), "coverage": len(common_decided)/len(ids), "abstention_rate": np.nan,
             **{m: v1_common_scores[m] for m in primary_metric_names}, "selective_risk": np.nan,
             "note": "Exact common_decided IDs; V1 scored on same records V2 decided"},
            {"model": model, "version": "v2", "view": "common_decided", "n_expected": len(ids),
             "n_available": len(common_available), "n_valid": len(common_available), "n_invalid": 0,
             "n_missing": len(ids)-len(common_available), "n_uncertain": 0, "n_decided": len(common_decided),
             "n_classification": len(common_decided), "coverage": len(common_decided)/len(ids),
             "abstention_rate": abstained/len(ids), **{m: v2_common_scores[m] for m in primary_metric_names},
             "selective_risk": np.nan, "note": "Exact common_decided IDs; conditional on V2 decision"},
        ])

        # Per-category primary views: V1 full, V2 decided, and both on common_decided.
        for version, records, view, scored in [
            ("v1", v1, "version_full_sample", v1_ids),
            ("v2", v2, "decided_subset", v2_decided),
            ("v1", v1, "common_decided", common_decided),
            ("v2", v2, "common_decided", common_decided),
        ]:
            result = score(expert_labels, {i: records[i]["labels"] for i in scored}, scored) if scored else {}
            cov = len(scored)/len(ids)
            for category in CATEGORIES:
                value = result.get("per_category", {}).get(category, {})
                per_category_rows.append({"model": model, "version": version, "view": view,
                                          "category": category, "support": value.get("support", np.nan),
                                          "TP": value.get("TP", np.nan), "FP": value.get("FP", np.nan),
                                          "FN": value.get("FN", np.nan), "precision": value.get("precision", np.nan),
                                          "recall": value.get("recall", np.nan), "F1": value.get("F1", np.nan),
                                          "n_scored": len(scored), "coverage": cov,
                                          "note": "V2 decided-subset/common-decided metrics conditional on coverage" if version == "v2" else "V1 category labels; blank set is not NONE"})

        # NONE confusion only for V2; V1 has no explicit NONE field.
        for version in ["v1", "v2"]:
            if version == "v1":
                none_rows.append({"model": model, "version": "v1", "view": "not_applicable", "positive_class": "expert NONE",
                                  "TP": np.nan, "FP": np.nan, "FN": np.nan, "TN": np.nan, "precision": np.nan,
                                  "recall": np.nan, "F1": np.nan, "n": 0, "note": "V1 has no explicit NONE outcome"})
            else:
                for view, set_ids in [("version_decided", v2_decided), ("common_decided", common_decided)]:
                    none_rows.append({"model": model, "version": "v2", "view": view,
                                      **none_confusion(expert_records, v2, set_ids)})

        # Full-sample abstention accounting; exact-set errors and abstentions remain separate.
        v1_stats = selective_counts(expert_records, v1, v1_ids)
        coverage_rows.append({"model": model, "version": "v1", "view": "full_sample",
                              "eligible_n": len(ids), "available_n": len(v1), "valid_n": len(v1_ids),
                              "invalid_n": len(v1)-len(v1_ids), "missing_n": len(set(ids)-set(v1)),
                              "abstentions": np.nan, "decided": len(v1_ids), "coverage": len(v1_ids)/len(ids),
                              "abstention_rate": np.nan, "correct_exact": v1_stats["correct"],
                              "incorrect_exact": v1_stats["incorrect"], "selective_risk": np.nan,
                              "full_sample_error_rate": v1_stats["incorrect"]/len(v1_ids),
                              "note": "Exact-set error over all V1 outputs; V1 has no abstention outcome"})
        coverage_rows.append({"model": model, "version": "v2", "view": "full_sample",
                              "eligible_n": len(ids), "available_n": len(v2), "valid_n": len(v2_ids),
                              "invalid_n": len(v2)-len(v2_ids), "missing_n": len(set(ids)-set(v2)),
                              "abstentions": v2_decisions["abstentions"], "decided": v2_decisions["decided"],
                              "coverage": v2_decisions["coverage"], "abstention_rate": v2_decisions["abstention_rate"],
                              "correct_exact": v2_decisions["correct"], "incorrect_exact": v2_decisions["incorrect"],
                              "selective_risk": v2_decisions["selective_risk"],
                              "full_sample_error_rate": np.nan,
                              "incorrect_fraction_of_eligible": v2_decisions["incorrect"]/len(ids),
                              "note": "V2 selective risk is among decided; abstentions not counted as errors"})
        for view, eligible in [("common_available", common_available), ("common_decided", common_decided)]:
            stats = selective_counts(expert_records, v2, common_available)
            coverage_rows.append({"model": model, "version": "v2", "view": view,
                                  "eligible_n": len(ids), "available_n": len(common_available),
                                  "valid_n": len(common_available), "invalid_n": 0,
                                  "missing_n": len(ids)-len(common_available),
                                  "abstentions": stats["abstentions"],
                                  "decided": len(common_decided),
                                  "coverage": len(common_decided)/len(ids),
                                  "abstention_rate": stats["abstentions"]/len(ids),
                                  "correct_exact": stats["correct"], "incorrect_exact": stats["incorrect"],
                                  "selective_risk": stats["selective_risk"] if view == "common_available" else np.nan,
                                  "full_sample_error_rate": np.nan,
                                  "incorrect_fraction_of_eligible": stats["incorrect"]/len(ids) if view == "common_available" else np.nan,
                                  "note": f"common IDs; n scored={len(eligible)}; common-decided conditional on V2 decision"})

        # Paired improvement table on identical common-decided rows.
        if common_decided:
            before, after, intervals = paired_bootstrap(expert_records, v1_labels, v2_labels, common_decided,
                                                        args.bootstrap_resamples, args.seed)
            v2_coverage = len(common_decided)/len(ids)
            for metric in ["macro_f1_primary", "macro_f1_all_seven", "micro_f1", "jaccard", "exact_match"]:
                delta = after[metric] - before[metric]
                paired_rows.append({"model": model, "metric": metric, "view": "common_decided",
                                    "v1": before[metric], "v2": after[metric], "absolute_change": delta,
                                    "relative_change": delta/before[metric] if before[metric] != 0 else np.nan,
                                    "paired_n": len(common_decided), "v2_coverage": v2_coverage,
                                    "change_ci95_low": intervals[metric][0], "change_ci95_high": intervals[metric][1],
                                    "bootstrap_resamples": args.bootstrap_resamples, "bootstrap_seed": args.seed,
                                    "note": "Paired incident-level percentile bootstrap; conditional on V2 deciding"})
            for i in common_decided:
                paired_case_rows.append({"model": model, "Incident ID": i,
                                         "expert_labels": ",".join(c for c in CATEGORIES if c in expert_labels[i]),
                                         "v1_labels": ",".join(c for c in CATEGORIES if c in v1_labels[i]),
                                         "v2_labels": ",".join(c for c in CATEGORIES if c in v2_labels[i]),
                                         "v1_exact": v1_labels[i] == expert_labels[i],
                                         "v2_exact": v2_labels[i] == expert_labels[i]})

        # Abstention-selection diagnostic: compare V1 quality within V2 outcome strata.
        for view, stratum_ids, condition, predictions in [
            ("v1_full_sample", v1_ids, "v1", v1_labels),
            ("v1_on_v2_decided", common_decided, "v1", v1_labels),
            ("v1_on_v2_abstained", [i for i in v2_abstained_ids if i in v1_valid], "v1", v1_labels),
            ("v2_on_decided", v2_decided, "v2", v2_labels),
        ]:
            diagnostic = score(expert_labels, predictions, stratum_ids) if stratum_ids else {}
            selection_rows.append({"model": model, "view": view, "n": len(stratum_ids),
                                   "fraction_of_full_sample": len(stratum_ids)/len(ids),
                                   "macro_f1_primary": diagnostic.get("macro_f1_primary", np.nan),
                                   "exact_match": diagnostic.get("exact_match", np.nan),
                                   "V2_full_sample_coverage": len(v2_decided)/len(ids)})

    notes = source_notes + audit_historical_expert_counts()
    warnings = notes + [
        "V2 uses author-designated final labels.csv tables; manifests/errors are archived metadata and do not invalidate them.",
        "V1 duplicate CSV copies were verified identical and are counted as one shared V1 dataset.",
    ]
    frames = {
        "v1_v2_version_specific.csv": pd.DataFrame(version_rows),
        "v1_v2_paired_changes.csv": pd.DataFrame(paired_rows),
        "v1_v2_coverage_abstention.csv": pd.DataFrame(coverage_rows),
        "v1_v2_per_category.csv": pd.DataFrame(per_category_rows),
        "v1_v2_none_detection.csv": pd.DataFrame(none_rows),
        "v1_v2_paired_cases.csv": pd.DataFrame(paired_case_rows),
        "v1_v2_selection_diagnostic.csv": pd.DataFrame(selection_rows),
        "v1_v2_warnings.csv": pd.DataFrame({"warning": warnings}),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    for filename, frame in frames.items():
        frame.to_csv(args.outdir / filename, index=False, na_rep="")
    write_tables(args.outdir, version_rows, paired_rows, per_category_rows, expert_counts)
    write_summary(args.outdir, expert_counts, expert_none_n, version_rows, paired_rows,
                  coverage_rows, selection_rows, warnings)
    print(f"Expert reference: n={len(ids)}, NONE={expert_none_n}, support={expert_counts}")
    for model, v1_name, v2_name in [("claude", "claude_v1", "claude_v2"), ("deepseek", "deepseek_v1", "deepseek_v2")]:
        uncertain = sum(record["is_uncertain"] for record in data[v2_name].values())
        print(f"{model}: shared V1 n={len(data[v1_name])}; final V2 n={len(data[v2_name])}; UNCERTAIN={uncertain}; coverage={1-uncertain/len(ids):.2f}")
    print(f"Outputs written under {args.outdir}")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ValueError, FileNotFoundError) as error:
        print(f"EVALUATION FAILED: {error}", file=sys.stderr)
        raise
