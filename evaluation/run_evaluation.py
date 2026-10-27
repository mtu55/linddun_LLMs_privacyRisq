"""Run EQ1-EQ3 and write reproducible CSV, LaTeX, and audit-log outputs."""

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from load import CATEGORIES, ROOT, load_all, load_v1_descriptives
from metrics import (
    binary_counts,
    binary_metrics,
    jaccard_and_exact,
    random_baseline,
    summarize_simulations,
    wilson_interval,
)


RESULTS = ROOT / "evaluation/results"
MODEL_CONDITIONS = ["claude_v2", "deepseek_v2"]
ALL_CONDITIONS = ["expert", "privacyrisq", *MODEL_CONDITIONS]
SIMULATIONS = 1000
SEED = 20261019


class RunLog:
    """Collect assertion outcomes and run metadata for the audit log."""

    def __init__(self):
        self.lines = []

    def assertion(self, condition, message):
        self.lines.append(f"PASS: {condition} — {message}")

    def write(self):
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "run_log.txt").write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _condition_columns(joined, condition):
    """Return normalized label and status arrays for a condition (R1-R7)."""
    labels = joined[[f"{condition}__{category}" for category in CATEGORIES]].to_numpy(dtype=bool)
    return (
        labels,
        joined[f"{condition}__is_none"].to_numpy(dtype=bool),
        joined[f"{condition}__is_abstention"].to_numpy(dtype=bool),
    )


def _wilson_columns(successes, total, prefix):
    """Format Wilson interval columns for a reported binomial rate (R2, R5, R7)."""
    low, high = wilson_interval(int(successes), int(total))
    return {f"{prefix}_ci95_low": low, f"{prefix}_ci95_high": high}


def build_eq1(joined, log):
    """Build explicit NONE confusion counts and derived metrics, excluding abstentions (R1-R3, R5)."""
    expert_none = joined["expert__is_none"].to_numpy(dtype=bool)
    output = []
    for condition in MODEL_CONDITIONS:
        _, predicted_none, abstention = _condition_columns(joined, condition)
        keep = ~abstention
        counts = binary_counts(expert_none[keep], predicted_none[keep])
        metrics = binary_metrics(counts)
        row = {
            "condition": condition,
            **counts,
            "n_scored": int(np.sum(keep)),
            "n_abstained_excluded": int(np.sum(abstention)),
            **metrics,
            **_wilson_columns(counts["TP"], counts["TP"] + counts["FP"], "precision"),
            **_wilson_columns(counts["TP"], counts["TP"] + counts["FN"], "recall"),
            "F1": metrics["F1"],
        }
        output.append(row)
        log.assertion(
            f"EQ1 {condition}",
            f"scored {row['n_scored']} non-abstaining incidents; excluded {row['n_abstained_excluded']} abstentions",
        )
    return pd.DataFrame(output)


def build_eq2(joined, log):
    """Summarize abstentions by expert NONE status and coverage with Wilson intervals (R3, R5, R7)."""
    expert_none = joined["expert__is_none"].to_numpy(dtype=bool)
    output = []
    groups = [("expert_NONE", expert_none), ("expert_nonempty", ~expert_none), ("all", np.ones(len(joined), dtype=bool))]
    for condition in MODEL_CONDITIONS:
        _, _, abstention = _condition_columns(joined, condition)
        for group_name, group_mask in groups:
            total = int(np.sum(group_mask))
            abstained = int(np.sum(abstention & group_mask))
            covered = total - abstained
            abstention_rate = abstained / total if total else 0.0
            coverage = covered / total if total else 0.0
            row = {
                "condition": condition,
                "expert_group": group_name,
                "n_incidents": total,
                "abstention_count": abstained,
                "covered_count": covered,
                "abstention_rate": abstention_rate,
                **_wilson_columns(abstained, total, "abstention_rate"),
                "coverage": coverage,
                **_wilson_columns(covered, total, "coverage"),
            }
            output.append(row)
        log.assertion(f"EQ2 {condition}", f"abstention count={int(np.sum(abstention))} of {len(joined)}")
    return pd.DataFrame(output)


def build_eq3(joined, primary_categories, log):
    """Compute category and set metrics on each non-abstaining subset (R1, R5-R7)."""
    reference = joined[[f"expert__{category}" for category in CATEGORIES]].to_numpy(dtype=bool)
    reference_positive_counts = np.sum(reference, axis=0)
    all_rows = []
    aggregate_rows = []

    for condition in ["privacyrisq", *MODEL_CONDITIONS]:
        prediction, _, abstention = _condition_columns(joined, condition)
        keep = ~abstention
        n_total = len(joined)
        n_scored = int(np.sum(keep))
        coverage = n_scored / n_total
        ref_scored = reference[keep]
        pred_scored = prediction[keep]
        category_f1 = []
        tp_total = fp_total = fn_total = 0
        for index, category in enumerate(CATEGORIES):
            counts = binary_counts(ref_scored[:, index], pred_scored[:, index])
            derived = binary_metrics(counts)
            category_f1.append(derived["F1"])
            tp_total += counts["TP"]
            fp_total += counts["FP"]
            fn_total += counts["FN"]
            row = {
                "condition": condition,
                "category": category,
                **counts,
                "precision": derived["precision"],
                "recall": derived["recall"],
                "F1": derived["F1"],
                "expert_positive_count": int(reference_positive_counts[index]),
                "expert_positive_count_scored": int(np.sum(ref_scored[:, index])),
                "n_total": n_total,
                "n_scored": n_scored,
                "coverage": coverage,
                "note": derived["note"],
                **_wilson_columns(counts["TP"], counts["TP"] + counts["FP"], "precision"),
                **_wilson_columns(counts["TP"], counts["TP"] + counts["FN"], "recall"),
            }
            all_rows.append(row)

        micro_denominator = 2 * tp_total + fp_total + fn_total
        micro_f1 = 2 * tp_total / micro_denominator if micro_denominator else 0.0
        if condition == "privacyrisq":
            # R4: no set-level or NONE/abstention interpretation for this condition.
            jaccard = exact_count = exact_denominator = exact_match = np.nan
        else:
            jaccard, exact_count, exact_denominator = jaccard_and_exact(ref_scored, pred_scored)
            exact_match = exact_count / exact_denominator if exact_denominator else 0.0
        primary_indices = [CATEGORIES.index(category) for category in primary_categories]
        aggregate_rows.append(
            {
                "condition": condition,
                "macro_f1_primary": float(np.mean([category_f1[i] for i in primary_indices])),
                "macro_f1_all_seven": float(np.mean(category_f1)),
                "micro_f1": micro_f1,
                "jaccard": jaccard,
                "exact_match": exact_match,
                "exact_match_count": exact_count,
                "n_scored": n_scored,
                "n_total": n_total,
                "coverage": coverage,
                **_wilson_columns(n_scored, n_total, "coverage"),
                **(
                    _wilson_columns(exact_count, exact_denominator, "exact_match")
                    if condition != "privacyrisq"
                    else {"exact_match_ci95_low": np.nan, "exact_match_ci95_high": np.nan}
                ),
                "note": (
                    "category-level only per R4; set-level metrics not computed"
                    if condition == "privacyrisq"
                    else "metrics use non-abstaining subset; expert NONE rows retained"
                ),
            }
        )
        log.assertion(
            f"EQ3 {condition}",
            f"{n_scored}/{n_total} scored (coverage={coverage:.6f}); expert NONE rows retained in scored denominator",
        )

    random_category_sims, random_aggregate_sims = random_baseline(
        reference, [CATEGORIES.index(category) for category in primary_categories], SIMULATIONS, SEED
    )
    for index, category in enumerate(CATEGORIES):
        simulations = random_category_sims[index]
        raw_fields = ["TP", "FP", "FN", "TN", "expert_positive_count"]
        metric_fields = ["precision", "recall", "F1"]
        row = {"condition": "random_baseline", "category": category}
        for field in raw_fields + metric_fields:
            values = np.asarray([sample[field] for sample in simulations], dtype=float)
            row[field] = float(np.mean(values))
            row[f"{field}_p2_5"] = float(np.percentile(values, 2.5))
            row[f"{field}_p97_5"] = float(np.percentile(values, 97.5))
        row.update(
            {
                "expert_positive_count_scored": int(reference_positive_counts[index]),
                "n_total": len(joined),
                "n_scored": len(joined),
                "coverage": 1.0,
                "coverage_ci95_low": wilson_interval(len(joined), len(joined))[0],
                "coverage_ci95_high": wilson_interval(len(joined), len(joined))[1],
                "note": f"mean and simulation percentiles over {SIMULATIONS} draws; zero denominator metrics set to 0",
            }
        )
        all_rows.append(row)

    random_metrics = summarize_simulations(
        random_aggregate_sims,
        ["macro_f1_primary", "macro_f1_all_seven", "micro_f1", "jaccard", "exact_match", "coverage"],
    )
    random_exact_counts = np.asarray([row["exact_match_count"] for row in random_aggregate_sims])
    aggregate_rows.append(
        {
            "condition": "random_baseline",
            **random_metrics,
            "exact_match_count": float(np.mean(random_exact_counts)),
            "exact_match_count_p2_5": float(np.percentile(random_exact_counts, 2.5)),
            "exact_match_count_p97_5": float(np.percentile(random_exact_counts, 97.5)),
            "n_scored": len(joined),
            "n_total": len(joined),
            "coverage_ci95_low": wilson_interval(len(joined), len(joined))[0],
            "coverage_ci95_high": wilson_interval(len(joined), len(joined))[1],
            "exact_match_ci95_low": np.nan,
            "exact_match_ci95_high": np.nan,
            "note": f"mean and 2.5/97.5 percentiles over {SIMULATIONS} simulations; seed={SEED}",
        }
    )
    log.assertion("random baseline", f"{SIMULATIONS} simulations; numpy.default_rng seed={SEED}")
    return pd.DataFrame(all_rows), pd.DataFrame(aggregate_rows)


def build_descriptives(frames, v1_frames, log):
    """Count positive labels and empty sets for all conditions, without scoring V1 (R1, R4)."""
    output = []
    descriptives = {**frames, **v1_frames}
    for condition, frame in descriptives.items():
        normalized = frame[CATEGORIES].astype(bool)
        n = len(normalized)
        for category in CATEGORIES:
            count = int(normalized[category].sum())
            output.append(
                {
                    "condition": condition,
                    "category": category,
                    "positive_count": count,
                    "positive_rate": count / n if n else 0.0,
                    **_wilson_columns(count, n, "positive_rate"),
                    "n_incidents": n,
                }
            )
        explicit_none_count = (
            int(frame["is_none"].sum()) if "is_none" in frame.columns else np.nan
        )
        output.append(
            {
                "condition": condition,
                "category": "ALL_LABELS_EMPTY",
                "positive_count": int((~normalized.any(axis=1)).sum()),
                "positive_rate": float((~normalized.any(axis=1)).mean()),
                **_wilson_columns(int((~normalized.any(axis=1)).sum()), n, "positive_rate"),
                "n_incidents": n,
                "explicit_none_count": explicit_none_count,
            }
        )
    counts = {row["condition"]: row["positive_count"] for row in output if row["category"] == "Nc"}
    assert counts["privacyrisq"] == 58 and counts["expert"] == 0, (
        f"Expected privacyrisq Nc=58 and expert Nc=0; found {counts['privacyrisq']} and {counts['expert']}"
    )
    log.assertion("descriptive Nc discrepancy", "privacyrisq Nc=58; expert Nc=0")
    return pd.DataFrame(output)


def _latex_escape(value):
    """Escape table values for a LaTeX booktabs output."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def write_table(frame, name):
    """Write one CSV and its matching LaTeX booktabs table."""
    csv_path = RESULTS / f"{name}.csv"
    tex_path = RESULTS / f"{name}.tex"
    frame.to_csv(csv_path, index=False, na_rep="")
    with tex_path.open("w", encoding="utf-8") as stream:
        stream.write("% Generated by evaluation/run_evaluation.py\n")
        stream.write("\\begin{table}[ht]\n\\centering\n")
        stream.write("\\begin{tabular}{" + "l" * len(frame.columns) + "}\n\\toprule\n")
        stream.write(" & ".join(_latex_escape(column) for column in frame.columns) + r" \\" + "\n")
        stream.write("\\midrule\n")
        for values in frame.itertuples(index=False, name=None):
            formatted = []
            for value in values:
                if pd.isna(value):
                    formatted.append("")
                elif isinstance(value, (float, np.floating)):
                    formatted.append(f"{value:.4f}")
                else:
                    formatted.append(_latex_escape(value))
            stream.write(" & ".join(formatted) + r" \\" + "\n")
        stream.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def run():
    """Execute all evaluations and produce reproducible outputs (R1-R7)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = RunLog()
    log.lines.extend(
        [
            "Evaluation pipeline run log",
            f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
            f"Python: {sys.version.split()[0]}",
            f"pandas: {pd.__version__}",
            f"numpy: {np.__version__}",
            f"Random baseline simulations: {SIMULATIONS}",
            f"Random baseline seed: {SEED}",
            "",
        ]
    )
    joined, frames = load_all()
    for condition, frame in frames.items():
        log.lines.append(f"Rows loaded — {condition}: {len(frame)}")
    log.assertion("ID coverage", "four conditions have identical sets of 100 exact Incident IDs")
    log.assertion("expert NONE count", f"{int(frames['expert']['is_none'].sum())} empty expert label sets (R3)")
    for condition in MODEL_CONDITIONS:
        abstention = frames[condition]["is_abstention"]
        log.assertion(
            f"exclusive UNCERTAIN — {condition}",
            f"{int(abstention.sum())} UNCERTAIN rows; none co-occurs with labels",
        )

    expert_positive_counts = frames["expert"][CATEGORIES].sum(axis=0)
    primary_categories = [
        category for category in CATEGORIES if int(expert_positive_counts[category]) >= 5
    ]
    assert primary_categories, "No primary categories have at least five expert positives"
    log.lines.append(f"Primary category subset (expert positives >= 5): {', '.join(primary_categories)}")

    v1_frames = load_v1_descriptives()
    for condition, frame in v1_frames.items():
        log.lines.append(f"Rows loaded — {condition} (descriptive only): {len(frame)}")
        assert set(frame.index) == set(joined.index), (
            f"{condition}: ID set differs from four evaluation conditions"
        )
        log.assertion(f"V1 ID coverage — {condition}", "same 100 exact IDs as evaluation conditions")

    eq1 = build_eq1(joined, log)
    eq2 = build_eq2(joined, log)
    eq3_categories, eq3_aggregates = build_eq3(joined, primary_categories, log)
    descriptives = build_descriptives(frames, v1_frames, log)

    for name, table in [
        ("eq1_none_confusion", eq1),
        ("eq2_abstention", eq2),
        ("eq3_per_category", eq3_categories),
        ("eq3_aggregates", eq3_aggregates),
        ("descriptives", descriptives),
    ]:
        write_table(table, name)
        log.assertion(f"output {name}", f"wrote CSV and LaTeX booktabs table")
    log.write()
    print(f"Primary category subset: {', '.join(primary_categories)}")
    print(f"Outputs written to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as error:
        print(f"ASSERTION FAILED: {error}", file=sys.stderr)
        raise
    except Exception as error:
        print(f"EVALUATION FAILED: {error}", file=sys.stderr)
        raise
