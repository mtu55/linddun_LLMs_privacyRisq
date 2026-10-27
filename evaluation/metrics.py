"""Explicit metric calculations for the paper evaluation (no sklearn)."""

import numpy as np


def wilson_interval(successes, total, z=1.959963984540054):
    """Return a two-sided Wilson score interval for a binomial rate (R2, R5, R7)."""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, center - half_width)
    high = 1.0 if successes == total else min(1.0, center + half_width)
    return low, high


def binary_counts(reference, prediction):
    """Compute TP/FP/FN/TN for one category, retaining all supplied rows (R1, R6, R7)."""
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    assert reference.shape == prediction.shape, "Reference and prediction shapes differ"
    tp = int(np.sum(reference & prediction))
    fp = int(np.sum(~reference & prediction))
    fn = int(np.sum(reference & ~prediction))
    tn = int(np.sum(~reference & ~prediction))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def binary_metrics(counts):
    """Derive precision, recall and F1 with explicit zero-denominator notes (R1, R6, R7)."""
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = tp / precision_denominator if precision_denominator else 0.0
    recall = tp / recall_denominator if recall_denominator else 0.0
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else 0.0
    notes = []
    if not precision_denominator:
        notes.append("precision denominator is zero; precision set to 0")
    if not recall_denominator:
        notes.append("recall denominator is zero; recall set to 0")
    return {"precision": precision, "recall": recall, "F1": f1, "note": "; ".join(notes)}


def jaccard_and_exact(reference, prediction):
    """Return mean per-incident Jaccard and exact match; both empty sets score 1 (R6, R7)."""
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    assert reference.shape == prediction.shape, "Reference and prediction matrices differ"
    assert reference.ndim == 2, "Set metrics require two-dimensional label matrices"
    intersections = np.sum(reference & prediction, axis=1)
    unions = np.sum(reference | prediction, axis=1)
    per_case_jaccard = np.ones(len(reference), dtype=float)
    nonempty_union = unions > 0
    per_case_jaccard[nonempty_union] = (
        intersections[nonempty_union] / unions[nonempty_union]
    )
    exact = np.all(reference == prediction, axis=1)
    return float(np.mean(per_case_jaccard)), int(np.sum(exact)), int(len(exact))


def random_baseline(reference, primary_indices, simulations=1000, seed=20261019):
    """Simulate independent category draws at expert prevalence and summarize all metrics (R1, R6, R7)."""
    reference = np.asarray(reference, dtype=bool)
    assert reference.ndim == 2, "Random baseline reference must be a matrix"
    assert len(reference) > 0, "Random baseline reference cannot be empty"
    assert simulations > 0, "Random baseline simulations must be positive"
    probabilities = np.mean(reference, axis=0)
    rng = np.random.default_rng(seed)
    category_simulations = [[] for _ in range(reference.shape[1])]
    aggregate_rows = []

    for _ in range(simulations):
        prediction = rng.random(reference.shape) < probabilities
        per_category = []
        tp_total = fp_total = fn_total = 0
        for category_index in range(reference.shape[1]):
            counts = binary_counts(reference[:, category_index], prediction[:, category_index])
            metrics = binary_metrics(counts)
            per_category.append(metrics["F1"])
            tp_total += counts["TP"]
            fp_total += counts["FP"]
            fn_total += counts["FN"]
            category_simulations[category_index].append(
                {
                    "category_index": category_index,
                    **counts,
                    **metrics,
                    "expert_positive_count": int(np.sum(reference[:, category_index])),
                }
            )
        jaccard, exact_count, total = jaccard_and_exact(reference, prediction)
        micro_denominator = 2 * tp_total + fp_total + fn_total
        aggregate_rows.append(
            {
                "macro_f1_primary": float(np.mean([per_category[i] for i in primary_indices])),
                "macro_f1_all_seven": float(np.mean(per_category)),
                "micro_f1": (
                    2 * tp_total / micro_denominator if micro_denominator else 0.0
                ),
                "jaccard": jaccard,
                "exact_match": exact_count / total,
                "coverage": 1.0,
                "exact_match_count": exact_count,
                "n_scored": total,
            }
        )
    return category_simulations, aggregate_rows


def summarize_simulations(rows, metric_columns):
    """Compute simulation means and 2.5/97.5 percentiles for each metric (R7)."""
    result = {}
    for column in metric_columns:
        values = np.asarray([row[column] for row in rows], dtype=float)
        result[column] = float(np.mean(values))
        result[f"{column}_p2_5"] = float(np.percentile(values, 2.5))
        result[f"{column}_p97_5"] = float(np.percentile(values, 97.5))
    return result
