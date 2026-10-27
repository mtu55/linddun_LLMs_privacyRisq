"""Input loading and strict normalization for the evaluation pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd


CATEGORIES = ["L", "I", "Nr", "D", "DD", "U", "Nc"]
EXPECTED_ROWS = 100
ROOT = Path(__file__).resolve().parents[1]

CONDITION_PATHS = {
    "expert": ROOT / "data/sample_experts.csv",
    "privacyrisq": ROOT / "data/sample_ref.csv",
    "claude_v2": ROOT / "data/llm_run/runs/v2_two_step/main/claude/labels.csv",
    "deepseek_v2": ROOT / "data/llm_run/runs/v2_two_step/main/deepseek/labels.csv",
}
V1_PATHS = {
    "claude_v1": ROOT / "data/llm_run/runs/v1_single_step/main/claude/labels.csv",
    "deepseek_v1": ROOT / "data/llm_run/runs/v1_single_step/main/deepseek/labels.csv",
}


def _positive_marker(value):
    """Normalize a binary label marker (R1, R4, R6)."""
    if pd.isna(value):
        return False
    return str(value).strip().casefold() == "x"


def read_condition(path, condition, require_outcome=False, is_expert=False):
    """Read and validate one condition, including explicit NONE semantics (R1-R3, R5)."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {condition} input: {path}")
    frame = pd.read_csv(
        path,
        sep=";",
        dtype={"Incident ID": "string"},
        keep_default_na=True,
    )
    required = ["Incident ID", *CATEGORIES]
    if require_outcome:
        required.append("Outcome")
    missing = [column for column in required if column not in frame.columns]
    assert not missing, f"{condition}: missing required columns: {missing}"

    ids = frame["Incident ID"]
    assert not ids.isna().any(), f"{condition}: Incident ID contains missing values"
    assert ids.map(lambda value: bool(str(value).strip())).all(), (
        f"{condition}: Incident ID contains an empty/whitespace value"
    )
    frame["Incident ID"] = ids.astype(str)
    duplicates = frame.loc[frame["Incident ID"].duplicated(keep=False), "Incident ID"].tolist()
    assert not duplicates, f"{condition}: duplicate Incident ID values: {duplicates[:10]}"
    assert len(frame) == EXPECTED_ROWS, (
        f"{condition}: expected {EXPECTED_ROWS} rows, found {len(frame)}"
    )

    normalized = pd.DataFrame({"Incident ID": frame["Incident ID"]})
    for category in CATEGORIES:
        normalized[category] = frame[category].map(_positive_marker).astype(bool)

    if require_outcome:
        outcomes = frame["Outcome"].fillna("").astype(str).str.strip().str.upper()
        allowed = {"NONE", "UNCERTAIN", "LABELS"}
        invalid = sorted(set(outcomes) - allowed)
        assert not invalid, f"{condition}: unexpected Outcome values: {invalid}"
        assert outcomes.ne("").all(), f"{condition}: Outcome contains an empty value"
        normalized["Outcome"] = outcomes
        uncertain_with_labels = normalized.loc[
            (normalized["Outcome"] == "UNCERTAIN")
            & normalized[CATEGORIES].any(axis=1),
            "Incident ID",
        ].tolist()
        assert not uncertain_with_labels, (
            f"{condition}: UNCERTAIN co-occurs with assigned labels for IDs: "
            f"{uncertain_with_labels[:10]}"
        )
        normalized["is_abstention"] = normalized["Outcome"].eq("UNCERTAIN")
        normalized["is_none"] = normalized["Outcome"].eq("NONE")
    elif is_expert:
        # R3: only the expert file uses an empty label row to define NONE.
        normalized["is_none"] = ~normalized[CATEGORIES].any(axis=1)
        normalized["is_abstention"] = False
        none_count = int(normalized["is_none"].sum())
        assert none_count == 50, (
            f"expert: expected 50 all-blank label rows for NONE (R3), found {none_count}"
        )
    else:
        # R4: empty PrivacyRISQ sets are not interpreted as explicit NONE.
        normalized["is_none"] = False
        normalized["is_abstention"] = False

    return normalized.set_index("Incident ID", drop=True)


def load_all():
    """Load the four evaluation conditions and enforce exact 100-ID coverage (R1-R7)."""
    frames = {}
    for condition, path in CONDITION_PATHS.items():
        frames[condition] = read_condition(
            path,
            condition,
            require_outcome=condition in {"claude_v2", "deepseek_v2"},
            is_expert=condition == "expert",
        )

    expert_ids = set(frames["expert"].index)
    for condition, frame in frames.items():
        missing = sorted(expert_ids - set(frame.index))
        extra = sorted(set(frame.index) - expert_ids)
        assert set(frame.index) == expert_ids, (
            f"{condition}: ID set differs from expert; missing={missing[:10]}, "
            f"extra={extra[:10]}"
        )
        assert len(frame) == EXPECTED_ROWS, (
            f"{condition}: expected {EXPECTED_ROWS} loaded rows, found {len(frame)}"
        )

    print("Rows loaded per condition:")
    for condition, frame in frames.items():
        print(f"  {condition}: {len(frame)}")
    print("All four conditions cover the identical set of 100 Incident IDs.")

    joined = pd.DataFrame(index=frames["expert"].index)
    joined.index.name = "Incident ID"
    for condition, frame in frames.items():
        for column in [*CATEGORIES, "is_none", "is_abstention"]:
            joined[f"{condition}__{column}"] = frame[column].reindex(joined.index)
        if "Outcome" in frame.columns:
            joined[f"{condition}__Outcome"] = frame["Outcome"].reindex(joined.index)
    return joined, frames


def load_v1_descriptives():
    """Load V1 files for descriptive label counts only; do not score them (R1, R7)."""
    frames = {}
    for condition, path in V1_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required descriptive V1 input: {path}")
        frame = pd.read_csv(
            path,
            sep=";",
            dtype={"Incident ID": "string"},
            keep_default_na=True,
        )
        required = ["Incident ID", *CATEGORIES]
        missing = [column for column in required if column not in frame.columns]
        assert not missing, f"{condition}: missing required columns: {missing}"
        assert not frame["Incident ID"].isna().any(), f"{condition}: missing Incident ID"
        frame["Incident ID"] = frame["Incident ID"].astype(str)
        assert not frame["Incident ID"].duplicated().any(), f"{condition}: duplicate Incident ID"
        assert len(frame) == EXPECTED_ROWS, (
            f"{condition}: expected {EXPECTED_ROWS} rows, found {len(frame)}"
        )
        normalized = pd.DataFrame({"Incident ID": frame["Incident ID"]})
        for category in CATEGORIES:
            normalized[category] = frame[category].map(_positive_marker).astype(bool)
        frames[condition] = normalized.set_index("Incident ID")
    return frames
