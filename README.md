# PrivacyRISQ LINDDUN evaluation — minimal reproducible package

This folder contains the data, evaluation code, result tables, and LINDDUN
reference files needed to inspect and reproduce the 100-incident V1/V2
evaluation reported in the paper.

## Structure

```text
data/                       Input data and model labels
evaluation/                 Evaluation code and frozen result tables
linddun/                    LINDDUN taxonomy and anchor files
```

## Data

- `data/raw.csv` — upstream incident table.
- `data/sample_experts.csv` — expert reference labels for the 100 sampled
  incidents.
- `data/sample_ref.csv` — PrivacyRISQ reference labels for the same incidents.
- `data/sample_work.csv` — working table with incident IDs and metadata.
- `data/llm_run/out/main/{claude,deepseek}/labels.csv` — shared V1 labels.
- `data/llm_run/runs/v1_single_step/main/{claude,deepseek}/labels.csv` —
  versioned copy of the V1 labels, used to verify that both V1 copies agree.
- `data/llm_run/runs/v2_two_step/main/{claude,deepseek}/labels.csv` — final V2
  labels, including explicit `Outcome` values (`LABELS`, `NONE`, `UNCERTAIN`).

The label files are semicolon-delimited and use `Incident ID` as the exact join
key. The V2 files additionally contain prompt version/hash and run metadata
columns.

## Results

`evaluation/results/` contains the frozen tables underlying the paper:

- `descriptives.csv` / `.tex`
- `eq1_none_confusion.csv` / `.tex`
- `eq2_abstention.csv` / `.tex`
- `eq3_aggregates.csv` / `.tex`
- `eq3_per_category.csv` / `.tex`
- `v1_v2_paired_cases.csv`
- `v1_v2_paired_changes.csv`
- `v1_v2_paired_table.tex`
- `v1_v2_version_specific.csv`
- `run_log.txt`

These files are copied from the evaluation run and are not regenerated during
packaging. Re-running the scripts will overwrite them.

## Reproduction

Requirements: Python 3, `pandas`, and `numpy`. No scikit-learn dependency is
used.

From this folder:

```bash
python3 evaluation/run_evaluation.py
python3 evaluation/compare_versions.py
```

`run_evaluation.py` computes the NONE, abstention, aggregate, per-category, and
descriptive tables. `compare_versions.py` computes the V1/V2 comparison and
paired bootstrap intervals. Both scripts resolve paths relative to this folder
and validate row counts, incident IDs, label schemas, and V2 outcomes before
writing results.

The paired bootstrap uses 10,000 incident-level resamples with seed `20261020`;
the random baseline uses 1,000 simulations with seed `20261019`.

## Notes

- The package is anonymized for review. Run manifests, state files, raw model
  logs, rationales, and error dumps are intentionally not included.
- Full incident texts are included in the data and label CSVs. Check the
  upstream data terms before public redistribution.
- The files in `linddun/` are derived from the LINDDUN privacy threat modeling
  framework. Attribution to the LINDDUN authors/KU Leuven should be retained;
  no separate license header was present in the source files.
- `linddun/trees.py` writes `linddun_anchors.yaml` into the current working
  directory when run. The packaged anchor file is already provided, so the
  script does not need to be executed for evaluation.
