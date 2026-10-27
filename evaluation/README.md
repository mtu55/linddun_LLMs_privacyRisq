# Reproducible evaluation pipeline

Run the pipeline from the repository root:

```bash
python evaluation/run_evaluation.py
```

To compare prompt versions, run:

```bash
python evaluation/compare_versions.py
```

This reads the authoritative expert reference, shared V1 CSVs and author-designated final V2 `labels.csv` files, joins by exact incident ID, validates schemas/outcomes, and writes `v1_v2_*.csv`, compact booktabs LaTeX tables, and `v1_v2_summary.md` under `evaluation/results/`. Optional `--bootstrap-resamples` and `--seed` flags control paired incident-level percentile bootstrap intervals (defaults: 10,000 resamples and seed 20261020). `--v1-claude`, `--v1-deepseek`, `--v2-claude`, `--v2-deepseek`, `--expert`, and `--sample` allow explicit source paths. Duplicate IDs or malformed labels stop execution; missing/invalid records are reported.

Requirements: Python 3, pandas, and NumPy. No scikit-learn dependency is used. The runner reads semicolon-delimited source CSVs using `Incident ID` as an exact string key, normalizes positive category markers (`x` or `X`, case-insensitive and whitespace-stripped), checks all required row/ID constraints, and stops with a clear error if an assertion fails.

The four scored conditions are:

| Condition | Source |
|---|---|
| `expert` | `data/sample_experts.csv` |
| `privacyrisq` | `data/sample_ref.csv` |
| `claude_v2` | `data/llm_run/runs/v2_two_step/main/claude/labels.csv` |
| `deepseek_v2` | `data/llm_run/runs/v2_two_step/main/deepseek/labels.csv` |

The versioned V2 files include the repository's `main/` run-mode directory. The corresponding V1 label files are read for descriptive counts and for the paired V1/V2 comparison. V1 is not evaluated as a standalone full-sample condition.

## Decision rules implemented

- **R1:** `expert` is the sole gold standard.
- **R2:** Model `NONE` requires explicit `Outcome == "NONE"`; an empty label set does not imply `NONE`.
- **R3:** The expert file has no `Outcome`; exactly 50 all-blank category rows are required and treated as expert `NONE`.
- **R4:** Empty `privacyrisq` sets are not `NONE`; this condition contributes only to category-level EQ3.
- **R5:** `Outcome == "UNCERTAIN"` is an exclusive abstention; the runner asserts it has no assigned categories.
- **R6:** Expert `NONE` cases remain negative category examples in EQ3.
- **R7:** EQ3 category and set metrics use the non-abstaining subset, with coverage reported alongside.

The canonical category order everywhere is `L, I, Nr, D, DD, U, Nc`. Precision and recall are zero when their denominators are zero; a note is emitted for affected category rows. Jaccard is 1.0 when both compared sets are empty. Wilson score 95% intervals are reported for binomial rates such as precision, recall, abstention, coverage, and exact match.

The random baseline draws each category independently using its prevalence in the expert reference, with 1,000 simulations and `numpy.random.default_rng(20261019)`. Means and 2.5th/97.5th simulation percentiles are included in the tables.

## Outputs

All outputs are written beneath `evaluation/results/`; each table is written as CSV and a LaTeX `booktabs` table (`.tex`).

- `eq1_none_confusion.csv` / `.tex`: explicit expert `NONE` versus model `NONE` 2×2 confusion counts, plus precision, recall, F1, and denominator/interval information for Claude V2 and DeepSeek V2. Abstentions are excluded.
- `eq2_abstention.csv` / `.tex`: abstention and coverage counts/rates with Wilson intervals, split into expert `NONE`, expert non-empty, and all incidents.
- `eq3_per_category.csv` / `.tex`: per-category TP, FP, FN, TN and derived precision, recall, F1, expert prevalence, coverage, and Wilson intervals. Includes the 1,000-simulation random baseline summary. PrivacyRISQ is evaluated here but is not given NONE/abstention semantics.
- `eq3_aggregates.csv` / `.tex`: primary-subset macro-F1 (categories with at least five expert positives), all-seven macro-F1, micro-F1, mean per-incident Jaccard, exact match, and coverage. Model label/set metrics use non-abstaining incidents; expert NONE cases remain included. Random baseline means and simulation percentiles are also included.
- `descriptives.csv` / `.tex`: positive-label counts/rates and empty-set counts for expert, PrivacyRISQ, Claude V2, DeepSeek V2, and the two V1 conditions. Explicit NONE count is reported separately where available. The PrivacyRISQ/expert `Nc` difference is shown as 58 versus 0.
- `run_log.txt`: timestamp, Python/pandas/NumPy versions, input row counts, successful validation assertions, primary category subset, random seed, and simulation count.

All tables contain a `condition` column. Run the script again to overwrite outputs with the same metric results; the run timestamp in the audit log changes on each invocation.
