# Notebooks

M3 implements `01_data_quality.ipynb` over the pinned M1 development sample. Acquire
the archives explicitly with `python scripts/verify_ingestion.py`, then run
`python scripts/execute_quality_notebook.py` from the repository root. No database
is needed. The runner starts a fresh kernel, fails on cell errors, and saves an
executed copy under ignored `artifacts/`; committed outputs remain cleared.
See [the data-quality report](../docs/DATA_QUALITY.md) for findings and scope caveats.

Recommended sequence:

- `01_data_quality.ipynb`
- `02_player_eda.ipynb`
- `03_player_statistics.ipynb`
- `04_market_value_modelling.ipynb`
- `05_match_eda.ipynb`
- `06_poisson_dixon_coles.ipynb`
- `07_match_ml.ipynb`
- `08_model_comparison.ipynb`

Keep reusable functions in `src/pl_analytics/`.
