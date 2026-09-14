# Data directories

- `raw/` — immutable downloaded/source snapshots; not committed.
- `interim/` — cleaned intermediate data; not committed.
- `processed/` — model-ready data; not committed.

Only small synthetic/test fixtures should be committed.

Every source snapshot should be registered in the database provenance table and include a checksum.
