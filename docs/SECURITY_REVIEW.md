# M9 security review — 2026-09-16

Scope: local Streamlit serving, artifact loading, packaging and existing ingestion.
This is a repository review, not a penetration test or a guarantee of no vulnerabilities.

| Boundary | Control / evidence | Residual limitation |
|---|---|---|
| Secrets | `.env` ignored; settings hide validation input; UI errors are sanitized; no database URL passed to app container | Operators must review logs and deployment environment |
| Models | Fixed local paths, SHA-256 verification before cached deserialization, no model uploads; tests reject tampering | Joblib executes code; trusted producer and read-only mounts are required; hashes are not signatures |
| Forecast context | Model and feature origin/competition checked; distinct clubs and unique states required | Future production refresh requires a new evaluated artifact set |
| Network | No UI downloads, fitting or database connection; tests prohibit these calls | Streamlit is a web server and must sit behind access control for remote use |
| Container | Non-root UID, read-only filesystem/mounts, dropped capabilities, loopback host port | Docker image and transitive dependencies are not immutable digest locks |
| Build leakage | Docker context allowlist excludes raw data, secrets, notebooks and `.git` | Review image contents before public distribution |
| Input/display | Fixed page/model-card paths; static CSS only; no HTML built from data; Streamlit escaping | CSV exports contain source names; treat them as untrusted spreadsheet input |
| Data rights | Published datasets with source manifests; no core Transfermarkt/FBref scraper | Public availability does not automatically grant unrestricted redistribution |

Only trusted users should write artifact files. Replacing both a bundle and its
catalog bypasses a checksum-only trust model; deployments must control filesystem
writes. Model accuracy failures are documented separately in model cards and must
not be presented as investment or betting certainty.
