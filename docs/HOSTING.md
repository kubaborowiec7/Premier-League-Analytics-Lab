# Private hosting

The app can run online for its owner without a computer switched on. It must stay
private: the prepared data comes from sources licensed for local use only (see
[DATA_SOURCES.md](DATA_SOURCES.md)), and the Betting Lab is not meant for the public.

## How it fits together

1. **This repository (public)** holds the code only. Prepared data is ignored by Git.
2. **A private repository** holds a deployment copy: the app code plus the prepared data
   files the dashboard reads (about 7 MB). `scripts/publish_private_app.py` writes it.
3. **Streamlit Community Cloud** runs the app from the private repository and restricts
   viewers to invited accounts. It redeploys whenever the private repository changes.
4. **The daily refresh workflow** rebuilds current data and, when configured, publishes
   it to the private repository, so the hosted app updates itself.

## One-time setup

1. Create an empty **private** repository on GitHub, for example `pl-analytics-app`.
2. From this repository, publish the first copy (it uses your normal Git login):

   ```bash
   python scripts/publish_private_app.py --remote https://github.com/<you>/pl-analytics-app.git
   ```

   The script refuses to push to a public repository or back to this one.
3. On <https://share.streamlit.io>, sign in with GitHub, choose **Create app**, pick the
   private repository, branch `main` and main file `app/Home.py`. Under advanced settings
   choose Python 3.12. An app from a private repository is private; add your own email
   under **Share** if it is not already listed.
4. Open the app's address on the phone and use the browser's **Add to Home Screen**.

## Automatic daily updates (optional)

Without this the hosted data only changes when step 2 is run again from a computer.

1. Create a fine-grained personal access token limited to the private repository with
   **Contents: read and write**.
2. In this repository's settings add:
   - secret `DEPLOY_REMOTE` =
     `https://x-access-token:<token>@github.com/<you>/pl-analytics-app.git`
   - variable `FPL_ENTRY_ID` = your public FPL team ID (optional).
3. The **Refresh current football data** workflow then publishes after every run. It only
   rebuilds current data (FPL, league, live match models, advanced profiles); the frozen
   model artifacts in the private repository stay as first published.

## What differs when hosted

- The bet log is a file on the host, which a restart or redeploy wipes. Use **Back up or
  restore the log** in the Bet log tab.
- Team and mini-league lookups are sent from the host. They use the same public
  endpoints and no credentials.
- The market benchmark and the frozen experiments are not rebuilt by the workflow; rerun
  step 2 from a computer after rebuilding them locally.
