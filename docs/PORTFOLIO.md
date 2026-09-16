# Portfolio description

## 30-second introduction

Premier League Analytics Lab combines football data engineering, SQL analytics and
chronological statistical/ML evaluation in a six-page Streamlit app. It compares
player profiles, estimates editorial market values and predicts match probabilities.
Its strongest result is a time-weighted Dixon–Coles model with 1.0263 final log loss
versus 1.0844 for league base rates. The project also reports what did not work:
fragile low-minute valuation estimates and an ML pilot that trails simpler goal models.

## CV entry — English

**Premier League Analytics Lab — Python, PostgreSQL, statistics and ML**

- Built reproducible ingestion with immutable SHA-256 archives, provenance, idempotent
  PostgreSQL loading and SQL marts using CTEs, window functions and prior-match histories.
- Implemented per-90 normalization, peer percentiles, shrinkage, bootstrap uncertainty
  and scouting similarity; evaluated valuation regressions and Elo/Poisson/Dixon–Coles,
  logistic and XGBoost match models with leakage tests and chronological holdouts.
- Delivered a six-page Streamlit dashboard, automated CI and a non-root Docker setup;
  144 local tests passed with 87.05% statement coverage, plus separate PostgreSQL CI checks.

## Opis do CV — polski

**Premier League Analytics Lab — Python, PostgreSQL, statystyka i ML**

- Zbudowałem odtwarzalny proces pozyskiwania danych z kontrolą SHA-256, metadanymi
  źródeł oraz idempotentnym ładowaniem do PostgreSQL i warstwą analityczną SQL.
- Opracowałem profile zawodników, normalizację per 90 minut, percentyle, estymację
  niepewności i wyszukiwanie podobnych profili; porównałem modele wycen oraz modele
  Elo, Poissona, Dixona–Colesa, regresję logistyczną i XGBoost na podziałach czasowych.
- Przygotowałem dashboard Streamlit, CI i kontener Docker; lokalnie 144 testy zakończyły
  się powodzeniem przy pokryciu kodu 87,05%, a PostgreSQL jest sprawdzany w CI.

## LinkedIn draft

I built Premier League Analytics Lab to connect reproducible data engineering with
honest model evaluation. The project includes PostgreSQL analytics, player profiles,
market-value research and match probability models in Streamlit.

The main lesson: complexity is not an outcome. Time-weighted Dixon–Coles performed
well against base rates, while the new 40-match ML pilot did worse than the statistical
benchmark. Valuation MAE looked promising, but RMSE exposed serious extrapolation
failures. Those limitations are visible in the dashboard and model cards.

Repository: https://github.com/kubaborowiec7/Premier-League-Analytics-Lab
Demo is local only; screenshots and reproduction commands are in the README.

These are drafts for the repository owner to adapt; no social post has been published.
