"""Build M4 player artifacts from pinned published appearance data."""

import argparse
import json
import logging
import re
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np

from pl_analytics.config import get_settings
from pl_analytics.data.appearances import COUNTS, parse_appearances
from pl_analytics.data.contracts import Scope
from pl_analytics.data.snapshots import Snapshot, SnapshotStore
from pl_analytics.data.transfermarkt import PUBLISHED_BASE_URL, TERMS
from pl_analytics.features.players import aggregate_players, fit_peers
from pl_analytics.statistics.players import bootstrap_per90, fit_player_space, similar_players

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    """Acquisition is opt-in; default execution requires the immutable local archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/m4_sources.json"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    scope = Scope(
        config["competition_id"],
        config["competition_name"],
        config["country"],
        config["source_competition_code"],
        config["season"],
        date.fromisoformat(config["start_date"]),
        date.fromisoformat(config["end_date"]),
    )
    if scope.competition_id not in settings.active_competitions:
        raise ValueError("Requested competition is not enabled")
    snapshots = []
    store = SnapshotStore(settings.data_dir / "raw")
    for table in ("games", "appearances", "players"):
        digest = config["sha256"][table]
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Each source requires a valid pinned SHA-256")
        source = f"transfermarkt-datasets.{table}"
        sidecar = store.raw_dir / source / digest / "metadata.json"
        if sidecar.exists():
            snapshot = Snapshot.read(sidecar)
        elif args.download:
            snapshot = store.download(
                f"{PUBLISHED_BASE_URL}/{table}.csv.gz",
                source_name=source,
                dataset_version="sha256:" + digest,
                license_note=TERMS,
                expected_sha256=digest,
                max_bytes=256 * 1024 * 1024,
            )
        else:
            raise FileNotFoundError(f"Missing {source}; acquire explicitly with --download")
        if snapshot.source_name != source or snapshot.sha256 != digest:
            raise ValueError("Source identity does not match manifest")
        snapshots.append(snapshot)
    appearances = parse_appearances(*snapshots, scope=scope, source_season=config["source_season"])
    raw = aggregate_players(appearances, config["reference_date"])
    fitted = fit_peers(raw)
    players = fitted.transform(raw)
    features = tuple(f"{metric}_zscore" for metric in COUNTS)
    eligible = players.loc[players.ranking_eligible & players.position_group.eq("ST")].copy()
    # The example is explicitly a striker cohort; model APIs have no fixed position/league.
    features = tuple(feature for feature in features if eligible[feature].std() > 0)
    eligible = eligible.dropna(subset=list(features))
    if len(features) < 2 or len(eligible) < 3:
        raise ValueError("Not enough complete striker observations for the documented PCA example")
    space = fit_player_space(eligible, features=features)
    projected = space.transform(eligible)
    example = eligible.sort_values(
        ["goals", "minutes", "player_id"], ascending=[False, False, True]
    ).iloc[0]
    neighbors = similar_players(
        example, players, candidate_competitions=(scope.competition_id,), features=features
    )
    history = appearances.loc[appearances.player_id.eq(example.player_id)]
    intervals = {
        metric: bootstrap_per90(history, metric, reference_date=config["reference_date"])
        for metric in COUNTS
    }
    counts = {
        "matches": int(appearances.match_id.nunique()),
        "appearances": len(appearances),
        "players": len(players),
        "eligible_players": int(players.ranking_eligible.sum()),
        "striker_pca_players": len(eligible),
    }
    for key, expected in config.get("expected_counts", {}).items():
        if counts[key] != expected:
            raise ValueError(f"Unexpected {key}: {counts[key]} != {expected}")
    destination = settings.artifact_dir / "m4"
    processed = settings.data_dir / "processed/m4"
    destination.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    appearances.to_parquet(processed / "appearances.parquet", index=False)
    players.to_parquet(processed / "player_features.parquet", index=False)
    projected.to_parquet(processed / "player_pca.parquet", index=False)
    # pandas serialization converts undefined statistics to JSON null.
    report = {
        "scope": {
            "competition_id": scope.competition_id,
            "season": scope.season,
            "reference_date": config["reference_date"],
        },
        "counts": counts,
        "parameters": {
            "min_minutes": fitted.min_minutes,
            "prior_minutes": fitted.prior_minutes,
            "min_peers": fitted.min_peers,
            "bootstrap_seed": 42,
        },
        "sources": [
            {
                "source": s.source_name,
                "sha256": s.sha256,
                "retrieved_at": s.retrieved_at,
                "url": s.source_url,
            }
            for s in snapshots
        ],
        "example": json.loads(example.to_frame().T.to_json(orient="records"))[0],
        "intervals": intervals,
        "similar_players": json.loads(
            neighbors[
                [
                    "player_id",
                    "player_name",
                    "competition_id",
                    "season",
                    "position_group",
                    "distance",
                ]
            ].to_json(orient="records")
        ),
        "pca": {
            "features": features,
            "explained_variance_ratio": space.pca.explained_variance_ratio_.tolist(),
            "components": space.pca.components_.tolist(),
            "scaler_mean": space.scaler.mean_.tolist(),
            "scaler_scale": space.scaler.scale_.tolist(),
        },
        "limitations": [
            "Snapshot position labels are historically unverified",
            "Only goals, assists and cards; not a complete player-quality model",
            "Bootstrap intervals describe observed match variation, not future guarantees",
            "Source identities remain separate from Football-Data canonical matches",
        ],
    }
    (destination / "profile.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    metrics = ["goals", "assists"]
    axes[0].bar(metrics, [example[f"{m}_percentile"] for m in metrics], color="#185A80")
    axes[0].set(
        ylim=(0, 100), ylabel="Percentile among eligible strikers", title=example.player_name
    )
    for i, metric in enumerate(metrics):
        interval = intervals[metric]
        estimate = interval["estimate"]
        if interval["lower"] is not None:
            axes[1].vlines(i, interval["lower"], interval["upper"], color="#B15E24", linewidth=3)
        axes[1].scatter(i, estimate, color="#185A80")
    axes[1].set(
        xticks=np.arange(2),
        xticklabels=metrics,
        ylabel="Count per 90 minutes",
        title="Match bootstrap: 95% intervals",
    )
    fig.suptitle(f"{scope.competition_id} {scope.season} · retrospective snapshot positions")
    fig.tight_layout()
    fig.savefig(destination / "player_profile.png", dpi=140)
    plt.close(fig)
    logging.basicConfig(level=logging.INFO)
    logging.info("M4 built: %s; example=%s", counts, example.player_name)


if __name__ == "__main__":
    main()
