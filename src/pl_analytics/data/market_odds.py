"""Bookmaker odds from the archived Football-Data.co.uk result files.

The results adapter (:mod:`pl_analytics.data.football_data`) deliberately ignores these
columns and is frozen with the match experiments, so odds are read here, separately.
They are a benchmark and a price reference only: no frozen model uses them as a feature.
Match identities are rebuilt with the same rule as the results adapter.
"""

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd

from pl_analytics.data.contracts import Scope
from pl_analytics.data.football_data import SOURCE, _club_id
from pl_analytics.data.match_history import _latest_mutable_snapshot
from pl_analytics.data.snapshots import Snapshot

# book -> source columns for home, draw, away decimal odds.
OUTCOME_BOOKS = {
    "avg_open": ("AvgH", "AvgD", "AvgA"),
    "avg_close": ("AvgCH", "AvgCD", "AvgCA"),
    "max_close": ("MaxCH", "MaxCD", "MaxCA"),
    "pinnacle_close": ("PSCH", "PSCD", "PSCA"),
    "bet365_close": ("B365CH", "B365CD", "B365CA"),
}
# book -> source columns for over and under 2.5 goals.
TOTALS_BOOKS = {
    "avg_close": ("AvgC>2.5", "AvgC<2.5"),
    "max_close": ("MaxC>2.5", "MaxC<2.5"),
}
OUTCOME_SIDES = ("home", "draw", "away")


def parse_odds(snapshot: Snapshot, scope: Scope) -> pd.DataFrame:
    """Return one row per match with decimal odds; absent or invalid prices stay missing."""
    snapshot.verify()
    try:
        source = pd.read_csv(snapshot.path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        source = pd.read_csv(snapshot.path, encoding="latin-1")
    required = {"Div", "HomeTeam", "AwayTeam"}
    if not required <= set(source):
        raise ValueError("Odds file is missing match identity columns")
    source = source.loc[
        source.Div.eq(scope.source_code) & source.HomeTeam.notna() & source.AwayTeam.notna()
    ]
    home = source.HomeTeam.astype(str).str.strip()
    away = source.AwayTeam.astype(str).str.strip()
    result = pd.DataFrame(
        {
            "match_id": [
                "fd:match:"
                + hashlib.sha256(
                    f"{scope.competition_id}\0{scope.season}\0"
                    f"{_club_id(h, scope.country)}\0{_club_id(a, scope.country)}".encode()
                ).hexdigest()[:32]
                for h, a in zip(home, away, strict=True)
            ],
            "competition_id": scope.competition_id,
            "season": scope.season,
        }
    )
    columns = {}
    for book, names in OUTCOME_BOOKS.items():
        columns.update(
            {f"{book}_{side}": name for side, name in zip(OUTCOME_SIDES, names, strict=True)}
        )
    for book, (over, under) in TOTALS_BOOKS.items():
        columns.update({f"{book}_over25": over, f"{book}_under25": under})
    for target, name in columns.items():
        values = pd.to_numeric(
            source.get(name, pd.Series(float("nan"), index=source.index)), errors="coerce"
        )
        # Decimal odds are strictly above 1; anything else is a source error, not a price.
        result[target] = values.where(values.gt(1.0)).to_numpy()
    if result.match_id.duplicated().any():
        raise ValueError("Duplicate match identities in the odds file")
    return result


def read_market_odds(config: dict, raw_dir: Path, *, end_date: str) -> pd.DataFrame:
    """Read odds for every configured season already archived locally; never downloads."""
    frames = []
    for season in config["seasons"]:
        if season["start_date"] > end_date:
            continue
        sidecar = raw_dir / SOURCE / season["sha256"] / "metadata.json"
        if sidecar.exists():
            snapshot = Snapshot.read(sidecar)
        elif season.get("mutable_current"):
            snapshot = _latest_mutable_snapshot(
                raw_dir, f"{config['source_code']}-{season['source_season']}"
            )
        else:
            snapshot = None
        if snapshot is None:
            raise FileNotFoundError(
                f"Missing archived season {season['season']}; run the match pipeline first"
            )
        scope = Scope(
            config["competition_id"],
            config["competition_name"],
            config["country"],
            config["source_code"],
            season["season"],
            date.fromisoformat(season["start_date"]),
            date.fromisoformat(season["end_date"]),
        )
        frames.append(parse_odds(snapshot, scope))
    if not frames:
        raise ValueError("No odds in the requested period")
    result = pd.concat(frames, ignore_index=True)
    if result.match_id.duplicated().any():
        raise ValueError("Duplicate match identities across seasons")
    return result
