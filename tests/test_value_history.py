"""Small published-schema fixtures test the historical adapter without downloads."""

from types import SimpleNamespace

import pandas as pd
import pytest

from pl_analytics.data.value_history import read_value_history


@pytest.fixture
def archives(tmp_path, monkeypatch):
    frames = {
        "games": pd.DataFrame(
            {
                "game_id": ["1", "2"],
                "competition_id": ["SRC", "SRC"],
                "date": ["2020-01-01", "2023-01-01"],
                "home_club_id": ["a", "a"],
                "away_club_id": ["b", "b"],
            }
        ),
        "appearances": pd.DataFrame(
            {
                "game_id": ["1", "2"],
                "player_id": ["7", "7"],
                "player_club_id": ["a", "a"],
                "date": ["2020-01-01", "2023-01-01"],
                "minutes_played": [90, 90],
                "goals": [1, 2],
                "assists": [0, 1],
            }
        ),
        "players": pd.DataFrame(
            {
                "player_id": ["7"],
                "name": ["Player"],
                "date_of_birth": ["2000-01-01"],
                "current_club_id": ["future"],
            }
        ),
        "player_valuations": pd.DataFrame(
            {
                "player_id": ["7", "7"],
                "date": ["2020-01-02", "2023-01-02"],
                "market_value_in_eur": [100, 200],
                "player_club_domestic_competition_id": ["OTHER", "OTHER"],
            }
        ),
    }
    config = dict(
        competition_id="CANON",
        source_competition_code="SRC",
        history_start="2019-01-01",
        sha256={table: "a" * 64 for table in frames},
    )
    snapshots = {}
    for table, frame in frames.items():
        folder = tmp_path / f"transfermarkt-datasets.{table}" / ("a" * 64)
        folder.mkdir(parents=True)
        sidecar = folder / "metadata.json"
        sidecar.touch()
        path = folder / "payload.csv.gz"
        frame.to_csv(path, index=False)
        snapshots[sidecar] = SimpleNamespace(
            path=path,
            source_name=f"transfermarkt-datasets.{table}",
            sha256="a" * 64,
            source_url="https://example.test/data",
            retrieved_at="2026-01-01",
        )
    monkeypatch.setattr("pl_analytics.data.value_history.Snapshot.read", snapshots.__getitem__)
    return config, tmp_path, snapshots


def test_adapter_limits_period_maps_scope_and_ignores_current_club(archives):
    config, root, _ = archives
    apps, values, profiles, sources = read_value_history(config, root, end_date="2020-12-31")
    assert len(apps) == len(values) == 1
    assert apps.iloc[0].competition_id == "CANON"
    assert values.iloc[0].player_id == "tm:player:7"
    assert "current_club_id" not in profiles
    assert "player_club_domestic_competition_id" not in values
    assert len(sources) == 4


@pytest.mark.parametrize("failure", ["source", "hash", "club", "duplicate_value"])
def test_adapter_rejects_inconsistent_history(archives, failure):
    config, root, snapshots = archives
    snapshot = next(iter(snapshots.values()))
    if failure == "source":
        snapshot.source_name = "wrong"
    elif failure == "hash":
        snapshot.sha256 = "b" * 64
    elif failure == "club":
        app = next(s for s in snapshots.values() if s.source_name.endswith(".appearances"))
        frame = pd.read_csv(app.path)
        frame.loc[0, "player_club_id"] = "outsider"
        frame.to_csv(app.path, index=False)
    else:
        value = next(s for s in snapshots.values() if s.source_name.endswith(".player_valuations"))
        frame = pd.read_csv(value.path)
        duplicate = frame.iloc[:1].assign(market_value_in_eur=999)
        pd.concat([frame, duplicate]).to_csv(value.path, index=False)
    with pytest.raises(ValueError):
        read_value_history(config, root, end_date="2020-12-31")
