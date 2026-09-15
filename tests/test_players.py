"""Synthetic player analytics: exposure, cohort boundaries and temporal leakage."""

import numpy as np
import pandas as pd
import pytest

from pl_analytics.features.players import aggregate_players, fit_peers
from pl_analytics.statistics.players import bootstrap_per90, fit_player_space, similar_players


@pytest.fixture
def appearances() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": f"p{p}",
                "player_name": f"Player {p}",
                "competition_id": "A",
                "season": "2023/24",
                "match_id": f"m{m}",
                "club_id": "c",
                "match_date": f"2024-01-{m + 1:02d}",
                "position_group": "ST",
                "position_context": "synthetic",
                "minutes": 90,
                "goals": p if m == 0 else 0,
                "assists": (p + 1) % 3,
                "yellow_cards": p % 2,
                "red_cards": 0,
            }
            for p in range(6)
            for m in range(3)
        ]
    )


def normalized(appearances: pd.DataFrame) -> pd.DataFrame:
    raw = aggregate_players(appearances, "2024-02-01")
    return fit_peers(raw, min_minutes=90, min_peers=3).transform(raw)


def test_per90_totals_and_temporal_boundary(appearances: pd.DataFrame) -> None:
    result = aggregate_players(appearances, "2024-01-03")
    row = result.loc[result.player_id.eq("p3")].iloc[0]
    assert row.minutes == 180 and row.goals == 3 and row.goals_per90 == 1.5
    changed = appearances.copy()
    changed.loc[changed.match_date.ge("2024-01-03"), "goals"] = 10000
    pd.testing.assert_frame_equal(result, aggregate_players(changed, "2024-01-03"))


def test_missing_and_zero_exposure(appearances: pd.DataFrame) -> None:
    appearances.loc[0, "minutes"] = np.nan
    appearances.loc[appearances.player_id.eq("p1"), "minutes"] = 0
    result = aggregate_players(appearances, "2024-02-01").set_index("player_id")
    assert pd.isna(result.loc["p0", "minutes"])
    assert pd.isna(result.loc["p0", "goals_per90"])
    assert pd.isna(result.loc["p1", "goals_per90"])


def test_invalid_and_duplicate_counts_rejected(appearances: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_players(pd.concat([appearances, appearances.iloc[:1]]), "2024-02-01")
    appearances.loc[0, "goals"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        aggregate_players(appearances, "2024-02-01")


def test_peer_statistics_and_shrinkage(appearances: pd.DataFrame) -> None:
    result = normalized(appearances)
    top = result.loc[result.player_id.eq("p5")].iloc[0]
    assert top.goals_percentile == pytest.approx(100 * 5.5 / 6)
    assert result.goals_zscore.mean() == pytest.approx(0)
    assert result.goals_zscore.std(ddof=0) == pytest.approx(1)
    assert top.reliability_weight == pytest.approx(270 / 1170)
    assert top.goals_shrunk_per90 == pytest.approx((270 / 1170) * 5 / 3 + (900 / 1170) * 2.5 / 3)
    assert result.red_cards_percentile.eq(50).all()
    assert result.red_cards_zscore.isna().all()


def test_small_peers_low_minutes_and_unknown_position(appearances: pd.DataFrame) -> None:
    raw = aggregate_players(appearances, "2024-02-01")
    raw.loc[0, "minutes"] = 20
    result = fit_peers(raw, min_minutes=90, min_peers=3).transform(raw)
    assert pd.isna(result.loc[0, "goals_percentile"])
    assert pd.notna(result.loc[0, "goals_shrunk_per90"])
    assert fit_peers(raw, min_minutes=90, min_peers=8).transform(raw).goals_percentile.isna().all()
    raw["position_group"] = "UNKNOWN"
    assert fit_peers(raw, min_minutes=90).transform(raw).goals_percentile.isna().all()


def test_reference_is_frozen_and_time_checked(appearances: pd.DataFrame) -> None:
    raw = aggregate_players(appearances, "2024-02-01")
    fitted = fit_peers(raw, min_minutes=90, min_peers=3)
    expected = fitted.transform(raw.iloc[:1])
    candidates = pd.concat([raw, raw.assign(player_id="outsider", goals_per90=999)])
    actual = fitted.transform(candidates).iloc[:1]
    pd.testing.assert_frame_equal(expected, actual)
    early = raw.assign(reference_date="2024-01-01")
    with pytest.raises(ValueError, match="after"):
        fitted.transform(early)
    for column, value in (("competition_id", "B"), ("season", "other"), ("position_group", "CB")):
        assert fitted.transform(raw.assign(**{column: value})).goals_percentile.isna().all()


def test_bootstrap_ratio_determinism_and_missingness(appearances: pd.DataFrame) -> None:
    rows = appearances.loc[appearances.player_id.eq("p2")].iloc[:2].copy()
    rows["minutes"] = [90, 10]
    rows["goals"] = [1, 1]
    first = bootstrap_per90(rows, "goals", reference_date="2024-02-01")
    assert first == bootstrap_per90(rows, "goals", reference_date="2024-02-01")
    assert first["estimate"] == 1.8
    assert first["lower"] <= first["estimate"] <= first["upper"]
    rows.loc[rows.index[0], "minutes"] = np.nan
    assert bootstrap_per90(rows, "goals", reference_date="2024-02-01")["lower"] is None
    with pytest.raises(ValueError, match="one player"):
        bootstrap_per90(appearances, "goals", reference_date="2024-02-01")


def test_bootstrap_future_rows_do_not_change_interval(appearances: pd.DataFrame) -> None:
    rows = appearances.loc[appearances.player_id.eq("p2")].copy()
    expected = bootstrap_per90(rows, "goals", reference_date="2024-01-03")
    rows.loc[rows.match_date.eq("2024-01-03"), "goals"] = 99
    assert bootstrap_per90(rows, "goals", reference_date="2024-01-03") == expected
    single = bootstrap_per90(rows.iloc[:1], "goals", reference_date="2024-02-01")
    assert single["lower"] is None


def test_pca_fit_transform_and_feature_restrictions(appearances: pd.DataFrame) -> None:
    players = normalized(appearances)
    features = ("goals_zscore", "assists_zscore", "yellow_cards_zscore")
    space = fit_player_space(players, features=features)
    expected = space.transform(players.iloc[:1])
    changed = players.copy()
    changed.loc[1:, "goals_zscore"] = 1000
    pd.testing.assert_frame_equal(expected, space.transform(changed).iloc[:1])
    assert space.pca.explained_variance_ratio_.sum() <= 1.000001
    with pytest.raises(ValueError, match="performance"):
        fit_player_space(players, features=("market_value_eur",))
    with pytest.raises(ValueError, match="later"):
        space.transform(players.assign(reference_date="2024-01-01"))


def test_similarity_competition_universe_and_exclusions(appearances: pd.DataFrame) -> None:
    players = normalized(appearances)
    foreign = players.assign(competition_id="B", player_id=lambda x: "foreign-" + x.player_id)
    candidates = pd.concat([players, foreign], ignore_index=True)
    result = similar_players(
        players.iloc[0],
        candidates,
        candidate_competitions=("B",),
        features=("goals_zscore", "assists_zscore"),
        min_minutes=90,
    )
    assert set(result.competition_id) == {"B"}
    assert result.iloc[0].distance == pytest.approx(0)
    assert result.distance.is_monotonic_increasing
    assert similar_players(
        players.iloc[0],
        players.iloc[:1],
        candidate_competitions=("A",),
        features=("goals_zscore",),
        min_minutes=90,
    ).empty
    with pytest.raises(ValueError, match="Explicit"):
        similar_players(
            players.iloc[0],
            players,
            candidate_competitions=(),
            features=("goals_zscore",),
            min_minutes=90,
        )
