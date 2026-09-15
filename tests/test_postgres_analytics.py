"""Execute actual PostgreSQL SQL against synthetic adversarial histories."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

from pl_analytics.data.analytics import check_analytics, install_analytics

SQL = Path(__file__).resolve().parents[1] / "sql"


def seed(connection: Connection) -> None:
    connection.exec_driver_sql("""
        INSERT INTO competitions VALUES ('A', 'League A', 'X', 'test'),
                                        ('B', 'League B', 'X', 'test');
        INSERT INTO clubs VALUES ('a', 'Club A', 'X'), ('b', 'Club B', 'X'),
                                 ('c', 'Club C', 'X');
        INSERT INTO players (player_id, name) VALUES ('p', 'Player'), ('v', 'Valuation only');
    """)


def match(connection: Connection, match_id: str, day: int, **overrides: object) -> None:
    row = {
        "id": match_id, "competition": "A", "season": "2023/24",
        "date": datetime(2024, 1, 1, 15, tzinfo=UTC) + timedelta(days=day),
        "home": "a", "away": "b", "hg": 2, "ag": 1,
        "status": "finished", "known": True,
    }
    row.update(overrides)
    connection.execute(text("""
        INSERT INTO matches (match_id, competition_id, season, match_date, home_club_id,
                             away_club_id, home_goals, away_goals, status, source, match_time_known)
        VALUES (:id, :competition, :season, :date, :home, :away, :hg, :ag, :status, 'test', :known)
    """), row)


def test_analytics_empty_and_repeatable(pg_engine: Engine) -> None:
    first = install_analytics(pg_engine, SQL)
    assert len(first) == 9 and not any(first.values())
    assert install_analytics(pg_engine, SQL) == first
    with pg_engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM mart_player_season")) == 0


def test_prematch_cutoff_scope_and_five_games(pg_engine: Engine) -> None:
    install_analytics(pg_engine, SQL)
    with pg_engine.begin() as connection:
        seed(connection)
        for day in range(7):
            match(connection, f"m{day}", day, hg=day)
        # Same timestamp and unknown-time preceding UTC date cannot enter target history.
        match(connection, "peer", 10, hg=99)
        match(connection, "unknown", 9, known=False, hg=99)
        match(connection, "other-competition", 7, competition="B", hg=99)
        match(connection, "other-season", 7, season="2024/25", hg=99)
        match(connection, "cancelled", 7, status="cancelled", hg=99)
        match(connection, "target", 10, hg=None, ag=None, status="scheduled")
        query = text("SELECT * FROM mart_team_prematch WHERE match_id='target' AND club_id='a'")
        before = dict(connection.execute(query).mappings().one())
        assert before["history_matches"] == 5
        assert before["goals_for_last_5"] == Decimal(4)  # days 2..6, not 0..6
        assert before["points_last_5"] == Decimal(3)
        assert before["last_history_date"] < before["history_cutoff"]
        connection.exec_driver_sql("""
            UPDATE matches SET status='finished', home_goals=88, away_goals=0
            WHERE match_id='target'
        """)
        match(connection, "future", 20, hg=99)
        assert dict(connection.execute(query).mappings().one()) == before
        first = connection.execute(text("""
            SELECT history_matches, points_last_5 FROM mart_team_prematch
            WHERE match_id='m0' AND club_id='a'
        """)).one()
        assert first == (0, None)
        # Results cannot depend on a caller's PostgreSQL timezone.
        connection.exec_driver_sql("SET LOCAL TIME ZONE 'Pacific/Auckland'")
        after = dict(connection.execute(query).mappings().one())
        assert after == before
        assert not any(check_analytics(connection, SQL).values())


def test_player_seasons_missing_values_transfers_and_sources(pg_engine: Engine) -> None:
    install_analytics(pg_engine, SQL)
    with pg_engine.begin() as connection:
        seed(connection)
        match(connection, "one", 0)
        match(connection, "two", 5, home="b", away="c")
        match(connection, "other", 6, competition="B")
        connection.exec_driver_sql("""
            INSERT INTO player_appearances
                (match_id, player_id, club_id, minutes, goals, assists, source)
            VALUES ('one','p','a',90,1,0,'test'), ('two','p','b',NULL,0,1,'test'),
                   ('other','p','a',10,5,0,'test');
            INSERT INTO player_market_values
                (player_id, competition_id, season, valuation_date, market_value_eur, source)
            VALUES ('p','A','2023/24','2024-01-01',100,'s1'),
                   ('p','A','2023/24','2024-02-01',80,'s1'),
                   ('p','A','2023/24','2024-02-01',200,'s2'),
                   ('v','A','2023/24','2024-02-01',50,'s1');
        """)
        perf = connection.execute(text("""
            SELECT appearances, recorded_clubs, minutes, goals, appearances_with_minutes
            FROM mart_player_season_performance WHERE player_id='p' AND competition_id='A'
        """)).one()
        assert perf == (2, 2, None, 1, 1)
        values = connection.execute(text("""
            SELECT latest_value_eur, maximum_value_eur, valuation_count, has_unverified_context
            FROM mart_player_season_value WHERE player_id='p' AND source='s1'
        """)).one()
        assert values == (80, 100, 2, True)
        missing = connection.execute(text("""
            SELECT has_performance, minutes, goals, valuation_count FROM mart_player_season
            WHERE player_id='v'
        """)).one()
        assert missing == (False, None, None, 1)
        assert connection.scalar(text("SELECT count(*) FROM mart_player_season")) == 3
        assert not any(check_analytics(connection, SQL).values())


@pytest.mark.parametrize("override", [
    {"hg": -1}, {"hg": None}, {"competition": None}, {"home": None}, {"season": " "},
])
def test_match_constraints(pg_engine: Engine, override: dict[str, object]) -> None:
    install_analytics(pg_engine, SQL)
    with pg_engine.begin() as connection:
        seed(connection)
    with pytest.raises(IntegrityError), pg_engine.begin() as connection:
        match(connection, "invalid", 0, **override)


def test_invalid_legacy_data_rolls_back_installation(pg_engine: Engine) -> None:
    with pg_engine.begin() as connection:
        seed(connection)
        match(connection, "bad", 0, hg=-1)
    with pytest.raises(IntegrityError):
        install_analytics(pg_engine, SQL)
    with pg_engine.connect() as connection:
        assert connection.scalar(text("SELECT home_goals FROM matches")) == -1
        assert connection.scalar(text("SELECT to_regclass('mart_team_match')")) is None


def test_appearance_invariant_and_examples(pg_engine: Engine) -> None:
    install_analytics(pg_engine, SQL)
    with pg_engine.begin() as connection:
        seed(connection)
        match(connection, "m", 0)
        examples = (SQL / "examples.sql").read_text(encoding="utf-8")
        for statement in examples.split(";"):
            if statement.strip():
                assert connection.execute(
                    text(statement), {"competition_id": "A", "season": "2023/24"}
                ).fetchall()
        connection.exec_driver_sql("""
            INSERT INTO player_appearances (match_id, player_id, club_id, source)
            VALUES ('m','p','c','test')
        """)
        with pytest.raises(ValueError, match="appearance_club_participates"):
            check_analytics(connection, SQL)
