"""Install/check M2 views against a populated database; never downloads data."""

import argparse
import json
import logging
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from pl_analytics.config import get_settings
from pl_analytics.data.analytics import check_analytics, install_analytics

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Optionally apply M2, then validate invariants and report scoped mart coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Apply M2 after the M1 schema")
    parser.add_argument("--manifest", type=Path, help="Assert coverage of the pinned M1 sample")
    parser.add_argument("--report", type=Path, default=Path("artifacts/m2-verification.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    engine = create_engine(
        settings.database_url, poolclass=NullPool,
        connect_args={"connect_timeout": settings.database_connect_timeout},
    )
    try:
        if args.install:
            install_analytics(engine, ROOT / "sql")
        with engine.connect() as connection:
            checks = check_analytics(connection, ROOT / "sql")
            coverage = [dict(row) for row in connection.execute(text("""
                SELECT competition_id, season, count(*) AS player_seasons,
                       count(*) FILTER (WHERE has_performance) AS players_with_performance
                FROM mart_player_season GROUP BY competition_id, season
                ORDER BY competition_id, season
            """)).mappings()]
            if args.manifest:
                manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
                scope = {"competition": manifest["competition_id"], "season": manifest["season"]}
                if scope["competition"] not in settings.active_competitions:
                    raise ValueError("Manifest competition is not enabled in ACTIVE_COMPETITIONS")
                counts = connection.execute(text("""
                    SELECT
                      (SELECT count(*) FROM mart_team_match
                       WHERE competition_id=:competition AND season=:season) AS team_matches,
                      (SELECT count(*) FROM mart_player_season
                       WHERE competition_id=:competition AND season=:season) AS player_seasons,
                      (SELECT coalesce(sum(valuation_count), 0) FROM mart_player_season_value
                       WHERE competition_id=:competition AND season=:season) AS valuations
                """), scope).mappings().one()
                expected = manifest["expected_counts"]
                for name, target in (
                    ("team_matches", 2 * expected["matches"]),
                    ("player_seasons", expected["players"]),
                    ("valuations", expected["player_market_values"]),
                ):
                    if counts[name] != target:
                        raise ValueError(f"Unexpected {name} coverage: {counts[name]} != {target}")
            report = {"checks": checks, "player_coverage": coverage}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logging.info("Verified M2 analytics; report: %s", args.report)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
