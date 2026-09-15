CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS competitions (
    competition_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT,
    source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_competition_mappings (
    source TEXT NOT NULL,
    source_competition_code TEXT NOT NULL,
    competition_id TEXT NOT NULL REFERENCES competitions(competition_id),
    PRIMARY KEY (source, source_competition_code)
);

CREATE TABLE IF NOT EXISTS clubs (
    club_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    date_of_birth DATE,
    position TEXT,
    nationality TEXT
);

-- Referenced entities must exist before this table during first-volume initialization.
CREATE TABLE IF NOT EXISTS player_season_features (
    player_id TEXT NOT NULL REFERENCES players(player_id),
    competition_id TEXT NOT NULL REFERENCES competitions(competition_id),
    season TEXT NOT NULL,
    reference_date DATE NOT NULL,
    position_group TEXT NOT NULL,
    club_id TEXT REFERENCES clubs(club_id),
    minutes NUMERIC(10, 2),
    feature_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    normalization_context TEXT NOT NULL DEFAULT 'competition-season-position',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, competition_id, season, reference_date, position_group)
);

CREATE INDEX IF NOT EXISTS idx_player_features_competition_position
    ON player_season_features(competition_id, season, position_group);

CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    competition_id TEXT REFERENCES competitions(competition_id),
    season TEXT NOT NULL,
    match_date TIMESTAMPTZ NOT NULL,
    match_time_known BOOLEAN NOT NULL DEFAULT FALSE,
    home_club_id TEXT REFERENCES clubs(club_id),
    away_club_id TEXT REFERENCES clubs(club_id),
    home_goals INTEGER,
    away_goals INTEGER,
    home_shots INTEGER,
    away_shots INTEGER,
    home_shots_on_target INTEGER,
    away_shots_on_target INTEGER,
    status TEXT,
    source TEXT NOT NULL,
    CHECK (home_club_id IS NULL OR away_club_id IS NULL OR home_club_id <> away_club_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_date
    ON matches(match_date);

CREATE INDEX IF NOT EXISTS idx_matches_competition_season
    ON matches(competition_id, season);

CREATE TABLE IF NOT EXISTS player_appearances (
    match_id TEXT NOT NULL REFERENCES matches(match_id),
    player_id TEXT NOT NULL REFERENCES players(player_id),
    club_id TEXT REFERENCES clubs(club_id),
    minutes INTEGER,
    goals INTEGER,
    assists INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,
    source TEXT NOT NULL,
    PRIMARY KEY (match_id, player_id),
    CHECK (minutes IS NULL OR minutes >= 0)
);

CREATE INDEX IF NOT EXISTS idx_appearances_player
    ON player_appearances(player_id);

CREATE TABLE IF NOT EXISTS player_market_values (
    player_id TEXT NOT NULL REFERENCES players(player_id),
    valuation_date DATE NOT NULL,
    market_value_eur NUMERIC(16, 2) NOT NULL,
    club_id TEXT REFERENCES clubs(club_id),
    source TEXT NOT NULL,
    competition_id TEXT REFERENCES competitions(competition_id),
    season TEXT,
    source_reported_club_id TEXT REFERENCES clubs(club_id),
    competition_context TEXT NOT NULL DEFAULT 'source_reported_unverified',
    PRIMARY KEY (player_id, valuation_date, source),
    CHECK (market_value_eur >= 0)
);

CREATE INDEX IF NOT EXISTS idx_market_values_date
    ON player_market_values(valuation_date);

CREATE TABLE IF NOT EXISTS source_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name TEXT NOT NULL,
    source_url TEXT,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    dataset_version TEXT,
    coverage_start DATE,
    coverage_end DATE,
    license_or_terms_note TEXT,
    sha256 TEXT,
    local_path TEXT,
    UNIQUE (source_name, sha256)
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL,
    task TEXT NOT NULL,
    git_sha TEXT,
    train_start DATE,
    train_end DATE,
    test_start DATE,
    test_end DATE,
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifact_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Example leakage-safe rolling match feature pattern:
-- Current match is excluded because the window ends at 1 PRECEDING.
--
-- SELECT
--     club_id,
--     match_date,
--     AVG(points) OVER (
--         PARTITION BY club_id
--         ORDER BY match_date
--         ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
--     ) AS points_last_5
-- FROM team_match_long;
