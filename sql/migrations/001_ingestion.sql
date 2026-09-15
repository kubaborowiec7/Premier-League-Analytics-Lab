-- Additive M1 upgrade; safe for existing M0 databases and repeated execution.
-- Existing valuation rows retain NULL scope rather than inventing historical context.
ALTER TABLE matches ADD COLUMN IF NOT EXISTS match_time_known BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE player_market_values
    ADD COLUMN IF NOT EXISTS competition_id TEXT REFERENCES competitions(competition_id),
    ADD COLUMN IF NOT EXISTS season TEXT,
    ADD COLUMN IF NOT EXISTS source_reported_club_id TEXT REFERENCES clubs(club_id),
    ADD COLUMN IF NOT EXISTS competition_context TEXT NOT NULL DEFAULT 'source_reported_unverified';

CREATE TABLE IF NOT EXISTS ingestion_batches (
    batch_id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    competition_id TEXT NOT NULL REFERENCES competitions(competition_id),
    season TEXT NOT NULL,
    coverage_start DATE NOT NULL,
    coverage_end DATE NOT NULL,
    loader_version TEXT NOT NULL,
    row_counts JSONB NOT NULL,
    skipped_rows INTEGER NOT NULL DEFAULT 0,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (coverage_end >= coverage_start),
    CHECK (skipped_rows >= 0)
);

CREATE TABLE IF NOT EXISTS ingestion_batch_snapshots (
    batch_id UUID NOT NULL REFERENCES ingestion_batches(batch_id),
    snapshot_id UUID NOT NULL REFERENCES source_snapshots(snapshot_id),
    PRIMARY KEY (batch_id, snapshot_id)
);

CREATE INDEX IF NOT EXISTS idx_ingestion_batches_scope
    ON ingestion_batches(competition_id, season);
