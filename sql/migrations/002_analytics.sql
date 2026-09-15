-- Additive, repeatable M2 migration. Invalid existing rows abort the transaction;
-- fix their provenance/ingestion explicitly rather than silently deleting them.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'matches'::regclass AND conname = 'matches_analytics_valid'
    ) THEN
        ALTER TABLE matches ADD CONSTRAINT matches_analytics_valid CHECK (
            competition_id IS NOT NULL AND btrim(season) <> ''
            AND home_club_id IS NOT NULL AND away_club_id IS NOT NULL
            AND home_goals >= 0 AND away_goals >= 0
            AND home_shots >= 0 AND away_shots >= 0
            AND home_shots_on_target >= 0 AND away_shots_on_target >= 0
            AND home_shots_on_target <= home_shots
            AND away_shots_on_target <= away_shots
            AND (status IS DISTINCT FROM 'finished'
                 OR (home_goals IS NOT NULL AND away_goals IS NOT NULL))
        );
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'player_appearances'::regclass
          AND conname = 'appearances_analytics_valid'
    ) THEN
        ALTER TABLE player_appearances ADD CONSTRAINT appearances_analytics_valid
            CHECK (goals >= 0 AND assists >= 0 AND yellow_cards >= 0 AND red_cards >= 0);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'player_market_values'::regclass
          AND conname = 'values_analytics_scope'
    ) THEN
        ALTER TABLE player_market_values ADD CONSTRAINT values_analytics_scope
            CHECK (competition_id IS NOT NULL AND season IS NOT NULL AND btrim(season) <> '');
    END IF;
END $$;

-- Two directional indexes support per-club history, without a fixed league list.
CREATE INDEX IF NOT EXISTS idx_matches_home_history
    ON matches(competition_id, season, home_club_id, match_date, match_id)
    WHERE status = 'finished';
CREATE INDEX IF NOT EXISTS idx_matches_away_history
    ON matches(competition_id, season, away_club_id, match_date, match_id)
    WHERE status = 'finished';
CREATE INDEX IF NOT EXISTS idx_values_player_scope_date
    ON player_market_values(competition_id, season, player_id, source, valuation_date DESC);
