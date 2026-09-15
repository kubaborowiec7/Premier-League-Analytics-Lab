-- Every row must report zero violations. Empty marts pass structural checks;
-- population/coverage expectations are verified separately against the M1 manifest.
SELECT 'team_match_grain' AS check_name, count(*) AS violations FROM (
    SELECT match_id, club_id FROM mart_team_match GROUP BY match_id, club_id HAVING count(*) <> 1
) duplicates
UNION ALL
SELECT 'two_sides_per_finished_match', count(*) FROM (
    SELECT m.match_id FROM matches m LEFT JOIN mart_team_match t USING (match_id)
    WHERE m.status = 'finished' GROUP BY m.match_id HAVING count(t.club_id) <> 2
) invalid
UNION ALL
SELECT 'prematch_grain', count(*) FROM (
    SELECT match_id, club_id FROM mart_team_prematch GROUP BY match_id, club_id HAVING count(*) <> 1
) duplicates
UNION ALL
SELECT 'history_cutoff', count(*) FROM mart_team_prematch
WHERE last_history_date >= history_cutoff OR history_matches NOT BETWEEN 0 AND 5
   OR (history_matches = 0 AND points_last_5 IS NOT NULL)
UNION ALL
SELECT 'player_season_grain', count(*) FROM (
    SELECT player_id, competition_id, season FROM mart_player_season
    GROUP BY player_id, competition_id, season HAVING count(*) <> 1
) duplicates
UNION ALL
SELECT 'value_grain', count(*) FROM (
    SELECT player_id, competition_id, season, source FROM mart_player_season_value
    GROUP BY player_id, competition_id, season, source HAVING count(*) <> 1
) duplicates
UNION ALL
SELECT 'appearance_club_participates', count(*) FROM player_appearances a
JOIN matches m USING (match_id)
WHERE a.club_id IS NOT NULL AND a.club_id NOT IN (m.home_club_id, m.away_club_id)
UNION ALL
SELECT 'missing_performance_not_zero', count(*) FROM mart_player_season
WHERE NOT has_performance AND (appearances IS NOT NULL OR minutes IS NOT NULL OR goals IS NOT NULL)
UNION ALL
SELECT 'points_and_goals_balance', count(*) FROM (
    SELECT match_id FROM mart_team_match GROUP BY match_id
    HAVING sum(goals_for) <> sum(goals_against) OR sum(points) NOT IN (2, 3)
) invalid;
