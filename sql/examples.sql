-- Execute each statement through SQLAlchemy text() with :competition_id and :season.
-- Descriptive league table: observed totals, no disciplinary point deductions.
WITH totals AS (
    SELECT competition_id, season, club_id, max(club_name) AS club_name,
           count(*) AS played, sum(points) AS points,
           sum(goals_for - goals_against) AS goal_difference, sum(goals_for) AS goals,
           count(*) FILTER (WHERE points = 3) AS wins,
           count(*) FILTER (WHERE points = 1) AS draws,
           count(*) FILTER (WHERE points = 0) AS losses
    FROM mart_team_match WHERE competition_id = :competition_id AND season = :season
    GROUP BY competition_id, season, club_id
)
SELECT *, dense_rank() OVER (ORDER BY points DESC, goal_difference DESC, goals DESC) AS rank
FROM totals ORDER BY rank, club_id;

-- Retrospective timeline: LEAD is deliberately descriptive, never a model feature.
SELECT competition_id, season, club_id, match_id, match_date, points,
       lag(points) OVER history AS previous_recorded_points,
       lead(match_date) OVER history AS next_recorded_match_date,
       sum(points) OVER (history ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
           AS season_points_to_date
FROM mart_team_match WHERE competition_id = :competition_id AND season = :season
WINDOW history AS (PARTITION BY competition_id, season, club_id ORDER BY match_date, match_id)
ORDER BY club_id, match_date, match_id;

-- Explicit feature projection: current outcomes cannot accidentally enter X.
SELECT competition_id, season, match_id, club_id, match_date, history_cutoff,
       history_matches, points_last_5, goals_for_last_5, goals_against_last_5
FROM mart_team_prematch WHERE competition_id = :competition_id AND season = :season
ORDER BY match_date, match_id, club_id;
