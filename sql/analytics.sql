-- Ordinary views: no refresh job, training or persistent derived copy required.
-- Grain: (match_id, club_id). These are OBSERVED outcomes, not pre-match inputs.
CREATE OR REPLACE VIEW mart_team_match AS
SELECT m.match_id, m.competition_id, m.season, m.match_date, m.match_time_known,
       m.source, side.club_id, c.name AS club_name, side.opponent_id,
       side.is_home, side.goals_for, side.goals_against, side.shots, side.shots_on_target,
       CASE WHEN side.goals_for > side.goals_against THEN 3
            WHEN side.goals_for = side.goals_against THEN 1 ELSE 0 END AS points
FROM matches m
CROSS JOIN LATERAL (
    VALUES (m.home_club_id, m.away_club_id, TRUE, m.home_goals, m.away_goals,
            m.home_shots, m.home_shots_on_target),
           (m.away_club_id, m.home_club_id, FALSE, m.away_goals, m.home_goals,
            m.away_shots, m.away_shots_on_target)
) AS side(club_id, opponent_id, is_home, goals_for, goals_against, shots, shots_on_target)
JOIN clubs c ON c.club_id = side.club_id
WHERE m.status = 'finished';

-- Grain: (match_id, club_id), including scheduled fixtures. No current outcomes.
-- UTC day boundary minus one full day is a conservative 24-48h embargo:
-- missing kickoffs can be local midnight, crossing the previous UTC date.
-- History is within the same competition AND season and limited to five games.
CREATE OR REPLACE VIEW mart_team_prematch AS
WITH fixtures AS (
    SELECT m.match_id, m.competition_id, m.season, m.match_date, side.club_id,
           side.opponent_id, side.is_home,
           (date_trunc('day', m.match_date AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')
               - INTERVAL '1 day' AS history_cutoff
    FROM matches m
    CROSS JOIN LATERAL (
        VALUES (m.home_club_id, m.away_club_id, TRUE),
               (m.away_club_id, m.home_club_id, FALSE)
    ) AS side(club_id, opponent_id, is_home)
    WHERE m.status IN ('finished', 'scheduled')
)
SELECT f.*, h.history_matches, h.last_history_date, h.points_last_5,
       h.goals_for_last_5, h.goals_against_last_5
FROM fixtures f
CROSS JOIN LATERAL (
    SELECT count(*) AS history_matches, max(previous.match_date) AS last_history_date,
           avg(previous.points) AS points_last_5,
           avg(previous.goals_for) AS goals_for_last_5,
           avg(previous.goals_against) AS goals_against_last_5
    FROM (
        SELECT t.match_date, t.points, t.goals_for, t.goals_against
        FROM mart_team_match t
        WHERE t.competition_id = f.competition_id AND t.season = f.season
          AND t.club_id = f.club_id AND t.match_date < f.history_cutoff
        ORDER BY t.match_date DESC, t.match_id DESC
        LIMIT 5
    ) previous
) h;

-- Retrospective full-season totals, NEVER use to predict an earlier observation.
-- Missing individual statistics make the corresponding total NULL, not zero.
CREATE OR REPLACE VIEW mart_player_season_performance AS
SELECT a.player_id, m.competition_id, m.season,
       count(*) AS appearances, count(DISTINCT a.club_id) AS recorded_clubs,
       min(m.match_date) AS first_match_date, max(m.match_date) AS last_match_date,
       count(a.minutes) AS appearances_with_minutes,
       CASE WHEN count(a.minutes) = count(*) THEN sum(a.minutes) END AS minutes,
       CASE WHEN count(a.goals) = count(*) THEN sum(a.goals) END AS goals,
       CASE WHEN count(a.assists) = count(*) THEN sum(a.assists) END AS assists,
       count(*) FILTER (WHERE a.minutes > 0) AS appearances_with_playing_time
FROM player_appearances a
JOIN matches m ON m.match_id = a.match_id
WHERE m.status = 'finished'
GROUP BY a.player_id, m.competition_id, m.season;

-- Keep each provider separate; no arbitrary averaging across valuation sources.
CREATE OR REPLACE VIEW mart_player_season_value AS
WITH ranked AS (
    SELECT v.*,
           row_number() OVER (
               PARTITION BY player_id, competition_id, season, source
               ORDER BY valuation_date DESC
           ) AS newest
    FROM player_market_values v
)
SELECT player_id, competition_id, season, source,
       count(*) AS valuation_count, min(valuation_date) AS first_valuation_date,
       max(valuation_date) AS last_valuation_date,
       min(market_value_eur) AS minimum_value_eur,
       max(market_value_eur) AS maximum_value_eur,
       max(market_value_eur) FILTER (WHERE newest = 1) AS latest_value_eur,
       bool_or(competition_context = 'source_reported_unverified')
           AS has_unverified_context,
       array_agg(DISTINCT competition_context ORDER BY competition_context) AS context_labels
FROM ranked
GROUP BY player_id, competition_id, season, source;

-- Coverage overview at (player_id, competition_id, season); no invented performance
-- for players represented only by valuations. Club identities remain unreconciled.
CREATE OR REPLACE VIEW mart_player_season AS
WITH universe AS (
    SELECT player_id, competition_id, season FROM mart_player_season_performance
    UNION
    SELECT player_id, competition_id, season FROM mart_player_season_value
), values_coverage AS (
    SELECT player_id, competition_id, season, sum(valuation_count) AS valuation_count,
           count(*) AS valuation_sources
    FROM mart_player_season_value
    GROUP BY player_id, competition_id, season
)
SELECT u.*, p.name AS player_name, perf.appearances, perf.minutes, perf.goals, perf.assists,
       perf.appearances_with_minutes, perf.last_match_date,
       (perf.player_id IS NOT NULL) AS has_performance,
       coalesce(v.valuation_count, 0) AS valuation_count,
       coalesce(v.valuation_sources, 0) AS valuation_sources
FROM universe u
JOIN players p USING (player_id)
LEFT JOIN mart_player_season_performance perf USING (player_id, competition_id, season)
LEFT JOIN values_coverage v USING (player_id, competition_id, season);
