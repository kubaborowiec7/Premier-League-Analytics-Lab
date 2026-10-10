"""Club-level match statistics: one row per club and match, then per-match averages.

Everything here is descriptive. Values are recorded match totals from the source, shown
as averages per match with their rank among the clubs of the same competition and season.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SCOPES = ("Season", "Last 5 matches", "Home", "Away")
LAST_MATCHES = 5
KEYS = ["competition_id", "season", "club_id"]


@dataclass(frozen=True)
class TeamStat:
    """One comparable club statistic.

    ``source`` is a recorded match total. With ``denominator`` the statistic is the
    percentage ``sum(source) / sum(denominator)`` over the period instead of a mean.
    ``against`` reads the opponent's value in the same matches.
    """

    label: str
    group: str
    source: str
    against: bool = False
    denominator: str | None = None
    higher_is_better: bool = True
    percentage: bool = False


STATS: dict[str, TeamStat] = {
    # Attack
    "goals": TeamStat("Goals", "Attack", "score"),
    "xg": TeamStat("xG", "Attack", "expected_goals_xg"),
    "np_xg": TeamStat("Non-penalty xG", "Attack", "non_penalty_xg"),
    "xg_open_play": TeamStat("Open-play xG", "Attack", "xg_open_play"),
    "shots": TeamStat("Shots", "Attack", "total_shots"),
    "shots_on_target": TeamStat("Shots on target", "Attack", "shots_on_target"),
    "shot_accuracy": TeamStat(
        "Shots on target %", "Attack", "shots_on_target", denominator="total_shots"
    ),
    "shots_inside_box": TeamStat("Shots inside box", "Attack", "shots_inside_box"),
    "big_chances": TeamStat("Big chances", "Attack", "big_chances"),
    "box_touches": TeamStat("Touches in opp. box", "Attack", "touches_in_opposition_box"),
    "dribbles": TeamStat("Successful dribbles", "Attack", "successful_dribbles"),
    # Defence
    "goals_against": TeamStat(
        "Goals conceded", "Defence", "score", against=True, higher_is_better=False
    ),
    "xg_against": TeamStat(
        "xG conceded", "Defence", "expected_goals_xg", against=True, higher_is_better=False
    ),
    "shots_against": TeamStat(
        "Shots conceded", "Defence", "total_shots", against=True, higher_is_better=False
    ),
    "shots_on_target_against": TeamStat(
        "Shots on target conceded",
        "Defence",
        "shots_on_target",
        against=True,
        higher_is_better=False,
    ),
    "big_chances_against": TeamStat(
        "Big chances conceded", "Defence", "big_chances", against=True, higher_is_better=False
    ),
    "box_touches_against": TeamStat(
        "Opp. touches in own box",
        "Defence",
        "touches_in_opposition_box",
        against=True,
        higher_is_better=False,
    ),
    "tackles_won": TeamStat("Tackles won", "Defence", "tackles_won"),
    "interceptions": TeamStat("Interceptions", "Defence", "interceptions"),
    "clearances": TeamStat("Clearances", "Defence", "clearances"),
    "blocks": TeamStat("Blocks", "Defence", "blocks"),
    "saves": TeamStat("Saves", "Defence", "keeper_saves"),
    "duels_won": TeamStat("Duels won", "Defence", "duels_won"),
    "aerial_duels_won": TeamStat("Aerial duels won", "Defence", "aerial_duels_won"),
    # Possession & passing
    "possession": TeamStat(
        "Possession %", "Possession & passing", "possession", percentage=True
    ),
    "passes": TeamStat("Passes", "Possession & passing", "passes"),
    "pass_accuracy": TeamStat(
        "Pass accuracy %", "Possession & passing", "accurate_passes", denominator="passes"
    ),
    "final_half_passes": TeamStat(
        "Passes in opp. half", "Possession & passing", "opposition_half"
    ),
    "long_balls": TeamStat("Accurate long balls", "Possession & passing", "accurate_long_balls"),
    "crosses": TeamStat("Accurate crosses", "Possession & passing", "accurate_crosses"),
    # Discipline & set pieces
    "corners": TeamStat("Corners", "Discipline & set pieces", "corners"),
    "corners_against": TeamStat(
        "Corners conceded",
        "Discipline & set pieces",
        "corners",
        against=True,
        higher_is_better=False,
    ),
    "xg_set_play": TeamStat("Set-piece xG", "Discipline & set pieces", "xg_set_play"),
    "xg_set_play_against": TeamStat(
        "Set-piece xG conceded",
        "Discipline & set pieces",
        "xg_set_play",
        against=True,
        higher_is_better=False,
    ),
    "fouls": TeamStat(
        "Fouls committed", "Discipline & set pieces", "fouls_committed", higher_is_better=False
    ),
    "fouls_won": TeamStat(
        "Fouls won", "Discipline & set pieces", "fouls_committed", against=True
    ),
    "yellow_cards": TeamStat(
        "Yellow cards", "Discipline & set pieces", "yellow_cards", higher_is_better=False
    ),
    "red_cards": TeamStat(
        "Red cards", "Discipline & set pieces", "red_cards", higher_is_better=False
    ),
    "offsides": TeamStat(
        "Offsides", "Discipline & set pieces", "offsides", higher_is_better=False
    ),
}
GROUPS = tuple(dict.fromkeys(stat.group for stat in STATS.values()))
SOURCES = tuple(
    dict.fromkeys(
        name for stat in STATS.values() for name in (stat.source, stat.denominator) if name
    )
)


def team_match_rows(
    matches: pd.DataFrame, teams: pd.DataFrame, *, competition_id: str, season: str
) -> pd.DataFrame:
    """Turn finished matches into two rows each: the club's totals and its opponent's.

    Statistics the source does not record stay missing; they are never filled with zero.
    """
    finished = matches.loc[matches.finished.astype(str).str.lower().isin(["true", "1"])]
    lookup = teams.assign(code=pd.to_numeric(teams.code, errors="raise").astype(int)).set_index(
        "code"
    )
    frames = []
    for side, other in (("home", "away"), ("away", "home")):
        code = pd.to_numeric(finished[f"{side}_team"], errors="raise").astype(int)
        opponent = pd.to_numeric(finished[f"{other}_team"], errors="raise").astype(int)
        if not set(code) <= set(lookup.index):
            raise ValueError("Match references an unknown team")
        frame = pd.DataFrame(
            {
                "competition_id": competition_id,
                "season": season,
                "match_id": finished.match_id.to_numpy(),
                "match_date": pd.to_datetime(
                    finished.kickoff_time, utc=True, format="mixed"
                ).to_numpy(),
                "club_id": ("fpl:club:" + code.astype(str)).to_numpy(),
                "club_name": lookup.name.reindex(code).to_numpy(),
                "club_short_name": lookup.short_name.reindex(code).to_numpy(),
                "opponent_short_name": lookup.short_name.reindex(opponent).to_numpy(),
                "venue": "Home" if side == "home" else "Away",
            }
        )
        for source in SOURCES:
            for suffix, prefix in (("", side), ("_against", other)):
                column = f"{prefix}_{source}"
                values = (
                    pd.to_numeric(finished[column], errors="coerce")
                    if column in finished
                    else pd.Series(np.nan, index=finished.index)
                )
                frame[source + suffix] = values.to_numpy()
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    if rows.duplicated(["match_id", "club_id"]).any():
        raise ValueError("Duplicate club match rows")
    return rows.sort_values(["match_date", "match_id", "venue"], ignore_index=True)


def _scope_rows(rows: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope not in SCOPES:
        raise ValueError(f"Unknown period: {scope}")
    if scope in ("Home", "Away"):
        return rows.loc[rows.venue.eq(scope)]
    if scope == "Last 5 matches":
        ordered = rows.sort_values(["match_date", "match_id"])
        return ordered.groupby(KEYS, sort=False).tail(LAST_MATCHES)
    return rows


def summarize_teams(rows: pd.DataFrame, scope: str = "Season") -> pd.DataFrame:
    """Average every statistic per match for one period and rank clubs within a season.

    ``{name}_percentile`` is 100 for the best club of the competition and season and 0
    for the worst, with "best" following each statistic's direction; ``{name}_rank``
    starts at 1. Clubs are only compared over the same period.
    """
    chosen = _scope_rows(rows, scope)
    grouped = chosen.groupby(KEYS, sort=False)
    result = grouped.agg(
        club_name=("club_name", "first"),
        club_short_name=("club_short_name", "first"),
        matches=("match_id", "nunique"),
        last_match=("match_date", "max"),
    )
    values: dict[str, pd.Series] = {}
    for name, stat in STATS.items():
        column = stat.source + ("_against" if stat.against else "")
        if stat.denominator:
            base = stat.denominator + ("_against" if stat.against else "")
            # Only matches recording both parts count, so a gap cannot distort the ratio.
            both = chosen[column].notna() & chosen[base].notna()
            parts = chosen.loc[both].groupby(KEYS, sort=False)[[column, base]].sum()
            ratio = 100 * parts[column] / parts[base].where(parts[base] > 0)
            values[name] = ratio.reindex(result.index)
        else:
            values[name] = grouped[column].mean()
    result = pd.concat([result, pd.DataFrame(values)], axis=1).reset_index()
    season_groups = result.groupby(["competition_id", "season"], sort=False)
    ranked: dict[str, pd.Series] = {}
    for name, stat in STATS.items():
        ranks = season_groups[name].rank(ascending=not stat.higher_is_better, method="min")
        count = season_groups[name].transform("count")
        ranked[f"{name}_rank"] = ranks
        ranked[f"{name}_percentile"] = (100 * (count - ranks) / (count - 1)).where(count > 1)
    return pd.concat([result, pd.DataFrame(ranked)], axis=1).assign(scope=scope)
