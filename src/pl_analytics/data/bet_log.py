"""Local, user-owned bet log: one CSV under ``data/user`` that is never committed.

This is the only file the dashboard writes. It holds what the user typed and nothing is
sent anywhere. Rows are validated before every write and the file is replaced atomically.
"""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

BET_COLUMNS = (
    "bet_id",
    "placed_at",
    "event",
    "market",
    "selection",
    "odds",
    "stake",
    "tax",
    "model_probability",
    "closing_odds",
    "result",
)
RESULTS = ("open", "won", "lost", "void")
TEXT_COLUMNS = ("bet_id", "placed_at", "event", "market", "selection", "result")


def bet_log_path(data_dir: Path) -> Path:
    return data_dir / "user" / "bets.csv"


def validate_bets(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a log to its schema; reject impossible prices, stakes, taxes and results."""
    missing = set(BET_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"Bet log is missing columns: {sorted(missing)}")
    result = frame.loc[:, list(BET_COLUMNS)].copy()
    for column in ("odds", "stake", "tax", "model_probability", "closing_odds"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in TEXT_COLUMNS:
        result[column] = result[column].fillna("").astype(str).str.strip()
    result["result"] = result.result.str.lower().replace("", "open")
    if (result.bet_id == "").any() or result.bet_id.duplicated().any():
        raise ValueError("Every bet needs a unique identifier")
    if result.odds.isna().any() or (result.odds <= 1).any():
        raise ValueError("Decimal odds must be above 1")
    if result.stake.isna().any() or (result.stake <= 0).any():
        raise ValueError("Stakes must be positive")
    if result.tax.isna().any() or not result.tax.between(0, 1, inclusive="left").all():
        raise ValueError("Stake tax must lie in [0, 1)")
    if not result.result.isin(RESULTS).all():
        raise ValueError(f"Results must be one of {RESULTS}")
    probability = result.model_probability.dropna()
    if not probability.between(0, 1).all():
        raise ValueError("Model probability must lie in [0, 1]")
    if (result.closing_odds.dropna() <= 1).any():
        raise ValueError("Closing odds must be above 1")
    return result.reset_index(drop=True)


def read_bets(path: Path) -> pd.DataFrame:
    """Return the saved log, or an empty one when nothing has been recorded yet."""
    if not path.exists():
        return pd.DataFrame({column: pd.Series(dtype=object) for column in BET_COLUMNS})
    return validate_bets(pd.read_csv(path, dtype={column: str for column in TEXT_COLUMNS}))


def write_bets(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, then replace the log in one step so a failed write cannot truncate it."""
    validated = validate_bets(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    validated.to_csv(temporary, index=False)
    temporary.replace(path)
    return validated


def add_bet(
    path: Path,
    *,
    event: str,
    market: str,
    selection: str,
    odds: float,
    stake: float,
    tax: float,
    model_probability: float | None = None,
) -> pd.DataFrame:
    """Append one open bet stamped with the current UTC time."""
    existing = read_bets(path)
    now = datetime.now(UTC)
    record = {
        "bet_id": f"{now:%Y%m%d%H%M%S%f}",
        "placed_at": now.isoformat(timespec="seconds"),
        "event": event,
        "market": market,
        "selection": selection,
        "odds": odds,
        "stake": stake,
        "tax": tax,
        "model_probability": model_probability,
        "closing_odds": None,
        "result": "open",
    }
    rows = [*existing.to_dict("records"), record]
    return write_bets(path, pd.DataFrame(rows, columns=list(BET_COLUMNS)))
