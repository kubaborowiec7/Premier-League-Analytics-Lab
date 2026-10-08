"""A transparent, rule-based reading of injury headlines.

This is not a model and has not been validated against outcomes. Each headline that
names the player is matched against fixed phrases and given a signal; the estimate is
the recency-weighted average of those signals. It exists to summarise what the headlines
say and to point at the sources, not to replace reading them.
"""

import re
import unicodedata
from datetime import datetime

# Checked in order: the first group with a matching phrase decides the headline.
SIGNALS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    (
        "Out",
        0.03,
        (
            "ruled out",
            "out for",
            "out until",
            "sidelined",
            "surgery",
            "will miss",
            "to miss",
            "set to miss",
            "misses",
            "season-ending",
            "long-term",
            "weeks out",
            "months out",
            "suspended",
            "banned",
            "not available",
            "unavailable",
            "injury blow",
            "setback",
        ),
    ),
    (
        "Doubt",
        0.5,
        (
            "doubt",
            "doubtful",
            "fitness test",
            "late test",
            "late call",
            "be assessed",
            "being assessed",
            "50-50",
            "touch and go",
            "race against time",
            "race to be fit",
            "uncertain",
            "injury scare",
            "injury worry",
            "injury concern",
            "limps off",
            "limped off",
            "forced off",
            "knock",
        ),
    ),
    (
        "Positive",
        0.9,
        (
            "passed fit",
            "declared fit",
            "fit to",
            "fit again",
            "fully fit",
            "returns to training",
            "returned to training",
            "return to training",
            "back in training",
            "back in contention",
            "in contention",
            "available",
            "injury boost",
            "fitness boost",
            "boost",
            "set to return",
            "ready to return",
            "recovered",
            "back from injury",
            "shakes off",
        ),
    ),
)
ROUNDUPS = (
    "injury update",
    "injury news",
    "injury list",
    "injury roundup",
    "injury round-up",
    "injury latest",
    "injuries and suspensions",
    "suspension list",
    "return dates",
    "team news",
    "predicted lineup",
    "predicted line-up",
    "predicted xi",
)
HALF_LIFE_DAYS = 2.0
MIXED_SPREAD = 0.3


def _plain(text: str) -> str:
    """Lower-case text without accents, so "Ødegaard" and "Odegaard" compare equal."""
    replaced = text.replace("ø", "o").replace("Ø", "O").replace("ß", "ss")
    decomposed = unicodedata.normalize("NFKD", replaced)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


def mentions(title: str, names: tuple[str, ...]) -> bool:
    """True when the headline contains any of the player's names as a whole word."""
    text = _plain(title)
    return any(
        re.search(rf"(?<![a-z]){re.escape(_plain(name))}(?![a-z])", text)
        for name in names
        if len(name.strip()) >= 3
    )


def classify_headline(title: str) -> tuple[str, float | None]:
    """Return the first matching signal as ``(label, chance of playing)``."""
    text = _plain(title)
    # A round-up lists several players under one headline, so its wording ("suspension
    # list", "ruled out") says nothing reliable about any single one of them.
    if title.count(",") >= 2 or any(phrase in text for phrase in ROUNDUPS):
        return "Round-up", None
    for label, chance, phrases in SIGNALS:
        # Whole words only, with an optional plural: "knock" must not match "knockout".
        if any(re.search(rf"(?<![a-z]){re.escape(phrase)}s?(?![a-z])", text) for phrase in phrases):
            return label, chance
    return "No signal", None


def estimate_availability(
    headlines: list[dict], names: tuple[str, ...], *, now: datetime
) -> dict:
    """Summarise headlines into a chance of playing, or none when they carry no signal.

    Headlines that do not name the player are ignored, and round-ups are never a signal.
    A headline naming two players still counts for both, which is the main remaining way
    this reading goes wrong.
    """
    rows, weights, chances = [], [], []
    for headline in headlines:
        relevant = mentions(headline["title"], names)
        label, chance = classify_headline(headline["title"]) if relevant else ("Not about", None)
        rows.append({**headline, "label": label, "chance": chance, "relevant": relevant})
        if chance is not None:
            age = max(0.0, (now - headline["published"]).total_seconds() / 86_400)
            weights.append(0.5 ** (age / HALF_LIFE_DAYS))
            chances.append(chance)
    if not chances:
        return {"estimate": None, "signals": 0, "mixed": False, "headlines": rows}
    total = sum(weights)
    estimate = sum(weight * chance for weight, chance in zip(weights, chances, strict=True)) / total
    return {
        "estimate": estimate,
        "signals": len(chances),
        "mixed": max(chances) - min(chances) > MIXED_SPREAD,
        "headlines": rows,
    }
