"""Headline feed parsing and the rule-based injury reading."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from pl_analytics.data.injury_news import fetch_headlines
from pl_analytics.features.injury_news import (
    classify_headline,
    estimate_availability,
    mentions,
)

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item><title>Palmer ruled out of Bournemouth clash - Evening Standard</title>
<link>https://example.test/a</link><pubDate>Thu, 08 Oct 2026 08:00:00 GMT</pubDate>
<source url="https://example.test">Evening Standard</source></item>
<item><title>Cole Palmer returns to training in injury boost - Football London</title>
<link>https://example.test/b</link><pubDate>Wed, 07 Oct 2026 08:00:00 GMT</pubDate>
<source url="https://example.test">Football London</source></item>
<item><title>Undated story - Nobody</title><link>https://example.test/c</link>
<source url="https://example.test">Nobody</source></item>
</channel></rss>"""


def test_feed_is_parsed_newest_first_without_publisher_suffix():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "Palmer Chelsea injury when:10d"
        return httpx.Response(200, content=FEED.encode())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        headlines = fetch_headlines("Palmer", "Chelsea", client=client)
    # The item without a date is dropped rather than guessed.
    assert [row["title"] for row in headlines] == [
        "Palmer ruled out of Bournemouth clash",
        "Cole Palmer returns to training in injury boost",
    ]
    assert headlines[0]["source"] == "Evening Standard"
    assert headlines[0]["published"].tzinfo is not None
    with pytest.raises(ValueError):
        fetch_headlines("", "Chelsea")
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"not xml"))
    ) as client, pytest.raises(ValueError):
        fetch_headlines("Palmer", "Chelsea", client=client)


def test_headline_rules_use_whole_words_and_the_first_matching_group():
    assert classify_headline("Saka ruled out for a month") == ("Out", 0.03)
    assert classify_headline("Rice a doubt for the weekend")[0] == "Doubt"
    assert classify_headline("Injury doubts over Rice")[0] == "Doubt"
    assert classify_headline("Haaland passed fit to face Liverpool")[0] == "Positive"
    # "not available" is an absence even though it contains "available".
    assert classify_headline("Manager confirms White is not available")[0] == "Out"
    # Whole words only: a cup knockout tie is not a knock.
    assert classify_headline("Arsenal reach the knockout stage") == ("No signal", None)
    assert classify_headline("Slot praises his midfield") == ("No signal", None)
    # Round-ups name several players, so their wording is not read as a signal for any.
    assert classify_headline(
        "Chelsea injury, suspension list and return dates: James, Caicedo, Palmer"
    ) == ("Round-up", None)
    assert classify_headline("Chelsea team news: Palmer ruled out") == ("Round-up", None)


def test_names_match_whole_words_and_ignore_accents():
    assert mentions("Odegaard back in training", ("Ødegaard",))
    assert mentions("Groß limps off", ("Gross",))
    assert not mentions("Whiteside makes debut", ("White",))
    assert not mentions("Anything at all", ("Jo",))  # too short to match safely


def test_estimate_weights_recent_headlines_and_ignores_other_players():
    now = datetime(2026, 10, 8, 12, tzinfo=UTC)

    def item(title: str, days: float) -> dict:
        return {
            "title": title,
            "source": "Test",
            "link": "",
            "published": now - timedelta(days=days),
        }

    reading = estimate_availability(
        [
            item("Palmer returns to training", 0.1),
            item("Palmer ruled out for two weeks", 6),
            item("Caicedo ruled out for the season", 0.1),
        ],
        ("Palmer",),
        now=now,
    )
    # The fresh positive report dominates the week-old absence; Caicedo's is ignored.
    assert reading["signals"] == 2 and reading["mixed"]
    assert 0.75 < reading["estimate"] < 0.9
    assert [row["label"] for row in reading["headlines"]] == ["Positive", "Out", "Not about"]
    silent = estimate_availability([item("Palmer signs new deal", 1)], ("Palmer",), now=now)
    assert silent["estimate"] is None and silent["signals"] == 0
