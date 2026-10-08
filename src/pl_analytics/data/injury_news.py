"""Recent news headlines about one player from the public Google News RSS search feed.

One request per explicit lookup, no account or key. Only the headline, publisher, date
and link are kept, in memory; article text is never fetched. Social networks are not
searched: they have no free feed.
"""

import xml.etree.ElementTree as ET
from datetime import UTC
from email.utils import parsedate_to_datetime

import httpx

FEED_URL = "https://news.google.com/rss/search"
MAX_HEADLINES = 30
MAX_FEED_BYTES = 2_000_000


def fetch_headlines(
    player: str,
    club: str,
    *,
    days: int = 10,
    client: httpx.Client | None = None,
) -> list[dict]:
    """Return recent headlines for ``player`` at ``club``, newest first."""
    if not player.strip() or not 1 <= days <= 30:
        raise ValueError("A player name and a window of 1 to 30 days are required")
    if client is None:
        with httpx.Client(timeout=15, follow_redirects=True) as owned:
            return fetch_headlines(player, club, days=days, client=owned)
    try:
        response = client.get(
            FEED_URL,
            params={
                "q": f"{player} {club} injury when:{days}d",
                "hl": "en-GB",
                "gl": "GB",
                "ceid": "GB:en",
            },
        )
        response.raise_for_status()
        if len(response.content) > MAX_FEED_BYTES:
            raise ValueError("The news feed response is unexpectedly large")
        root = ET.fromstring(response.content)
    except (httpx.HTTPError, ET.ParseError) as error:
        raise ValueError("The news feed could not be read") from error
    headlines = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=UTC)
        # The feed appends " - Publisher" to every title.
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        if title:
            headlines.append(
                {
                    "title": title,
                    "source": source,
                    "published": published.astimezone(UTC),
                    "link": (item.findtext("link") or "").strip(),
                }
            )
    headlines.sort(key=lambda row: row["published"], reverse=True)
    return headlines[:MAX_HEADLINES]
