"""Capture real local dashboard pages after preparing artifacts and starting Streamlit."""

import argparse
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--channel", default="chromium")
    parser.add_argument("--output", type=Path, default=Path("docs/screenshots"))
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost"}:
        parser.error("Screenshot capture is limited to the local demo")
    args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200}, device_scale_factor=1)
        for route, title, filename in [
            ("", "Overview", "overview"),
            ("Player_Explorer", "Player Explorer", "player-explorer"),
            ("Match_Predictor", "Match Predictor", "match-predictor"),
        ]:
            page.set_viewport_size({"width": 1440, "height": 1400 if route else 800})
            page.goto(f"{args.url.rstrip('/')}/{route}")
            page.get_by_role("heading", name=title, exact=True).wait_for(timeout=30000)
            if route == "Player_Explorer":
                # Preserve the documented M4 example when newer profile datasets exist.
                page.get_by_label("Season", exact=True).click()
                page.get_by_role("option", name="2023/24", exact=True).click()
                page.get_by_label("Player", exact=True).wait_for()
                page.get_by_label("Player", exact=True).click()
                page.get_by_label("Player", exact=True).fill("Erling Haaland")
                page.get_by_role("option", name="Erling Haaland", exact=True).click()
                page.get_by_role("heading", name="Erling Haaland", exact=True).wait_for()
                page.locator(".js-plotly-plot").first.wait_for()
            if route == "Match_Predictor":
                page.get_by_role("button", name="Predict match", exact=True).click()
                page.locator(".js-plotly-plot").wait_for()
            if page.get_by_test_id("stException").count():
                raise RuntimeError(f"Dashboard failed while capturing {title}")
            page.get_by_test_id("stMetricValue").first.wait_for()
            page.get_by_test_id("stMetricValue").first.locator("div").wait_for()
            page.wait_for_function("document.fonts.status === 'loaded'")
            page.screenshot(
                path=str(args.output / f"{filename}.png"), full_page=True, animations="disabled"
            )
        browser.close()


if __name__ == "__main__":
    main()
