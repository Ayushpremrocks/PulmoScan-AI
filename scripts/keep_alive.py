"""Visit the deployed Streamlit app in a headless browser so it doesn't go to sleep.

If the app is asleep, click Streamlit Community Cloud's wake-up button and wait
until the app has loaded. Exits with status 1 if the app never appears, so the
GitHub Actions run fails and GitHub emails you.

Usage:
    python scripts/keep_alive.py            # uses APP_URL / APP_TEXT env vars or the defaults below

Requires: pip install playwright && python -m playwright install chromium
"""

from __future__ import annotations

import os
import re
import sys
import time

from playwright.sync_api import Page, sync_playwright

APP_URL = os.environ.get("APP_URL") or "https://pulmoscan-ai-ayushpremrocks.streamlit.app/"
APP_TEXT = os.environ.get("APP_TEXT") or ""  # optional extra text that must appear once loaded
WAKE_BUTTON = re.compile(r"get this app back up", re.IGNORECASE)
APP_CONTAINER = '[data-testid="stAppViewContainer"]'  # exists only once the Streamlit app itself renders
LOAD_TIMEOUT_S = 300


def app_has_loaded(page: Page) -> bool:
    # Community Cloud renders the app inside an iframe, so check every frame.
    for frame in page.frames:
        try:
            if frame.locator(APP_CONTAINER).count() == 0:
                continue
            if not APP_TEXT or frame.get_by_text(APP_TEXT).count() > 0:
                return True
        except Exception:
            continue
    return False


def click_wake_button(page: Page) -> bool:
    for frame in page.frames:
        try:
            button = frame.get_by_role("button", name=WAKE_BUTTON)
            if button.count() > 0:
                button.first.click()
                return True
        except Exception:
            continue
    return False


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get("BROWSER_CHANNEL") or None)
        page = browser.new_page()
        print(f"Opening {APP_URL}")
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(10_000)

        if click_wake_button(page):
            print("App was asleep: clicked the wake-up button.")

        deadline = time.time() + LOAD_TIMEOUT_S
        while time.time() < deadline:
            if app_has_loaded(page):
                print("App is up.")
                browser.close()
                return 0
            # The wake-up button can appear late, so keep checking for it too.
            if click_wake_button(page):
                print("App was asleep: clicked the wake-up button.")
            page.wait_for_timeout(5_000)

        page.screenshot(path="keep-alive-failure.png", full_page=True)
        print(f"App did not load within {LOAD_TIMEOUT_S}s. Saved keep-alive-failure.png.")
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
