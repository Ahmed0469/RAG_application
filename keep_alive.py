import os
from playwright.sync_api import sync_playwright

URL = os.environ["APP_URL"]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60000)

    # If the app already went to sleep, click the wake-up button
    wake_button = page.get_by_role("button", name="Yes, get this app back up!")
    try:
        wake_button.wait_for(timeout=10000)
        wake_button.click()
        print("App was asleep, wake button clicked")
        page.wait_for_timeout(30000)  # give it time to boot
    except Exception:
        print("App already awake")

    # Stay on the page long enough for the websocket session to register
    page.wait_for_timeout(15000)

    page.screenshot(path="screenshot.png")
    browser.close()
