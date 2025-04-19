import subprocess
import sys
import json
import time

# 1) Install dependencies if missing
def install_if_missing(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_if_missing("selenium")
install_if_missing("bs4", "bs4")

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# 2) Fetch & save raw HTML
def fetch_and_save_raw_html(product_url, output_path="raw_page.html", headless=True):
    chrome_opts = Options()
    if headless:
        chrome_opts.add_argument("--headless")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.page_load_strategy = "none"

    driver = webdriver.Chrome(options=chrome_opts)
    print(f"[DEBUG] Loading {product_url}")
    driver.get(product_url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "ul#cm-cr-dp-review-list li.review"))
    )
    time.sleep(0.5)

    # Expand all “Read more” links
    driver.execute_script("""
      document
        .querySelectorAll("a[data-hook='expand-collapse-read-more-less'][aria-expanded='false']")
        .forEach(el => el.click());
    """)
    time.sleep(0.5)

    html = driver.page_source
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DEBUG] Saved raw HTML to {output_path}")

    driver.quit()

# 3) Parse reviews from saved HTML
def parse_reviews_from_saved_html(html_path="raw_page.html", output_json="parsed_reviews.json"):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    items = soup.select("ul#cm-cr-dp-review-list li.review")
    print(f"[DEBUG] Found {len(items)} review items in saved HTML")

    reviews = []
    for idx, li in enumerate(items, start=1):
        author_el = li.select_one(".a-profile-name")
        title_el  = li.select_one("a[data-hook='review-title']")
        rating_el = li.select_one("i[data-hook='review-star-rating'] span.a-icon-alt")
        date_el    = li.select_one("span[data-hook='review-date']")
        body_el    = li.select_one("span[data-hook='review-body']")

        if not all([author_el, title_el, rating_el, date_el, body_el]):
            print(f"[DEBUG] Skipping item #{idx}: missing element")
            continue

        reviews.append({
            "author": author_el.get_text(strip=True),
            "title":  title_el.get_text(strip=True),
            "rating": rating_el.get_text(strip=True).split(" out")[0],
            "date":   date_el.get_text(strip=True),
            "body":   body_el.get_text(" ", strip=True)
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    print(f"[DEBUG] Parsed and saved {len(reviews)} reviews to {output_json}")

if __name__ == "__main__":
    PRODUCT_URL = "https://www.amazon.in/dp/B0CG5STQFQ?ref=MARS_NAV_desktop&th=1"
    # Step 1: fetch & save raw HTML
    fetch_and_save_raw_html(PRODUCT_URL, output_path="raw_page.html", headless=True)

    # Step 2: parse saved HTML to JSON
    parse_reviews_from_saved_html(html_path="raw_page.html", output_json="parsed_reviews.json")
