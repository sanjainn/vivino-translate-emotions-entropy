# streamlit run vivino_compare_vintages_selenium.py.py
# Simple, auto-seeking app: paste Vivino URL → get table + chart(s) with download buttons.

import re, time, io, json
import pandas as pd
import streamlit as st

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

# import matplotlib.pyplot as plt
# from matplotlib.ticker import StrMethodFormatter, FixedLocator, FuncFormatter

import altair as alt
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from typing import Optional

DEFAULT_URL = "https://www.vivino.com/US/en/domaine-de-la-romanee-conti-romanee-conti-grand-cru/w/83912?ref=nav-search#all_reviews"


# ---------- Selenium helpers ----------
def first_present(driver, selectors, timeout=12, visible=True):
    last_err = None
    for how, sel in selectors:
        try:
            cond = EC.visibility_of_element_located if visible else EC.presence_of_element_located
            return WebDriverWait(driver, timeout).until(cond((how, sel)))
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise TimeoutError(f"None of the selectors became present/visible: {selectors}")


def click_js(driver, el):
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    time.sleep(0.15)
    try:
        el.click()
    except Exception:
        try:
            ActionChains(driver).move_to_element(el).pause(0.05).click().perform()
        except Exception:
            driver.execute_script("arguments[0].click();", el)


def close_popups(driver):
    texts = ["Accept all", "Accept All", "I agree", "Got it", "Accept cookies", "Allow all", "OK", "Accept"]
    for t in texts:
        try:
            btn = WebDriverWait(driver, 2.5).until(EC.element_to_be_clickable((By.XPATH, f"//button[contains(.,'{t}')]")))
            click_js(driver, btn)
            break
        except Exception:
            pass


def find_compare_block(driver):
    """Return (section_like_node, 'Show all' element or None)."""
    container = None
    show_all = None
    for xp in [
        "//h2[contains(.,'Compare Vintages')]",
        "//h3[contains(.,'Compare Vintages')]",
        "//*[self::h1 or self::h2 or self::h3][contains(.,'Compare Vintages')]",
    ]:
        try:
            container = WebDriverWait(driver, 2).until(EC.visibility_of_element_located((By.XPATH, xp)))
            break
        except Exception:
            pass
    if container is not None:
        try:
            show_all = container.find_element(By.XPATH, ".//following::*[(self::a or self::button) and contains(.,'Show all')][1]")
        except Exception:
            pass
    if show_all is None:
        try:
            node = driver.execute_script("""
                const lower=s=>(s||'').toLowerCase();
                for (const el of document.querySelectorAll('body *')) {
                  if (lower(el.innerText).includes('compare vintages')) return el;
                }
                return null;
            """)
            if node:
                container = node
                show_all = driver.execute_script("""
                    const root = arguments[0];
                    const nearby = root.closest('section,div')?.querySelectorAll('a,button') || [];
                    for (const el of nearby) if ((el.innerText||'').toLowerCase().includes('show all')) return el;
                    for (const el of document.querySelectorAll('a,button')) {
                      if ((el.innerText||'').toLowerCase().includes('show all')) return el;
                    }
                    return null;
                """, node)
        except Exception:
            pass
    return container, show_all


def wait_for_modal(driver):
    return first_present(driver, [
        (By.CSS_SELECTOR, "div[role='dialog']"),
        (By.CSS_SELECTOR, "div[aria-modal='true']"),
        (By.XPATH, "//div[contains(@class,'modal') or contains(@class,'Modal')]"),
    ], timeout=15, visible=True)


def scroll_modal_to_bottom(driver, modal, max_loops=80):
    last_h = 0
    for _ in range(max_loops):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", modal)
        time.sleep(0.45)
        h = driver.execute_script("return arguments[0].scrollHeight;", modal)
        if h == last_h:
            break
        last_h = h


def extract_rows_text(modal):
    items = modal.find_elements(By.CSS_SELECTOR, "*")
    out, seen = [], set()
    for it in items:
        try:
            txt = it.text.strip()
        except Exception:
            continue
        if not txt:
            continue
        if re.search(r"\b(19\d{2}|20\d{2})\b", txt):
            if txt not in seen:
                seen.add(txt)
                out.append(txt)
    return out


# ---- helpers to pull metrics from text ----
def _extract_avg_rating(txt: str) -> Optional[float]:
    """
    Find a float like 4.6 that plausibly represents an average rating (0–5).
    Avoid years/prices; look for 0.0–5.0 with one decimal.
    """
    # prefer numbers near the word 'ratings' or 'rating'
    near = re.search(r"([0-5]\.\d)\s*(?:★|stars?)?\s*(?:\d[\d,\.]*\s+ratings?)?", txt, flags=re.IGNORECASE)
    if near:
        try:
            val = float(near.group(1))
            if 0.0 <= val <= 5.0:
                return val
        except Exception:
            pass
    # fallback: any decimal 0–5
    cands = []
    for m in re.findall(r"(?<!\d)([0-5]\.\d)(?!\d)", txt):
        try:
            v = float(m)
            if 0.0 <= v <= 5.0:
                cands.append(v)
        except Exception:
            pass
    return cands[0] if cands else None


def _extract_reviews_count_from_text(txt: str) -> Optional[int]:
    """
    If the row explicitly says '123 reviews', capture it (some locales do).
    """
    m = re.search(r"\b([0-9][\d,\.]*)\s+reviews?\b", txt, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1).replace(",", "").replace(".", ""))
        except Exception:
            return None
    return None


def parse_row(txt):
    if "No average price available" in txt:
        return None
    keep = ("Available for purchase" in txt) or ("Only available from external shops" in txt)
    if not keep:
        return None

    m_year = re.search(r"\b(19\d{2}|20\d{2})\b", txt)
    if not m_year:
        return None
    year = int(m_year.group(1))

    m_prices = re.findall(r"(?:HK\$|A\$|C\$|US\$|€|\$|£|₹)\s?[\d,]+(?:\.\d{2})?", txt)
    if not m_prices:
        return None
    price_str = m_prices[-1].replace("US$", "$").strip()

    if price_str.startswith("HK$"):
        currency = "HKD"
    elif price_str.startswith("A$"):
        currency = "AUD"
    elif price_str.startswith("C$"):
        currency = "CAD"
    elif price_str.startswith("€"):
        currency = "EUR"
    elif price_str.startswith("£"):
        currency = "GBP"
    elif price_str.startswith("₹"):
        currency = "INR"
    else:
        currency = "USD"

    amount = re.sub(r"^[^\d]+", "", price_str)
    availability = "Available for purchase" if "Available for purchase" in txt else \
                   "Only available from external shops (avg price)"

    avg_rating = _extract_avg_rating(txt)
    reviews_count = _extract_reviews_count_from_text(txt)  # often not present; we’ll fill via API

    return {
        "Vintage": year,
        "Price": amount,
        "Currency": currency,
        "Availability": availability,
        "AvgRating": avg_rating,
        "ReviewsCount": reviews_count,
    }


def seek_show_all_adaptive(driver, max_seconds=75, per_step_pause=0.7):
    """Auto-scrolls until 'Compare Vintages → Show all' appears or bottom is reached."""
    deadline = time.time() + max_seconds
    last_height = -1
    while time.time() < deadline:
        _, show_all = find_compare_block(driver)
        if show_all:
            return show_all
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(per_step_pause)
        close_popups(driver)
        h = driver.execute_script("return document.documentElement.scrollHeight;")
        if h == last_height:
            for _ in range(3):
                driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(0.6)
            _, show_all = find_compare_block(driver)
            if show_all:
                return show_all
            break
        last_height = h
    return None


# ---------- Per-vintage URL helper ----------
def build_vintage_url(input_url: str, vintage_year: int) -> str:
    """
    Return the same product URL but forced to the given vintage:
    - sets/overwrites ?year=YYYY
    - removes price_id (if present)
    - preserves other query params and fragment
    """
    parts = urlsplit(input_url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    q["year"] = str(int(vintage_year))
    q.pop("price_id", None)
    new_query = urlencode(q, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


# ---------- Browser-side JSON fetch + diggers (reliable with site cookies) ----------
_WINE_ID_RE = re.compile(r"/w/(\d+)(?:[/?#]|$)")
def extract_wine_id_from_url(product_url: str) -> Optional[int]:
    m = _WINE_ID_RE.search(urlsplit(product_url).path)
    return int(m.group(1)) if m else None


def browser_get_json(driver, url: str):
    """
    Run fetch() inside the page (with cookies/headers). Returns dict or None.
    """
    script = """
    const url = arguments[0];
    const done = arguments[arguments.length - 1];
    fetch(url, {credentials: 'include'})
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(data => done({ok:true, data}))
      .catch(err => done({ok:false, error: String(err)}));
    """
    try:
        res = driver.execute_async_script(script, url)
        if res and res.get("ok") and isinstance(res.get("data"), dict):
            return res["data"]
    except Exception:
        pass
    return None


def dig_metrics_for_year(obj, target_year: int):
    """
    Look for:
      - ratings average (e.g., 4.6) under keys like ratings_average / average_rating
      - reviews_count (count of written reviews)
    within nodes whose 'year' == target_year or in 'statistics' for that year.
    """
    avg = None
    reviews = None

    def maybe_update(d):
        nonlocal avg, reviews
        if not isinstance(d, dict):
            return
        if avg is None:
            for k in ("ratings_average", "average_rating", "vintage_rating", "rating"):
                v = d.get(k)
                if isinstance(v, (int, float)) and 0.0 <= float(v) <= 5.0:
                    avg = float(v)
                    break
        if reviews is None:
            for k in ("reviews_count", "review_count", "reviewsTotalCount"):
                v = d.get(k)
                if isinstance(v, (int, float)):
                    reviews = int(v)
                    break

    def walk(node, cur_year=None):
        nonlocal avg, reviews
        if avg is not None and reviews is not None:
            return
        if isinstance(node, dict):
            y = cur_year
            if "year" in node and str(node["year"]).isdigit():
                y = int(node["year"])
            if y == target_year:
                maybe_update(node)
                for key in ("statistics", "vintage_statistics", "aggregated_statistics"):
                    maybe_update(node.get(key, {}))
            for v in node.values():
                walk(v, y)
        elif isinstance(node, list):
            for it in node:
                walk(it, cur_year)

    walk(obj, None)
    return avg, reviews


def fill_avg_and_reviews_via_api(driver, base_product_url: str, df: pd.DataFrame, logs):
    wine_id = extract_wine_id_from_url(base_product_url)
    if not wine_id:
        logs.append("[vivino] Could not parse wine_id from URL; skipping API metrics fill.")
        return df

    years = df["Vintage"].astype(int).tolist()
    for y in years:
        # Skip if already parsed from text
        if pd.notna(df.loc[df["Vintage"] == y, "AvgRating"]).all() and pd.notna(df.loc[df["Vintage"] == y, "ReviewsCount"]).all():
            continue

        # Try the most informative endpoint first
        endpoints = [
            f"https://www.vivino.com/api/wines/{wine_id}?year={y}",
            f"https://www.vivino.com/api/wines/{wine_id}/vintages?year={y}",
        ]
        avg, reviews = None, None
        for api in endpoints:
            data = browser_get_json(driver, api)
            if not data:
                continue
            a, r = dig_metrics_for_year(data, y)
            if a is not None:
                avg = a
            if r is not None:
                reviews = r
            if avg is not None and reviews is not None:
                break

        if avg is not None:
            df.loc[df["Vintage"] == y, "AvgRating"] = float(avg)
        if reviews is not None:
            df.loc[df["Vintage"] == y, "ReviewsCount"] = int(reviews)

        time.sleep(0.15)  # tiny delay to be polite
    return df


def scrape_compare_vintages(url):
    logs = []
    def log(msg): logs.append(msg); print(msg, flush=True)

    opts = webdriver.ChromeOptions()
    # Always non-headless for reliability (kept simple per your request)
    opts.add_argument("--window-size=1400,1000")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    try:
        log("[vivino] Opening page…")
        driver.get(url); time.sleep(1.0)
        close_popups(driver)

        log("[vivino] Auto-seeking 'Compare Vintages'…")
        show_all = seek_show_all_adaptive(driver)
        if not show_all:
            raise TimeoutError("Could not find the Compare Vintages section / Show all link.")

        log("[vivino] Found — opening modal.")
        click_js(driver, show_all)

        log("[vivino] Waiting for modal…")
        modal = wait_for_modal(driver)
        driver.execute_script("arguments[0].scrollTop = 0;", modal); time.sleep(0.3)

        log("[vivino] Scrolling modal to bottom…")
        scroll_modal_to_bottom(driver, modal)

        log("[vivino] Parsing rows…")
        rows_text = extract_rows_text(modal)

        records = {}
        for txt in rows_text:
            rec = parse_row(txt)
            if rec:
                records.setdefault(rec["Vintage"], rec)

        if not records:
            raise RuntimeError("No priced vintages found.")

        # Base DF & per-vintage URL
        df = pd.DataFrame(sorted(records.values(), key=lambda r: r["Vintage"], reverse=True))
        # Ensure column order
        base_cols = ["Vintage", "Price", "Currency", "Availability", "AvgRating", "ReviewsCount"]
        for c in base_cols:
            if c not in df.columns:
                df[c] = None
        df = df[base_cols]
        df["URL"] = df["Vintage"].apply(lambda y: build_vintage_url(url, y))

        # Fill missing AvgRating/ReviewsCount via browser-side JSON APIs
        missing = df[df["AvgRating"].isna() | df["ReviewsCount"].isna()]
        if not missing.empty:
            log("[vivino] Filling AvgRating/ReviewsCount via site JSON…")
            df = fill_avg_and_reviews_via_api(driver, url, df, logs)

        return df, "\n".join(logs)
    finally:
        driver.quit()


# ---------- Charting (+ download buttons) ----------
def show_charts_with_downloads(df: pd.DataFrame):
    """
    Streamlit + Altair:
    - Bullet markers for every priced year
    - Text labels on each point: "YYYY — 12,345"
    - X-axis shows exactly the years we have
    """
    df2 = df.copy()
    df2["PriceFloat"] = df2["Price"].astype(str).str.replace(",", "", regex=False).astype(float)
    df2 = df2.dropna(subset=["PriceFloat"])
    df2["Vintage"] = df2["Vintage"].astype(int)
    df2["Label"] = df2.apply(lambda r: f"{r['Vintage']} — {r['PriceFloat']:,.0f}", axis=1)

    for cur, sub in df2.groupby("Currency"):
        sub = sub.sort_values("Vintage")
        if sub.empty:
            continue
        years = sub["Vintage"].tolist()
        label_angle = -40 if len(years) > 12 else 0

        base = alt.Chart(sub).encode(
            x=alt.X(
                "Vintage:Q",
                axis=alt.Axis(values=years, format="d", labelAngle=label_angle, title="Vintage Year"),
                scale=alt.Scale(nice=False),
            ),
            y=alt.Y("PriceFloat:Q", axis=alt.Axis(title=f"Price ({cur})", format=",.0f")),
            tooltip=[
                alt.Tooltip("Vintage:Q", title="Year", format="d"),
                alt.Tooltip("PriceFloat:Q", title="Price", format=",.0f"),
                alt.Tooltip("Currency:N", title="Currency"),
            ],
        )

        line = base.mark_line()
        points = base.mark_point(size=70)
        text = base.mark_text(align="center", dy=-10, fontSize=11).encode(text="Label:N")

        chart = (line + points + text).properties(
            title=f"Vintage year vs price ({cur})",
            width="container",
            height=380,
        ).configure_axis(gridColor="rgba(128,128,128,0.25)").interactive()

        st.altair_chart(chart, use_container_width=True)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Vivino: Compare Vintages Scraper", page_icon="🍷", layout="wide")
st.title("🍷 Vivino – Compare Vintages → Years & Prices")
st.caption("Paste a Vivino product URL, then click the button. The app finds the Compare Vintages modal, extracts priced vintages, adds per-vintage URLs, AvgRating & ReviewsCount, and plots years vs price.")

url = st.text_input("Vivino product URL", value=DEFAULT_URL)
go = st.button("🚀 Fetch Years & Prices", type="primary", use_container_width=True)

if go:
    with st.spinner("Working…"):
        try:
            df, logs = scrape_compare_vintages(url)
        except Exception as e:
            st.error(f"Failed: {e}")
            with st.expander("Debug log"):
                st.text(logs if 'logs' in locals() else "(no logs captured)")
            st.stop()

    st.success(f"Found {len(df)} priced vintage(s).")
    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="vivino_romanee_conti_vintages.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Chart(s): Years vs Price")
    show_charts_with_downloads(df)

    with st.expander("Raw log"):
        st.text(logs)
else:
    st.info("Paste the link above and click **Fetch Years & Prices**.")
