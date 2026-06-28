"""
CareerLens AI — Internshala JD Scraper
=======================================
Scrapes fresher and internship job listings from Internshala
Saves results to data/job_descriptions.csv

Run: python scraper.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import re

# ── Settings ─────────────────────────────────────────────────
OUTPUT_PATH = "data/job_descriptions.csv"
SLEEP_TIME  = 2   # seconds between requests (be polite to server)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Search Queries ────────────────────────────────────────────
# Format: (search_keyword, domain)
QUERIES = [
    # ML / AI
    ("machine-learning",         "ML / AI"),
    ("artificial-intelligence",  "ML / AI"),
    ("data-science",             "ML / AI"),
    ("deep-learning",            "ML / AI"),
    ("natural-language-processing", "ML / AI"),
    ("computer-vision",          "ML / AI"),

    # Data Analytics
    ("data-analytics",           "Data Analytics"),
    ("data-analyst",             "Data Analytics"),
    ("business-analytics",       "Data Analytics"),
    ("business-analyst",         "Data Analytics"),

    # Web Development
    ("web-development",          "Web Development"),
    ("frontend-development",     "Web Development"),
    ("backend-development",      "Web Development"),
    ("full-stack",               "Web Development"),
    ("python-development",       "Web Development"),
    ("react-js",                 "Web Development"),

    # DevOps / Cloud
    ("devops",                   "DevOps / Cloud"),
    ("cloud-computing",          "DevOps / Cloud"),

    # Mobile Dev
    ("android-development",      "Mobile Dev"),
    ("flutter",                  "Mobile Dev"),

    # Management
    ("product-management",       "Management"),
    ("business-development",     "Management"),
    ("human-resources",          "Management"),
]


def get_listings_from_page(keyword: str, domain: str, page: int = 1) -> list:
    """
    Scrape one page of Internshala listings for a given keyword.
    Returns list of dicts with JD data.
    """
    url = f"https://internshala.com/internships/{keyword}-internship/page-{page}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"  Skipping {keyword} page {page} — status {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all internship cards
        cards = soup.find_all("div", class_="individual_internship")
        if not cards:
            return []

        results = []
        for card in cards:
            try:
                # Company name
                company_tag = card.find("p", class_="company-name")
                company = company_tag.get_text(strip=True) if company_tag else "Unknown"

                # Role / title
                role_tag = card.find("h3", class_="job-internship-name")
                if not role_tag:
                    role_tag = card.find("a", class_="job-title-href")
                role = role_tag.get_text(strip=True) if role_tag else "Internship"

                # Location
                loc_tag = card.find("p", id=re.compile("location_names"))
                if not loc_tag:
                    loc_tag = card.find("a", class_="location_link")
                location = loc_tag.get_text(strip=True) if loc_tag else "Bangalore"

                # Stipend / duration (use as experience proxy)
                duration_tag = card.find("div", class_="item_body", string=re.compile(r"Month|Week"))
                experience = "Internship"

                # Skills from card if available
                skills_tags = card.find_all("span", class_="round_tabs")
                skills = ", ".join([s.get_text(strip=True) for s in skills_tags]) if skills_tags else ""

                # Full JD link
                link_tag = card.find("a", class_="job-title-href")
                if not link_tag:
                    link_tag = card.find("h3", class_="job-internship-name")
                    if link_tag:
                        link_tag = link_tag.find("a")

                full_jd_text = ""
                if link_tag and link_tag.get("href"):
                    jd_url = "https://internshala.com" + link_tag["href"]
                    full_jd_text, extra_skills = get_full_jd(jd_url)
                    if extra_skills and not skills:
                        skills = extra_skills
                    time.sleep(SLEEP_TIME)

                if not full_jd_text or len(full_jd_text) < 50:
                    continue

                results.append({
                    "Company":      company,
                    "Role":         role,
                    "Domain":       domain,
                    "Experience":   experience,
                    "Location":     location if location else "Bangalore",
                    "Skills":       skills,
                    "Full_JD_Text": full_jd_text,
                    "Type":         "Internship",
                })

            except Exception as e:
                continue

        return results

    except Exception as e:
        print(f"  Error fetching {keyword} page {page}: {e}")
        return []


def get_full_jd(url: str) -> tuple:
    """
    Fetch the full job description text from an individual listing page.
    Returns (full_text, skills_string)
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return "", ""

        soup = BeautifulSoup(response.text, "html.parser")

        # Main JD text
        jd_div = soup.find("div", class_="internship_details")
        if not jd_div:
            jd_div = soup.find("div", id="about_internship")
        if not jd_div:
            jd_div = soup.find("div", class_="section_heading")

        full_text = jd_div.get_text(separator=" ", strip=True) if jd_div else ""

        # Skills
        skills_div = soup.find_all("div", class_="round_tabs_container")
        skills = ""
        if skills_div:
            skill_tags = []
            for div in skills_div:
                skill_tags += div.find_all("span", class_="round_tabs")
            skills = ", ".join([s.get_text(strip=True) for s in skill_tags])

        return full_text, skills

    except Exception as e:
        return "", ""


def run_scraper():
    os.makedirs("data", exist_ok=True)

    all_jds  = []
    seen     = set()   # deduplicate by (company + role)

    print("=" * 55)
    print("  CareerLens AI — Internshala Scraper")
    print("=" * 55)

    for keyword, domain in QUERIES:
        print(f"\nScraping: {keyword} ({domain})")

        for page in range(1, 4):   # scrape up to 3 pages per keyword
            print(f"  Page {page}...", end=" ")
            listings = get_listings_from_page(keyword, domain, page)

            new = 0
            for jd in listings:
                key = (jd["Company"].lower(), jd["Role"].lower())
                if key not in seen:
                    seen.add(key)
                    all_jds.append(jd)
                    new += 1

            print(f"{new} new JDs collected (total so far: {len(all_jds)})")

            if not listings:
                break   # no more pages for this keyword

            time.sleep(SLEEP_TIME)

        # Stop early if we have enough
        if len(all_jds) >= 200:
            print("\n  200 JDs collected. Stopping early.")
            break

    # ── Save ─────────────────────────────────────────────────
    if all_jds:
        df = pd.DataFrame(all_jds)
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

        print("\n" + "=" * 55)
        print(f"  DONE! {len(all_jds)} JDs saved to {OUTPUT_PATH}")
        print("\n  Breakdown by domain:")
        print(df["Domain"].value_counts().to_string())
        print("=" * 55)
    else:
        print("\n  No JDs collected. Check your internet connection.")


if __name__ == "__main__":
    run_scraper()
