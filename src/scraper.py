import requests
import pandas as pd
import time
from bs4 import BeautifulSoup

TAGS = [
    "machine-learning",
    "data-science",
    "artificial-intelligence",
    "mlops",
    "data-engineering",
]

TAG_TO_ROLE = {
    "machine-learning":        "ml_engineer",
    "data-science":            "data_scientist",
    "artificial-intelligence": "ai_engineer",
    "mlops":                   "mlops_engineer",
    "data-engineering":        "data_engineer",
}

# Must contain at least 2 of these in description to be considered ML
ML_SIGNALS = [
    "machine learning", "deep learning", "neural network",
    "python", "scikit-learn", "xgboost", "pytorch", "tensorflow",
    "data pipeline", "etl", "feature engineering", "model training",
    "nlp", "llm", "langchain", "mlflow", "data warehouse",
    "sql", "pandas", "spark", "huggingface", "random forest",
    "classification", "regression", "clustering", "data science",
    "data engineer", "ml model", "ai model", "gradient boosting"
]

# If title contains any of these — reject immediately
JUNK_TITLES = [
    "volunteer", "conference", "business development", "sales",
    "recruiter", "attorney", "therapist", "social worker",
    "designer", "ux", "frontend", "ios", "android",
    "marketing", "finance", "payroll", "localization",
    "translator", "proofreader", "crypto trader", "billing",
    "account manager", "sourcer", "devops", "security engineer"
]

def clean_html(html_text):
    if not html_text:
        return ""
    return BeautifulSoup(html_text, "html.parser").get_text(separator=" ")

def is_genuine_ml_job(title, description):
    # Reject junk titles immediately
    t = title.lower()
    if any(junk in t for junk in JUNK_TITLES):
        return False
    
    # Check description for ML signal strength
    desc_lower = description.lower()
    matches = sum(1 for signal in ML_SIGNALS if signal in desc_lower)
    return matches >= 3  # needs at least 3 ML keywords in description

def fetch_jobs_for_tag(tag, max_pages=3):
    all_jobs = []
    for page in range(1, max_pages + 1):
        url = f"https://www.arbeitnow.com/api/job-board-api?tags[]={tag}&page={page}"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                break
            data = response.json()
            jobs = data.get("data", [])
            if not jobs:
                break
            print(f"  [{tag}] page {page}: {len(jobs)} raw jobs")
            all_jobs.extend(jobs)
            time.sleep(1)
        except Exception as e:
            print(f"  [{tag}] error: {e}")
            break
    return all_jobs

def parse_job(job, role_category):
    raw_desc = job.get("description", "")
    clean_desc = clean_html(raw_desc)
    title = job.get("title", "")

    if not is_genuine_ml_job(title, clean_desc):
        return None

    return {
        "job_title":     title,
        "company":       job.get("company_name", ""),
        "role_category": role_category,
        "tags":          ", ".join(job.get("tags", [])),
        "description":   clean_desc,
        "url":           job.get("url", ""),
        "date":          job.get("created_at", ""),
        "location":      job.get("location", "Remote"),
        "source":        "arbeitnow"
    }

if __name__ == "__main__":
    # Install beautifulsoup4 if needed
    print("=== Fetching & Filtering ML/Data Jobs ===\n")

    all_jobs = []
    seen_urls = set()

    for tag in TAGS:
        print(f"\nFetching: {tag}")
        raw_jobs = fetch_jobs_for_tag(tag, max_pages=3)

        kept = 0
        for job in raw_jobs:
            url = job.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                parsed = parse_job(job, TAG_TO_ROLE[tag])
                if parsed:
                    all_jobs.append(parsed)
                    kept += 1

        print(f"  [{tag}] kept after filtering: {kept}")
        time.sleep(2)

    df = pd.DataFrame(all_jobs)

    print(f"\n=== Final Results ===")
    print(f"Total genuine ML/Data jobs: {len(df)}")
    print(f"\nBy category:")
    print(df["role_category"].value_counts())
    print(f"\nSample titles:")
    for t in df["job_title"].head(20).tolist():
        print(f"  - {t}")

    df.to_csv("/Users/kishorekumar/Desktop/skillgap analysis/data/raw/ml_jobs.csv", index=False)
    print(f"\nSaved to data/raw/ml_jobs.csv")