import pandas as pd

df = pd.read_csv("/data/raw/postings.csv")

# ML/Data role keywords for title filtering
ML_TITLES = [
    "machine learning", "data scientist", "data science",
    "data engineer", "data analyst", "ml engineer", "mlops",
    "ai engineer", "applied scientist", "research scientist",
    "analytics engineer", "business intelligence", "bi engineer",
    "deep learning", "nlp engineer", "computer vision"
]

def is_ml_role(title):
    if pd.isna(title):
        return False
    t = title.lower()
    return any(kw in t for kw in ML_TITLES)

def assign_category(title):
    t = title.lower()
    if any(x in t for x in ["mlops", "ml platform", "ml infra"]):
        return "mlops_engineer"
    elif any(x in t for x in ["machine learning", "ml engineer"]):
        return "ml_engineer"
    elif any(x in t for x in ["data scientist", "data science", "research scientist", "applied scientist"]):
        return "data_scientist"
    elif any(x in t for x in ["data engineer", "analytics engineer"]):
        return "data_engineer"
    elif any(x in t for x in ["data analyst", "business intelligence", "bi engineer"]):
        return "data_analyst"
    elif any(x in t for x in ["ai engineer", "deep learning", "nlp", "computer vision"]):
        return "ai_engineer"
    else:
        return "ml_engineer"

ml_df = df[df["title"].apply(is_ml_role)].copy()
ml_df["role_category"] = ml_df["title"].apply(assign_category)

# Keep only useful columns
ml_df = ml_df[[
    "job_id", "title", "company_name", "description",
    "location", "formatted_experience_level", "skills_desc",
    "role_category", "job_posting_url"
]].rename(columns={
    "title": "job_title",
    "formatted_experience_level": "experience_level",
    "job_posting_url": "url"
})

# Drop rows with no description
ml_df = ml_df[ml_df["description"].notna()]
ml_df = ml_df[ml_df["description"].str.len() > 200]

print(f"Total LinkedIn jobs: {len(df)}")
print(f"ML/Data jobs filtered: {len(ml_df)}")
print(f"\nBy category:")
print(ml_df["role_category"].value_counts())
print(f"\nTop titles:")
print(ml_df["job_title"].value_counts().head(15))
print(f"\nExperience levels:")
print(ml_df["experience_level"].value_counts())
print(f"\nSample description (first 300 chars):")
print(ml_df["description"].iloc[0][:300])

ml_df.to_csv("/data/raw/ml_jobs.csv", index=False)
