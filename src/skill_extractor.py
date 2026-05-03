import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

# ── Master Skills List ─────────────────────────────────────────────
SKILLS = {
    # Languages
    "python": ["python"],
    "sql": ["sql"],
    "r": ["r programming", " r ,", " r."],
    "scala": ["scala"],
    "java": ["java"],
    "javascript": ["javascript", "js"],

    # ML Frameworks
    "scikit_learn": ["scikit-learn", "sklearn"],
    "pytorch": ["pytorch"],
    "tensorflow": ["tensorflow"],
    "keras": ["keras"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"],
    "catboost": ["catboost"],
    "huggingface": ["huggingface", "hugging face", "transformers"],

    # ML Concepts
    "machine_learning": ["machine learning"],
    "deep_learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"],
    "computer_vision": ["computer vision"],
    "llm": ["llm", "large language model"],
    "rag": ["rag", "retrieval augmented generation"],
    "feature_engineering": ["feature engineering"],
    "model_deployment": ["model deployment", "model serving"],
    "classification": ["classification"],
    "regression": ["regression"],
    "clustering": ["clustering"],
    "time_series": ["time series", "time-series"],
    "recommendation": ["recommendation system", "recommender"],
    "reinforcement_learning": ["reinforcement learning"],
    "generative_ai": ["generative ai", "gen ai", "genai"],

    # Data Engineering
    "spark": ["apache spark", "pyspark", "spark"],
    "kafka": ["kafka", "apache kafka"],
    "airflow": ["airflow", "apache airflow"],
    "dbt": ["dbt"],
    "etl": ["etl", "elt"],
    "data_pipeline": ["data pipeline"],
    "data_warehouse": ["data warehouse"],
    "data_lake": ["data lake"],

    # Databases
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb"],
    "redis": ["redis"],
    "snowflake": ["snowflake"],
    "bigquery": ["bigquery", "big query"],
    "redshift": ["redshift"],
    "elasticsearch": ["elasticsearch"],
    "pinecone": ["pinecone"],

    # Cloud
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud"],
    "azure": ["azure", "microsoft azure"],

    # MLOps & DevOps
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "mlflow": ["mlflow"],
    "kubeflow": ["kubeflow"],
    "git": ["git", "github"],
    "cicd": ["ci/cd", "cicd", "continuous integration"],

    # Visualization
    "tableau": ["tableau"],
    "power_bi": ["power bi", "powerbi"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "plotly": ["plotly"],

    # Stats & Math
    "statistics": ["statistics", "statistical"],
    "probability": ["probability"],
    "linear_algebra": ["linear algebra"],
    "hypothesis_testing": ["hypothesis testing", "a/b testing"],
    "bayesian": ["bayesian"],

    # Soft/Process
    "agile": ["agile", "scrum"],
    "communication": ["communication skills"],
    "stakeholder": ["stakeholder"],
}


def build_matcher(skills_dict):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for skill_key, phrases in skills_dict.items():
        patterns = [nlp.make_doc(p.lower()) for p in phrases]
        matcher.add(skill_key, patterns)
    return matcher


def extract_skills(description, matcher):
    doc = nlp.make_doc(description[:50000])  # spaCy limit safeguard
    matches = matcher(doc)
    found = set()
    for match_id, start, end in matches:
        skill_key = nlp.vocab.strings[match_id]
        found.add(skill_key)
    return found


def build_feature_matrix(df, matcher):
    skill_keys = list(SKILLS.keys())
    rows = []

    for _, row in df.iterrows():
        found = extract_skills(row["description"], matcher)
        skill_vector = {skill: int(skill in found) for skill in skill_keys}
        skill_vector["job_id"]        = row["job_id"]
        skill_vector["job_title"]     = row["job_title"]
        skill_vector["company_name"]  = row["company_name"]
        skill_vector["role_category"] = row["role_category"]
        skill_vector["experience_level"] = row.get("experience_level", "")
        rows.append(skill_vector)

    cols = ["job_id", "job_title", "company_name",
            "role_category", "experience_level"] + skill_keys
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(r"/Users/kishorekumar/Desktop/skillgap analysis/data/raw/ml_jobs.csv")

    print("Building matcher...")
    matcher = build_matcher(SKILLS)

    print("Extracting skills from 1,570 job descriptions...")
    features_df = build_feature_matrix(df, matcher)

    # ── Quick EDA ──────────────────────────────────────────────────
    skill_cols = list(SKILLS.keys())
    skill_counts = features_df[skill_cols].sum().sort_values(ascending=False)

    print(f"\nTop 20 skills across all jobs:")
    print(skill_counts.head(20).to_string())

    print(f"\nSkill presence by role:")
    for role in features_df["role_category"].unique():
        role_df = features_df[features_df["role_category"] == role]
        top = role_df[skill_cols].sum().sort_values(ascending=False).head(5)
        print(f"\n  {role}:")
        for skill, count in top.items():
            print(f"    {skill}: {count}/{len(role_df)}")

    features_df.to_csv(r"/Users/kishorekumar/Desktop/skillgap analysis/data/processed/ml_job_features.csv", index=False,)
    print(f"\nSaved to data/processed/ml_jobs_features.csv")
    print(f"Shape: {features_df.shape}")