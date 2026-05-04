from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import xgboost as xgb
import spacy
from spacy.matcher import PhraseMatcher
from pdfminer.high_level import extract_text
import io
import os
import pandas as pd

# Load exact skill columns the model was trained on

from dotenv import load_dotenv

load_dotenv()



app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

app.secret_key = os.environ.get("FLASK_SECRET_KEY")


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

features_df = pd.read_csv(os.path.join(BASE, "data/processed/ml_job_features.csv"))

SKILL_COLS = [c for c in features_df.columns if c not in [
    "job_id", "job_title", "company_name",
    "role_category", "experience_level", "url"
]]
print(f"Loaded {len(SKILL_COLS)} skill columns from training data")
# ── Load Artifacts ────────────────────────────────────────────────
model = xgb.XGBClassifier()
model.load_model(os.path.join(BASE, "data/processed/xgb_model.json"))

with open(os.path.join(BASE, "data/processed/label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

nlp = spacy.load("en_core_web_sm")

# ── Skills ────────────────────────────────────────────────────────
SKILLS = {
    # Programming Languages (6)
    "python": ["python", "python3"],
    "sql": ["sql", "t-sql", "pl/sql"],
    "r": ["r programming", "r language"],
    "scala": ["scala"],
    "java": ["java"],
    "javascript": ["javascript", "js", "node.js"],

    # ML Libraries (8)
    "scikit_learn": ["scikit-learn", "sklearn", "scikit learn"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf", "tensor flow"],
    "keras": ["keras"],
    "xgboost": ["xgboost", "xgb"],
    "lightgbm": ["lightgbm", "lgbm"],
    "catboost": ["catboost"],
    "huggingface": ["huggingface", "hugging face", "transformers", "bert", "gpt"],

    # ML/AI Domains (14)
    "machine_learning": ["machine learning", "ml", "predictive modeling"],
    "deep_learning": ["deep learning", "neural network", "neural networks", "lstm", "cnn", "rnn"],
    "nlp": ["nlp", "natural language processing", "text mining", "sentiment analysis"],
    "computer_vision": ["computer vision", "image recognition", "object detection"],
    "llm": ["llm", "large language model", "llms", "gpt-4", "claude", "gemini"],
    "rag": ["rag", "retrieval augmented generation", "vector search", "semantic search"],
    "feature_engineering": ["feature engineering", "feature extraction", "feature selection"],
    "model_deployment": ["model deployment", "model serving", "production ml"],
    "classification": ["classification", "classifier"],
    "regression": ["regression", "linear regression", "logistic regression"],
    "clustering": ["clustering", "k-means", "kmeans", "dbscan"],
    "time_series": ["time series", "forecasting", "demand forecasting", "arima"],
    "recommendation": ["recommendation system", "recommender system", "collaborative filtering"],
    "reinforcement_learning": ["reinforcement learning", "rl", "q-learning"],
    "generative_ai": ["generative ai", "gen ai", "genai", "diffusion model"],

    # Data Engineering (8)
    "spark": ["apache spark", "pyspark", "spark"],
    "kafka": ["kafka", "apache kafka"],
    "airflow": ["airflow", "apache airflow"],
    "dbt": ["dbt", "data build tool"],
    "etl": ["etl", "elt", "data ingestion"],
    "data_pipeline": ["data pipeline", "data pipelines"],
    "data_warehouse": ["data warehouse", "data warehousing", "dwh"],
    "data_lake": ["data lake", "data lakehouse", "delta lake"],

    # Databases (10)
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis", "redis cache"],
    "snowflake": ["snowflake"],
    "bigquery": ["bigquery", "big query"],
    "redshift": ["redshift"],
    "elasticsearch": ["elasticsearch", "opensearch"],
    "pinecone": ["pinecone", "vector database", "vector db"],

    # Cloud & DevOps (8) — note: no wandb, terraform, databricks, cpp, opencv
    "aws": ["aws", "amazon web services", "sagemaker", "s3", "ec2"],
    "gcp": ["gcp", "google cloud", "vertex ai"],
    "azure": ["azure", "microsoft azure", "azure ml"],
    "docker": ["docker", "dockerfile", "containerization"],
    "kubernetes": ["kubernetes", "k8s", "kubectl"],
    "mlflow": ["mlflow"],
    "kubeflow": ["kubeflow"],
    "git": ["git", "github", "gitlab"],
    "cicd": ["ci/cd", "cicd", "continuous integration", "github actions"],

    # Visualization (5)
    "tableau": ["tableau"],
    "power_bi": ["power bi", "powerbi"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "plotly": ["plotly"],

    # Math & Stats (5)
    "statistics": ["statistics", "statistical", "statistical analysis"],
    "probability": ["probability", "probabilistic"],
    "linear_algebra": ["linear algebra", "matrix", "numpy", "pandas"],
    "hypothesis_testing": ["hypothesis testing", "a/b testing", "ab testing", "p-value"],
    "bayesian": ["bayesian", "bayes"],

    # Soft Skills (3)
    "agile": ["agile", "scrum", "kanban"],
    "communication": ["communication", "presentation", "storytelling"],
    "stakeholder": ["stakeholder management", "stakeholder", "business stakeholder"],
}

SKILL_COLS = list(SKILLS.keys())

ROLE_INFO = {
    "data_analyst":   {"icon": "◈", "label": "Data Analyst",   "color": "#00d4ff"},
    "data_scientist": {"icon": "◉", "label": "Data Scientist", "color": "#a78bfa"},
    "data_engineer":  {"icon": "◎", "label": "Data Engineer",  "color": "#34d399"},
    "ml_engineer":    {"icon": "◆", "label": "ML Engineer",    "color": "#f59e0b"},
    "ai_engineer":    {"icon": "◈", "label": "AI Engineer",    "color": "#f472b6"},
    "mlops_engineer": {"icon": "◉", "label": "MLOps Engineer", "color": "#60a5fa"},
}

ROLE_CORE_SKILLS = {
    "data_analyst":   ["sql", "python", "tableau", "power_bi", "statistics", "hypothesis_testing"],
    "data_scientist": ["python", "statistics", "machine_learning", "scikit_learn", "sql", "pytorch"],
    "data_engineer":  ["sql", "python", "spark", "etl", "aws", "airflow", "docker"],
    "ml_engineer":    ["python", "pytorch", "tensorflow", "deep_learning", "machine_learning", "docker", "mlflow"],
    "ai_engineer":    ["python", "llm", "generative_ai", "rag", "huggingface", "pytorch"],
    "mlops_engineer": ["python", "mlflow", "docker", "kubernetes", "cicd", "aws", "terraform"],
}

LEARNING_RESOURCES = {
    "python":           {"title": "Python for ML",          "url": "https://www.fast.ai",                        "source": "fast.ai"},
    "sql":              {"title": "SQL for Data Science",   "url": "https://mode.com/sql-tutorial",              "source": "Mode"},
    "pytorch":          {"title": "PyTorch Tutorials",      "url": "https://pytorch.org/tutorials",              "source": "pytorch.org"},
    "tensorflow":       {"title": "TensorFlow Basics",      "url": "https://www.tensorflow.org/tutorials",       "source": "tensorflow.org"},
    "scikit_learn":     {"title": "Scikit-learn Guide",     "url": "https://scikit-learn.org/stable/tutorial",   "source": "sklearn.org"},
    "docker":           {"title": "Docker for ML",          "url": "https://docs.docker.com/get-started",        "source": "docker.com"},
    "kubernetes":       {"title": "K8s Crash Course",       "url": "https://kubernetes.io/docs/tutorials",       "source": "kubernetes.io"},
    "mlflow":           {"title": "MLflow Quickstart",      "url": "https://mlflow.org/docs/latest/quickstart",  "source": "mlflow.org"},
    "aws":              {"title": "AWS ML Specialty",       "url": "https://aws.amazon.com/training/learn-about/machine-learning", "source": "AWS"},
    "spark":            {"title": "PySpark Tutorial",       "url": "https://spark.apache.org/docs/latest/api/python", "source": "Apache"},
    "airflow":          {"title": "Airflow Tutorial",       "url": "https://airflow.apache.org/docs/apache-airflow/stable/tutorial", "source": "Apache"},
    "statistics":       {"title": "Statistics for DS",      "url": "https://www.khanacademy.org/math/statistics-probability", "source": "Khan Academy"},
    "machine_learning": {"title": "ML Crash Course",        "url": "https://developers.google.com/machine-learning/crash-course", "source": "Google"},
    "deep_learning":    {"title": "Deep Learning Book",     "url": "https://www.deeplearningbook.org",           "source": "Goodfellow et al."},
    "nlp":              {"title": "HuggingFace NLP Course", "url": "https://huggingface.co/learn/nlp-course",    "source": "HuggingFace"},
    "llm":              {"title": "LLM Bootcamp",           "url": "https://fullstackdeeplearning.com/llm-bootcamp", "source": "FSDL"},
    "rag":              {"title": "RAG from Scratch",       "url": "https://github.com/langchain-ai/rag-from-scratch", "source": "LangChain"},
    "tableau":          {"title": "Tableau Public Training","url": "https://public.tableau.com/app/learn/training", "source": "Tableau"},
    "power_bi":         {"title": "Power BI Learning",      "url": "https://learn.microsoft.com/en-us/power-bi", "source": "Microsoft"},
    "git":              {"title": "Git Handbook",           "url": "https://guides.github.com/introduction/git-handbook", "source": "GitHub"},
    "cicd":             {"title": "GitHub Actions Guide",   "url": "https://docs.github.com/en/actions",         "source": "GitHub"},
    "snowflake":        {"title": "Snowflake Quickstart",   "url": "https://quickstarts.snowflake.com",          "source": "Snowflake"},
    "dbt":              {"title": "dbt Learn",              "url": "https://courses.getdbt.com",                 "source": "dbt Labs"},
    "hypothesis_testing":{"title": "Stats Testing Guide",  "url": "https://www.statstest.com",                  "source": "statstest.com"},
    "generative_ai":    {"title": "Generative AI for Devs","url": "https://www.cloudskillsboost.google/paths/183", "source": "Google"},
    "databricks":       {"title": "Databricks Academy",    "url": "https://www.databricks.com/learn/training",  "source": "Databricks"},
    "terraform":        {"title": "Terraform Learn",        "url": "https://developer.hashicorp.com/terraform/tutorials", "source": "HashiCorp"},
}

# ── Matcher ───────────────────────────────────────────────────────
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
for skill_key, phrases in SKILLS.items():
    patterns = [nlp.make_doc(p.lower()) for p in phrases]
    matcher.add(skill_key, patterns)

def extract_skills_from_text(text):
    doc = nlp.make_doc(text.lower()[:50000])
    matches = matcher(doc)
    found = set()
    for match_id, _, _ in matches:
        found.add(nlp.vocab.strings[match_id])
    return found

def text_to_vector(text):
    found = extract_skills_from_text(text)
    return np.array([int(s in found) for s in SKILL_COLS]), found

# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    roles = list(ROLE_CORE_SKILLS.keys())
    return render_template("index.html", roles=roles, role_info=ROLE_INFO)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        text = ""

        if "pdf" in request.files and request.files["pdf"].filename:
            pdf_file = request.files["pdf"]
            pdf_bytes = pdf_file.read()
            text = extract_text(io.BytesIO(pdf_bytes))
        else:
            text = request.form.get("resume_text", "")

        target_role = request.form.get("target_role", "data_scientist")

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        vector, found_skills = text_to_vector(text)
        proba = model.predict_proba([vector])[0]
        pred_idx = int(np.argmax(proba))
        pred_role = le.classes_[pred_idx]

        # All role scores
        role_scores = {
            le.classes_[i]: round(float(proba[i]) * 100, 1)
            for i in range(len(le.classes_))
        }

        # Skill gap
        core = ROLE_CORE_SKILLS[target_role]
        present = [s for s in core if s in found_skills]
        missing = [s for s in core if s not in found_skills]
        readiness = round(len(present) / len(core) * 100)

        # All detected skills
        detected = sorted(list(found_skills))

        # Learning recommendations for missing skills
        recommendations = []
        for skill in missing:
            if skill in LEARNING_RESOURCES:
                rec = LEARNING_RESOURCES[skill].copy()
                rec["skill"] = skill.replace("_", " ").title()
                recommendations.append(rec)

        return jsonify({
            "predicted_role":  pred_role,
            "confidence":      round(float(proba[pred_idx]) * 100, 1),
            "role_scores":     role_scores,
            "target_role":     target_role,
            "readiness":       readiness,
            "present_skills":  present,
            "missing_skills":  missing,
            "detected_skills": detected,
            "recommendations": recommendations,
            "role_info":       ROLE_INFO,
        })
    except Exception as e:
        print(f"DEBUG ERROR: {e}") 
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
        host='0.0.0.0',
        port=port
    )
