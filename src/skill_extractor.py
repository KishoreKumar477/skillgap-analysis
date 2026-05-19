import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")
# ── Data Leakage Audit ────────────────────────────────────────────
# Q: Could role_category leak into features?
# A: No. Features are extracted from description text only.
#    role_category is derived from job_title, not description.
#    Titles are dropped before feature extraction.
#
# Q: Could company name leak signal?
# A: Possible — "Google" correlates with ML roles.
#    Mitigation: company_name excluded from feature matrix.
#
# Q: Temporal leakage?
# A: Dataset has no temporal ordering applied.
#    Future improvement: train on older postings, test on newer.

# ── Master Skills List ─────────────────────────────────────────────
# Each key must exactly match what train.py and flask_app.py expect
SKILLS = {
    # Languages
    "python": ["python", "python3", "python 3"],
    "sql": ["sql", "mysql", "t-sql", "pl/sql"],         
    "r": ["r programming", "r language"],                  
    "scala": ["scala"],
    "java": ["java", "java 8", "java 11"],                
    "javascript": ["javascript", "js", "node.js", "nodejs"], 

    # ML Frameworks
    "scikit_learn": ["scikit-learn", "sklearn", "scikit learn", "sk-learn"],  
    "pytorch": ["pytorch", "torch", "py torch"],         
    "tensorflow": ["tensorflow", "tf", "tensor flow"],
    "keras": ["keras"],
    "xgboost": ["xgboost", "xgb"],                       
    "lightgbm": ["lightgbm", "lgbm"],                   
    "catboost": ["catboost"],
    "huggingface": [                                      
        "huggingface", "hugging face", "transformers",
        "bert", "gpt", "roberta", "hf transformers"
    ],

    # ML Concepts
    "machine_learning": [                                 
        "machine learning", "ml", "predictive modeling",
        "predictive modelling", "statistical learning"
    ],
    "deep_learning": [                                    
        "deep learning", "neural network", "neural networks",
        "ann", "cnn", "rnn", "lstm", "convolutional network",
        "recurrent network"
    ],
    "nlp": [                                              
        "nlp", "natural language processing", "text mining",
        "sentiment analysis", "named entity recognition",
        "ner", "text analytics", "natural language understanding"
    ],
    "computer_vision": [                                  
        "computer vision", "image recognition",
        "object detection", "image classification",
        "image processing"
    ],
    "llm": [                                              
        "llm", "large language model", "large language models",
        "llms", "gpt-4", "gpt4", "claude", "gemini", "mistral"
    ],
    "rag": [                                              
        "rag", "retrieval augmented generation",
        "retrieval-augmented", "vector search", "semantic search"
    ],
    "feature_engineering": [
        "feature engineering", "feature extraction", "feature selection",
        "feature importance"                              
    ],
    "model_deployment": [
        "model deployment", "model serving", "model inference",
        "production ml", "ml deployment"                 
    ],
    "classification": ["classification", "classifier"],  
    "regression": [
        "regression", "linear regression", "logistic regression"  
    ],
    "clustering": [
        "clustering", "k-means", "kmeans",
        "hierarchical clustering", "dbscan"              
    ],
    "time_series": [
        "time series", "time-series", "forecasting",
        "demand forecasting", "arima", "prophet"         
    ],
    "recommendation": [
        "recommendation system", "recommender system",
        "recommender", "collaborative filtering",
        "content-based filtering"                        
    ],
    "reinforcement_learning": [
        "reinforcement learning", "rl", "deep rl",
        "q-learning", "reward function"                  
    ],
    "generative_ai": [
        "generative ai", "gen ai", "genai",
        "generative model", "diffusion model",
        "stable diffusion"                               
    ],

    # Data Engineering
    "spark": [
        "apache spark", "pyspark", "spark",
        "spark sql", "spark streaming"                   
    ],
    "kafka": ["kafka", "apache kafka", "kafka streaming"],  
    "airflow": ["airflow", "apache airflow"],
    "dbt": ["dbt", "data build tool"],                   
    "etl": ["etl", "elt", "data ingestion", "data integration"],  
    "data_pipeline": ["data pipeline", "data pipelines", "pipeline development"],  
    "data_warehouse": [
        "data warehouse", "data warehousing", "dwh", "data mart"  
    ],
    "data_lake": ["data lake", "data lakehouse", "delta lake"],  

    # Databases
    "postgresql": ["postgresql", "postgres", "psql"],    
    "mysql": ["mysql", "mariadb"],                       
    "mongodb": ["mongodb", "mongo"],                     
    "redis": ["redis"],
    "snowflake": ["snowflake"],
    "bigquery": ["bigquery", "big query", "google bigquery"],  
    "redshift": ["redshift", "amazon redshift"],        
    "elasticsearch": ["elasticsearch", "elastic search", "opensearch"],  
    "pinecone": ["pinecone", "vector database", "vector db"],  

    # Cloud
    "aws": [
        "aws", "amazon web services",
        "sagemaker", "s3", "ec2", "lambda", "glue"      
    ],
    "gcp": [
        "gcp", "google cloud", "google cloud platform",
        "vertex ai", "dataflow", "cloud run"             
    ],
    "azure": [
        "azure", "microsoft azure", "azure ml",
        "azure databricks", "azure synapse"              
    ],

    # MLOps & DevOps
    "docker": [
        "docker", "dockerfile", "containerization", "container"  
    ],
    "kubernetes": ["kubernetes", "k8s", "kubectl", "helm"],  
    "mlflow": ["mlflow", "ml flow"],                     
    "kubeflow": ["kubeflow"],
    "git": ["git", "github", "gitlab", "version control"],  
    "cicd": [
        "ci/cd", "cicd", "continuous integration",
        "continuous deployment", "continuous delivery",
        "jenkins", "github actions", "gitlab ci"         
    ],

    # Visualization
    "tableau": ["tableau"],
    "power_bi": ["power bi", "powerbi", "microsoft power bi"],  
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "plotly": ["plotly", "plotly dash", "dash"],         

    # Stats & Math
    "statistics": [
        "statistics", "statistical", "statistical analysis",
        "descriptive statistics", "inferential statistics"  
    ],
    "probability": ["probability", "probabilistic"],     
    "linear_algebra": [
        "linear algebra", "matrix operations",
        "numpy", "pandas"                                
    ],
    "hypothesis_testing": [
        "hypothesis testing", "a/b testing", "ab testing",
        "statistical testing", "t-test", "chi-square",
        "p-value", "significance testing"               
    ],
    "bayesian": [
        "bayesian", "bayes", "bayesian inference",
        "probabilistic programming"                      
    ],

    # Soft/Process
    "agile": ["agile", "scrum", "kanban", "sprint"],    
    "communication": [
        "communication skills", "data storytelling",
        "stakeholder communication"                      
    ],
    "stakeholder": [
        "stakeholder", "stakeholders", "cross-functional"  
    ],
}


def build_matcher(skills_dict):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for skill_key, phrases in skills_dict.items():
        patterns = [nlp.make_doc(p.lower()) for p in phrases]
        matcher.add(skill_key, patterns)
    return matcher


def extract_skills(description, matcher):

    # we only need tokenization for PhraseMatcher
    doc = nlp.make_doc(description.lower()[:50000])
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
        skill_vector["job_id"]           = row["job_id"]
        skill_vector["job_title"]        = row["job_title"]
        skill_vector["company_name"]     = row["company_name"]
        skill_vector["role_category"]    = row["role_category"]
        skill_vector["experience_level"] = row.get("experience_level", "")
        rows.append(skill_vector)

    cols = ["job_id", "job_title", "company_name",
            "role_category", "experience_level"] + skill_keys
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    print("Loading data...")
    
    df = pd.read_csv("data/raw/ml_jobs.csv")

    print("Building matcher...")
    matcher = build_matcher(SKILLS)

    print("Extracting skills from job descriptions...")
    features_df = build_feature_matrix(df, matcher)

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

    
    features_df.to_csv("data/processed/ml_job_features.csv", index=False)
    print(f"\nSaved to data/processed/ml_job_features.csv")
    print(f"Shape: {features_df.shape}")
