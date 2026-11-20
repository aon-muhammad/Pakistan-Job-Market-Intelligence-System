import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

os.makedirs("models", exist_ok=True)

# ==========================================
# 1. SALARY REGRESSION
# ==========================================
df_sal = pd.read_csv("pakistan_tech_jobs_final.csv")
df_sal.drop("job_description", inplace=True, axis=1)
df_sal["education"] = df_sal["education"].fillna(df_sal["education"].mode()[0])
df_sal["job_title"] = df_sal["job_title"].fillna(df_sal["job_title"].mode()[0])

cols_for_sal = [
    "experience_min",
    "city",
    "job_category",
    "work_mode",
    "education",
    "experience_max",
    "salary_min",
    "salary_max",
    "required_skills",
]
reg_df = df_sal[cols_for_sal].copy()
reg_df["experience"] = (reg_df["experience_min"] + reg_df["experience_max"]) / 2
reg_df["avg_sal"] = (reg_df["salary_min"] + reg_df["salary_max"]) / 2

senior_kw = [
    "senior",
    "sr",
    "lead",
    "principal",
    "architect",
    "head",
    "manager",
    "director",
]
reg_df["is_senior"] = (
    df_sal["job_title"]
    .fillna("")
    .str.lower()
    .apply(lambda t: int(any(k in t for k in senior_kw)))
)

reg_df["num_skills"] = (
    reg_df["required_skills"]
    .fillna("")
    .apply(lambda s: len([x for x in str(s).split(",") if x.strip()]))
)
high_skills = [
    "aws",
    "cloud",
    "devops",
    "docker",
    "kubernetes",
    "python",
    "machine learning",
    "react",
]
reg_df["has_high_skill"] = (
    reg_df["required_skills"]
    .fillna("")
    .str.lower()
    .apply(lambda s: int(any(k in s for k in high_skills)))
)

top_cities = ["Karachi", "Lahore", "Islamabad", "Rawalpindi"]
reg_df["city_tier"] = reg_df["city"].apply(lambda x: x if x in top_cities else "Other")
reg_df = pd.get_dummies(
    reg_df, drop_first=True, columns=["work_mode", "city_tier", "job_category"], dtype=int
)
reg_df["education"] = (
    reg_df["education"].replace({"Master's": 1, "Bachelor's": 0}).astype(int)
)
reg_df.drop(
    columns=[
        "city",
        "required_skills",
        "experience_min",
        "experience_max",
        "salary_min",
        "salary_max",
    ],
    inplace=True,
)

cap_sal = reg_df["avg_sal"].quantile(0.98)
reg_df = reg_df[reg_df["avg_sal"] <= cap_sal].copy()

X_sal = reg_df.drop(columns=["avg_sal"])
y_sal = reg_df["avg_sal"]
X_sal_train, X_sal_test, y_sal_train, y_sal_test = train_test_split(
    X_sal, y_sal, test_size=0.25, random_state=42
)
model_sal = LinearRegression()
model_sal.fit(X_sal_train, y_sal_train)
y_sal_pred = model_sal.predict(X_sal_test)

salary_r2 = r2_score(y_sal_test, y_sal_pred)
salary_mae = mean_absolute_error(y_sal_test, y_sal_pred)

assert salary_r2 == 0.8122284015181946, f"Salary R2 mismatch: {salary_r2}"
assert abs(salary_mae - 15360.436) < 0.01, f"Salary MAE mismatch: {salary_mae}"
print("PASS")

joblib.dump(
    {
        "model": model_sal,
        "columns": list(X_sal.columns),
        "mae": salary_mae,
        "r2": salary_r2,
        "cap_sal": cap_sal,
        "top_cities": top_cities,
        "high_skills": high_skills,
    },
    os.path.join("models", "salary.pkl"),
)

# ==========================================
# 2. KNN MODEL
# ==========================================
df_knn = pd.read_csv("pakistan_tech_jobs_final.csv")
df_knn.drop("job_description", inplace=True, axis=1)
df_knn["education"] = df_knn["education"].fillna(df_knn["education"].mode()[0])
df_knn["job_title"] = df_knn["job_title"].fillna(df_knn["job_title"].mode()[0])

knn_cols = [
    "salary_min",
    "salary_max",
    "experience_min",
    "experience_max",
    "job_title",
    "education",
]
skills_ohe = df_knn["required_skills"].str.lower().str.get_dummies(sep=", ")
knn_df = df_knn[knn_cols].copy()
knn_df["is_senior"] = (
    knn_df["job_title"]
    .fillna("")
    .str.lower()
    .apply(lambda t: int(any(k in t for k in senior_kw)))
)
knn_df["avg_salary"] = (knn_df["salary_min"] + knn_df["salary_max"]) / 2
knn_df["avg_exp"] = (knn_df["experience_min"] + knn_df["experience_max"]) / 2
knn_df.drop(
    columns=["salary_min", "salary_max", "experience_max", "experience_min", "job_title"],
    inplace=True,
)
knn_df["education"] = knn_df["education"].replace({"Bachelor's": 0, "Master's": 1})

knn_scaler = StandardScaler()
knn_df[["avg_salary", "avg_exp"]] = knn_scaler.fit_transform(knn_df[["avg_salary", "avg_exp"]])
knn_df = pd.concat([knn_df, skills_ohe], axis=1)

nn_model = NearestNeighbors(metric="euclidean")
nn_model.fit(knn_df)

jobs_needed = df_knn[
    [
        "job_title",
        "company",
        "city",
        "work_mode",
        "job_category",
        "experience_min",
        "experience_max",
        "salary_min",
        "salary_max",
        "education",
        "required_skills",
    ]
].copy()

joblib.dump(
    {
        "nn": nn_model,
        "scaler": knn_scaler,
        "columns": list(knn_df.columns),
        "jobs": jobs_needed,
    },
    os.path.join("models", "knn.pkl"),
)

# ==========================================
# 3. LIKELIHOOD MODEL
# ==========================================
df_like = pd.read_csv("pakistan_tech_jobs_final.csv")
df_like.drop("job_description", inplace=True, axis=1)
df_like["education"] = df_like["education"].fillna(df_like["education"].mode()[0])
df_like["job_title"] = df_like["job_title"].fillna(df_like["job_title"].mode()[0])

cols_for_like = [
    "experience_min",
    "city",
    "job_category",
    "work_mode",
    "education",
    "experience_max",
    "required_skills",
]
reg_df_like = df_like[cols_for_like].copy()
reg_df_like["experience"] = (
    reg_df_like["experience_min"] + reg_df_like["experience_max"]
) / 2
reg_df_like["num_skills"] = (
    reg_df_like["required_skills"]
    .fillna("")
    .apply(lambda s: len([x for x in str(s).split(",") if x.strip()]))
)
reg_df_like["education"] = (
    reg_df_like["education"].replace({"Master's": 1, "Bachelor's": 0}).astype(int)
)

scaler_mm = MinMaxScaler()
exp_norm = scaler_mm.fit_transform(reg_df_like[["experience"]]).flatten()
skills_norm = scaler_mm.fit_transform(reg_df_like[["num_skills"]]).flatten()
edu_norm = scaler_mm.fit_transform(reg_df_like[["education"]]).flatten()

reg_df_like["likelihood"] = np.round(
    (exp_norm * 50) + (skills_norm * 35) + (edu_norm * 15), 2
)

reg_df_like["is_senior"] = (
    df_like["job_title"]
    .fillna("")
    .str.lower()
    .apply(lambda t: int(any(k in t for k in senior_kw)))
)
reg_df_like["has_high_skill"] = (
    reg_df_like["required_skills"]
    .fillna("")
    .str.lower()
    .apply(lambda s: int(any(k in s for k in high_skills)))
)

reg_df_like["city_tier"] = reg_df_like["city"].apply(
    lambda x: x if x in top_cities else "Other"
)
reg_df_like = pd.get_dummies(
    reg_df_like,
    drop_first=True,
    columns=["work_mode", "city_tier", "job_category"],
    dtype=int,
)
reg_df_like.drop(
    columns=["city", "required_skills", "experience_min", "experience_max"],
    inplace=True,
)

X_like = reg_df_like.drop(columns=["likelihood"])
y_like = reg_df_like["likelihood"]
X_like_train, X_like_test, y_like_train, y_like_test = train_test_split(
    X_like, y_like, test_size=0.25, random_state=42
)
scaler_like = StandardScaler()
X_like_train_scaled = scaler_like.fit_transform(X_like_train)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
    early_stopping=True,
)
mlp_model.fit(X_like_train_scaled, y_like_train)

joblib.dump(
    {
        "mlp": mlp_model,
        "scaler": scaler_like,
        "columns": list(X_like.columns),
    },
    os.path.join("models", "likelihood.pkl"),
)

# ==========================================
# 4. CLUSTERING MODEL
# ==========================================
df_clust = pd.read_csv("pakistan_tech_jobs_final.csv")
df_clust.drop("job_description", inplace=True, axis=1)
df_clust["education"] = df_clust["education"].fillna(df_clust["education"].mode()[0])
df_clust["job_title"] = df_clust["job_title"].fillna(df_clust["job_title"].mode()[0])

clustering_df = df_clust.copy()
clustering_df["is_senior"] = (
    df_clust["job_title"]
    .fillna("")
    .str.lower()
    .apply(lambda t: int(any(k in t for k in senior_kw)))
)
cols_to_drop = [
    "job_id",
    "job_title",
    "company",
    "required_skills",
    "city",
    "province",
    "employment_type",
    "work_mode",
    "posted_date",
    "job_category",
    "source_website",
]
clustering_df.drop(columns=cols_to_drop, inplace=True)
clustering_df["avg_experience"] = (
    clustering_df["experience_min"] + clustering_df["experience_max"]
) / 2
clustering_df["avg_sal"] = (
    clustering_df["salary_min"] + clustering_df["salary_max"]
) / 2
clustering_df.drop(
    columns=["experience_min", "experience_max", "salary_min", "salary_max"], inplace=True
)
clustering_df["education"] = clustering_df["education"].replace(
    {"Bachelor's": 0, "Master's": 1}
)

clustering_scaler = StandardScaler()
scaling_cols = ["avg_experience", "avg_sal"]
clustering_df[scaling_cols] = clustering_scaler.fit_transform(clustering_df[scaling_cols])

optimal_k = 3
km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
km_final.fit(clustering_df)

# derive cluster names from centroid salary
# lowest = Entry / Junior Level, middle = Mid-Level Specialist, highest = Lead / Senior Specialist
sal_col_idx = list(clustering_df.columns).index("avg_sal")
centroid_salaries = km_final.cluster_centers_[:, sal_col_idx]
sorted_indices = np.argsort(centroid_salaries)
cluster_names = {
    int(sorted_indices[0]): "Entry / Junior Level",
    int(sorted_indices[1]): "Mid-Level Specialist",
    int(sorted_indices[2]): "Lead / Senior Specialist",
}

joblib.dump(
    {
        "kmeans": km_final,
        "scaler": clustering_scaler,
        "columns": list(clustering_df.columns),
        "cluster_names": cluster_names,
    },
    os.path.join("models", "cluster.pkl"),
)

# ==========================================
# 5. PERSONA MODEL (LOGISTIC REGRESSION)
# ==========================================
df_persona = pd.read_csv("pakistan_tech_jobs_final.csv")
df_persona.drop("job_description", inplace=True, axis=1)
df_persona["education"] = df_persona["education"].fillna(df_persona["education"].mode()[0])
df_persona["job_title"] = df_persona["job_title"].fillna(df_persona["job_title"].mode()[0])
df_persona["required_skills"] = df_persona["required_skills"].str.strip()
df_persona["required_skills"] = df_persona["required_skills"].str.lower()

skill_clean_map = {
    "solid programming (python": "python",
    "or java)": "java",
    "c/c++": "c++",
    "rest api": "rest apis",
    "html": "html/css",
    "creative ui/ux": "ui/ux design",
    "dart dazzle": "dart",
    "devops foundations": "devops",
    "core ml expertise": "machine learning",
    "multi-platform jack with flutter development": "flutter",
    "driven sqa automation": "selenium",
    "datacraft": "data analysis",
    "webwarp": "web development",
    "webweave": "web development",
    "webwizard": "web development",
    "advance python": "python",
}


def clean_skills(txt):
    tokens = [t.strip().lower() for t in str(txt).split(",") if t.strip()]
    cleaned = []
    for token in tokens:
        token = skill_clean_map.get(token, token)
        token = token.replace("(", "").replace(")", "").strip()
        if token.startswith("or "):
            token = token[3:].strip()
        if token:
            cleaned.append(token)
    return ", ".join(sorted(list(set(cleaned))))


df_persona["required_skills"] = df_persona["required_skills"].apply(clean_skills)
df_persona["role_features"] = (
    df_persona["job_title"] + " " + df_persona["required_skills"]
)
y_persona = df_persona["job_category"]
X_train_raw, X_test_raw, y_persona_train, y_persona_test = train_test_split(
    df_persona["role_features"], y_persona, test_size=0.25, random_state=42, stratify=y_persona
)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train_raw)
model_persona = LogisticRegression(max_iter=1000, C=1.0)
model_persona.fit(X_train_tfidf, y_persona_train)

joblib.dump(
    {
        "tfidf": tfidf,
        "model": model_persona,
        "skill_clean_map": skill_clean_map,
    },
    os.path.join("models", "persona.pkl"),
)
