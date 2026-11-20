import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load the pickles once at startup. Only ever load files from models/.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
sal_data = joblib.load(os.path.join(MODELS_DIR, "salary.pkl"))
knn_data = joblib.load(os.path.join(MODELS_DIR, "knn.pkl"))
like_data = joblib.load(os.path.join(MODELS_DIR, "likelihood.pkl"))
clust_data = joblib.load(os.path.join(MODELS_DIR, "cluster.pkl"))
pers_data = joblib.load(os.path.join(MODELS_DIR, "persona.pkl"))

app = FastAPI(title="CareerML API")


class ProfileRequest(BaseModel):
    name: Optional[str] = "Ali"
    targetRole: Optional[str] = "Python Backend Developer"
    experience: float = Field(..., ge=0, description="Experience in years (>= 0)")
    city: str
    education: str
    workMode: str
    isSenior: Optional[int] = 0
    category: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    targetSkills: Optional[List[str]] = Field(default_factory=list)

    model_config = {"extra": "allow"}


def get_source_logo(category: str) -> str:
    cat = str(category).lower()
    if "cloud" in cat or "devops" in cat:
        return "cloud"
    elif "mobile" in cat or "android" in cat:
        return "android"
    elif "data" in cat or "ai" in cat or "ml" in cat:
        return "data"
    elif "cyber" in cat or "qa" in cat or "test" in cat or "system" in cat:
        return "system"
    else:
        return "python"


def clean_skills(txt: str, skill_clean_map: dict) -> str:
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


@app.post("/api/analyze")
async def analyze(profile: ProfileRequest):
    exp = float(profile.experience)
    is_senior = int(profile.isSenior or 0)
    edu_num = 1 if "master" in profile.education.lower() else 0
    raw_skills = [s.strip().lower() for s in profile.skills if s.strip()]
    normalized_skills = list(dict.fromkeys(raw_skills))

    # --- 1. PERSONA MODEL ---
    pers_tfidf = pers_data["tfidf"]
    pers_model = pers_data["model"]
    skill_clean_map = pers_data["skill_clean_map"]
    cleaned_sk = clean_skills(", ".join(normalized_skills), skill_clean_map)
    role_text = f"{profile.targetRole or ''} {cleaned_sk}".strip()
    role_vec = pers_tfidf.transform([role_text])
    predicted_category = str(pers_model.predict(role_vec)[0])
    effective_category = profile.category if profile.category else predicted_category

    # --- 2. SALARY REGRESSION ---
    sal_model = sal_data["model"]
    sal_cols = sal_data["columns"]
    mae = sal_data["mae"]
    r2 = sal_data["r2"]
    cap_sal = sal_data["cap_sal"]
    top_cities = sal_data["top_cities"]
    high_skills = sal_data["high_skills"]

    sal_query = {col: 0 for col in sal_cols}
    sal_query["experience"] = exp
    sal_query["education"] = edu_num
    sal_query["is_senior"] = is_senior
    sal_query["num_skills"] = len(normalized_skills)
    sal_query["has_high_skill"] = int(any(k in normalized_skills for k in high_skills))

    city_tier = profile.city if profile.city in top_cities else "Other"
    if f"city_tier_{city_tier}" in sal_query:
        sal_query[f"city_tier_{city_tier}"] = 1
    if f"work_mode_{profile.workMode}" in sal_query:
        sal_query[f"work_mode_{profile.workMode}"] = 1
    if f"job_category_{effective_category}" in sal_query:
        sal_query[f"job_category_{effective_category}"] = 1

    sal_df = pd.DataFrame([sal_query], columns=sal_cols)
    raw_sal = float(sal_model.predict(sal_df)[0])

    if raw_sal > cap_sal:
        raw_sal = cap_sal
    if raw_sal < 60000:
        raw_sal = 60000

    expected_salary = int(round(raw_sal))
    min_salary = max(50000, int(round(expected_salary - 2.5 * mae)))
    max_salary = int(round(expected_salary + 2.5 * mae))
    median_salary = int(round(expected_salary / 5000.0) * 5000)
    confidence = int(round(r2 * 100))

    known_weights = {
        "python": 95,
        "sql": 85,
        "django": 80,
        "aws": 92,
        "docker": 88,
        "kubernetes": 90,
        "react": 86,
        "machine learning": 94,
        "system design": 89,
        "fastapi": 82,
        "java": 84,
    }
    skill_impacts = [
        {"skill": "Python", "impactPct": 95},
        {"skill": "SQL", "impactPct": 85},
        {"skill": "Django", "impactPct": 80},
    ]
    if normalized_skills:
        custom_impacts = []
        for s in normalized_skills[:4]:
            w = known_weights.get(s, min(98, 75 + (len(s) * 3) % 20))
            custom_impacts.append({"skill": s.capitalize(), "impactPct": min(98, w)})
        if len(custom_impacts) >= 3:
            skill_impacts = custom_impacts

    salary_res = {
        "expectedSalary": expected_salary,
        "minSalary": min_salary,
        "maxSalary": max_salary,
        "medianSalary": median_salary,
        "confidence": confidence,
        "skillImpacts": skill_impacts,
    }

    # --- 3. CLUSTERING MODEL ---
    clust_km = clust_data["kmeans"]
    clust_scaler = clust_data["scaler"]
    clust_cols = clust_data["columns"]
    cluster_names = clust_data["cluster_names"]

    scaled_exp_sal = clust_scaler.transform(
        pd.DataFrame([[exp, expected_salary]], columns=["avg_experience", "avg_sal"])
    )
    clust_df = pd.DataFrame(
        [[edu_num, is_senior, scaled_exp_sal[0, 0], scaled_exp_sal[0, 1]]],
        columns=clust_cols,
    )
    cid = int(clust_km.predict(clust_df)[0])
    cluster_name = cluster_names.get(cid, "Mid-Level Specialist")

    persona_res = {
        "category": predicted_category,
        "clusterName": cluster_name,
    }

    # --- 4. CANDIDATE LIKELIHOOD ---
    like_mlp = like_data["mlp"]
    like_scaler = like_data["scaler"]
    like_cols = like_data["columns"]

    cand_like_query = {col: 0 for col in like_cols}
    cand_like_query["experience"] = exp
    cand_like_query["education"] = edu_num
    cand_like_query["num_skills"] = len(normalized_skills)
    cand_like_query["is_senior"] = is_senior
    cand_like_query["has_high_skill"] = int(
        any(k in normalized_skills for k in high_skills)
    )
    if f"city_tier_{city_tier}" in cand_like_query:
        cand_like_query[f"city_tier_{city_tier}"] = 1
    if f"work_mode_{profile.workMode}" in cand_like_query:
        cand_like_query[f"work_mode_{profile.workMode}"] = 1
    if f"job_category_{effective_category}" in cand_like_query:
        cand_like_query[f"job_category_{effective_category}"] = 1

    cand_like_df = pd.DataFrame([cand_like_query], columns=like_cols)
    cand_like_scaled = like_scaler.transform(cand_like_df)
    cand_like_val = float(like_mlp.predict(cand_like_scaled)[0])
    candidate_likelihood = int(round(max(0.0, min(100.0, cand_like_val))))

    # --- 5. TOP 5 PICKS VIA KNN ---
    knn_nn = knn_data["nn"]
    knn_scaler = knn_data["scaler"]
    knn_cols = knn_data["columns"]
    knn_jobs = knn_data["jobs"]

    knn_query = {col: 0 for col in knn_cols}
    scaled_knn_exp_sal = knn_scaler.transform(
        pd.DataFrame([[expected_salary, exp]], columns=["avg_salary", "avg_exp"])
    )
    knn_query["education"] = edu_num
    knn_query["is_senior"] = is_senior
    knn_query["avg_salary"] = scaled_knn_exp_sal[0, 0]
    knn_query["avg_exp"] = scaled_knn_exp_sal[0, 1]
    for s in normalized_skills:
        if s in knn_query:
            knn_query[s] = 1

    knn_query_df = pd.DataFrame([knn_query], columns=knn_cols)
    distances, indices = knn_nn.kneighbors(knn_query_df, n_neighbors=30)

    user_skills_set = set(normalized_skills)
    max_d = 12.0

    seen_keys = set()
    picks = []

    for d, idx in zip(distances[0], indices[0]):
        row = knn_jobs.iloc[idx]
        dedup_key = (
            str(row["job_title"]).strip().lower(),
            str(row["company"]).strip().lower(),
            str(row["city"]).strip().lower(),
        )
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        job_skills = [
            s.strip().lower()
            for s in str(row["required_skills"]).split(",")
            if s.strip()
        ]
        job_skills_set = set(job_skills)
        matched_skills = [s for s in normalized_skills if s in job_skills_set]
        skills_to_learn = [s for s in job_skills if s not in user_skills_set][:3]

        match_pct = int(
            round(max(0.0, min(100.0, 100.0 * (1.0 - float(d) / max_d))))
        )

        # per-job likelihood
        job_city_tier = row["city"] if row["city"] in top_cities else "Other"
        job_like_query = {col: 0 for col in like_cols}
        job_like_query["experience"] = exp
        job_like_query["education"] = edu_num
        job_like_query["num_skills"] = len(normalized_skills)
        job_like_query["is_senior"] = is_senior
        job_like_query["has_high_skill"] = int(
            any(k in normalized_skills for k in high_skills)
        )
        if f"city_tier_{job_city_tier}" in job_like_query:
            job_like_query[f"city_tier_{job_city_tier}"] = 1
        if f"work_mode_{row['work_mode']}" in job_like_query:
            job_like_query[f"work_mode_{row['work_mode']}"] = 1
        if f"job_category_{row['job_category']}" in job_like_query:
            job_like_query[f"job_category_{row['job_category']}"] = 1

        job_like_df = pd.DataFrame([job_like_query], columns=like_cols)
        job_like_scaled = like_scaler.transform(job_like_df)
        job_like_val = float(like_mlp.predict(job_like_scaled)[0])
        job_likelihood = int(round(max(0.0, min(100.0, job_like_val))))

        pick_obj = {
            "id": len(picks) + 1,
            "title": str(row["job_title"]),
            "company": str(row["company"]),
            "city": str(row["city"]),
            "workMode": str(row["work_mode"]),
            "salaryMin": int(row["salary_min"]),
            "salaryMax": int(row["salary_max"]),
            "expMin": float(row["experience_min"]),
            "expMax": float(row["experience_max"]),
            "matchPct": match_pct,
            "likelihood": job_likelihood,
            "matchedSkills": matched_skills,
            "skillsToLearn": skills_to_learn,
            "category": str(row["job_category"]),
            "education": str(row["education"]),
            "sourceLogo": get_source_logo(row["job_category"]),
        }
        picks.append(pick_obj)
        if len(picks) == 5:
            break

    return {
        "salary": salary_res,
        "likelihood": candidate_likelihood,
        "picks": picks,
        "persona": persona_res,
    }


# Serve the project folder as static files AFTER the API routes
app.mount("/", StaticFiles(directory=".", html=True), name="static")
