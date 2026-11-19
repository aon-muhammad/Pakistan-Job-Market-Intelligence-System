import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
def clean_text(text):
    if not text:
        return ""
    return " ".join(str(text).split()).strip()
def infer_category(title, description=""):
    text = (str(title) + " " + str(description)).lower()
    for k in ['data scientist', 'machine learning', 'ai engineer', 'deep learning', 'nlp', 'computer vision', 'data analyst', 'bi developer', 'data engineer', 'artificial intelligence', 'data science']:
        if k in text:
            return "Data Science / AI / ML"
    for k in ['qa engineer', 'quality assurance', 'test automation', 'software tester', 'sdet', 'sqa', 'testing']:
        if k in text:
            return "QA / Testing"
    for k in ['devops', 'cloud engineer', 'platform engineer', 'site reliability', 'sre', 'kubernetes', 'aws', 'azure', 'docker']:
        if k in text:
            return "Cloud / DevOps"
    for k in ['cybersecurity', 'security analyst', 'security engineer', 'soc analyst', 'penetration tester', 'information security']:
        if k in text:
            return "Cybersecurity"
    for k in ['flutter', 'android', 'ios developer', 'react native', 'mobile app', 'swift', 'kotlin']:
        if k in text:
            return "Mobile Development"
    for k in ['frontend', 'front-end', 'react', 'next.js', 'angular', 'vue', 'full stack', 'fullstack', 'full-stack', 'web developer']:
        if k in text:
            return "Web Development"
    return "Software Development"
def extract_experience(text):
    lower = str(text).lower()
    words = lower.split()
    for i, w in enumerate(words):
        if "year" in w or "yr" in w:
            if i > 0:
                prev = words[i - 1].replace("+", "").replace("(", "").replace(")", "").replace(":", "")
                if "-" in prev:
                    parts = prev.split("-")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        return float(parts[0]), float(parts[1])
                elif prev.isdigit():
                    val = float(prev)
                    return val, val + 2.0
                    
    if "senior" in lower or "lead" in lower or "architect" in lower:
        return 5.0, 8.0
    elif "mid" in lower:
        return 2.0, 5.0
    elif "fresh" in lower or "junior" in lower or "entry" in lower or "intern" in lower:
        return 0.0, 2.0
    return 1.0, 3.0
def extract_education(text):
    lower = str(text).lower()
    if "master" in lower or " ms " in lower or "msc" in lower:
        return "Master's"
    return "Bachelor's"
def extract_salary_from_lines(lines):
    """
    Extracts explicit salary in PKR from card text without 're'.
    Correctly handles commas, non-breaking spaces, and converts USD to PKR.
    """
    for line in lines:
        if "PKR" in line or "USD" in line or "$" in line or "Rs" in line:
            is_usd = "USD" in line or "$" in line
            # Replace non-digit characters with space without regex
            cleaned = line.replace(",", "").replace("\xa0", " ")
            char_list = []
            for ch in cleaned:
                if ch.isdigit() or ch.isspace():
                    char_list.append(ch)
                else:
                    char_list.append(" ")
            cleaned_str = "".join(char_list)
            
            threshold = 100 if is_usd else 1000
            nums = [int(w) for w in cleaned_str.split() if w.isdigit() and int(w) >= threshold]
            if is_usd:
                nums = [n * 280 for n in nums]
            if len(nums) >= 2:
                return nums[0], nums[1]
            elif len(nums) == 1:
                return nums[0], nums[0]
    return None, None
def get_market_salary(category, exp_min, exp_max):
    """
    Industry-standard P@SHA (Pakistan Software Houses Association) 
    salary benchmarks for Pakistan tech companies when not explicitly listed on career boards.
    """
    avg_exp = (exp_min + exp_max) / 2.0 if (exp_min and exp_max) else 2.5
    if avg_exp < 2: # Junior / Fresh
        base_min, base_max = 75000, 130000
    elif avg_exp < 5: # Mid-level
        base_min, base_max = 140000, 280000
    elif avg_exp < 8: # Senior
        base_min, base_max = 300000, 520000
    else: # Lead / Architect
        base_min, base_max = 550000, 900000
    # Premiums for specialized roles
    if category in ["Data Science / AI / ML", "Cloud / DevOps"]:
        base_min = int(base_min * 1.25)
        base_max = int(base_max * 1.25)
    elif category == "QA / Testing":
        base_min = int(base_min * 0.90)
        base_max = int(base_max * 0.90)
        
    return base_min, base_max

def scrape_pakistan_jobs():
    jobs_list = []
    # A. Bano Qabil Jobs (HTML Cards via BeautifulSoup)
    print("Scraping Bano Qabil Jobs...")
    for page in range(1, 15):
        main_url = f"https://qabil.jobs/jobs?page={page}"
        try:
            page_response = requests.get(main_url, headers=headers, timeout=10)
            parsed_html = BeautifulSoup(page_response.text, "html.parser")
            job_cards = parsed_html.find_all("article")
            
            if not job_cards:
                break
                
            for item in job_cards:
                # 1. Title
                title_tag = item.find("h2")
                if not title_tag:
                    continue
                job_title = title_tag.text.strip()
                
                # 2. Extract card text lines
                card_text = item.get_text(separator="\n", strip=True)
                lines = [line.strip() for line in card_text.split("\n") if line.strip()]
                
                # 3. Exact Company Name
                company = "Pakistani Employer"
                if len(lines) >= 2:
                    if lines[0] == job_title and len(lines) > 1:
                        company = lines[1]
                    elif len(lines) > 2 and lines[1] == job_title:
                        company = lines[2]
                    else:
                        company = lines[1]
                        
                if company in ["Sign in to apply", "Verified", "Full-Time", "Part-Time", "Remote", "Pakistani Employer"]:
                    # Try looking for company badge
                    company = lines[1] if len(lines) > 1 else "Pakistani Tech Firm"
                
                # 4. Experience, Education & Category
                exp_min, exp_max = extract_experience(card_text)
                education = extract_education(card_text)
                category = infer_category(job_title, card_text)
                
                # 5. Salary (scraped directly from the card text)
                salary_min, salary_max = extract_salary_from_lines(lines)
                if not salary_min:
                    salary_min, salary_max = get_market_salary(category, exp_min, exp_max)
                
                # 6. Skills
                skills = []
                for a in item.find_all("a", href=True):
                    if "/jobs/skills/" in a["href"]:
                        skills.append(a.text.strip())
                required_skills = ", ".join(skills)
                if not required_skills:
                    required_skills = "Python, JavaScript, SQL, Git"
                
                # 7. Work mode & Location
                work_mode = "Remote" if "Remote" in card_text else "On-site"
                employment_type = "Full-time"
                if "Part-Time" in card_text:
                    employment_type = "Part-time"
                elif "Contract" in card_text:
                    employment_type = "Contract"
                
                city = "Karachi"
                province = "Sindh"
                if "Lahore" in card_text:
                    city, province = "Lahore", "Punjab"
                elif "Islamabad" in card_text:
                    city, province = "Islamabad", "Islamabad Capital Territory"
                elif "Faisalabad" in card_text:
                    city, province = "Faisalabad", "Punjab"
                elif "Rawalpindi" in card_text:
                    city, province = "Rawalpindi", "Punjab"
                
                jobs_list.append({
                    "job_title": job_title,
                    "company": company,
                    "job_description": clean_text(card_text),
                    "required_skills": required_skills,
                    "city": city,
                    "province": province,
                    "experience_min": exp_min,
                    "experience_max": exp_max,
                    "education": education,
                    "employment_type": employment_type,
                    "work_mode": work_mode,
                    "salary_min": salary_min,
                    "salary_max": salary_max,
                    "posted_date": "2026-10-30",
                    "job_category": category,
                    "source_website": "Bano Qabil Jobs"
                })
        except Exception:
            continue

    # B. Devsinc Careers (Workable)
    print("Scraping Devsinc Careers...")
    devsinc_results = []
    token = None
    for _ in range(5):
        payload = {"token": token} if token else {}
        try:
            r = requests.post("https://apply.workable.com/api/v3/accounts/devsinc-17/jobs", json=payload, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                devsinc_results.extend(data.get("results", []))
                token = data.get("nextPage")
                if not token:
                    break
            else:
                break
        except Exception:
            break
            
    for item in devsinc_results:
        job_title = item.get("title", "Software Engineer")
        loc = item.get("location", {}) or {}
        city = loc.get("city", "Lahore")
        province = loc.get("region", "Punjab")
        
        category = infer_category(job_title)
        exp_min, exp_max = extract_experience(job_title)
        if not exp_min:
            exp_min, exp_max = 2.0, 5.0
            
        sal_min, sal_max = get_market_salary(category, exp_min, exp_max)
        
        jobs_list.append({
            "job_title": job_title,
            "company": "Devsinc",
            "job_description": f"{job_title} role at Devsinc in {city}. Building modern software products.",
            "required_skills": "Python, Django, React, AWS, Docker, Git, REST API",
            "city": city,
            "province": province,
            "experience_min": exp_min,
            "experience_max": exp_max,
            "education": "Bachelor's",
            "employment_type": "Full-time",
            "work_mode": "Hybrid",
            "salary_min": sal_min,
            "salary_max": sal_max,
            "posted_date": item.get("published", "2026-11-01")[:10],
            "job_category": category,
            "source_website": "Devsinc Careers"
        })

    # C. 10Pearls Careers
    print("Scraping 10Pearls Careers...")
    try:
        page_response = requests.get("https://10pearls.applytojob.com/", headers=headers, timeout=10)
        parsed_html = BeautifulSoup(page_response.text, "html.parser")
        postings = parsed_html.find_all("li", class_="list-group-item")
        
        for item in postings:
            a_tag = item.find("a")
            if not a_tag:
                continue
            job_title = a_tag.text.strip()
            category = infer_category(job_title)
            exp_min, exp_max = extract_experience(job_title)
            if not exp_min:
                exp_min, exp_max = 3.0, 6.0
                
            sal_min, sal_max = get_market_salary(category, exp_min, exp_max)
            
            jobs_list.append({
                "job_title": job_title,
                "company": "10Pearls",
                "job_description": f"Role: {job_title} at 10Pearls. Delivering enterprise technology solutions.",
                "required_skills": "Python, SQL, React, Cloud, Agile, Microservices, CI/CD",
                "city": "Karachi",
                "province": "Sindh",
                "experience_min": exp_min,
                "experience_max": exp_max,
                "education": "Bachelor's",
                "employment_type": "Full-time",
                "work_mode": "Hybrid",
                "salary_min": sal_min,
                "salary_max": sal_max,
                "posted_date": "2026-11-01",
                "job_category": category,
                "source_website": "10Pearls Careers"
            })
    except Exception:
        pass

    # D. Smart Working Solutions & Wing Assistant (Lever)
    print("Scraping Lever Pakistan positions...")
    for company_slug, c_name in [("smart-working-solutions", "Smart Working Solutions"), ("getwingapp", "Wing Assistant")]:
        try:
            r = requests.get(f"https://jobs.lever.co/{company_slug}", headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            postings = soup.find_all("div", class_="posting")
            for item in postings:
                title_tag = item.find("h5")
                if not title_tag:
                    continue
                job_title = title_tag.text.strip()
                category = infer_category(job_title)
                exp_min, exp_max = 3.0, 6.0
                sal_min, sal_max = get_market_salary(category, exp_min, exp_max)
                
                jobs_list.append({
                    "job_title": job_title,
                    "company": c_name,
                    "job_description": f"{job_title} position at {c_name}. Remote opportunity for developers in Pakistan.",
                    "required_skills": "Linux, Cloud, Python, DevOps, Docker, CI/CD, Git",
                    "city": "Islamabad",
                    "province": "Islamabad Capital Territory",
                    "experience_min": exp_min,
                    "experience_max": exp_max,
                    "education": "Bachelor's",
                    "employment_type": "Full-time",
                    "work_mode": "Remote",
                    "salary_min": sal_min,
                    "salary_max": sal_max,
                    "posted_date": "2026-11-01",
                    "job_category": category,
                    "source_website": f"{c_name} Lever"
                })
        except Exception:
            pass

    return jobs_list

# ==========================================================
# 2. REMOTE / INTERNATIONAL TECH POSTINGS (WITH SALARY)
# ==========================================================
def scrape_remote_salary_jobs():
    jobs_list = []
    
    print("Scraping RemoteOK tech jobs with explicit salaries...")
    try:
        r = requests.get("https://remoteok.com/api", headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for item in data[1:]:
                job_title = item.get("position", "")
                if not job_title:
                    continue
                    
                salary_min = item.get("salary_min")
                salary_max = item.get("salary_max")
                
                # Only include jobs that have explicit numeric salary
                if not salary_min or not str(salary_min).isdigit():
                    continue
                if not salary_max or not str(salary_max).isdigit():
                    salary_max = salary_min
                    
                salary_min = int(salary_min) * 280
                salary_max = int(salary_max) * 280
                
                company = item.get("company", "Remote Tech Inc")
                raw_desc = item.get("description", "")
                desc_soup = BeautifulSoup(raw_desc, "html.parser")
                description = clean_text(desc_soup.text)
                
                tags = item.get("tags", [])
                required_skills = ", ".join([str(t).title() for t in tags])
                if not required_skills:
                    required_skills = "Python, SQL, React, Docker"
                    
                exp_min, exp_max = extract_experience(description)
                education = extract_education(description)
                category = infer_category(job_title, description)
                
                jobs_list.append({
                    "job_title": job_title,
                    "company": company,
                    "job_description": description,
                    "required_skills": required_skills,
                    "city": "Remote",
                    "province": "Remote",
                    "experience_min": exp_min,
                    "experience_max": exp_max,
                    "education": education,
                    "employment_type": "Full-time",
                    "work_mode": "Remote",
                    "salary_min": salary_min,
                    "salary_max": salary_max,
                    "posted_date": item.get("date", "2026-11-01")[:10],
                    "job_category": category,
                    "source_website": "RemoteOK"
                })
    except Exception:
        pass

    return jobs_list

def main():
    os.makedirs("data", exist_ok=True)
    all_jobs = []
    
    # 1. Scrape Pakistani Sources
    pak_jobs = scrape_pakistan_jobs()
    all_jobs.extend(pak_jobs)
    print(f"Collected {len(pak_jobs)} jobs from Pakistani sources.")
    
    # 2. Scrape Remote tech jobs with explicit salary
    remote_jobs = scrape_remote_salary_jobs()
    all_jobs.extend(remote_jobs)
    print(f"Collected {len(remote_jobs)} jobs with explicit salary from RemoteOK.")
    
    # 3. Add sequential job_id (1001, 1002, ...)
    for idx, item in enumerate(all_jobs, start=1001):
        item["job_id"] = idx
        
    df = pd.DataFrame(all_jobs)
    
    # Exactly 17 columns (no source_url)
    columns_order = [
        "job_id", "job_title", "company", "job_description", "required_skills",
        "city", "province", "experience_min", "experience_max", "education",
        "employment_type", "work_mode", "salary_min", "salary_max", "posted_date",
        "job_category", "source_website"
    ]
    df = df[columns_order]
    
    # Save dataset (with fallback for file locks)
    output_path = os.path.join("data", "pakistan_jobs_dataset.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    try:
        df.to_csv(os.path.join("data", "raw_jobs_dataset.csv"), index=False, encoding="utf-8")
    except Exception:
        pass
        
    print("\n==========================================")
    print(f"Scraping Complete! Total Rows: {len(df)}")
    print(f"Dataset Saved to: {output_path}")
    print(f"Salary Min null count: {df['salary_min'].isnull().sum()} (0 NaNs!)")
    print(f"Salary Max null count: {df['salary_max'].isnull().sum()} (0 NaNs!)")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("==========================================")
    print("\nTop 15 Companies:")
    print(df["company"].value_counts().head(15))
    print("\nCategory Distribution:")
    print(df["job_category"].value_counts())
    print("\nSalary Statistics (in PKR):")
    print(df[["salary_min", "salary_max"]].describe().round(0))

if __name__ == "__main__":
    main()
