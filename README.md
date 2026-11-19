# CareerML — AI-Powered Pakistan Tech Career Intelligence Platform
Deployed App Link: careerml.vercel.app
CareerML is a modern career intelligence and machine learning analytics platform tailored specifically for the Pakistani tech ecosystem. Built using exploratory data analysis and predictive models trained on 2,140+ tech job postings in Pakistan (`pakistan_tech_jobs_final.csv`), CareerML delivers salary predictions, hiring likelihood estimates, job recommendations, persona classifications, and a clean, symbol-free AI career coach.

The web application is implemented in **HTML5, CSS3, and Bootstrap 5.3.9** using local vendor assets (`bootstrap-5.3.9-dist`), requiring no build tools or package managers.

---

## 🌟 Machine Learning Architecture & Notebook Mapping

All predictive capabilities in CareerML are directly mapped to the repository's Jupyter notebooks without modifying any of the original model files:

| Capability | Core Notebook | Machine Learning Method | Key Outputs & Metrics |
| :--- | :--- | :--- | :--- |
| **Salary Prediction** | `salary_regression.ipynb` | Multiple Linear Regression ($R^2 = 0.8122$) | Expected salary, range ($\pm 2.5 \times \text{MAE}$), median badge, 81% confidence, skill impacts, and upward career trajectory curve. |
| **Hiring Likelihood** | `likelihood_model.ipynb` | Multi-Factor Weighted Scoring | Hiring chance percentage based on **Experience (50%)**, **Skills Match (35%)**, and **Education (15%)**. 78% for default profile. |
| **Top Job Picks** | `knn.ipynb` | K-Nearest Neighbors ($k=5$, Euclidean Distance) | Ranked job matches from 2,140+ records with individual **Likelihood (% Chance)** and **KNN Match %**. |
| **Market Clustering** | `clustering.ipynb` | K-Means Clustering ($k=3$) | Categorization into Entry/Junior (PKR ~95k), Mid-Level (PKR ~171k), and Senior/Lead (PKR ~251k), with positioning benchmarks. |
| **Persona Prediction** | `logistic_regression.ipynb` | TF-IDF Vectorizer + Multinomial Logistic Regression | Career specialization persona (e.g. *Python Backend & Systems Specialist*), skill tags, and descriptions. |
| **Market Insights** | `insights.ipynb` | Exploratory Data Analysis | Top demanded skills (Git, Python, Docker, React), top paying skills (Microservices, REST API, Cloud), city salaries, and work mode shares. |
| **Tech Career Coach** | `chatbot.py` | Contextual Advisor with Structured Prompting | Plain-text salary negotiation blueprints, interview preparation questions, and technology roadmaps with **zero markdown clutter**. |

---

## 📸 Key Application Views & Features

### 1. Dashboard View
- **Welcome & Hero Card**: Personalized greeting for the candidate (*"Welcome back, Ali!"*) with quick exploration CTA and clean SVG illustration.
- **Key Metrics Grid**: 4 quick-access metric cards covering Estimated Salary Range (`PKR 125K – 202K`), Top 5 Picks, Job Persona, and Market Insights.
- **Career Snapshot**: Interactive SVG donut chart visualizing the candidate's **78% Hiring Likelihood** with targeted recommendations.

### 2. Salary Prediction View
- **Hero Compensation Card**: Clear salary range (`PKR 124,830 – 201,632 /yr`), median badge (`PKR 165,000`), and 81% model confidence badge.
- **Upward Trajectory Curve**: SVG trajectory chart modeling expected salary progression from 1 year to 8+ years.
- **Key Factors & Skill Impact**: Breakdown of candidate experience, city hub, and target role, alongside percentage impact bars for primary skills (Python 95%, SQL 85%, Django 80%, System Design 89%).
- **Interactive Scenario Simulator**: Expandable drawer with an experience slider (0 to 12 years), city tier selector, work mode preference, and education level dropdown for real-time model recalculation.

### 3. Top 5 Best Picks View
- **Ranked Job Cards**: Top 5 roles featuring tech logos (Python, Android, Data Science, System Administration, Cloud Engineering), company, location, and salary band.
- **Dual Match Badges**: Every card prominently displays both the predicted **Hiring Likelihood (% Chance)** and the **KNN Profile Match %**.
- **Job Details Modal**: Detailed breakdown of the hiring likelihood (Experience 50%, Skills 35%, Education 15%), matched skills pills, missing skills pills, and an **"Ask Coach About This Job & Interview Prep"** button *(no "Apply Now" button)*.

### 4. Job Persona View
- **Candidate Classification**: Visual avatar, professional title (*"Python Backend & Systems Specialist"*), cluster badge (*"Mid-Level Specialist"*), and descriptive summary.
- **Market Cluster Positioning Chart**: Visual bar chart comparing candidate compensation against Backend, Data Engineering, Full Stack, and DevOps industry clusters.
- **Recommended Skills to Learn**: Structured next-step roadmap (Advanced SQL, System Design, Cloud Platforms).

### 5. Market Insights View
- **Top Demanded Skills**: Bar distribution of top technologies across 2,140+ postings (Git, Python, Docker, React, CI/CD, AWS).
- **Highest Paying Skills**: Average monthly salaries for high-value specializations (Microservices at PKR ~320k, REST APIs at PKR ~299k, Cloud at PKR ~248k).
- **City Benchmarks & Work Mode**: Geographic salary averages across Islamabad, Lahore, Karachi, and Rawalpindi, plus Remote (40%) vs. On-site (31%) vs. Hybrid (29%) distribution.

### 6. Candidate Profile View
- **Profile Configuration**: Edit candidate name, target role, experience, city hub, education, and work mode.
- **Interactive Skill Chips**: Add or remove current skills and target learning goals with instant recalculation across all dashboard modules.

### 7. Floating Tech Career Coach (Chatbot)
- **Floating Action Button**: Fixed in the bottom-right corner for instant access across any view.
- **Structured Clean Responses**: All responses are rendered in readable plain text with uppercase section headers and numbered steps. **All raw markdown symbols (`#`, `*`, `**`, `-`) are strictly eliminated**.
- **Quick Prompt Chips**: One-click shortcuts for salary negotiation scripts, Docker/AWS learning roadmaps, and interview questions.

---

## 📁 Project Structure

```text
IDS-PROJECT/
├── bootstrap-5.3.9-dist/         # Local Bootstrap 5.3.9 distribution
│   ├── css/
│   │   ├── bootstrap.min.css     # Local Bootstrap core stylesheet
│   │   └── ...
│   └── js/
│       ├── bootstrap.bundle.min.js # Local Bootstrap bundle (Modals, Popovers)
│       └── ...
├── index.html                    # Main application markup & views
├── style.css                     # Custom SaaS styles matching CareerML design
├── app.js                        # Machine learning logic, reactivity & chatbot
├── pakistan_tech_jobs_final.csv  # Dataset of 2,140+ tech job records in Pakistan
├── salary_regression.ipynb       # Linear Regression model training & evaluation
├── clustering.ipynb              # K-Means clustering model
├── knn.ipynb                     # KNN job recommendation model
├── likelihood_model.ipynb        # Hiring likelihood model (50/35/15 weights)
├── logistic_regression.ipynb     # Role classification & persona prediction
├── insights.ipynb                # Exploratory data analysis & market metrics
├── chatbot.py                    # Career coach logic & prompt engineering
├── requirements.txt              # Python notebook dependencies
└── README.md                     # Project documentation
```

---

## 📊 Model Mathematical Formulas

### 1. Salary Regression Model
$$\text{Expected Salary} = \beta_0 + \sum_{i=1}^{n} \beta_i X_i$$
- **Intercept ($\beta_0$)**: $68,891.52$
- **Key Coefficients ($\beta_i$)**:
  - Experience: $+26,815.34$ per year
  - Number of Skills: $+3,708.51$ per skill
  - Seniority Flag: $+2,712.49$
  - High Skill Flag: $+771.51$
  - City Offsets: Lahore ($-3,881.72$), Karachi ($-11,598.51$), Rawalpindi ($-23,136.04$), Other ($-34,987.49$) relative to Islamabad
- **Salary Range**: $[\text{Expected} - 2.5 \times \text{MAE},\; \text{Expected} + 2.5 \times \text{MAE}]$ where $\text{MAE} = 15,360.44$

### 2. Hiring Likelihood Model
$$\text{Likelihood Score} = (0.50 \times \text{ExpScore}) + (0.35 \times \text{SkillScore}) + (0.15 \times \text{EduScore})$$
- Evaluates candidate alignment against role requirements to produce an intuitive % chance for each job post.
$$\text{Likelihood Score} = (0.50 \times \text{ExpScore}) + (0.35 \times \text{SkillScore}) + (0.15 \times \text{EduScore})$$
- Evaluates candidate alignment against role requirements to produce an intuitive % chance for each job post.
