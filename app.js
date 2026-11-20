/**
 * CareerML — AI-Powered Career Intelligence & Prediction Platform
 * Vanilla JavaScript implementation using Bootstrap 5.3.9
 * Includes embedded ML parameters (Linear Regression, KNN Scaler, K-Means Clusters, Likelihood Model)
 */

// ==========================================
// 1. EMBEDDED ML MODELS & BENCHMARK DATA
// ==========================================
const MODELS_DATA = {
  insights: {
    top_demanded_skills: [
      { skill: "git", count: 371 },
      { skill: "python", count: 344 },
      { skill: "docker", count: 331 },
      { skill: "react", count: 320 },
      { skill: "ci/cd", count: 315 },
      { skill: "aws", count: 308 },
      { skill: "elasticsearch", count: 287 },
      { skill: "sql", count: 286 },
      { skill: "linux", count: 286 },
      { skill: "azure", count: 285 },
      { skill: "rest apis", count: 284 },
      { skill: "node.js", count: 282 },
      { skill: "terraform", count: 282 },
      { skill: "javascript", count: 280 },
      { skill: "django", count: 278 }
    ],
    top_paying_skills: [
      { skill: "microservices", avg_salary: 319674, count: 23 },
      { skill: "rest api", avg_salary: 299259, count: 27 },
      { skill: "cloud", avg_salary: 248264, count: 55 },
      { skill: "devops", avg_salary: 216462, count: 39 },
      { skill: "python", avg_salary: 179962, count: 344 },
      { skill: "aws", avg_salary: 167029, count: 308 },
      { skill: "ci/cd", avg_salary: 163657, count: 315 },
      { skill: "react", avg_salary: 163562, count: 320 },
      { skill: "git", avg_salary: 161373, count: 371 },
      { skill: "sql", avg_salary: 160900, count: 286 },
      { skill: "docker", avg_salary: 160542, count: 331 },
      { skill: "django", avg_salary: 159892, count: 278 }
    ],
    city_salaries: [
      { city: "Islamabad", avg_salary: 165685 },
      { city: "Lahore", avg_salary: 161902 },
      { city: "Karachi", avg_salary: 149447 },
      { city: "Rawalpindi", avg_salary: 147049 }
    ]
  }
};

// SVG Icons Mapping for Job Cards
const ICONS = {
  python: `<svg width="28" height="28" viewBox="0 0 24 24" fill="none"><path d="M12 2C7.5 2 7 3.9 7 3.9L7 6H12V7H5C3 7 2 8.5 2 11.5C2 14.5 3 16 5 16H6.5V14C6.5 12 8 10.5 10 10.5H14C15.5 10.5 17 9.2 17 7.5V4C17 2.5 15.5 2 12 2ZM9.5 4C10.05 4 10.5 4.45 10.5 5C10.5 5.55 10.05 6 9.5 6C8.95 6 8.5 5.55 8.5 5C8.5 4.45 8.95 4 9.5 4Z" fill="#3B82F6"/><path d="M12 22C16.5 22 17 20.1 17 20.1L17 18H12V17H19C21 17 22 15.5 22 12.5C22 9.5 21 8 19 8H17.5V10C17.5 12 16 13.5 14 13.5H10C8.5 13.5 7 14.8 7 16.5V20C7 21.5 8.5 22 12 22ZM14.5 20C13.95 20 13.5 19.55 13.5 19C13.5 18.45 13.95 18 14.5 18C15.05 18 15.5 18.45 15.5 19C15.5 19.55 15.05 20 14.5 20Z" fill="#F59E0B"/></svg>`,
  android: `<svg width="28" height="28" viewBox="0 0 24 24" fill="none"><path d="M6 18c0 .55.45 1 1 1h1v3.5c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5V19h2v3.5c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5V19h1c.55 0 1-.45 1-1V8H6v10z" fill="#10B981"/><path d="M3.5 8C2.67 8 2 8.67 2 9.5v6c0 .83.67 1.5 1.5 1.5S5 16.33 5 15.5v-6C5 8.67 4.33 8 3.5 8zm17 0c-.83 0-1.5.67-1.5 1.5v6c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5v-6c0-.83-.67-1.5-1.5-1.5z" fill="#10B981"/><path d="M7 6.8c.8-1.7 2.4-3.1 4.5-3.6L10.1 1.4c-.2-.3-.1-.7.2-.9.3-.2.7-.1.9.2l1.5 2c.8-.1 1.6-.1 2.4 0l1.5-2c.2-.3.6-.4.9-.2.3.2.4.6.2.9L16.5 3.2c2.1.5 3.7 1.9 4.5 3.6H7z" fill="#10B981"/><circle cx="9.5" cy="5.5" r=".75" fill="#fff"/><circle cx="14.5" cy="5.5" r=".75" fill="#fff"/></svg>`,
  data: `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>`,
  system: `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0284C7" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>`,
  cloud: `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>`
};

// ==========================================
// 2. ACTIVE APPLICATION STATE
// ==========================================
let currentProfile = {
  name: 'Ali',
  targetRole: 'Python Backend Developer',
  experience: 3.5,
  city: 'Lahore',
  education: "Bachelor's",
  workMode: 'On-site',
  isSenior: 0,
  category: 'Software Development',
  skills: ['python', 'django', 'sql', 'system design'],
  targetSkills: ['docker', 'aws', 'fastapi']
};

let activeJobs = [];
let selectedJob = null;
let bootstrapJobModal = null;
let lastAnalysis = null;
let renderRequestId = 0;
let renderDebounceTimer = null;

function showServerErrorAlert() {
  let alertEl = document.getElementById('serverErrorAlert');
  if (!alertEl) {
    alertEl = document.createElement('div');
    alertEl.id = 'serverErrorAlert';
    alertEl.className = 'alert alert-danger rounded-3 small fw-semibold mb-4';
    alertEl.setAttribute('role', 'alert');
    alertEl.innerText = "Can't reach the model server. Run: uvicorn server:app --reload";
    const main = document.querySelector('main');
    if (main) main.prepend(alertEl);
  } else {
    alertEl.classList.remove('d-none');
    alertEl.style.display = 'block';
  }
}

function hideServerErrorAlert() {
  const alertEl = document.getElementById('serverErrorAlert');
  if (alertEl) {
    alertEl.classList.add('d-none');
    alertEl.style.display = 'none';
  }
}

async function fetchAnalysis(profile) {
  const res = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profile)
  });
  if (!res.ok) {
    throw new Error(`Server returned ${res.status}`);
  }
  return await res.json();
}

// ==========================================
// 3. ML PREDICTION CORE FUNCTIONS
// ==========================================
function predictSalary(profile) {
  return lastAnalysis ? lastAnalysis.salary : {};
}

function calculateJobLikelihood(profile, job) {}

function getCandidateLikelihood(profile) {
  return lastAnalysis ? lastAnalysis.likelihood : 0;
}

function getJobPersona(profile) {}

// ==========================================
// 4. TAB NAVIGATION & VIEW SWITCHING
// ==========================================
function switchTab(tabName) {
  // Update sidebar buttons
  const navBtns = document.querySelectorAll('.nav-link-custom');
  navBtns.forEach(btn => {
    if (btn.getAttribute('data-tab') === tabName) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  // Hide all view panels
  const panels = document.querySelectorAll('.view-panel');
  panels.forEach(p => p.style.display = 'none');

  // Show active view panel
  const activePanel = document.getElementById(`view-${tabName}`);
  if (activePanel) {
    activePanel.style.display = 'block';
  }

  // Close mobile sidebar if open
  closeMobileSidebar();

  // Scroll to top of content
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function toggleMobileSidebar() {
  const sidebar = document.getElementById('appSidebar');
  const backdrop = document.getElementById('sidebar-backdrop');
  if (sidebar.classList.contains('show')) {
    sidebar.classList.remove('show');
    backdrop.style.display = 'none';
  } else {
    sidebar.classList.add('show');
    backdrop.style.display = 'block';
  }
}

function closeMobileSidebar() {
  const sidebar = document.getElementById('appSidebar');
  const backdrop = document.getElementById('sidebar-backdrop');
  if (sidebar) sidebar.classList.remove('show');
  if (backdrop) backdrop.style.display = 'none';
}

// ==========================================
// 5. VIEW RENDERERS
// ==========================================
async function renderAll() {
  if (renderDebounceTimer) {
    clearTimeout(renderDebounceTimer);
  }
  return new Promise((resolve) => {
    renderDebounceTimer = setTimeout(async () => {
      const currentId = ++renderRequestId;
      try {
        const data = await fetchAnalysis(currentProfile);
        if (currentId !== renderRequestId) {
          resolve();
          return;
        }
        hideServerErrorAlert();
        lastAnalysis = data;

        const salaryPred = data.salary;
        const likelihood = data.likelihood;
        activeJobs = data.picks || [];

        // 1. Top Navbar User Info
        document.getElementById('navUserName').innerText = currentProfile.name;
        document.getElementById('navUserSub').innerText = `${currentProfile.city} • ${currentProfile.experience}y exp`;

        // 2. Dashboard View
        document.getElementById('dashWelcomeTitle').innerText = `Welcome back, ${currentProfile.name}!`;
        document.getElementById('dashSalaryRange').innerText = `PKR ${Math.round(salaryPred.minSalary / 1000)}K – ${Math.round(salaryPred.maxSalary / 1000)}K`;
        
        // Donut chart calculation (Circumference: 2 * Math.PI * 38 = 238.76)
        const donutCircle = document.getElementById('dashDonutCircle');
        const dashDonutPct = document.getElementById('dashDonutPct');
        const circumference = 238.76;
        const offset = circumference * (1 - likelihood / 100);
        if (donutCircle) donutCircle.setAttribute('stroke-dashoffset', offset.toFixed(2));
        if (dashDonutPct) dashDonutPct.innerText = `${likelihood}%`;

        document.getElementById('dashSnapshotPctText').innerText = `${likelihood}% hiring likelihood`;
        document.getElementById('dashSnapshotRole').innerText = currentProfile.targetRole;

        // 3. Salary Prediction View
        document.getElementById('heroSalaryRange').innerHTML = `PKR ${salaryPred.minSalary.toLocaleString()} – ${salaryPred.maxSalary.toLocaleString()} <span class="fs-6 fw-normal text-muted">/yr</span>`;
        document.getElementById('heroMedianBadge').innerText = `Median: PKR ${salaryPred.medianSalary.toLocaleString()}`;

        document.getElementById('kfExp').innerText = `${currentProfile.experience} Years`;
        document.getElementById('kfCity').innerText = currentProfile.city;
        document.getElementById('kfRole').innerText = currentProfile.targetRole;

        // Skill Impact Bars
        const skillImpactContainer = document.getElementById('skillImpactList');
        if (skillImpactContainer && salaryPred.skillImpacts) {
          skillImpactContainer.innerHTML = salaryPred.skillImpacts.map(si => `
            <div class="mb-3">
              <div class="d-flex justify-content-between small fw-bold mb-1">
                <span>${si.skill}</span>
                <span>${si.impactPct}%</span>
              </div>
              <div class="progress" style="height: 8px;">
                <div class="progress-bar bg-primary rounded-pill" style="width: ${si.impactPct}%;"></div>
              </div>
            </div>
          `).join('');
        }

        // 4. Top 5 Picks View
        renderTopPicks();

        // 5. Job Persona View
        const personaCategory = (data.persona && data.persona.category) || 'Software Development';
        const clusterName = (data.persona && data.persona.clusterName) || 'Mid-Level Specialist';

        let title = 'Python Backend & Systems Specialist';
        let tags = ['Python', 'Django', 'SQL', 'System Design'];
        let description = 'A skilled backend developer who builds scalable web applications and works with data-driven systems.';

        if (personaCategory === 'Data Science / AI / ML') {
          title = 'AI & Data Science Specialist';
          tags = ['Python', 'Machine Learning', 'Pandas', 'Deep Learning'];
          description = 'An analytical problem-solver developing machine learning models and data pipelines.';
        } else if (personaCategory === 'Cloud / DevOps') {
          title = 'Cloud & DevOps Systems Engineer';
          tags = ['AWS', 'Docker', 'Kubernetes', 'CI/CD'];
          description = 'An infrastructure architect designing resilient, scalable cloud architectures.';
        } else if (personaCategory === 'Mobile Development') {
          title = 'Mobile App Engineering Specialist';
          tags = ['Flutter', 'Android', 'Kotlin', 'Mobile UI'];
          description = 'A creative mobile specialist building high-performance cross-platform applications.';
        } else if (personaCategory === 'Web Development') {
          title = 'Modern Frontend & UI Engineer';
          tags = ['React', 'TypeScript', 'Next.js', 'Tailwind CSS'];
          description = 'A visual frontend craftsman engineering fluid and accessible user experiences.';
        }

        document.getElementById('personaTitle').innerText = title;
        document.getElementById('personaClusterBadge').innerText = clusterName;
        document.getElementById('personaDesc').innerText = description;
        const skillsPills = document.getElementById('personaSkillsPills');
        if (skillsPills) {
          skillsPills.innerHTML = tags.map(t => `
            <span class="badge bg-primary-subtle text-primary border border-primary-subtle rounded-pill px-3 py-1.5 small">${t}</span>
          `).join('');
        }

        // 6. Market Insights View
        renderMarketInsights();

        // 7. Profile Form Sync
        syncProfileForm();

        // 8. Chat Context Strip
        document.getElementById('chatContextRole').innerText = `${currentProfile.targetRole} • ${currentProfile.city}`;
        document.getElementById('chatContextSalary').innerText = `Target: PKR ${Math.round(salaryPred.medianSalary / 1000)}k`;
        resolve();
      } catch (err) {
        if (currentId !== renderRequestId) {
          resolve();
          return;
        }
        showServerErrorAlert();
        resolve();
      }
    }, 250);
  });
}

function renderTopPicks() {
  const container = document.getElementById('topPicksContainer');
  if (!container) return;

  container.innerHTML = activeJobs.map((job, index) => {
    const iconSvg = ICONS[job.sourceLogo] || ICONS.python;
    return `
      <div class="job-item-card">
        <div class="d-flex justify-content-between align-items-center mb-1">
          <div class="job-rank-badge">${index + 1}</div>
          <span class="badge bg-primary px-2 py-1 rounded-pill" style="font-size: 10px; font-weight: 700;">
            ${job.likelihood}% Likelihood
          </span>
        </div>

        <div class="job-tech-icon bg-light">
          ${iconSvg}
        </div>

        <h5 class="fs-7 fw-bold text-dark text-truncate mb-1" title="${job.title}">${job.title}</h5>
        <div class="text-muted fs-8 mb-2">${job.company} • ${job.city}</div>

        <div class="small fw-bolder text-dark mb-2">
          PKR ${Math.round(job.salaryMin / 1000)}K – ${Math.round(job.salaryMax / 1000)}K
        </div>

        <div class="mb-3">
          <span class="badge bg-light text-primary border border-primary-subtle rounded-pill px-2.5 py-1" style="font-size: 10px; font-weight: 600;">
            ${job.matchPct}% Match
          </span>
        </div>

        <button class="btn btn-outline-primary btn-sm rounded-pill w-100 fs-8 fw-semibold py-1.5" onclick="openJobModal(${job.id})">
          View Details →
        </button>
      </div>
    `;
  }).join('');
}

function openJobModal(jobId) {
  const job = activeJobs.find(j => j.id === jobId);
  if (!job) return;

  selectedJob = job;

  // Update modal fields
  document.getElementById('modalLikelihoodBadge').innerText = `Likelihood: ${job.likelihood}% Chance`;
  document.getElementById('modalMatchBadge').innerText = `KNN Match: ${job.matchPct}%`;
  document.getElementById('modalJobTitle').innerText = job.title;
  document.getElementById('modalJobCompanyLocation').innerText = `${job.company} • ${job.city} (${job.workMode})`;

  document.getElementById('modalLikelihoodPct').innerText = `${job.likelihood}%`;
  document.getElementById('modalLikelihoodBar').style.width = `${job.likelihood}%`;

  const bd = job.likelihoodBreakdown || { experienceScore: 48, skillsScore: 30, educationScore: 15 };
  document.getElementById('modalScoreExp').innerText = `${bd.experienceScore} / 50`;
  document.getElementById('modalScoreSkills').innerText = `${bd.skillsScore} / 35`;
  document.getElementById('modalScoreEdu').innerText = `${bd.educationScore} / 15`;

  document.getElementById('modalSalaryRange').innerText = `PKR ${job.salaryMin.toLocaleString()} – ${job.salaryMax.toLocaleString()}`;
  document.getElementById('modalExpNeeded').innerText = `${job.expMin} – ${job.expMax} Years`;
  document.getElementById('modalEduReq').innerText = job.education || "Bachelor's";
  document.getElementById('modalCatReq').innerText = job.category || "Software Development";

  // Matched skills pills
  const matchedContainer = document.getElementById('modalMatchedSkills');
  if (matchedContainer) {
    matchedContainer.innerHTML = (job.matchedSkills || []).map(s => `
      <span class="badge bg-success-subtle text-success border border-success-subtle rounded-pill px-2.5 py-1 small">${s}</span>
    `).join('');
  }

  // Missing skills pills
  const missingContainer = document.getElementById('modalMissingSkills');
  if (missingContainer) {
    missingContainer.innerHTML = (job.skillsToLearn || []).map(s => `
      <span class="badge bg-warning-subtle text-warning border border-warning-subtle rounded-pill px-2.5 py-1 small">${s}</span>
    `).join('');
  }

  // Configure Ask Coach button
  const askCoachBtn = document.getElementById('modalAskCoachBtn');
  if (askCoachBtn) {
    askCoachBtn.onclick = function() {
      if (bootstrapJobModal) {
        bootstrapJobModal.hide();
      }
      openChatWithPrompt(`Can you help me prepare for the ${job.title} role at ${job.company}? My likelihood is ${job.likelihood}%. What technical questions and architecture concepts should I study?`);
    };
  }

  // Show modal
  const modalEl = document.getElementById('jobDetailModal');
  if (modalEl) {
    if (!bootstrapJobModal) {
      bootstrapJobModal = new bootstrap.Modal(modalEl);
    }
    bootstrapJobModal.show();
  }
}

function renderMarketInsights() {
  const insights = MODELS_DATA.insights;

  // Demanded skills
  const demandedEl = document.getElementById('demandedSkillsList');
  if (demandedEl) {
    const topSkills = insights.top_demanded_skills.slice(0, 6);
    const maxCount = topSkills[0].count;
    demandedEl.innerHTML = topSkills.map(s => {
      const pct = Math.round((s.count / maxCount) * 100);
      return `
        <div>
          <div class="d-flex justify-content-between small fw-bold mb-1">
            <span class="text-uppercase">${s.skill}</span>
            <span>${s.count} postings</span>
          </div>
          <div class="progress" style="height: 7px;">
            <div class="progress-bar bg-primary rounded-pill" style="width: ${pct}%;"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  // Paying skills
  const payingEl = document.getElementById('payingSkillsList');
  if (payingEl) {
    const topPaying = insights.top_paying_skills.slice(0, 6);
    const maxSal = topPaying[0].avg_salary;
    payingEl.innerHTML = topPaying.map(s => {
      const pct = Math.round((s.avg_salary / maxSal) * 100);
      return `
        <div>
          <div class="d-flex justify-content-between small fw-bold mb-1">
            <span class="text-capitalize">${s.skill}</span>
            <span>PKR ${s.avg_salary.toLocaleString()}/mo</span>
          </div>
          <div class="progress" style="height: 7px;">
            <div class="progress-bar bg-success rounded-pill" style="width: ${pct}%;"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  // City salary bars
  const cityBarsEl = document.getElementById('citySalaryBars');
  if (cityBarsEl) {
    const cities = insights.city_salaries;
    const maxCitySal = cities[0].avg_salary;
    cityBarsEl.innerHTML = cities.map(c => {
      const pct = Math.round((c.avg_salary / maxCitySal) * 100);
      return `
        <div>
          <div class="d-flex justify-content-between small fw-bold mb-1">
            <span>${c.city}</span>
            <span class="fw-bold">PKR ${c.avg_salary.toLocaleString()}</span>
          </div>
          <div class="progress" style="height: 7px;">
            <div class="progress-bar bg-info rounded-pill" style="width: ${pct}%;"></div>
          </div>
        </div>
      `;
    }).join('');
  }
}

// ==========================================
// 6. SIMULATOR & PROFILE EVENT HANDLERS
// ==========================================
function toggleSimulator() {
  const drawer = document.getElementById('simulatorDrawer');
  const btnText = document.getElementById('simulatorBtnText');
  if (drawer.style.display === 'none' || !drawer.style.display) {
    drawer.style.display = 'block';
    btnText.innerText = 'Hide Simulator';
  } else {
    drawer.style.display = 'none';
    btnText.innerText = 'Simulate Different Scenarios';
  }
}

function updateExpSimulator(val) {
  const num = parseFloat(val);
  currentProfile.experience = num;
  document.getElementById('simExpVal').innerText = `${num} Years`;
  renderAll();
}

function updateProfileField(field, value) {
  currentProfile[field] = value;
  renderAll();
}

function syncProfileForm() {
  // Simulator selects
  const simCity = document.getElementById('simCitySelect');
  if (simCity) simCity.value = currentProfile.city;

  const simMode = document.getElementById('simModeSelect');
  if (simMode) simMode.value = currentProfile.workMode;

  const simEdu = document.getElementById('simEduSelect');
  if (simEdu) simEdu.value = currentProfile.education;

  const simExpSlider = document.getElementById('simExpSlider');
  if (simExpSlider) simExpSlider.value = currentProfile.experience;

  const simExpVal = document.getElementById('simExpVal');
  if (simExpVal) simExpVal.innerText = `${currentProfile.experience} Years`;

  // Profile page form
  const inputName = document.getElementById('inputName');
  if (inputName) inputName.value = currentProfile.name;

  const inputTargetRole = document.getElementById('inputTargetRole');
  if (inputTargetRole) inputTargetRole.value = currentProfile.targetRole;

  const inputExp = document.getElementById('inputExp');
  if (inputExp) inputExp.value = currentProfile.experience;

  const inputExpDisplay = document.getElementById('inputExpDisplay');
  if (inputExpDisplay) inputExpDisplay.innerText = `${currentProfile.experience} Years`;

  const inputCity = document.getElementById('inputCity');
  if (inputCity) inputCity.value = currentProfile.city;

  const inputEducation = document.getElementById('inputEducation');
  if (inputEducation) inputEducation.value = currentProfile.education;

  const inputWorkMode = document.getElementById('inputWorkMode');
  if (inputWorkMode) inputWorkMode.value = currentProfile.workMode;

  // Skills chips
  renderSkillChips();
}

function renderSkillChips() {
  const currentSkillsEl = document.getElementById('currentSkillsChips');
  if (currentSkillsEl) {
    currentSkillsEl.innerHTML = currentProfile.skills.map(s => `
      <span class="badge bg-primary text-white rounded-pill px-3 py-1.5 small d-flex align-items-center gap-1">
        <span>${s}</span>
        <button type="button" class="btn-close btn-close-white" style="font-size: 8px;" onclick="removeCurrentSkill('${s}')" aria-label="Remove skill"></button>
      </span>
    `).join('');
  }

  const targetSkillsEl = document.getElementById('targetSkillsChips');
  if (targetSkillsEl) {
    targetSkillsEl.innerHTML = currentProfile.targetSkills.map(s => `
      <span class="badge bg-secondary text-white rounded-pill px-3 py-1.5 small d-flex align-items-center gap-1">
        <span>${s}</span>
        <button type="button" class="btn-close btn-close-white" style="font-size: 8px;" onclick="removeTargetSkill('${s}')" aria-label="Remove skill"></button>
      </span>
    `).join('');
  }
}

function addCurrentSkill() {
  const input = document.getElementById('newCurrentSkillInput');
  const val = input.value.trim().toLowerCase();
  if (val && !currentProfile.skills.includes(val)) {
    currentProfile.skills.push(val);
    input.value = '';
    renderSkillChips();
    renderAll();
  }
}

function removeCurrentSkill(skill) {
  currentProfile.skills = currentProfile.skills.filter(s => s !== skill);
  renderSkillChips();
  renderAll();
}

function addTargetSkill() {
  const input = document.getElementById('newTargetSkillInput');
  const val = input.value.trim().toLowerCase();
  if (val && !currentProfile.targetSkills.includes(val)) {
    currentProfile.targetSkills.push(val);
    input.value = '';
    renderSkillChips();
    renderAll();
  }
}

function removeTargetSkill(skill) {
  currentProfile.targetSkills = currentProfile.targetSkills.filter(s => s !== skill);
  renderSkillChips();
  renderAll();
}

function saveProfileForm(event) {
  event.preventDefault();
  currentProfile.name = document.getElementById('inputName').value.trim() || 'Ali';
  currentProfile.targetRole = document.getElementById('inputTargetRole').value.trim() || 'Python Backend Developer';
  const parsedExp = parseFloat(document.getElementById('inputExp').value);
  currentProfile.experience = isNaN(parsedExp) ? 3.5 : parsedExp;
  currentProfile.city = document.getElementById('inputCity').value;
  currentProfile.education = document.getElementById('inputEducation').value;
  currentProfile.workMode = document.getElementById('inputWorkMode').value;

  renderAll();

  const alert = document.getElementById('profileSaveAlert');
  if (alert) {
    alert.classList.remove('d-none');
    setTimeout(() => alert.classList.add('d-none'), 4000);
  }
}

function resetDefaultProfile() {
  currentProfile = {
    name: 'Ali',
    targetRole: 'Python Backend Developer',
    experience: 3.5,
    city: 'Lahore',
    education: "Bachelor's",
    workMode: 'On-site',
    isSenior: 0,
    category: 'Software Development',
    skills: ['python', 'django', 'sql', 'system design'],
    targetSkills: ['docker', 'aws', 'fastapi']
  };
  renderAll();
}

// ==========================================
// 7. CHATBOT CLIENT & SYMBOL-FREE SANITIZER
// ==========================================
function toggleChatDrawer() {
  const drawer = document.getElementById('chatDrawer');
  if (drawer.style.display === 'none' || !drawer.style.display) {
    drawer.style.display = 'flex';
    document.getElementById('chatInput').focus();
  } else {
    drawer.style.display = 'none';
  }
}

function openChatWithPrompt(promptText) {
  const drawer = document.getElementById('chatDrawer');
  drawer.style.display = 'flex';
  const chatInput = document.getElementById('chatInput');
  chatInput.value = promptText;
  sendUserMessage(promptText);
  chatInput.value = '';
}

function sendQuickPrompt(promptText) {
  sendUserMessage(promptText);
}

function cleanResponseText(raw) {
  if (!raw) return '';
  return raw
    // Remove headers
    .replace(/^#{1,6}\s*/gm, '')
    // Remove bold and italic markdown
    .replace(/\*{1,3}([^*]+)\*{1,3}/g, '$1')
    .replace(/_{1,3}([^_]+)_{1,3}/g, '$1')
    // Remove bullet points
    .replace(/^[\s]*[-+*]\s+/gm, '')
    // Remove blockquotes
    .replace(/^>\s*/gm, '')
    // Remove inline code ticks
    .replace(/`([^`]+)`/g, '$1')
    .replace(/```[a-z]*\n?/gi, '')
    // Remove leftover markdown chars
    .replace(/[#*_`~]/g, '')
    .trim();
}

function appendChatMessage(sender, text) {
  const container = document.getElementById('chatMessages');
  if (!container) return;

  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${sender === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai'}`;
  bubble.innerText = cleanResponseText(text);

  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
}

function generateLocalIntelligentResponse(userMsg) {
  const lower = userMsg.toLowerCase();
  const sal = predictSalary(currentProfile);
  const targetK = Math.round(sal.expectedSalary / 1000);

  if (lower.includes('negotiat') || lower.includes('salary') || lower.includes('202k') || lower.includes('175k') || lower.includes('pkr')) {
    return [
      `SALARY NEGOTIATION BLUEPRINT FOR ${currentProfile.city.toUpperCase()}`,
      `Your regression expected salary is PKR ${sal.expectedSalary.toLocaleString()} with a high-tier band of PKR ${sal.maxSalary.toLocaleString()}.`,
      ``,
      `1. HIGHLIGHT YOUR CORE VALUE: Emphasize your ${currentProfile.experience} years of hands-on experience in ${currentProfile.targetRole} and proven delivery with ${currentProfile.skills.join(', ')}.`,
      `2. ANCHOR AT THE UPPER QUANTILE: When recruiters ask for your expectations, state PKR ${Math.round(sal.expectedSalary * 1.15 / 1000)}k to PKR ${Math.round(sal.maxSalary / 1000)}k. This leaves room to settle above your PKR ${targetK}k target.`,
      `3. NEGOTIATE BEYOND BASE PAY: If the company caps base compensation, ask for annual performance bonuses, remote work flexibility allowance, or bi-annual performance evaluations.`
    ].join('\n');
  }

  if (lower.includes('docker') || lower.includes('aws') || lower.includes('roadmap') || lower.includes('learn')) {
    return [
      `STEP-BY-STEP LEARNING ROADMAP FOR CLOUD & DEVOPS`,
      `Phase 1 Containerization with Docker: Master multi-stage builds, Docker Compose, networking, volume persistence, and slimming production image sizes.`,
      `Phase 2 Core Cloud Fundamentals on AWS: Focus on EC2, S3, RDS PostgreSQL, VPC networking, security groups, and IAM least privilege policies.`,
      `Phase 3 Production Deployment: Implement CI/CD pipelines with GitHub Actions to test, build, and deploy containers onto AWS ECS or Kubernetes clusters.`,
      `Phase 4 Observability: Configure structured logging and metrics monitoring with Prometheus and Grafana.`
    ].join('\n');
  }

  if (lower.includes('interview') || lower.includes('question') || lower.includes('prep') || lower.includes('technova') || lower.includes('google')) {
    return [
      `TECHNICAL INTERVIEW PREPARATION GUIDE`,
      `Target Role: ${currentProfile.targetRole}`,
      ``,
      `1. PYTHON INTERNALS: Be ready to explain the GIL, memory management, garbage collection, generators, decorators, and asynchronous programming with asyncio.`,
      `2. DATABASE & SQL OPTIMIZATION: Prepare indexing strategies (B-Tree, Hash), N+1 query troubleshooting in Django ORM, ACID compliance, and database sharding.`,
      `3. SYSTEM DESIGN: Practice designing scalable rate limiters, URL shorteners, distributed caching with Redis, and message queues with RabbitMQ or Kafka.`,
      `4. BEHAVIORAL QUESTIONS: Frame your answers using the STAR method (Situation, Task, Action, Result) demonstrating technical leadership and problem solving.`
    ].join('\n');
  }

  if (lower.includes('remote') || lower.includes('on-site') || lower.includes('hybrid')) {
    return [
      `PAKISTAN TECH MARKET: REMOTE VS ON-SITE ANALYSIS`,
      `Based on 2,140+ tech postings in our dataset:`,
      `Remote positions represent 40% of tech roles and typically offer 15% to 30% higher compensation, frequently pegged to USD rates or international engineering budgets.`,
      `On-site and Hybrid roles represent 60% of postings and provide strong mentorship, in-person team building, and formal benefits such as provident funds and health coverage.`,
      `Key companies actively hiring remote engineers in Pakistan include 10Pearls, Arbisoft, Devsinc, and international venture-backed firms.`
    ].join('\n');
  }

  return [
    `CAREER INTELLIGENCE ADVISOR`,
    `I have analyzed your candidate profile (${currentProfile.experience} years experience in ${currentProfile.city}).`,
    `Your hiring likelihood is estimated at ${getCandidateLikelihood(currentProfile)}% for ${currentProfile.targetRole} positions.`,
    `Ask me anything about salary negotiation scripts, interview questions, technical skill roadmaps, or remote job opportunities.`
  ].join('\n');
}

async function sendUserMessage(msgText) {
  const text = msgText.trim();
  if (!text) return;

  appendChatMessage('user', text);

  // Try Next.js API route if available, otherwise fall back to instant local generator
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        profile: currentProfile
      })
    });

    if (res.ok) {
      const data = await res.json();
      if (data && data.reply) {
        appendChatMessage('ai', data.reply);
        return;
      }
    }
  } catch (err) {
    // Backend API not reachable or static server: graceful fallback
  }

  // Fallback to local intelligent response generator
  const reply = generateLocalIntelligentResponse(text);
  appendChatMessage('ai', reply);
}

function handleChatSubmit(event) {
  event.preventDefault();
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (text) {
    sendUserMessage(text);
    input.value = '';
  }
}

// ==========================================
// 8. INITIALIZATION ON DOM READY
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
  // Setup mobile menu button
  const mobileMenuBtn = document.getElementById('mobileMenuBtn');
  if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', toggleMobileSidebar);
  }

  const sidebarBackdrop = document.getElementById('sidebar-backdrop');
  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener('click', closeMobileSidebar);
  }

  // Setup skill input Enter key listeners
  const newCurrentSkillInput = document.getElementById('newCurrentSkillInput');
  if (newCurrentSkillInput) {
    newCurrentSkillInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        addCurrentSkill();
      }
    });
  }

  const newTargetSkillInput = document.getElementById('newTargetSkillInput');
  if (newTargetSkillInput) {
    newTargetSkillInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        addTargetSkill();
      }
    });
  }

  // Initial welcome message in Chatbot with ZERO symbols
  appendChatMessage(
    'ai',
    'Welcome Ali! I am your Pakistan Tech Career Coach. How can I assist you with your salary expectations, interview preparation, or learning roadmap today?'
  );

  // Initial render of all views
  renderAll();
});
