import streamlit as st
import pdfplumber
from docx import Document
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
import spacy
from dotenv import load_dotenv


# load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY in .env file!")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)


# clean text
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# extract text from uploaded file
def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return clean_text(text)
    
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        text = " ".join([para.text for para in doc.paragraphs])
        return clean_text(text)
    
    else:
        raise ValueError("Unsupported file type")
    

# load NLP model
nlp = spacy.load("en_core_web_sm")

# skill keywords
SKILL_KEYWORDS = {
    # --- Programming Languages ---
    'python', 'java', 'javascript', 'typescript', 'c', 'c++', 'c#', 'go', 'rust', 'kotlin', 'swift',
    'scala', 'r', 'matlab', 'php', 'ruby', 'perl', 'sql', 'pl/sql', 'bash', 'shell', 'powershell',

    # --- Web & Frontend ---
    'html', 'css', 'react', 'angular', 'vue.js', 'svelte', 'next.js', 'nuxt.js', 'jquery', 'bootstrap',
    'tailwind css', 'sass', 'less', 'webpack', 'vite', 'redux', 'mobx', 'graphql', 'rest api',

    # --- Backend & Frameworks ---
    'node.js', 'django', 'flask', 'fastapi', 'spring', 'spring boot', 'express.js', 'laravel', 'rails',
    'asp.net', 'dotnet', 'nestjs', 'gin', 'fiber', 'quarkus',

    # --- Databases ---
    'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'microsoft sql server', 'mariadb', 'cassandra',
    'redis', 'elasticsearch', 'neo4j', 'dynamodb', 'firebase', 'bigquery', 'snowflake', 'clickhouse',

    # --- Cloud & DevOps ---
    'aws', 'amazon web services', 'azure', 'google cloud platform', 'gcp', 'cloudflare', 'digitalocean',
    'docker', 'kubernetes', 'helm', 'jenkins', 'gitlab ci', 'github actions', 'circleci', 'terraform',
    'ansible', 'puppet', 'chef', 'prometheus', 'grafana', 'elk stack', 'splunk', 'linux', 'unix',
    'bash scripting', 'shell scripting', 'nginx', 'apache', 'istio', 'argo cd',

    # --- Data Science & AI/ML ---
    'machine learning', 'deep learning', 'artificial intelligence', 'ai', 'nlp', 'computer vision',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'hugging face',
    'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'opencv', 'spacy', 'nltk', 'transformers',
    'llm', 'large language models', 'rag', 'fine-tuning', 'prompt engineering', 'data mining', 'feature engineering',

    # --- Data Engineering & Big Data ---
    'spark', 'hadoop', 'kafka', 'airflow', 'dbt', 'etl', 'elt', 'data pipeline', 'data warehouse',
    'data lake', 'redshift', 'databricks', 'looker', 'tableau', 'power bi', 'qlik', 'metabase',
    'apache beam', 'flink', 'storm', 'presto', 'trino',

    # --- Mobile Development ---
    'android', 'ios', 'flutter', 'react native', 'xamarin', 'swiftui', 'kotlin multiplatform', 'cordova',
    'ionic', 'firebase cloud messaging', 'push notifications',

    # --- Testing & QA ---
    'selenium', 'cypress', 'playwright', 'jest', 'mocha', 'chai', 'pytest', 'unittest', 'junit',
    'testng', 'postman', 'swagger', 'manual testing', 'automation testing', 'tdd', 'bdd', 'unit testing',
    'integration testing', 'end-to-end testing', 'performance testing', 'load testing', 'security testing',

    # --- Project & Product Management ---
    'agile', 'scrum', 'kanban', 'lean', 'waterfall', 'jira', 'trello', 'asana', 'monday.com', 'notion',
    'product management', 'product owner', 'project management', 'stakeholder management', 'roadmapping',
    'user stories', 'epics', 'backlog grooming', 'sprint planning', 'retrospectives',

    # --- Business & Soft Skills ---
    'communication', 'verbal communication', 'written communication', 'presentation', 'public speaking',
    'leadership', 'team leadership', 'team management', 'conflict resolution', 'negotiation', 'problem solving',
    'critical thinking', 'time management', 'organization', 'adaptability', 'collaboration', 'teamwork',
    'customer service', 'client management', 'stakeholder engagement', 'emotional intelligence', 'mentoring',
    'coaching', 'decision making', 'strategic thinking', 'business analysis', 'requirements gathering',

    # --- Design & UX/UI ---
    'figma', 'adobe xd', 'sketch', 'invision', 'balsamiq', 'ux design', 'ui design', 'user research',
    'wireframing', 'prototyping', 'usability testing', 'information architecture', 'interaction design',
    'graphic design', 'illustrator', 'photoshop', 'after effects', 'motion design',

    # --- Cybersecurity ---
    'cybersecurity', 'penetration testing', 'ethical hacking', 'network security', 'application security',
    'owasp', 'siem', 'firewalls', 'ids/ips', 'zero trust', 'iam', 'oauth', 'saml', 'encryption', 'ssl/tls',
    'vulnerability assessment', 'risk management', 'compliance', 'gdpr', 'hipaa', 'iso 27001', 'nist',

    # --- Finance & Analytics ---
    'financial modeling', 'excel', 'google sheets', 'powerpoint', 'financial analysis', 'budgeting',
    'forecasting', 'accounting', 'bookkeeping', 'sap', 'oracle erp', 'workday', 'tableau', 'qlik sense',
    'power query', 'dax', 'vba', 'pivot tables', 'dashboards', 'kpi tracking', 'roi analysis',

    # --- Other Tools & Platforms ---
    'git', 'github', 'gitlab', 'bitbucket', 'mercurial', 'svn', 'vs code', 'intellij', 'eclipse', 'vim',
    'emacs', 'jupyter notebook', 'colab', 'visual studio', 'xcode', 'android studio', 'postman', 'swagger',
    'docker hub', 'aws console', 'gcp console', 'azure portal', 'slack', 'microsoft teams', 'zoom'
}

def extract_skills(text):
    # normalizing text
    doc = nlp(text.lower())
    words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    text_lower = " ".join(words)
    
    found_skills = set()
    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            found_skills.add(skill.title())
    return sorted(found_skills)


# generate AI feedback
def get_gemini_summary(resume_text, job_desc):
    model = genai.GenerativeModel('models/gemini-2.0-flash-001')
    prompt = f"""
    You are an expert HR recruiter. Evaluate the candidate based on the resume and job description below.

    Job Description:
    {job_desc}

    Resume:
    {resume_text}

    Respond in exactly 3 short sentences:
    1. Overall match level.
    2. Key strengths relevant to the job.
    3. Any major missing qualifications or concerns.
    """
    try:
        response = model.generate_content(prompt, request_options={"timeout": 60})
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {str(e)}"


# === UI ===
st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="centered")
st.title("📄 Intelligent Resume Screener")
st.markdown("### Powered by Google Gemini AI")
st.text("Made by Dhrubendu Das")

job_desc = st.text_area("📋 Paste Job Description", height=150)
uploaded_files = st.file_uploader(
    "📎 Upload Resume(s) (PDF or DOCX)", 
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if st.button("🔍 Analyze Resume(s)", type="primary"):
    if not job_desc.strip():
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume (PDF or DOCX).")
    elif len(uploaded_files) > 5:
        st.warning("Please don't upload more than 5 resumes for best performance.")
    else:
        all_results = []
        
        with st.spinner(f"Processing {len(uploaded_files)} resume(s)..."):
            for uploaded_file in uploaded_files:
                try:
                    resume_text = extract_text(uploaded_file)
                    if not resume_text:
                        st.warning(f"Could not extract text from {uploaded_file.name}. Skipping.")
                        continue

                    # TF-IDF Matching
                    tfidf = TfidfVectorizer(stop_words='english')
                    vectors = tfidf.fit_transform([job_desc, resume_text])
                    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                    score = max(0, min(100, round(similarity * 100, 1)))

                    # Extract skills
                    skills = extract_skills(resume_text)

                    # AI Summary (only for top 3 to avoid rate limits)
                    ai_feedback = ""
                    if len(uploaded_files) <= 3:
                        ai_feedback = get_gemini_summary(resume_text, job_desc)
                    else:
                        ai_feedback = "AI summary skipped (batch mode)."

                    all_results.append({
                        "filename": uploaded_file.name,
                        "score": score,
                        "skills": skills,
                        "feedback": ai_feedback,
                        "text": resume_text
                    })

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue

        # Sort by score (descending)
        all_results.sort(key=lambda x: x["score"], reverse=True)

        # Display results
        st.divider()
        st.subheader(f"🏆 Top Matches ({len(all_results)} resumes processed)")
        
        for i, res in enumerate(all_results):
            with st.expander(f"📄 {res['filename']} — **{res['score']}% Match**", expanded=(i == 0)):
                st.markdown(f"**Skills Detected:** {', '.join(res['skills']) if res['skills'] else 'None'}")
                st.markdown("**AI Recruiter Feedback:**")
                st.info(res["feedback"])
                
                with st.expander("Full Extracted Text (Preview)"):
                    st.text_area(
                        f"Text from {res['filename']}",
                        res["text"][:800] + "...",
                        height=150,
                        disabled=True,
                        label_visibility="hidden"
                    )