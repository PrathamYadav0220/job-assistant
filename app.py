import streamlit as st
# Set page config first, before any other st commands
st.set_page_config(page_title="ResumeATS Pro", layout="wide")

import configparser
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
import re
from collections import Counter
import hashlib
import plotly.graph_objects as go
import numpy as np
from database import init_db, create_user, verify_user, get_data

import time
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Custom CSS for Apple-inspired design
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f7;
        color: #1d1d1f;
    }
    .stButton>button {
        background-color: #0071e3;
        color: white;
        border-radius: 20px;
    }
    .stProgress > div > div {
        background-color: #0071e3;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize database
init_db()

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = None

# Authentication UI
def show_auth_ui():
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.header("Login")
        login_email = st.text_input("Email", key="login_email")  # Changed to email
        login_password = st.text_input("Password", type="password", key="login_password")
        edgedriver_path = st.text_input("Edge Driver Path", key="edgeDriver_path")
        if st.button("Login"):
            if verify_user(login_email, login_password):  # Use email for verification
                st.session_state.authenticated = True
                st.session_state.username = login_email  # Store email in session state
                st.session_state.password = login_password  # Store password in session state
                st.session_state.edgedriver_path = edgedriver_path
                st.success("Successfully logged in!")
                st.rerun()
            else:
                st.error("Invalid email or password")

    with tab2:
        st.header("Sign Up")
        new_email = st.text_input("Email", key="new_email")  # Changed to email
        new_username = st.text_input("Username (optional)", key="new_username")
        new_password = st.text_input("Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        edgedriver_path = st.text_input("Edge Driver Path", key="edgedriver_path_signup")
        
        if st.button("Sign Up"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long")
            else:
                if create_user(new_username, new_password, new_email):  # Use email for account creation
                    st.session_state.edgedriver_path = edgedriver_path
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Email already exists")

# Main app logic
if not st.session_state.authenticated:
    show_auth_ui()
else:
    # Show logout button in sidebar
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()
        
    st.title(f"Welcome {st.session_state.username}!")
    
    # Add feature selector
    feature = st.selectbox(
        "Select Feature",
        ["Resume ATS Pro", "Auto Apply"],
        index=0
    )
    
    if feature == "Resume ATS Pro":
        # Load environment variables and configure API
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Add new scoring helper functions
        def calculate_keyword_match(text, keywords):
            """Calculate keyword match percentage"""
            text = text.lower()
            found_keywords = sum(1 for keyword in keywords if keyword.lower() in text)
            return (found_keywords / len(keywords)) * 100 if keywords else 0

        def normalize_score(score):
            """Normalize score to prevent outliers"""
            return min(max(score, 0), 100)

        def get_cached_score(pdf_text, job_description=None):
            """Get cached score or None"""
            if not pdf_text:
                return None
            # Create unique hash for the resume and job description combination
            content_hash = hashlib.md5((pdf_text + (job_description or "")).encode()).hexdigest()
            return st.session_state.get(f'score_{content_hash}')

        def cache_score(pdf_text, score, job_description=None):
            """Cache the score"""
            content_hash = hashlib.md5((pdf_text + (job_description or "")).encode()).hexdigest()
            st.session_state[f'score_{content_hash}'] = score

        class ATSScoreComponents:
            def __init__(self):  # Fix: Add __init__ method
                self.format_score = 0
                self.content_score = 0
                self.keyword_score = 0
                self.match_score = 0
                self.total_score = 0

        def calculate_base_ats_score(pdf_text, job_description=None):
            """Calculate unified ATS score and return components"""
            score_components = ATSScoreComponents()
            
            # Basic Resume Structure (40 points)
            sections = ['experience', 'education', 'skills']
            for section in sections:
                if section in pdf_text.lower():
                    score_components.format_score += 10
            
            # Clean formatting check
            if len(re.findall(r'[^\x00-\x7F]', pdf_text)) == 0:
                score_components.format_score += 5
            if len(re.findall(r'[^\S\n]{2,}', pdf_text)) == 0:
                score_components.format_score += 5

            # Content Quality
            action_verbs = ['achieved', 'implemented', 'developed', 'managed', 'created', 'increased']
            score_components.keyword_score = calculate_keyword_match(pdf_text, action_verbs)
            score_components.content_score = score_components.keyword_score * 0.2

            # Job description matching
            if job_description:
                job_terms = set(re.findall(r'\b\w+\b', job_description.lower()))
                resume_terms = set(re.findall(r'\b\w+\b', pdf_text.lower()))
                score_components.match_score = len(job_terms.intersection(resume_terms)) / len(job_terms) * 30
                score_components.content_score += score_components.match_score
            else:
                score_components.content_score += 30 if len(pdf_text.split()) > 200 else 15

            score_components.total_score = normalize_score(score_components.format_score + score_components.content_score)
            return score_components

        def display_score_visualization(score_components, analysis_components):
            """Display visual representation of ATS score"""
            ats_score = score_components.total_score
            
            # Single consistent score display
            st.markdown(f"### ATS Compatibility Score: {ats_score:.1f}/100")
            st.progress(ats_score/100)
            
            # Create radar chart for components
            categories = list(analysis_components.keys())
            values = list(analysis_components.values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Score Components'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=False,
                title="Score Component Breakdown",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

        def get_gemini_output(pdf_text, prompt):
            """Enhanced Gemini output with score visualization"""
            cached_score = get_cached_score(pdf_text, prompt)
            if cached_score:
                return cached_score
            
            score_components = calculate_base_ats_score(pdf_text, job_description if use_jd else None)
            
            try:
                response = model.generate_content([pdf_text, prompt])
                response_text = response.text
                
                # Calculate component scores
                analysis_components = {
                    'Resume Structure': normalize_score(score_components.format_score * 2.5),
                    'Content Quality': normalize_score(score_components.content_score * 1.67),
                    'Keyword Match': score_components.keyword_score
                }
                
                if use_jd:
                    analysis_components['Job Description Match'] = normalize_score(score_components.match_score * 3.33)
                
                # Display visualization with consistent scoring
                display_score_visualization(score_components, analysis_components)
                
                enhanced_response = f"""
        Score Summary:
        ATS Compatibility Score: {score_components.total_score:.1f}/100

        Detailed Analysis:
        {response_text}
        """
                cache_score(pdf_text, enhanced_response, prompt)
                return enhanced_response
                
            except Exception as e:
                st.error(f"Error in generating response: {str(e)}")
                return None

        # Initialize session state for caching if not exists
        if 'score_cache' not in st.session_state:
            st.session_state.score_cache = {}

        # Function to read PDF
        def read_pdf(uploaded_file):
            if uploaded_file is not None:
                pdf_reader = PdfReader(uploaded_file)
                return "".join([page.extract_text() for page in pdf_reader.pages])
            raise FileNotFoundError("No file uploaded")

        # File upload
        upload_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

        # Job description toggle and input
        st.markdown("### 📋 Job Description Analysis")
        use_jd = st.checkbox("Include job description for targeted analysis")
        job_description = ""
        if use_jd:
            job_description = st.text_area("Enter the job description", height=200)
            st.info("💡 Paste the complete job description for more accurate matching and ATS scoring")

        # Analysis options
        analysis_option = st.radio("Choose analysis type:", 
                                ["Quick Scan", "Detailed Analysis", "ATS Optimization"])

        if st.button("Analyze Resume"):
            if upload_file is not None:
                st.session_state.pdf_text = read_pdf(upload_file)
                pdf_text = st.session_state.pdf_text
                
                if analysis_option == "Quick Scan":
                    prompt = f"""
                    Analyze this resume and provide:
                    1. Key strengths (3-4 points)
                    2. Critical improvements needed (2-3 points)
                    3. Keyword optimization suggestions
                    
                    Focus on actionable feedback without numerical scores.
                    
                    Resume text: {pdf_text}
                    {f'Job Description: {job_description}' if use_jd else ''}
                    """
                elif analysis_option == "Detailed Analysis":
                    prompt = f"""
                    You are an expert ATS analyzer. Provide a comprehensive analysis:

                    {f'''Job Alignment Analysis (40 points):
                    1. Required Skills Coverage (15 points):
                    - Must-have skills presence
                    - Nice-to-have skills presence
                    - Technology stack matching
                    
                    2. Experience Match (15 points):
                    - Years of experience alignment
                    - Role responsibility matching
                    - Industry-specific requirements
                    
                    3. Qualification Match (10 points):
                    - Education requirements
                    - Certification requirements
                    - Special qualification matching''' if use_jd else ''}

                    Technical ATS Analysis (60 points):
                    1. Keyword Optimization (20 points):
                    - Industry-standard terminology
                    - Technical skill formatting
                    - Keyword density and placement
                    
                    2. Format & Structure (20 points):
                    - Section header standardization
                    - Consistent formatting
                    - ATS-friendly layout
                    
                    3. Content Quality (20 points):
                    - Quantified achievements
                    - Role-specific accomplishments
                    - Professional impact metrics

                    Provide:
                    1. Category-wise scoring with justification
                    2. {'Job fitness score and' if use_jd else ''} ATS compatibility score
                    3. Detailed keyword analysis
                    4. Section-by-section improvement recommendations
                    5. Format optimization guide
                    
                    Resume text: {pdf_text}
                    {f'Job Description: {job_description}' if use_jd else ''}
                    """
                else:
                    prompt = f"""
                    You are an expert ATS optimization specialist. Analyze with enhanced criteria:

                    {f'''Job-Specific Optimization (50 points):
                    1. Key Requirements Match (20 points):
                    - Must-have skills coverage
                    - Experience level alignment
                    - Industry-specific keywords
                    
                    2. Role Alignment (20 points):
                    - Job title optimization
                    - Responsibility matching
                    - Achievement relevance
                    
                    3. Qualification Alignment (10 points):
                    - Education requirements
                    - Certification matches
                    - Special requirements''' if use_jd else ''}

                    Technical Optimization (50 points):
                    1. Keyword Placement (20 points):
                    - Strategic keyword distribution
                    - Contextual usage
                    - Natural integration
                    
                    2. Format Optimization (15 points):
                    - ATS-friendly sections
                    - Consistent structure
                    - Clean formatting
                    
                    3. Content Enhancement (15 points):
                    - Achievement metrics
                    - Role descriptions
                    - Skill demonstrations

                    Provide:
                    1. Detailed optimization score
                    2. {'Job-specific keyword recommendations and' if use_jd else ''} general keywords
                    3. Section-wise formatting improvements
                    4. Content enhancement suggestions
                    5. Priority action items
                    
                    Resume text: {pdf_text}
                    {f'Job Description: {job_description}' if use_jd else ''}
                    """
                response = get_gemini_output(pdf_text, prompt)
                
                st.subheader("Analysis Results")
                st.write(response)
                
                # Option to chat about the resume
                st.subheader("Have questions about your resume?")
                user_question = st.text_input("Ask me anything about your resume or the analysis:")
                if user_question:
                    chat_prompt = f"""
                    Based on the resume and analysis above, answer the following question:
                    {user_question}
                    
                    Resume text: {pdf_text}
                    Previous analysis: {response}
                    """
                    chat_response = get_gemini_output(pdf_text, chat_prompt)
                    st.write(chat_response)
            else:
                st.error("Please upload a resume to analyze.")

        # Additional resources
        st.sidebar.title("Resources")
        st.sidebar.markdown("""
        - [Resume Writing Tips](https://www.jobbank.gc.ca/findajob/resources/write-good-resume)
        - [ATS Optimization Guide](https://career.io/career-advice/create-an-optimized-ats-resume)
        - [Interview Preparation](https://hbr.org/2021/11/10-common-job-interview-questions-and-how-to-answer-them)
        """)

        # Feedback form
        st.sidebar.title("Feedback")
        st.sidebar.text_area("Help us improve! Leave your feedback:")
        st.sidebar.button("Submit Feedback")
        
    elif feature == "Auto Apply":
        st.title("Auto Apply")
        st.subheader("Automatically Apply to Jobs on Naukri.com")
        
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Resume upload for Auto Apply
        auto_apply_resume = st.file_uploader("Upload Resume for Auto Apply", type=["pdf"])
        if auto_apply_resume:
            def read_pdf(uploaded_file):
                if uploaded_file is not None:
                    pdf_reader = PdfReader(uploaded_file)
                    return "".join([page.extract_text() for page in pdf_reader.pages])
                raise FileNotFoundError("No file uploaded")
            st.session_state.pdf_text = read_pdf(auto_apply_resume)
            
        if 'pdf_text' not in st.session_state or not st.session_state.pdf_text:
            st.error("Please upload a resume first")
            st.stop()

        with st.form("auto_apply_form"):
            job_type = st.selectbox("Job Type", options=["job", "internship"], index=0)
            designation_input = st.text_input("Designation (comma separated)")
            location_input = st.text_input("Location (comma separated)")
            max_applications = st.number_input("Max Applications per Day", min_value=1, step=1)
            yoe = st.number_input("Years of Experience", min_value=0, step=1)
            salary = st.number_input("Expected Salary", min_value=0)
            max_pages = st.number_input("Max Pages to Search", min_value=1, step=1)
            min_match_score = st.number_input("Minimum Job Description Match Score (0 - 1)", min_value=0.0, max_value=1.0, step=0.1, value=0.0)
            submitted = st.form_submit_button("Start Auto Apply")
        
        if submitted:
            designations = [d.strip() for d in designation_input.split(",") if d.strip()]
            locations = [l.strip() for l in location_input.split(",") if l.strip()]

            def login_naukri(driver, wait, credentials):
                """Log into Naukri.com using provided credentials."""
                driver.get('https://login.naukri.com/')
                st.write("Checkpoint: Navigated to login page.")
                try:
                    wait.until(EC.presence_of_element_located((By.ID, 'usernameField'))).send_keys(credentials['email'])
                    wait.until(EC.presence_of_element_located((By.ID, 'passwordField'))).send_keys(credentials['password'])
                    wait.until(EC.element_to_be_clickable((By.XPATH, "//button[text()='Login']"))).click()
                    st.write("Checkpoint: Login successful.")
                except Exception as e:
                    st.write(f"Checkpoint: Login failed: {e}")
                    driver.quit()
                    exit()

            def construct_url_for_combo(designation, location, job_type, page):
                """Helper function to generate a URL for a single designation, location, and page."""
                base_url = "https://www.naukri.com"
                designation_slug = designation.lower().replace(' ', '-')
                location_slug = location.lower().replace(' ', '-') if location else ""
                
                if job_type == "internship":
                    if location_slug:
                        url = (f"{base_url}/{designation_slug}-internship-jobs-in-{location_slug}"
                            if page == 1 else
                            f"{base_url}/internship/{designation_slug}-internship-jobs-in-{location_slug}-{page}")
                    else:
                        url = (f"{base_url}/{designation_slug}-internship-jobs"
                            if page == 1 else
                            f"{base_url}/internship/{designation_slug}-internship-jobs-{page}")
                else:
                    if location_slug:
                        url = (f"{base_url}/{designation_slug}-jobs-in-{location_slug}"
                            if page == 1 else
                            f"{base_url}/{designation_slug}-jobs-in-{location_slug}-{page}")
                    else:
                        url = (f"{base_url}/{designation_slug}-jobs"
                            if page == 1 else
                            f"{base_url}/{designation_slug}-jobs-{page}")
                return url

            def construct_search_urls(designations, locations, job_type, max_pages):
                """
                For each combination of designation and location (both are lists of strings),
                and for each page up to max_pages, generate a Naukri search URL.
                """
                urls = []
                
                for designation in designations:
                    if locations:
                        for location in locations:
                            for page in range(1, max_pages + 1):
                                url = construct_url_for_combo(designation, location, job_type, page)
                                st.write(f"Checkpoint: Constructed URL: {url}")
                                urls.append(url)
                    else:
                        for page in range(1, max_pages + 1):
                            url = construct_url_for_combo(designation, "", job_type, page)
                            st.write(f"Checkpoint: Constructed URL: {url}")
                            urls.append(url)
                            
                return urls

            def scrape_job_links(driver, wait, designations, locations, job_type, max_pages):
                """Collect job links from search results for each designation and location combination."""
                job_links = []
                urls = construct_search_urls(designations, locations, job_type, max_pages)
                
                for url in urls:
                    driver.get(url)
                    st.write(f"Checkpoint: Navigated to search results: {url}")
                    
                    try:
                        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "span[title='Close']"))).click()
                        st.write("Checkpoint: Closed a popup.")
                    except Exception:
                        pass
                    
                    try:
                        jobs = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.title")))
                        for job in jobs:
                            job_url = job.get_attribute('href')
                            if job_url and job_url not in job_links:
                                job_links.append(job_url)
                        st.write(f"Checkpoint: Found {len(jobs)} jobs on {url}")
                    except TimeoutException:
                        st.write(f"Checkpoint: No jobs found on {url}")
                
                st.write(f"Checkpoint: Total unique job links collected: {len(job_links)}")
                return job_links

            def extract_job_skills(driver, wait):
                """
                Attempt to locate and extract the 'Key Skills' from the job listing.
                """
                info = {
                    'skill': [],
                    'yoe': 0,
                    'salary': [],
                    'company_name': "Unknown Company",
                    'designation': "Unknown Designation"
                }
                skill_texts = []
                
                try:
                    parent_div = wait.until(EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div.styles_key-skill_GIPn")
                    ))
                    child_div = parent_div.find_element(By.XPATH, ".//div[not(@class)]")
                    skill_spans = child_div.find_elements(By.TAG_NAME, "span")
                    for span in skill_spans:
                        text = span.text.strip().lower()
                        if text:
                            skill_texts.append(text)
                    if skill_texts:
                        st.write(f"Checkpoint: Found {len(skill_texts)} skills from primary structure.")
                        info['skill'] = skill_texts
                except Exception as e:
                    info['skill'] = skill_texts

                try:
                    company_div = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.styles_jd-header-comp-name__MvqAI")
                        )
                    )
                    try:
                        company_name = company_div.find_element(By.TAG_NAME, "a").text.strip()
                    except Exception as e:
                        company_name = company_div.text.strip()
                    info['company_name'] = company_name
                except Exception as e:
                    info['company_name'] = "Unknown Company"

                try:
                    designation_elem = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "h1.styles_jd-header-title__rZwM1")
                        )
                    )
                    designation = designation_elem.text.strip()
                    info['designation'] = designation
                except Exception as e:
                    info['designation'] = "Unknown Designation"

                try:
                    exp_div = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.styles_jhc_exp_k_giM")
                        )
                    )
                    try:
                        yoe_text = exp_div.find_element(By.TAG_NAME, "span").text.strip()
                        info['yoe'] = int(yoe_text.split()[0])
                    except Exception as e:
                        try:
                            info['yoe'] = int(exp_div.text.strip().split()[0])
                        except Exception as e:
                            info['yoe'] = 0
                except Exception as e:
                    info['yoe'] = 0

                try:
                    salary_div = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.styles_jhc_salary_jdfEC")
                        )
                    )
                    try:
                        salary_text = salary_div.find_element(By.TAG_NAME, "span").text.strip()
                        info['salary'] = list(map(float, salary_text.split()[0].split('-')))
                    except Exception as e:
                        try:
                            info['salary'] = list(map(float, salary_div.text.strip().split()[0].split('-')))
                        except Exception as e:
                            info['salary'] = [0, 0]
                except Exception as e:
                    info['salary'] = [0, 0]

                return info

            def skills_match(job_skills, user_skills):
                """
                Calculate the percentage of user skills that are mentioned in the job's skills.
                Returns the match percentage.
                """
                if not job_skills:
                    return 0
                count = 0
                for sk in job_skills:
                    if sk in user_skills:
                        count += 1
                percentage = (count / len(job_skills)) * 100
                st.write(f"Checkpoint: {percentage:.2f}% of user skills matched.")
                return percentage

            def apply_to_jobs(driver, wait, job_links, max_applications, yoe, salary, user_skills, min_match_score, expected_domain):
                """Apply to jobs after checking the skills match and update the application log in the DB."""
                applied = 0
                failed = []
                for job_url in job_links:
                    if applied >= max_applications:
                        st.write("Checkpoint: Reached daily application limit.")
                        break
                    driver.get(job_url)
                    st.write(f"Checkpoint: Navigated to job posting: {job_url}")
                    try:
                        driver.find_element(By.XPATH, "//div[contains(text(), 'Applied')]")
                        st.write(f"Checkpoint: Already applied to {job_url}")
                        continue
                    except NoSuchElementException:
                        pass
                    job_text = extract_job_skills(driver, wait)

                    if yoe < job_text['yoe']:
                        continue

                    if salary > job_text['salary'][1]:
                        continue

                    match_percentage = skills_match(job_text['skill'], user_skills)
                    if match_percentage < min_match_score * 100:
                        st.write(f"Checkpoint: Skipping {job_url}: Only {match_percentage:.2f}% user skills matched.")
                        continue
                    try:
                        apply_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Apply')]")))
                        apply_btn.click()
                        st.write("Checkpoint: Clicked Apply button.")
                        current_url = driver.current_url
                        if expected_domain not in current_url:
                            st.write(f"Checkpoint: Redirected externally from {job_url}. Skipping application.")
                            driver.back()
                            continue
                        try:
                            submit_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Submit')]")))
                            submit_btn.click()
                            st.write(f"Checkpoint: Successfully applied to {job_url}")
                        except Exception:
                            st.write(f"Checkpoint: Quick applied to {job_url}")
                        applied += 1
                        try:
                            limit_msg = driver.find_element(By.XPATH, "//*[contains(text(), 'daily quota')]")
                            st.write(f"Checkpoint: Daily quota reached message detected: {limit_msg.text}")
                            break
                        except NoSuchElementException:
                            pass
                    except Exception as e:
                        st.write(f"Checkpoint: Failed to apply to {job_url}: {str(e)}")
                        failed.append(job_url)
                st.write(f"Checkpoint: Applied to {applied} jobs.")
                return applied, failed

            def extract_skills_from_resume():
                prompt = "Extract technical skills from this resume:"
                response = model.generate_content([st.session_state.pdf_text, prompt])
                return [skill.lower() for skill in response.text.split(", ")]

            def main(job_type, designations, locations, max_applications, yoe, max_pages, min_match_score, salary):
                edge_driver_path = st.session_state.get('edgedriver_path')
                credentials = {}
                credentials['email'] = st.session_state.get('username')
                credentials['password'] = st.session_state.get('password')
                
                user_skills = extract_skills_from_resume()
                expected_domain = "naukri.com"
                
                options = webdriver.EdgeOptions()
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_argument("--start-maximized")
                service = Service(edge_driver_path)
                driver = webdriver.Edge(service=service, options=options)
                wait = WebDriverWait(driver, 20)
                
                try:
                    login_naukri(driver, wait, credentials)
                    job_links = scrape_job_links(driver, wait, designations, locations, job_type, max_pages)
                    st.write(f"Checkpoint: Total job links found: {len(job_links)}")
                    if job_links:
                        applied_count, failed_applications = apply_to_jobs(driver, wait, job_links, max_applications, yoe, salary, user_skills, min_match_score, expected_domain)
                        st.write(f"Checkpoint: Applied to {applied_count} jobs. Failed: {len(failed_applications)}")
                    else:
                        st.write("Checkpoint: No job links found. Check search parameters.")
                finally:
                    driver.quit()
                    st.write("Checkpoint: WebDriver session ended.")

            st.info("Auto apply process started. Check the checkpoints below for progress updates.")
            main(job_type, designations, locations, max_applications, yoe, max_pages, min_match_score, salary)
            st.success("Auto apply process completed.")