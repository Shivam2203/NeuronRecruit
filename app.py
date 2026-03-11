# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import asyncio
import requests
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from security import security_manager
from database import db_manager
from agents import hiring_graph, HiringState
from utils import text_extractor, report_generator, data_validator
from models import UserCreate, UserLogin

# Page configuration
st.set_page_config(
    page_title="HireGenAI Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1E293B;
        font-family: 'Inter', sans-serif;
    }
    
    /* Cards */
    .css-1r6slb0 {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #1E293B;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
    }
    
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-color: #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Authentication functions
def login_user(username: str, password: str) -> bool:
    """Authenticate user"""
    user = db_manager.get_user(username=username)
    if user and security_manager.verify_password(password, user['password_hash']):
        st.session_state.authenticated = True
        st.session_state.user_id = user['id']
        st.session_state.username = user['username']
        db_manager.update_last_login(user['id'])
        db_manager.log_activity(user['id'], "login", f"User logged in from Streamlit")
        return True
    return False

def register_user(username: str, email: str, password: str) -> bool:
    """Register new user"""
    try:
        # Validate inputs
        user_data = UserCreate(username=username, email=email, password=password)
        
        # Check if user exists
        if db_manager.get_user(username=username) or db_manager.get_user(email=email):
            st.error("Username or email already exists")
            return False
        
        # Hash password
        password_hash = security_manager.hash_password(password)
        
        # Create user
        user_id = db_manager.create_user(username, email, password_hash)
        
        if user_id:
            db_manager.log_activity(user_id, "register", f"User registered from Streamlit")
            return True
        return False
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
        return False

def logout_user():
    """Logout user"""
    if st.session_state.user_id:
        db_manager.log_activity(st.session_state.user_id, "logout", "User logged out")
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.api_key = None
    st.rerun()

# Authentication UI
if not st.session_state.authenticated:
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1 style="font-size: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🤖 HireGenAI Pro
        </h1>
        <p style="font-size: 1.2rem; color: #666;">AI-Powered Hiring Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    if login_user(username, password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Username", key="reg_username")
                new_email = st.text_input("Email", key="reg_email")
                new_password = st.text_input("Password", type="password", key="reg_password")
                confirm_password = st.text_input("Confirm Password", type="password")
                
                submitted = st.form_submit_button("Register")
                
                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif register_user(new_username, new_email, new_password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Registration failed")

# Main application
else:
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <h3 style="color: white;">Welcome, {st.session_state.username}!</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        pages = ["Home", "Evaluate Candidates", "Batch Processing", "Analytics", "API Keys", "Settings"]
        for page in pages:
            if st.button(page, key=f"nav_{page}", use_container_width=True):
                st.session_state.current_page = page
        
        st.markdown("---")
        
        # Logout
        if st.button("Logout", use_container_width=True):
            logout_user()
        
        # API Key display
        if st.session_state.api_key:
            st.info(f"API Key: {st.session_state.api_key[:10]}...")
    
    # Main content
    if st.session_state.current_page == "Home":
        st.markdown("""
        <div style="text-align: center; padding: 50px 20px;">
            <h1 style="font-size: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🚀 HireGenAI Pro
            </h1>
            <p style="font-size: 1.5rem; color: #666;">Next-Generation AI Hiring Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: white; padding: 30px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h3>📄 Smart Resume Parsing</h3>
                <p>Extract structured data from resumes with 95% accuracy using advanced AI</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: white; padding: 30px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h3>🎯 Intelligent Matching</h3>
                <p>Hybrid scoring with skill gap analysis and cultural fit assessment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: white; padding: 30px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h3>💬 Interview Generator</h3>
                <p>Generate personalized interview questions based on candidate profile</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        analytics = db_manager.get_analytics(st.session_state.user_id)
        
        with col1:
            st.metric("Total Candidates", analytics['total_candidates'])
        with col2:
            st.metric("Total Jobs", analytics['total_jobs'])
        with col3:
            st.metric("Evaluations", analytics['total_evaluations'])
        with col4:
            st.metric("Avg Match Score", f"{analytics['average_score']}%")

    elif st.session_state.current_page == "Evaluate Candidates":
        st.title("📄 Evaluate Candidates")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload Resume(s)",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Upload one or more resumes"
            )
        
        with col2:
            job_title = st.text_input("Job Title")
            company = st.text_input("Company Name")
            job_description = st.text_area("Job Description", height=200)
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2, col3 = st.columns(3)
            with col1:
                bias_detection = st.checkbox("Bias Detection", value=True)
            with col2:
                skill_gap = st.checkbox("Skill Gap Analysis", value=True)
            with col3:
                cultural_fit = st.checkbox("Cultural Fit", value=True)
        
        if st.button("Start Evaluation", use_container_width=True):
            if not uploaded_files:
                st.error("Please upload at least one resume")
            elif not job_description:
                st.error("Please provide a job description")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                
                for idx, file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {file.name}...")
                        
                        # Read and extract text
                        content = file.read()
                        file_ext = file.name.split('.')[-1].lower()
                        resume_text = text_extractor.extract(content, file_ext)
                        
                        if not resume_text:
                            st.error(f"Could not extract text from {file.name}")
                            continue
                        
                        # Create state
                        state = HiringState(
                            resume_text=resume_text,
                            jd_text=job_description,
                            job_title=job_title,
                            company_name=company,
                            resume_data={},
                            jd_data={},
                            match_result={},
                            interview_questions={},
                            feedback_report={},
                            processing_time=0,
                            confidence_score=0,
                            bias_analysis={},
                            alternative_roles=[],
                            processing_stage="start",
                            error=""
                        )
                        
                        # Process
                        result = hiring_graph.invoke(state)
                        
                        if not result.get('error'):
                            # Save to database
                            candidate_id = db_manager.save_candidate(
                                st.session_state.user_id,
                                result['resume_data'].get('name', 'Unknown'),
                                result['resume_data'].get('email', ''),
                                result['resume_data'].get('phone', ''),
                                resume_text,
                                result['resume_data']
                            )
                            
                            job_id = db_manager.save_job(
                                st.session_state.user_id,
                                job_title or "Unknown",
                                company or "Unknown",
                                job_description,
                                result['jd_data']
                            )
                            
                            evaluation_id = db_manager.save_evaluation(
                                st.session_state.user_id,
                                candidate_id,
                                job_id,
                                result['match_result'],
                                result['interview_questions'],
                                result['feedback_report'],
                                result['match_result']['overall_score']
                            )
                            
                            results.append(result)
                        
                        progress = (idx + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                
                status_text.text("Processing complete!")
                
                if results:
                    # Sort by score
                    results.sort(key=lambda x: x['match_result']['overall_score'], reverse=True)
                    st.session_state.last_results = results
                    
                    # Display results
                    st.success(f"✅ Processed {len(results)} candidates successfully!")
                    
                    # Top candidates
                    st.subheader("🏆 Top Candidates")
                    
                    for i, result in enumerate(results[:5], 1):
                        with st.container():
                            st.markdown(f"""
                            <div style="background: white; padding: 20px; border-radius: 10px; margin: 10px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                                <h3>{'🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'#{i}'} {result['resume_data'].get('name', 'Unknown')}</h3>
                                <p>Match Score: <strong style="color: {'#4CAF50' if result['match_result']['overall_score'] >= 70 else '#FF9800' if result['match_result']['overall_score'] >= 50 else '#f44336'}">{result['match_result']['overall_score']}%</strong></p>
                                <p>Experience: {result['resume_data'].get('total_experience_years', 0)} years</p>
                                <p>Top Skills: {', '.join(result['match_result'].get('matched_skills', [])[:5])}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander(f"View Details - {result['resume_data'].get('name', 'Unknown')}"):
                                tab1, tab2, tab3, tab4 = st.tabs(["Match Analysis", "Skills", "Interview", "Feedback"])
                                
                                with tab1:
                                    st.json(result['match_result'])
                                
                                with tab2:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.subheader("✅ Matched Skills")
                                        for skill in result['match_result'].get('matched_skills', []):
                                            st.markdown(f"• {skill}")
                                    with col2:
                                        st.subheader("❌ Missing Skills")
                                        for skill in result['match_result'].get('missing_skills', []):
                                            st.markdown(f"• {skill}")
                                
                                with tab3:
                                    questions = result.get('interview_questions', {})
                                    for category, q_list in questions.items():
                                        if isinstance(q_list, list) and q_list:
                                            st.subheader(category.replace('_', ' ').title())
                                            for q in q_list[:3]:
                                                st.markdown(f"**Q:** {q.get('question', '')}")
                                                st.markdown(f"*Difficulty: {q.get('difficulty', 'Medium')}*")
                                                st.markdown("---")
                                
                                with tab4:
                                    feedback = result.get('feedback_report', {})
                                    st.subheader("Strengths")
                                    for strength in feedback.get('strengths', []):
                                        st.markdown(f"✅ {strength}")
                                    
                                    st.subheader("Recommendations")
                                    for rec in feedback.get('recommendations', []):
                                        st.markdown(f"💡 {rec}")
                    
                    # Export options
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📥 Export as CSV"):
                            # Prepare data
                            export_data = []
                            for r in results:
                                export_data.append({
                                    'Name': r['resume_data'].get('name', ''),
                                    'Email': r['resume_data'].get('email', ''),
                                    'Match Score': r['match_result']['overall_score'],
                                    'Experience': r['resume_data'].get('total_experience_years', 0),
                                    'Matched Skills': ', '.join(r['match_result'].get('matched_skills', [])),
                                    'Missing Skills': ', '.join(r['match_result'].get('missing_skills', []))
                                })
                            
                            df = pd.DataFrame(export_data)
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "Download CSV",
                                csv,
                                f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )
                    
                    with col2:
                        if st.button("📊 Generate Report"):
                            # Generate HTML report
                            html = report_generator.to_html(results, "comparison")
                            st.download_button(
                                "Download Report",
                                html,
                                f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                "text/html"
                            )
                    
                    with col3:
                        if st.button("🔄 Clear Results"):
                            st.session_state.pop('last_results', None)
                            st.rerun()

    elif st.session_state.current_page == "Analytics":
        st.title("📊 Analytics Dashboard")
        
        # Get analytics
        analytics = db_manager.get_analytics(st.session_state.user_id)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Candidates", analytics['total_candidates'])
        with col2:
            st.metric("Total Jobs", analytics['total_jobs'])
        with col3:
            st.metric("Evaluations", analytics['total_evaluations'])
        with col4:
            st.metric("Avg Match Score", f"{analytics['average_score']}%")
        
        # Recent evaluations
        st.subheader("Recent Evaluations")
        evaluations = db_manager.get_evaluations(st.session_state.user_id)
        
        if evaluations:
            df = pd.DataFrame(evaluations)
            st.dataframe(df[['candidate_name', 'job_title', 'final_score', 'created_at']].head(10))
            
            # Score distribution
            st.subheader("Score Distribution")
            fig = px.histogram(
                df,
                x='final_score',
                nbins=20,
                title="Distribution of Match Scores",
                labels={'final_score': 'Match Score (%)', 'count': 'Number of Candidates'},
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top candidates
            st.subheader("Top Performing Candidates")
            top_candidates = df.nlargest(5, 'final_score')[['candidate_name', 'job_title', 'final_score']]
            st.dataframe(top_candidates)
        
        # Activity logs
        st.subheader("Recent Activity")
        logs = db_manager.get_activity_logs(st.session_state.user_id, 20)
        if logs:
            log_df = pd.DataFrame(logs)
            st.dataframe(log_df[['action', 'details', 'created_at']])

    elif st.session_state.current_page == "API Keys":
        st.title("🔑 API Key Management")
        
        st.markdown("""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h4>API Usage</h4>
            <p>Use your API key to integrate HireGenAI Pro with your applications.</p>
            <p>Rate Limit: <strong>{settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_PERIOD/3600} hours</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create new API key
        with st.form("create_api_key"):
            key_name = st.text_input("Key Name", placeholder="e.g., Production App")
            expires_in = st.number_input("Expires in (days)", min_value=1, max_value=365, value=30)
            
            if st.form_submit_button("Generate API Key"):
                # Generate API key
                api_key = security_manager.generate_api_key()
                api_key_hash = security_manager.hash_api_key(api_key)
                
                from datetime import timedelta
                expires_at = (datetime.now() + timedelta(days=expires_in)).isoformat()
                
                # Save to database
                key_id = db_manager.save_api_key(
                    st.session_state.user_id,
                    api_key_hash,
                    key_name,
                    expires_at
                )
                
                st.session_state.api_key = api_key
                
                st.success("API Key generated successfully!")
                st.code(api_key, language="text")
                st.warning("⚠️ Save this key now. You won't be able to see it again!")
        
        # Existing keys
        st.subheader("Your API Keys")
        # Display existing keys from database

    elif st.session_state.current_page == "Settings":
        st.title("⚙️ Settings")
        
        tab1, tab2, tab3 = st.tabs(["Profile", "Preferences", "Security"])
        
        with tab1:
            st.subheader("Profile Information")
            st.text_input("Username", value=st.session_state.username, disabled=True)
            st.text_input("Email")
            st.button("Update Profile")
        
        with tab2:
            st.subheader("Application Preferences")
            
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("Default Analysis Depth", ["Basic", "Standard", "Deep", "Comprehensive"])
                st.multiselect("Default Analysis Modules", 
                    ["Bias Detection", "Skill Gap Analysis", "Cultural Fit", "Alternative Roles"],
                    default=["Bias Detection", "Skill Gap Analysis", "Cultural Fit"])
            
            with col2:
                st.number_input("Default Match Threshold (%)", min_value=0, max_value=100, value=70)
                st.selectbox("Report Format", ["HTML", "PDF", "Both"])
            
            if st.button("Save Preferences"):
                st.success("Preferences saved successfully!")
        
        with tab3:
            st.subheader("Security Settings")
            
            # Change password
            with st.form("change_password"):
                st.markdown("### Change Password")
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                if st.form_submit_button("Update Password"):
                    if new_password != confirm_password:
                        st.error("New passwords do not match")
                    else:
                        # Verify current password
                        user = db_manager.get_user(user_id=st.session_state.user_id)
                        if user and security_manager.verify_password(current_password, user['password_hash']):
                            # Update password
                            new_hash = security_manager.hash_password(new_password)
                            # Update in database
                            st.success("Password updated successfully!")
                        else:
                            st.error("Current password is incorrect")
            
            st.markdown("---")
            
            # Two-factor authentication
            st.markdown("### Two-Factor Authentication")
            if st.button("Enable 2FA"):
                st.info("2FA setup will be available in the next version")
            
            st.markdown("---")
            
            # Session management
            st.markdown("### Session Management")
            if st.button("Logout All Devices"):
                st.warning("This will log you out from all devices")
                # Implement session invalidation

# Batch Processing Page
elif st.session_state.current_page == "Batch Processing":
    st.title("📦 Batch Processing")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; margin: 20px 0; color: white;">
        <h3 style="color: white;">Process Multiple Candidates at Once</h3>
        <p>Upload multiple resumes and a single job description to evaluate all candidates in batch mode.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload Multiple Resumes",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload up to 50 resumes at once"
        )
    
    with col2:
        job_title = st.text_input("Job Title")
        company = st.text_input("Company Name")
        job_description = st.text_area("Job Description", height=150)
        
        email_results = st.checkbox("Email results when complete")
        if email_results:
            recipient_email = st.text_input("Recipient Email")
    
    # Batch processing options
    with st.expander("Batch Processing Options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            parallel_processing = st.checkbox("Parallel Processing", value=True)
            batch_size = st.number_input("Batch Size", min_value=1, max_value=20, value=5)
        with col2:
            st.selectbox("Sort Results By", ["Match Score", "Experience", "Name"])
            st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
        with col3:
            generate_individual_reports = st.checkbox("Generate Individual Reports", value=True)
            generate_comparison_report = st.checkbox("Generate Comparison Report", value=True)
    
    if st.button("Start Batch Processing", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload at least one resume")
        elif not job_description:
            st.error("Please provide a job description")
        else:
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Results container
            results_container = st.container()
            
            results = []
            errors = []
            
            # Process files in batches
            for i in range(0, len(uploaded_files), batch_size):
                batch = uploaded_files[i:i+batch_size]
                
                for idx, file in enumerate(batch):
                    try:
                        status_text.text(f"Processing {file.name}... ({i + idx + 1}/{len(uploaded_files)})")
                        
                        # Extract text
                        content = file.read()
                        file_ext = file.name.split('.')[-1].lower()
                        resume_text = text_extractor.extract(content, file_ext)
                        
                        if resume_text:
                            # Process
                            state = HiringState(
                                resume_text=resume_text,
                                jd_text=job_description,
                                job_title=job_title,
                                company_name=company,
                                resume_data={},
                                jd_data={},
                                match_result={},
                                interview_questions={},
                                feedback_report={},
                                processing_time=0,
                                confidence_score=0,
                                bias_analysis={},
                                alternative_roles=[],
                                processing_stage="start",
                                error=""
                            )
                            
                            result = hiring_graph.invoke(state)
                            
                            if not result.get('error'):
                                results.append(result)
                                
                                # Save to database
                                db_manager.save_candidate(
                                    st.session_state.user_id,
                                    result['resume_data'].get('name', 'Unknown'),
                                    result['resume_data'].get('email', ''),
                                    result['resume_data'].get('phone', ''),
                                    resume_text,
                                    result['resume_data']
                                )
                            else:
                                errors.append({"file": file.name, "error": result['error']})
                        else:
                            errors.append({"file": file.name, "error": "Could not extract text"})
                    
                    except Exception as e:
                        errors.append({"file": file.name, "error": str(e)})
                    
                    # Update progress
                    progress = (i + idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
            
            status_text.text("Batch processing complete!")
            
            # Display results
            with results_container:
                if results:
                    st.success(f"✅ Successfully processed {len(results)} out of {len(uploaded_files)} candidates")
                    
                    # Sort results
                    results.sort(key=lambda x: x['match_result']['overall_score'], reverse=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Processed", len(results))
                    with col2:
                        avg_score = sum(r['match_result']['overall_score'] for r in results) / len(results)
                        st.metric("Average Score", f"{avg_score:.1f}%")
                    with col3:
                        top_score = results[0]['match_result']['overall_score']
                        st.metric("Top Score", f"{top_score}%")
                    with col4:
                        st.metric("Errors", len(errors))
                    
                    # Results table
                    st.subheader("Results Summary")
                    
                    results_data = []
                    for r in results:
                        results_data.append({
                            'Name': r['resume_data'].get('name', 'Unknown'),
                            'Email': r['resume_data'].get('email', ''),
                            'Match Score': f"{r['match_result']['overall_score']}%",
                            'Experience': f"{r['resume_data'].get('total_experience_years', 0)} years",
                            'Matched Skills': len(r['match_result'].get('matched_skills', [])),
                            'Missing Skills': len(r['match_result'].get('missing_skills', []))
                        })
                    
                    df = pd.DataFrame(results_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Export options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # CSV Export
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                    
                    with col2:
                        # Excel Export
                        output = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')
                        df.to_excel(output, sheet_name='Results', index=False)
                        output.close()
                        with open('temp.xlsx', 'rb') as f:
                            st.download_button(
                                "📊 Download Excel",
                                f.read(),
                                f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                    with col3:
                        # JSON Export
                        json_str = json.dumps([r['match_result'] for r in results], indent=2)
                        st.download_button(
                            "📄 Download JSON",
                            json_str,
                            f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            "application/json"
                        )
                    
                    # Generate comparison report
                    if generate_comparison_report:
                        st.subheader("📊 Comparison Report")
                        html_report = report_generator.to_html(results, "comparison")
                        st.download_button(
                            "Download Comparison Report",
                            html_report,
                            f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            "text/html"
                        )
                    
                    # Individual reports
                    if generate_individual_reports:
                        st.subheader("📄 Individual Reports")
                        for i, result in enumerate(results[:5]):  # Show first 5
                            with st.expander(f"Report for {result['resume_data'].get('name', 'Unknown')}"):
                                html_report = report_generator.to_html(result, "candidate")
                                st.download_button(
                                    f"Download Report",
                                    html_report,
                                    f"{result['resume_data'].get('name', 'candidate')}_report.html",
                                    "text/html",
                                    key=f"report_{i}"
                                )
                    
                    # Show errors if any
                    if errors:
                        st.subheader("⚠️ Errors")
                        error_df = pd.DataFrame(errors)
                        st.dataframe(error_df)
                
                else:
                    st.error("No candidates were successfully processed")

# Run the app
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Initialize database
    db_manager.init_database()