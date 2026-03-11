# models.py
from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import re

# User Models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def validate_password(cls, v):
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool

# API Key Models
class APIKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)

class APIKeyResponse(BaseModel):
    id: int
    name: str
    api_key: str  # Only returned once on creation
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    is_active: bool

# Resume Models
class Experience(BaseModel):
    title: str
    company: str
    location: Optional[str]
    start_date: date
    end_date: Optional[date]
    current: bool = False
    duration_years: Optional[float]
    description: str
    achievements: List[str] = []
    technologies: List[str] = []
    
    @validator('duration_years', always=True)
    def calculate_duration(cls, v, values):
        if v is not None:
            return v
        if 'start_date' in values and 'end_date' in values:
            if values['end_date']:
                delta = values['end_date'] - values['start_date']
                return delta.days / 365.25
            elif values.get('current'):
                delta = datetime.now().date() - values['start_date']
                return delta.days / 365.25
        return 0

class Education(BaseModel):
    degree: str
    field: str
    institution: str
    location: Optional[str]
    start_date: date
    end_date: Optional[date]
    graduation_year: Optional[int]
    gpa: Optional[float]
    honors: List[str] = []
    courses: List[str] = []

class Certification(BaseModel):
    name: str
    issuer: str
    issue_date: date
    expiration_date: Optional[date]
    credential_id: Optional[str]
    credential_url: Optional[str]
    is_expired: bool = False
    
    @validator('is_expired', always=True)
    def check_expiry(cls, v, values):
        if 'expiration_date' in values and values['expiration_date']:
            return values['expiration_date'] < datetime.now().date()
        return False

class Project(BaseModel):
    name: str
    description: str
    technologies: List[str]
    role: str
    duration_months: Optional[int]
    team_size: Optional[int]
    impact: Optional[str]
    link: Optional[str]
    github: Optional[str]

class Publication(BaseModel):
    title: str
    publisher: str
    date: date
    authors: List[str]
    url: Optional[str]
    citation_count: Optional[int]

class ResumeData(BaseModel):
    # Personal Information
    name: str
    email: EmailStr
    phone: Optional[str]
    location: Optional[str]
    linkedin: Optional[str]
    github: Optional[str]
    portfolio: Optional[str]
    
    # Professional Summary
    summary: str
    
    # Skills
    skills: List[str]
    technical_skills: Dict[str, List[str]] = {
        "programming_languages": [],
        "frameworks": [],
        "databases": [],
        "tools": [],
        "cloud_platforms": []
    }
    soft_skills: List[str] = []
    languages: List[Dict[str, str]] = []  # [{"language": "English", "proficiency": "Native"}]
    
    # Experience
    experience: List[Experience] = []
    total_experience_years: float = 0
    
    # Education
    education: List[Education] = []
    highest_education: Optional[str]
    
    # Additional
    certifications: List[Certification] = []
    projects: List[Project] = []
    publications: List[Publication] = []
    awards: List[str] = []
    volunteer_experience: List[Experience] = []
    
    # Metadata
    parsed_date: datetime = Field(default_factory=datetime.now)
    confidence_score: float = 0.0
    
    @validator('total_experience_years', always=True)
    def calculate_total_experience(cls, v, values):
        if 'experience' in values and values['experience']:
            total = 0
            for exp in values['experience']:
                if exp.duration_years:
                    total += exp.duration_years
            return round(total, 1)
        return 0
    
    @validator('highest_education', always=True)
    def determine_highest_education(cls, v, values):
        if 'education' not in values or not values['education']:
            return None
        
        degree_rank = {
            'phd': 5, 'doctorate': 5,
            'master': 4, 'mba': 4,
            'bachelor': 3, 'b.tech': 3, 'b.e.': 3,
            'associate': 2,
            'diploma': 1
        }
        
        highest = None
        highest_rank = 0
        
        for edu in values['education']:
            degree_lower = edu.degree.lower()
            for key, rank in degree_rank.items():
                if key in degree_lower and rank > highest_rank:
                    highest_rank = rank
                    highest = edu.degree
        
        return highest

# Job Description Models
class JDData(BaseModel):
    # Company Information
    company: str = "Unknown"
    department: Optional[str]
    location: str
    remote_policy: str  # Remote, Hybrid, On-site
    
    # Role Details
    role: str
    employment_type: str  # Full-time, Part-time, Contract, Internship
    seniority_level: str  # Entry, Junior, Mid, Senior, Lead, Manager
    reports_to: Optional[str]
    
    # Requirements
    required_skills: List[str]
    preferred_skills: List[str] = []
    min_experience: float
    max_experience: Optional[float]
    education_requirements: List[str] = []
    certifications_required: List[str] = []
    
    # Responsibilities
    responsibilities: List[str] = []
    day_to_day_tasks: List[str] = []
    
    # Benefits
    salary_range: Optional[str]
    benefits: List[str] = []
    perks: List[str] = []
    
    # Company Culture
    company_culture: List[str] = []
    team_size: Optional[int]
    
    # Metadata
    posted_date: Optional[date]
    application_deadline: Optional[date]
    source_url: Optional[str]

# Match Result Models
class SkillGapAnalysis(BaseModel):
    existing_skills: List[str]
    missing_skills: List[str]
    partial_skills: List[str]
    recommended_courses: List[Dict[str, str]]
    estimated_learning_time_hours: int
    difficulty_level: str
    priority: str  # High, Medium, Low

class CulturalFitScore(BaseModel):
    communication_score: float = 0
    teamwork_score: float = 0
    leadership_score: float = 0
    adaptability_score: float = 0
    problem_solving_score: float = 0
    overall_fit_score: float = 0
    strengths: List[str] = []
    areas_for_development: List[str] = []
    culture_notes: List[str] = []

class MatchResult(BaseModel):
    # Individual Scores
    skill_match_score: float
    experience_match_score: float
    education_match_score: float
    certification_match_score: float
    project_relevance_score: float
    technical_depth_score: float
    cultural_fit_score: float
    
    # Overall
    overall_score: float
    
    # Detailed Analysis
    matched_skills: List[str]
    missing_skills: List[str]
    matched_certifications: List[str]
    missing_certifications: List[str]
    
    # Advanced Analysis
    skill_gap_analysis: SkillGapAnalysis
    cultural_fit: CulturalFitScore
    bias_analysis: Dict[str, Any]
    
    # Metadata
    match_confidence: float
    processing_time_ms: int

# Interview Models
class InterviewQuestion(BaseModel):
    question: str
    category: str  # Technical, Behavioral, System Design, Problem Solving
    difficulty: str  # Easy, Medium, Hard
    skill_assessed: str
    expected_answer_points: List[str]
    follow_up_questions: List[str]
    time_allocation_minutes: int
    evaluation_criteria: List[str]

class InterviewQuestions(BaseModel):
    technical_questions: List[InterviewQuestion] = []
    behavioral_questions: List[InterviewQuestion] = []
    system_design_questions: List[InterviewQuestion] = []
    problem_solving_questions: List[InterviewQuestion] = []
    
    total_questions: int = 0
    estimated_duration_minutes: int = 0
    difficulty_breakdown: Dict[str, int] = {"Easy": 0, "Medium": 0, "Hard": 0}
    
    @validator('total_questions', always=True)
    def calculate_total(cls, v, values):
        return len(values.get('technical_questions', [])) + \
               len(values.get('behavioral_questions', [])) + \
               len(values.get('system_design_questions', [])) + \
               len(values.get('problem_solving_questions', []))
    
    @validator('estimated_duration_minutes', always=True)
    def calculate_duration(cls, v, values):
        duration = 0
        for category in ['technical_questions', 'behavioral_questions', 
                        'system_design_questions', 'problem_solving_questions']:
            for q in values.get(category, []):
                duration += q.time_allocation_minutes
        return duration

# Feedback Models
class DevelopmentPlan(BaseModel):
    short_term_30_days: List[str]
    medium_term_60_days: List[str]
    long_term_90_days: List[str]
    recommended_courses: List[Dict[str, str]]
    mentorship_areas: List[str]

class FeedbackReport(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    suggested_roles: List[str]
    development_plan: DevelopmentPlan
    interview_tips: List[str]
    resume_improvements: List[str]
    
    # Market Insights
    market_demand: str
    salary_expectations: Dict[str, float]
    companies_to_target: List[str]

# API Request/Response Models
class EvaluationRequest(BaseModel):
    resume_file: bytes
    job_description: str
    job_title: Optional[str]
    company_name: Optional[str]
    options: Dict[str, bool] = {
        "bias_detection": True,
        "skill_gap_analysis": True,
        "cultural_fit": True,
        "alternative_roles": True
    }

class EvaluationResponse(BaseModel):
    evaluation_id: int
    candidate_name: str
    job_title: str
    match_score: float
    summary: str
    report_url: str
    created_at: datetime

class BatchEvaluationRequest(BaseModel):
    resume_files: List[bytes]
    job_description: str
    job_title: Optional[str]
    company_name: Optional[str]
    email_results: bool = False
    recipient_email: Optional[EmailStr]

class BatchEvaluationResponse(BaseModel):
    batch_id: str
    total_candidates: int
    estimated_time_seconds: int
    status: str
    results_url: Optional[str]