# agents.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

from models import (
    ResumeData, JDData, MatchResult, InterviewQuestions,
    FeedbackReport, SkillGapAnalysis, CulturalFitScore
)
from utils import text_extractor, skill_extractor, experience_analyzer
from config import settings

# LLM initialization
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    temperature=0.2,
    max_tokens=2048,
    timeout=30,
    max_retries=2
)

# State definition
class HiringState(TypedDict):
    # Input
    resume_text: str
    jd_text: str
    job_title: str
    company_name: str
    
    # Processed data
    resume_data: Dict[str, Any]
    jd_data: Dict[str, Any]
    match_result: Dict[str, Any]
    interview_questions: Dict[str, Any]
    feedback_report: Dict[str, Any]
    
    # Metadata
    processing_time: float
    confidence_score: float
    bias_analysis: Dict[str, Any]
    alternative_roles: List[str]
    processing_stage: str
    error: str

# Agent Nodes
class ResumeParserNode:
    """Advanced resume parsing with error handling and validation"""
    
    def __call__(self, state: HiringState) -> HiringState:
        try:
            start_time = time.time()
            
            # Extract text if not already extracted
            resume_text = state.get('resume_text', '')
            if not resume_text:
                raise ValueError("No resume text provided")
            
            # Use structured LLM for parsing
            structured_llm = llm.with_structured_output(ResumeData)
            
            prompt = f"""
            Extract comprehensive structured data from this resume.
            Pay attention to:
            1. Personal information and contact details
            2. Work experience with achievements and technologies
            3. Education with dates and GPA if available
            4. Skills categorization (programming languages, frameworks, etc.)
            5. Projects with technical details
            6. Certifications and their validity
            
            Resume Text:
            {resume_text}
            """
            
            result = structured_llm.invoke(prompt)
            resume_data = result.model_dump()
            
            # Calculate confidence score
            confidence = self._calculate_confidence(resume_data)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                **state,
                'resume_data': resume_data,
                'confidence_score': confidence,
                'processing_time': processing_time,
                'processing_stage': 'resume_parsed'
            }
            
        except Exception as e:
            return {
                **state,
                'error': f"Resume parsing failed: {str(e)}",
                'processing_stage': 'error'
            }
    
    def _calculate_confidence(self, data: Dict) -> float:
        """Calculate confidence score for extraction"""
        weights = {
            'name': 0.15,
            'email': 0.10,
            'skills': 0.20,
            'experience': 0.25,
            'education': 0.20,
            'projects': 0.10
        }
        
        confidence = 0
        for field, weight in weights.items():
            if data.get(field) and len(str(data.get(field))) > 0:
                confidence += weight
        
        # Bonus for detailed extraction
        if data.get('technical_skills') and len(data['technical_skills']) > 0:
            confidence += 0.05
        
        if data.get('certifications') and len(data['certifications']) > 0:
            confidence += 0.05
        
        return round(min(confidence, 1.0) * 100, 2)

class JDAnalyzerNode:
    """Advanced job description analysis"""
    
    def __call__(self, state: HiringState) -> HiringState:
        try:
            start_time = time.time()
            
            jd_text = state.get('jd_text', '')
            job_title = state.get('job_title', '')
            company = state.get('company_name', '')
            
            if not jd_text:
                raise ValueError("No job description provided")
            
            structured_llm = llm.with_structured_output(JDData)
            
            prompt = f"""
            Analyze this job description thoroughly.
            Job Title: {job_title}
            Company: {company}
            
            Job Description:
            {jd_text}
            
            Extract:
            1. Company details and role specifics
            2. Required and preferred skills
            3. Experience requirements (min and max)
            4. Education and certification needs
            5. Responsibilities and daily tasks
            6. Benefits and company culture
            """
            
            result = structured_llm.invoke(prompt)
            jd_data = result.model_dump()
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                **state,
                'jd_data': jd_data,
                'processing_time': processing_time,
                'processing_stage': 'jd_analyzed'
            }
            
        except Exception as e:
            return {
                **state,
                'error': f"JD analysis failed: {str(e)}",
                'processing_stage': 'error'
            }

class MatchingAgentNode:
    """Advanced matching with hybrid approach"""
    
    def __call__(self, state: HiringState) -> HiringState:
        try:
            start_time = time.time()
            
            resume = state.get('resume_data', {})
            jd = state.get('jd_data', {})
            
            if not resume or not jd:
                raise ValueError("Missing resume or job data")
            
            # Calculate skill match
            skill_match = self._calculate_skill_match(
                resume.get('skills', []),
                resume.get('technical_skills', {}),
                jd.get('required_skills', []),
                jd.get('preferred_skills', [])
            )
            
            # Calculate experience match
            exp_match = self._calculate_experience_match(
                resume.get('total_experience_years', 0),
                jd.get('min_experience', 0),
                jd.get('max_experience')
            )
            
            # Calculate education match
            edu_match = self._calculate_education_match(
                resume.get('education', []),
                jd.get('education_requirements', [])
            )
            
            # Calculate certification match
            cert_match = self._calculate_certification_match(
                resume.get('certifications', []),
                jd.get('certifications_required', [])
            )
            
            # Calculate project relevance
            project_relevance = self._calculate_project_relevance(
                resume.get('projects', []),
                jd.get('required_skills', [])
            )
            
            # Calculate technical depth
            technical_depth = self._calculate_technical_depth(resume)
            
            # Analyze skill gap
            skill_gap = self._analyze_skill_gap(resume, jd, skill_match['missing'])
            
            # Calculate cultural fit
            cultural_fit = self._calculate_cultural_fit(resume, jd)
            
            # Detect bias
            bias_analysis = self._detect_bias(resume, jd)
            
            # Calculate overall score with weighted factors
            overall_score = (
                skill_match['score'] * 0.35 +
                exp_match * 0.20 +
                edu_match * 0.10 +
                cert_match * 0.05 +
                project_relevance * 0.15 +
                technical_depth * 0.10 +
                cultural_fit['overall_fit_score'] * 0.05
            )
            
            match_result = {
                'skill_match_score': skill_match['score'],
                'experience_match_score': exp_match,
                'education_match_score': edu_match,
                'certification_match_score': cert_match,
                'project_relevance_score': project_relevance,
                'technical_depth_score': technical_depth,
                'cultural_fit_score': cultural_fit['overall_fit_score'],
                'overall_score': round(overall_score, 2),
                'matched_skills': skill_match['matched'],
                'missing_skills': skill_match['missing'],
                'matched_certifications': cert_match['matched'],
                'missing_certifications': cert_match['missing'],
                'skill_gap_analysis': skill_gap,
                'cultural_fit': cultural_fit,
                'bias_analysis': bias_analysis,
                'match_confidence': state.get('confidence_score', 90),
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
            
            # Find alternative roles
            alternative_roles = self._suggest_alternative_roles(resume, jd)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                **state,
                'match_result': match_result,
                'bias_analysis': bias_analysis,
                'alternative_roles': alternative_roles,
                'processing_time': processing_time,
                'processing_stage': 'matched'
            }
            
        except Exception as e:
            return {
                **state,
                'error': f"Matching failed: {str(e)}",
                'processing_stage': 'error'
            }
    
    def _calculate_skill_match(self, resume_skills, resume_tech_skills, required_skills, preferred_skills):
        """Calculate skill match with fuzzy logic"""
        all_resume_skills = set(resume_skills)
        
        # Add categorized skills
        for category, skills in resume_tech_skills.items():
            all_resume_skills.update(skills)
        
        all_resume_skills = {s.lower() for s in all_resume_skills}
        required_lower = {s.lower() for s in required_skills}
        preferred_lower = {s.lower() for s in preferred_skills}
        
        # Find matches
        matched = [s for s in required_skills if s.lower() in all_resume_skills]
        missing = [s for s in required_skills if s.lower() not in all_resume_skills]
        
        # Calculate score (required skills: 1.0, preferred skills: 0.5)
        required_matches = len(matched)
        preferred_matches = sum(1 for s in preferred_lower if s in all_resume_skills)
        
        total_weight = len(required_skills) + (len(preferred_skills) * 0.5)
        weighted_score = (required_matches + preferred_matches * 0.5) / total_weight if total_weight > 0 else 0
        
        return {
            'score': round(weighted_score * 100, 2),
            'matched': matched,
            'missing': missing
        }
    
    def _calculate_experience_match(self, resume_exp, min_req, max_req):
        """Calculate experience match with range consideration"""
        if resume_exp < min_req:
            # Below minimum - scale down
            return max(0, (resume_exp / min_req) * 80)
        elif max_req and resume_exp > max_req:
            # Above maximum - slightly penalize (overqualified)
            return max(80, 100 - ((resume_exp - max_req) * 5))
        else:
            # Within range - perfect match
            return 100
    
    def _calculate_education_match(self, education, requirements):
        """Calculate education match"""
        if not requirements:
            return 100
        
        edu_text = ' '.join([f"{e.get('degree', '')} {e.get('field', '')}" for e in education]).lower()
        
        matches = 0
        for req in requirements:
            if req.lower() in edu_text:
                matches += 1
        
        return (matches / len(requirements)) * 100 if requirements else 100
    
    def _calculate_certification_match(self, certifications, required):
        """Calculate certification match"""
        if not required:
            return {'score': 100, 'matched': [], 'missing': []}
        
        cert_names = [c.get('name', '').lower() for c in certifications]
        cert_text = ' '.join(cert_names)
        
        matched = []
        missing = []
        
        for req in required:
            if req.lower() in cert_text:
                matched.append(req)
            else:
                missing.append(req)
        
        score = (len(matched) / len(required)) * 100 if required else 100
        
        return {
            'score': score,
            'matched': matched,
            'missing': missing
        }
    
    def _calculate_project_relevance(self, projects, required_skills):
        """Calculate project relevance based on technologies"""
        if not projects:
            return 0
        
        project_tech = []
        for project in projects:
            project_tech.extend(project.get('technologies', []))
        
        project_tech_lower = [t.lower() for t in project_tech]
        required_lower = [s.lower() for s in required_skills]
        
        matches = sum(1 for req in required_lower if any(req in pt for pt in project_tech_lower))
        
        return (matches / len(required_skills)) * 100 if required_skills else 0
    
    def _calculate_technical_depth(self, resume):
        """Calculate technical depth score"""
        depth_score = 50  # Base score
        
        # Experience bonus
        exp_years = resume.get('total_experience_years', 0)
        if exp_years > 5:
            depth_score += 20
        elif exp_years > 3:
            depth_score += 15
        elif exp_years > 1:
            depth_score += 10
        
        # Project bonus
        projects = resume.get('projects', [])
        if len(projects) > 5:
            depth_score += 15
        elif len(projects) > 3:
            depth_score += 10
        
        # Certification bonus
        certifications = resume.get('certifications', [])
        if len(certifications) > 3:
            depth_score += 15
        elif len(certifications) > 1:
            depth_score += 10
        
        # Publication bonus
        publications = resume.get('publications', [])
        if len(publications) > 2:
            depth_score += 10
        elif len(publications) > 0:
            depth_score += 5
        
        return min(100, depth_score)
    
    def _analyze_skill_gap(self, resume, jd, missing_skills):
        """Analyze skill gaps and provide recommendations"""
        # Course database (in production, use API or database)
        course_db = {
            'python': {'name': 'Complete Python Bootcamp', 'platform': 'Udemy', 'hours': 40, 'level': 'Beginner'},
            'java': {'name': 'Java Masterclass', 'platform': 'Coursera', 'hours': 60, 'level': 'Intermediate'},
            'javascript': {'name': 'The Complete JavaScript Course', 'platform': 'Udemy', 'hours': 50, 'level': 'Beginner'},
            'react': {'name': 'React - The Complete Guide', 'platform': 'Udemy', 'hours': 45, 'level': 'Intermediate'},
            'node': {'name': 'Node.js Developer Course', 'platform': 'Udemy', 'hours': 35, 'level': 'Intermediate'},
            'aws': {'name': 'AWS Certified Solutions Architect', 'platform': 'AWS Training', 'hours': 80, 'level': 'Advanced'},
            'docker': {'name': 'Docker Mastery', 'platform': 'Udemy', 'hours': 25, 'level': 'Intermediate'},
            'kubernetes': {'name': 'Certified Kubernetes Administrator', 'platform': 'Linux Academy', 'hours': 70, 'level': 'Advanced'},
            'machine learning': {'name': 'Machine Learning Specialization', 'platform': 'Coursera', 'hours': 120, 'level': 'Advanced'},
            'sql': {'name': 'The Complete SQL Bootcamp', 'platform': 'Udemy', 'hours': 25, 'level': 'Beginner'},
            'mongodb': {'name': 'MongoDB - The Complete Guide', 'platform': 'Udemy', 'hours': 30, 'level': 'Intermediate'},
            'devops': {'name': 'DevOps Bootcamp', 'platform': 'edX', 'hours': 90, 'level': 'Advanced'},
        }
        
        # Find existing skills
        all_skills = set(resume.get('skills', []))
        for cat_skills in resume.get('technical_skills', {}).values():
            all_skills.update(cat_skills)
        
        existing_skills = list(all_skills)
        
        # Categorize missing skills
        partial_skills = []  # Skills that exist but need improvement
        
        # Generate recommendations
        recommended_courses = []
        total_hours = 0
        levels = []
        
        for skill in missing_skills:
            skill_lower = skill.lower()
            if skill_lower in course_db:
                course = course_db[skill_lower]
                recommended_courses.append({
                    'skill': skill,
                    'course': course['name'],
                    'platform': course['platform'],
                    'hours': course['hours'],
                    'level': course['level']
                })
                total_hours += course['hours']
                levels.append(course['level'])
        
        # Determine overall difficulty
        if levels:
            if 'Advanced' in levels:
                difficulty = 'Advanced'
            elif 'Intermediate' in levels:
                difficulty = 'Intermediate'
            else:
                difficulty = 'Beginner'
        else:
            difficulty = 'Beginner'
        
        # Determine priority
        if len(missing_skills) <= 2:
            priority = 'Low'
        elif len(missing_skills) <= 4:
            priority = 'Medium'
        else:
            priority = 'High'
        
        return {
            'existing_skills': existing_skills,
            'missing_skills': missing_skills,
            'partial_skills': partial_skills,
            'recommended_courses': recommended_courses,
            'estimated_learning_time_hours': total_hours,
            'difficulty_level': difficulty,
            'priority': priority
        }
    
    def _calculate_cultural_fit(self, resume, jd):
        """Calculate cultural fit score"""
        # Analyze resume for cultural indicators
        experience = resume.get('experience', [])
        projects = resume.get('projects', [])
        summary = resume.get('summary', '')
        
        # Simple heuristic-based scoring
        communication_score = 70
        teamwork_score = 70
        leadership_score = 50
        adaptability_score = 60
        problem_solving_score = 75
        
        # Adjust based on experience
        for exp in experience:
            desc = exp.get('description', '').lower()
            if 'team' in desc or 'collaborat' in desc:
                teamwork_score += 5
            if 'lead' in desc or 'mentor' in desc or 'manage' in desc:
                leadership_score += 10
            if 'adapt' in desc or 'fast' in desc or 'quick' in desc:
                adaptability_score += 5
            if 'solv' in desc or 'resolv' in desc or 'fix' in desc:
                problem_solving_score += 5
        
        # Adjust based on projects
        for project in projects:
            desc = project.get('description', '').lower()
            if 'team' in desc:
                teamwork_score += 3
            if 'lead' in desc:
                leadership_score += 5
        
        # Cap scores
        communication_score = min(100, communication_score)
        teamwork_score = min(100, teamwork_score)
        leadership_score = min(100, leadership_score)
        adaptability_score = min(100, adaptability_score)
        problem_solving_score = min(100, problem_solving_score)
        
        # Calculate overall
        overall = (communication_score + teamwork_score + leadership_score + 
                  adaptability_score + problem_solving_score) / 5
        
        return {
            'communication_score': round(communication_score, 1),
            'teamwork_score': round(teamwork_score, 1),
            'leadership_score': round(leadership_score, 1),
            'adaptability_score': round(adaptability_score, 1),
            'problem_solving_score': round(problem_solving_score, 1),
            'overall_fit_score': round(overall, 1),
            'strengths': self._identify_strengths(resume),
            'areas_for_development': self._identify_weaknesses(resume),
            'culture_notes': ["Good team player", "Shows leadership potential"]
        }
    
    def _identify_strengths(self, resume):
        """Identify candidate strengths"""
        strengths = []
        
        # Check experience
        if resume.get('total_experience_years', 0) > 5:
            strengths.append("Extensive industry experience")
        
        # Check projects
        if len(resume.get('projects', [])) > 3:
            strengths.append("Strong project portfolio")
        
        # Check certifications
        if len(resume.get('certifications', [])) > 2:
            strengths.append("Industry certifications")
        
        # Check publications
        if len(resume.get('publications', [])) > 0:
            strengths.append("Research experience")
        
        # Default strengths
        if not strengths:
            strengths = ["Good communication skills", "Team player"]
        
        return strengths[:3]  # Return top 3
    
    def _identify_weaknesses(self, resume):
        """Identify areas for development"""
        weaknesses = []
        
        # Check experience
        if resume.get('total_experience_years', 0) < 2:
            weaknesses.append("Limited professional experience")
        
        # Check projects
        if len(resume.get('projects', [])) < 2:
            weaknesses.append("Few practical projects")
        
        # Check certifications
        if len(resume.get('certifications', [])) == 0:
            weaknesses.append("No professional certifications")
        
        # Default weaknesses
        if not weaknesses:
            weaknesses = ["Could benefit from leadership experience"]
        
        return weaknesses
    
    def _detect_bias(self, resume, jd):
        """Detect potential bias in evaluation"""
        bias_report = {
            'potential_bias_detected': False,
            'bias_factors': [],
            'recommendations': []
        }
        
        # Check for name-based bias
        name = resume.get('name', '')
        if name:
            # Simple check for non-Latin characters
            if any(ord(char) > 127 for char in name):
                bias_report['bias_factors'].append("Name may indicate non-Western origin")
        
        # Check for education bias
        education = resume.get('education', [])
        if not any('university' in e.get('institution', '').lower() for e in education):
            bias_report['bias_factors'].append("Non-traditional educational background")
        
        # Check for experience gap
        if resume.get('total_experience_years', 0) < jd.get('min_experience', 0):
            bias_report['bias_factors'].append("Experience gap may lead to underestimation")
        
        if bias_report['bias_factors']:
            bias_report['potential_bias_detected'] = True
            bias_report['recommendations'] = [
                "Focus on demonstrated skills rather than pedigree",
                "Consider alternative paths to gaining experience",
                "Evaluate based on potential, not just history"
            ]
        
        return bias_report
    
    def _suggest_alternative_roles(self, resume, jd):
        """Suggest alternative roles based on skills"""
        skills = set(resume.get('skills', []))
        
        role_mappings = {
            'Full Stack Developer': ['python', 'javascript', 'react', 'node', 'html', 'css'],
            'Data Scientist': ['python', 'machine learning', 'sql', 'statistics', 'pandas'],
            'DevOps Engineer': ['aws', 'docker', 'kubernetes', 'jenkins', 'linux'],
            'Frontend Developer': ['javascript', 'react', 'angular', 'vue', 'html', 'css'],
            'Backend Developer': ['python', 'java', 'node', 'sql', 'mongodb'],
            'Mobile Developer': ['android', 'ios', 'swift', 'kotlin', 'flutter'],
            'Data Engineer': ['python', 'spark', 'hadoop', 'sql', 'etl'],
            'Cloud Architect': ['aws', 'azure', 'gcp', 'kubernetes', 'terraform'],
            'Security Engineer': ['security', 'network', 'penetration testing', 'firewall'],
            'Machine Learning Engineer': ['python', 'tensorflow', 'pytorch', 'ml', 'ai']
        }
        
        suggestions = []
        current_role = jd.get('role', '').lower()
        
        for role, required in role_mappings.items():
            if role.lower() != current_role:
                matching = skills.intersection(set(required))
                if len(matching) >= len(required) * 0.5:  # 50% match
                    suggestions.append({
                        'role': role,
                        'match_percentage': round(len(matching) / len(required) * 100, 1),
                        'matching_skills': list(matching)
                    })
        
        # Sort by match percentage
        suggestions.sort(key=lambda x: x['match_percentage'], reverse=True)
        
        return suggestions[:5]  # Return top 5

class InterviewAgentNode:
    """Generate comprehensive interview questions"""
    
    def __call__(self, state: HiringState) -> HiringState:
        try:
            start_time = time.time()
            
            resume = state.get('resume_data', {})
            jd = state.get('jd_data', {})
            match = state.get('match_result', {})
            
            if not resume or not jd:
                raise ValueError("Missing resume or job data")
            
            # Generate questions based on profile
            questions = self._generate_questions(resume, jd, match)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                **state,
                'interview_questions': questions,
                'processing_time': processing_time,
                'processing_stage': 'questions_generated'
            }
            
        except Exception as e:
            return {
                **state,
                'error': f"Interview question generation failed: {str(e)}",
                'processing_stage': 'error'
            }
    
    def _generate_questions(self, resume, jd, match):
        """Generate tailored interview questions"""
        
        # Technical questions based on required skills
        technical = []
        for skill in jd.get('required_skills', [])[:3]:  # Top 3 skills
            question = self._create_technical_question(skill, resume)
            technical.append(question)
        
        # Behavioral questions based on experience
        behavioral = []
        for exp in resume.get('experience', [])[:2]:  # Last 2 experiences
            question = self._create_behavioral_question(exp)
            behavioral.append(question)
        
        # System design questions for senior roles
        system_design = []
        if resume.get('total_experience_years', 0) > 3:
            sd_question = self._create_system_design_question(jd)
            system_design.append(sd_question)
        
        # Problem solving questions
        problem_solving = []
        ps_question = self._create_problem_solving_question(jd)
        problem_solving.append(ps_question)
        
        # Fill up to 5 questions each
        while len(technical) < 5:
            technical.append(self._create_generic_technical_question())
        
        while len(behavioral) < 5:
            behavioral.append(self._create_generic_behavioral_question())
        
        # Calculate difficulty breakdown
        difficulty_counts = {'Easy': 0, 'Medium': 0, 'Hard': 0}
        all_questions = technical + behavioral + system_design + problem_solving
        
        for q in all_questions:
            difficulty_counts[q['difficulty']] = difficulty_counts.get(q['difficulty'], 0) + 1
        
        return {
            'technical_questions': technical,
            'behavioral_questions': behavioral,
            'system_design_questions': system_design,
            'problem_solving_questions': problem_solving,
            'total_questions': len(all_questions),
            'estimated_duration_minutes': len(all_questions) * 10,  # 10 min per question
            'difficulty_breakdown': difficulty_counts
        }
    
    def _create_technical_question(self, skill, resume):
        """Create technical question for a skill"""
        questions_db = {
            'python': {
                'question': 'Explain Python decorators and provide a practical example.',
                'difficulty': 'Medium',
                'answer_points': ['Function that modifies another function', '@ syntax', 'Real-world use cases'],
                'follow_ups': ['How do you implement a decorator with arguments?']
            },
            'javascript': {
                'question': 'Explain closures in JavaScript and give an example.',
                'difficulty': 'Medium',
                'answer_points': ['Function with access to outer scope', 'Lexical scoping', 'Practical applications'],
                'follow_ups': ['What are the memory implications?']
            },
            'react': {
                'question': 'Explain the virtual DOM and React\'s rendering process.',
                'difficulty': 'Medium',
                'answer_points': ['In-memory representation', 'Diffing algorithm', 'Reconciliation'],
                'follow_ups': ['How does it compare to real DOM?']
            },
            'sql': {
                'question': 'Explain the difference between INNER JOIN and LEFT JOIN with examples.',
                'difficulty': 'Easy',
                'answer_points': ['Matching rows only', 'All from left table', 'NULL for non-matches'],
                'follow_ups': ['When would you use a FULL OUTER JOIN?']
            },
            'aws': {
                'question': 'Explain the differences between EC2, Lambda, and ECS.',
                'difficulty': 'Hard',
                'answer_points': ['Virtual servers', 'Serverless functions', 'Container orchestration'],
                'follow_ups': ['When would you choose each?']
            }
        }
        
        # Get question for skill or use generic
        if skill.lower() in questions_db:
            q = questions_db[skill.lower()]
        else:
            q = {
                'question': f'Explain your experience with {skill} and describe a challenging problem you solved using it.',
                'difficulty': 'Medium',
                'answer_points': ['Project examples', 'Technical challenges', 'Solutions implemented'],
                'follow_ups': ['What alternatives did you consider?']
            }
        
        return {
            'question': q['question'],
            'category': 'Technical',
            'difficulty': q['difficulty'],
            'skill_assessed': skill,
            'expected_answer_points': q['answer_points'],
            'follow_up_questions': q['follow_ups'],
            'time_allocation_minutes': 10,
            'evaluation_criteria': ['Technical accuracy', 'Practical experience', 'Communication']
        }
    
    def _create_behavioral_question(self, experience):
        """Create behavioral question based on experience"""
        title = experience.get('title', '')
        company = experience.get('company', '')
        
        questions = [
            {
                'question': f"Tell me about a challenging project you worked on at {company}.",
                'difficulty': 'Medium',
                'answer_points': ['Project description', 'Challenges faced', 'Solutions implemented', 'Results achieved'],
                'follow_ups': ['How did you handle setbacks?', 'What would you do differently?']
            },
            {
                'question': f"Describe a situation where you had to collaborate with a difficult team member at {company}.",
                'difficulty': 'Medium',
                'answer_points': ['Situation context', 'Actions taken', 'Communication approach', 'Outcome'],
                'follow_ups': ['What did you learn?', 'How would you handle it now?']
            },
            {
                'question': f"Tell me about a time you had to learn a new technology quickly at {company}.",
                'difficulty': 'Easy',
                'answer_points': ['Technology context', 'Learning approach', 'Application', 'Results'],
                'follow_ups': ['How do you stay updated with new technologies?']
            }
        ]
        
        import random
        return random.choice(questions)
    
    def _create_system_design_question(self, jd):
        """Create system design question"""
        domain = jd.get('company', 'a large-scale')
        
        return {
            'question': f"Design {jd.get('company', 'a social media')} platform like {jd.get('company', 'Twitter/Facebook')}. Discuss architecture, data modeling, and scaling considerations.",
            'category': 'System Design',
            'difficulty': 'Hard',
            'skill_assessed': 'System Architecture',
            'expected_answer_points': ['Requirements gathering', 'High-level design', 'Data flow', 'Scaling strategies', 'Trade-offs'],
            'follow_up_questions': ['How would you handle peak loads?', 'What database would you choose?'],
            'time_allocation_minutes': 20,
            'evaluation_criteria': ['Architecture knowledge', 'Scalability considerations', 'Problem decomposition']
        }
    
    def _create_problem_solving_question(self, jd):
        """Create problem solving question"""
        return {
            'question': "Given an array of integers, find the longest increasing subsequence. Explain your approach and complexity analysis.",
            'category': 'Problem Solving',
            'difficulty': 'Medium',
            'skill_assessed': 'Algorithms',
            'expected_answer_points': ['Problem understanding', 'Algorithm choice', 'Complexity analysis', 'Edge cases'],
            'follow_up_questions': ['How would you optimize it?', 'Can you handle duplicates?'],
            'time_allocation_minutes': 15,
            'evaluation_criteria': ['Problem-solving approach', 'Code quality', 'Optimization thinking']
        }
    
    def _create_generic_technical_question(self):
        """Create generic technical question"""
        return {
            'question': "Explain a design pattern you've used recently and why you chose it.",
            'category': 'Technical',
            'difficulty': 'Medium',
            'skill_assessed': 'Software Design',
            'expected_answer_points': ['Pattern name', 'Problem context', 'Implementation details', 'Benefits'],
            'follow_up_questions': ['What alternatives did you consider?'],
            'time_allocation_minutes': 10,
            'evaluation_criteria': ['Pattern knowledge', 'Practical application', 'Decision making']
        }
    
    def _create_generic_behavioral_question(self):
        """Create generic behavioral question"""
        return {
            'question': "Tell me about a time you had to handle a difficult situation with a stakeholder.",
            'category': 'Behavioral',
            'difficulty': 'Medium',
            'skill_assessed': 'Communication',
            'expected_answer_points': ['Situation context', 'Actions taken', 'Communication approach', 'Outcome'],
            'follow_up_questions': ['What did you learn?', 'How would you handle it now?'],
            'time_allocation_minutes': 10,
            'evaluation_criteria': ['Communication', 'Problem-solving', 'Professionalism']
        }

class FeedbackAgentNode:
    """Generate comprehensive feedback report"""
    
    def __call__(self, state: HiringState) -> HiringState:
        try:
            start_time = time.time()
            
            resume = state.get('resume_data', {})
            jd = state.get('jd_data', {})
            match = state.get('match_result', {})
            
            if not resume or not jd:
                raise ValueError("Missing resume or job data")
            
            # Generate feedback report
            feedback = self._generate_feedback(resume, jd, match)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                **state,
                'feedback_report': feedback,
                'processing_time': processing_time,
                'processing_stage': 'completed'
            }
            
        except Exception as e:
            return {
                **state,
                'error': f"Feedback generation failed: {str(e)}",
                'processing_stage': 'error'
            }
    
    def _generate_feedback(self, resume, jd, match):
        """Generate comprehensive feedback"""
        
        # Identify strengths
        strengths = []
        if match['skill_match_score'] > 80:
            strengths.append("Excellent technical skills match")
        if match['experience_match_score'] > 80:
            strengths.append("Strong relevant experience")
        if len(resume.get('projects', [])) > 3:
            strengths.append("Diverse project portfolio")
        if len(resume.get('certifications', [])) > 2:
            strengths.append("Industry certifications")
        
        # Identify weaknesses
        weaknesses = []
        if match['skill_match_score'] < 60:
            weaknesses.append("Significant skill gaps")
        if match['experience_match_score'] < 60:
            weaknesses.append("Limited relevant experience")
        if len(resume.get('projects', [])) < 2:
            weaknesses.append("Limited project experience")
        if len(resume.get('certifications', [])) == 0:
            weaknesses.append("No professional certifications")
        
        # Generate recommendations
        recommendations = []
        if match.get('missing_skills'):
            recommendations.append(f"Focus on developing: {', '.join(match['missing_skills'][:3])}")
        if weaknesses:
            recommendations.append("Consider taking online courses to address gaps")
        recommendations.append("Update resume with more quantifiable achievements")
        
        # Suggest roles
        suggested_roles = state.get('alternative_roles', [])
        if suggested_roles:
            suggested_roles = [r['role'] for r in suggested_roles[:3]]
        else:
            suggested_roles = [jd.get('role', 'Similar roles')]
        
        # Create development plan
        dev_plan = {
            'short_term_30_days': [
                "Complete online courses for missing skills",
                "Update LinkedIn profile",
                "Network with industry professionals"
            ],
            'medium_term_60_days': [
                "Work on personal projects to demonstrate skills",
                "Obtain relevant certifications",
                "Attend industry conferences"
            ],
            'long_term_90_days': [
                "Contribute to open source projects",
                "Build a professional portfolio",
                "Seek mentorship opportunities"
            ],
            'recommended_courses': match.get('skill_gap_analysis', {}).get('recommended_courses', []),
            'mentorship_areas': ["Technical skills", "Career guidance", "Industry insights"]
        }
        
        # Market insights
        market_demand = "High" if match['overall_score'] > 70 else "Medium"
        salary_expectations = {
            'entry': 70000,
            'mid': 95000,
            'senior': 130000
        }
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations,
            'suggested_roles': suggested_roles,
            'development_plan': dev_plan,
            'interview_tips': [
                "Prepare specific examples from your experience",
                "Research the company thoroughly",
                "Practice explaining your projects clearly"
            ],
            'resume_improvements': [
                "Quantify achievements with metrics",
                "Highlight relevant projects",
                "Use action verbs consistently"
            ],
            'market_demand': market_demand,
            'salary_expectations': salary_expectations,
            'companies_to_target': ["Tech startups", "Established enterprises", "Consulting firms"]
        }

# Build the graph
def build_hiring_graph():
    """Build the LangGraph for hiring pipeline"""
    
    graph = StateGraph(HiringState)
    
    # Add nodes
    graph.add_node("resume_parser", ResumeParserNode())
    graph.add_node("jd_analyzer", JDAnalyzerNode())
    graph.add_node("matching_agent", MatchingAgentNode())
    graph.add_node("interview_agent", InterviewAgentNode())
    graph.add_node("feedback_agent", FeedbackAgentNode())
    
    # Add edges
    graph.set_entry_point("resume_parser")
    graph.add_edge("resume_parser", "jd_analyzer")
    graph.add_edge("jd_analyzer", "matching_agent")
    graph.add_edge("matching_agent", "interview_agent")
    graph.add_edge("interview_agent", "feedback_agent")
    
    # Conditional edge for error handling
    def check_error(state):
        if state.get('error'):
            return 'error'
        return 'continue'
    
    # Compile graph
    return graph.compile()

# Initialize graph
hiring_graph = build_hiring_graph()