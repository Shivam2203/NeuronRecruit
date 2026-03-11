# utils.py
import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import PyPDF2
import docx2txt
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class TextExtractor:
    """Extract text from various file formats"""
    
    @staticmethod
    def from_pdf(file_content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf = PyPDF2.PdfReader(file_content)
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    @staticmethod
    def from_docx(file_content: bytes) -> str:
        """Extract text from DOCX"""
        try:
            # Save temporarily
            with open('temp.docx', 'wb') as f:
                f.write(file_content)
            text = docx2txt.process('temp.docx')
            return text
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {str(e)}")
    
    @staticmethod
    def from_txt(file_content: bytes) -> str:
        """Extract text from TXT"""
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            # Try different encoding
            return file_content.decode('latin-1')
    
    @classmethod
    def extract(cls, file_content: bytes, file_type: str) -> str:
        """Extract text based on file type"""
        extractors = {
            'pdf': cls.from_pdf,
            'docx': cls.from_docx,
            'txt': cls.from_txt
        }
        
        if file_type.lower() not in extractors:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return extractors[file_type.lower()](file_content)

class TextProcessor:
    """Process and clean text"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\-\@]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    @staticmethod
    def extract_phones(text: str) -> List[str]:
        """Extract phone numbers"""
        phone_pattern = r'\b[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}\b'
        return re.findall(phone_pattern, text)
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs"""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*'
        return re.findall(url_pattern, text)

class SkillExtractor:
    """Extract skills from text"""
    
    def __init__(self):
        self.skills_db = self._load_skills_db()
        self.nlp = spacy.load('en_core_web_sm')
    
    def _load_skills_db(self) -> Dict[str, List[str]]:
        """Load skills database"""
        return {
            'programming_languages': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'typescript', 'scala', 'perl'],
            'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'rails', 'laravel', 'tensorflow', 'pytorch', 'keras'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'elasticsearch', 'cassandra', 'dynamodb'],
            'tools': ['git', 'docker', 'kubernetes', 'jenkins', 'aws', 'azure', 'gcp', 'jira', 'confluence', 'vscode'],
            'cloud_platforms': ['aws', 'azure', 'gcp', 'heroku', 'digitalocean', 'linode'],
            'soft_skills': ['communication', 'teamwork', 'leadership', 'problem solving', 'time management', 'adaptability', 'creativity']
        }
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from text"""
        text_lower = text.lower()
        found_skills = {category: [] for category in self.skills_db}
        
        for category, skills in self.skills_db.items():
            for skill in skills:
                if skill in text_lower:
                    found_skills[category].append(skill)
        
        return found_skills

class ExperienceAnalyzer:
    """Analyze work experience"""
    
    @staticmethod
    def extract_years(text: str) -> float:
        """Extract years of experience from text"""
        patterns = [
            r'(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?',
            r'experience\s+of\s+(\d+)\+?\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return float(match.group(1))
        
        return 0
    
    @staticmethod
    def calculate_duration(start_date: str, end_date: str = None) -> float:
        """Calculate duration in years between dates"""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.now() if not end_date else datetime.strptime(end_date, '%Y-%m-%d')
            delta = end - start
            return round(delta.days / 365.25, 1)
        except:
            return 0

class DataValidator:
    """Validate and sanitize data"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format"""
        # Remove common separators
        cleaned = re.sub(r'[\s\-\(\)]', '', phone)
        return cleaned.isdigit() and len(cleaned) >= 10
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_')
        
        # Remove special characters
        filename = re.sub(r'[^\w\-_\. ]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1)
            filename = name[:250] + '.' + ext
        
        return filename

class ReportGenerator:
    """Generate reports in various formats"""
    
    @staticmethod
    def to_html(data: Dict[str, Any], template: str = "basic") -> str:
        """Generate HTML report"""
        if template == "candidate":
            return ReportGenerator._candidate_html(data)
        elif template == "comparison":
            return ReportGenerator._comparison_html(data)
        else:
            return ReportGenerator._basic_html(data)
    
    @staticmethod
    def _basic_html(data: Dict) -> str:
        """Basic HTML template"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HireGenAI Pro Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .score {{ font-size: 24px; color: #4CAF50; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>HireGenAI Pro Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <pre>{json.dumps(data, indent=2)}</pre>
        </body>
        </html>
        """
        return html
    
    @staticmethod
    def _candidate_html(data: Dict) -> str:
        """Candidate report template"""
        candidate = data.get('resume_data', {})
        match = data.get('match_result', {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Candidate Report - {candidate.get('name', 'Unknown')}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .score-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .score {{ font-size: 48px; font-weight: bold; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .skills {{ margin: 20px 0; }}
                .skill-badge {{ display: inline-block; padding: 5px 10px; margin: 2px; background: #4CAF50; color: white; border-radius: 20px; }}
                .missing-skill {{ background: #f44336; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .footer {{ margin-top: 50px; text-align: center; color: #777; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Candidate Report: {candidate.get('name', 'Unknown')}</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="score-card">
                    <div class="metric">
                        <div>Overall Match</div>
                        <div class="score">{match.get('overall_score', 0)}%</div>
                    </div>
                    <div class="metric">
                        <div>Skills Match</div>
                        <div>{match.get('skill_match_score', 0)}%</div>
                    </div>
                    <div class="metric">
                        <div>Experience Match</div>
                        <div>{match.get('experience_match_score', 0)}%</div>
                    </div>
                </div>
                
                <h2>Contact Information</h2>
                <table>
                    <tr><th>Email</th><td>{candidate.get('email', 'N/A')}</td></tr>
                    <tr><th>Phone</th><td>{candidate.get('phone', 'N/A')}</td></tr>
                    <tr><th>Location</th><td>{candidate.get('location', 'N/A')}</td></tr>
                    <tr><th>Experience</th><td>{candidate.get('total_experience_years', 0)} years</td></tr>
                </table>
                
                <h2>Skills Analysis</h2>
                <div class="skills">
                    <h3>Matched Skills</h3>
                    {''.join([f'<span class="skill-badge">{skill}</span>' for skill in match.get('matched_skills', [])])}
                    
                    <h3>Missing Skills</h3>
                    {''.join([f'<span class="skill-badge missing-skill">{skill}</span>' for skill in match.get('missing_skills', [])])}
                </div>
                
                <h2>Experience</h2>
                <table>
                    <tr><th>Title</th><th>Company</th><th>Duration</th></tr>
                    {''.join([f'<tr><td>{exp.get("title", "")}</td><td>{exp.get("company", "")}</td><td>{exp.get("duration_years", 0)} years</td></tr>' for exp in candidate.get('experience', [])])}
                </table>
                
                <h2>Education</h2>
                <table>
                    <tr><th>Degree</th><th>Field</th><th>Institution</th></tr>
                    {''.join([f'<tr><td>{edu.get("degree", "")}</td><td>{edu.get("field", "")}</td><td>{edu.get("institution", "")}</td></tr>' for edu in candidate.get('education', [])])}
                </table>
                
                <h2>Projects</h2>
                <table>
                    <tr><th>Name</th><th>Technologies</th></tr>
                    {''.join([f'<tr><td>{proj.get("name", "")}</td><td>{", ".join(proj.get("technologies", []))}</td></tr>' for proj in candidate.get('projects', [])])}
                </table>
                
                <h2>Certifications</h2>
                <ul>
                    {''.join([f'<li>{cert.get("name", "")} - {cert.get("issuer", "")}</li>' for cert in candidate.get('certifications', [])])}
                </ul>
                
                <div class="footer">
                    <p>Generated by HireGenAI Pro - AI-Powered Hiring Intelligence</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    @staticmethod
    def _comparison_html(data: List[Dict]) -> str:
        """Comparison report template"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Candidate Comparison Report</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th { background-color: #4CAF50; color: white; padding: 12px; }
                td { padding: 10px; border-bottom: 1px solid #ddd; }
                tr:hover { background-color: #f5f5f5; }
                .score-high { color: #4CAF50; font-weight: bold; }
                .score-medium { color: #FF9800; font-weight: bold; }
                .score-low { color: #f44336; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Candidate Comparison Report</h1>
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Name</th>
                    <th>Overall Score</th>
                    <th>Skills Match</th>
                    <th>Experience</th>
                    <th>Education</th>
                    <th>Top Skills</th>
                </tr>
        """
        
        for idx, candidate in enumerate(data, 1):
            name = candidate.get('resume_data', {}).get('name', 'Unknown')
            overall = candidate.get('match_result', {}).get('overall_score', 0)
            skills_match = candidate.get('match_result', {}).get('skill_match_score', 0)
            experience = candidate.get('resume_data', {}).get('total_experience_years', 0)
            education = candidate.get('resume_data', {}).get('highest_education', 'N/A')
            top_skills = ', '.join(candidate.get('match_result', {}).get('matched_skills', [])[:3])
            
            score_class = 'score-high' if overall >= 80 else 'score-medium' if overall >= 60 else 'score-low'
            
            html += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{name}</td>
                    <td class="{score_class}">{overall}%</td>
                    <td>{skills_match}%</td>
                    <td>{experience} years</td>
                    <td>{education}</td>
                    <td>{top_skills}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        return html

# Initialize utilities
text_extractor = TextExtractor()
text_processor = TextProcessor()
skill_extractor = SkillExtractor()
experience_analyzer = ExperienceAnalyzer()
data_validator = DataValidator()
report_generator = ReportGenerator()