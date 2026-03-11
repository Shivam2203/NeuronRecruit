# database.py
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import contextmanager
import pandas as pd
from config import settings

class DatabaseManager:
    """Handle all database operations"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DATABASE_URL.replace('sqlite:///', '')
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    api_key TEXT UNIQUE,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # API keys table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    api_key_hash TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    last_used TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Candidates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    resume_text TEXT,
                    resume_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Job descriptions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_descriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    company TEXT,
                    description TEXT,
                    jd_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Evaluations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    candidate_id INTEGER NOT NULL,
                    job_id INTEGER NOT NULL,
                    match_result TEXT,
                    interview_questions TEXT,
                    feedback_report TEXT,
                    final_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (candidate_id) REFERENCES candidates (id),
                    FOREIGN KEY (job_id) REFERENCES job_descriptions (id)
                )
            ''')
            
            # Activity logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activity_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
    
    # User operations
    def create_user(self, username: str, email: str, password_hash: str) -> Optional[int]:
        """Create a new user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash)
                    VALUES (?, ?, ?)
                ''', (username, email, password_hash))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None
    
    def get_user(self, user_id: int = None, username: str = None, email: str = None):
        """Get user by ID, username, or email"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            elif username:
                cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            elif email:
                cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            else:
                return None
            return cursor.fetchone()
    
    def update_last_login(self, user_id: int):
        """Update user's last login time"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (user_id,))
            conn.commit()
    
    # API Key operations
    def save_api_key(self, user_id: int, api_key_hash: str, name: str, expires_at: str = None):
        """Save API key for user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO api_keys (user_id, api_key_hash, name, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (user_id, api_key_hash, name, expires_at))
            conn.commit()
            return cursor.lastrowid
    
    def validate_api_key(self, api_key_hash: str) -> Optional[int]:
        """Validate API key and return user_id"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id FROM api_keys
                WHERE api_key_hash = ? AND is_active = 1
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            ''', (api_key_hash,))
            result = cursor.fetchone()
            if result:
                # Update last used
                cursor.execute('''
                    UPDATE api_keys SET last_used = CURRENT_TIMESTAMP
                    WHERE api_key_hash = ?
                ''', (api_key_hash,))
                conn.commit()
                return result['user_id']
            return None
    
    # Candidate operations
    def save_candidate(self, user_id: int, name: str, email: str, phone: str, 
                      resume_text: str, resume_data: dict) -> int:
        """Save candidate information"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO candidates (user_id, name, email, phone, resume_text, resume_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, name, email, phone, resume_text, json.dumps(resume_data)))
            conn.commit()
            return cursor.lastrowid
    
    def get_candidates(self, user_id: int) -> List[Dict]:
        """Get all candidates for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM candidates WHERE user_id = ? ORDER BY created_at DESC
            ''', (user_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # Job operations
    def save_job(self, user_id: int, title: str, company: str, description: str, jd_data: dict) -> int:
        """Save job description"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO job_descriptions (user_id, title, company, description, jd_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, title, company, description, json.dumps(jd_data)))
            conn.commit()
            return cursor.lastrowid
    
    def get_jobs(self, user_id: int) -> List[Dict]:
        """Get all jobs for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM job_descriptions WHERE user_id = ? ORDER BY created_at DESC
            ''', (user_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # Evaluation operations
    def save_evaluation(self, user_id: int, candidate_id: int, job_id: int,
                       match_result: dict, interview_questions: dict, 
                       feedback_report: dict, final_score: float) -> int:
        """Save evaluation results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO evaluations (user_id, candidate_id, job_id, match_result,
                                       interview_questions, feedback_report, final_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, candidate_id, job_id, json.dumps(match_result),
                  json.dumps(interview_questions), json.dumps(feedback_report), final_score))
            conn.commit()
            return cursor.lastrowid
    
    def get_evaluations(self, user_id: int) -> List[Dict]:
        """Get all evaluations for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT e.*, c.name as candidate_name, j.title as job_title
                FROM evaluations e
                JOIN candidates c ON e.candidate_id = c.id
                JOIN job_descriptions j ON e.job_id = j.id
                WHERE e.user_id = ?
                ORDER BY e.created_at DESC
            ''', (user_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # Activity logging
    def log_activity(self, user_id: Optional[int], action: str, details: str = None,
                    ip_address: str = None, user_agent: str = None):
        """Log user activity"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO activity_logs (user_id, action, details, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, action, details, ip_address, user_agent))
            conn.commit()
    
    def get_activity_logs(self, user_id: Optional[int] = None, limit: int = 100) -> List[Dict]:
        """Get activity logs"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute('''
                    SELECT * FROM activity_logs
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (user_id, limit))
            else:
                cursor.execute('''
                    SELECT * FROM activity_logs
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    # Analytics
    def get_analytics(self, user_id: int) -> Dict:
        """Get analytics for user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total candidates
            cursor.execute('SELECT COUNT(*) FROM candidates WHERE user_id = ?', (user_id,))
            total_candidates = cursor.fetchone()[0]
            
            # Total jobs
            cursor.execute('SELECT COUNT(*) FROM job_descriptions WHERE user_id = ?', (user_id,))
            total_jobs = cursor.fetchone()[0]
            
            # Total evaluations
            cursor.execute('SELECT COUNT(*) FROM evaluations WHERE user_id = ?', (user_id,))
            total_evaluations = cursor.fetchone()[0]
            
            # Average score
            cursor.execute('''
                SELECT AVG(final_score) FROM evaluations WHERE user_id = ?
            ''', (user_id,))
            avg_score = cursor.fetchone()[0] or 0
            
            # Top candidates
            cursor.execute('''
                SELECT c.name, e.final_score
                FROM evaluations e
                JOIN candidates c ON e.candidate_id = c.id
                WHERE e.user_id = ?
                ORDER BY e.final_score DESC
                LIMIT 5
            ''', (user_id,))
            top_candidates = [dict(row) for row in cursor.fetchall()]
            
            return {
                'total_candidates': total_candidates,
                'total_jobs': total_jobs,
                'total_evaluations': total_evaluations,
                'average_score': round(avg_score, 2),
                'top_candidates': top_candidates
            }

db_manager = DatabaseManager()