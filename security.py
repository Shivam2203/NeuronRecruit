# security.py
import hashlib
import jwt
import bcrypt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from cryptography.fernet import Fernet
from config import settings

class SecurityManager:
    """Handle all security-related operations"""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.cipher = Fernet(Fernet.generate_key())
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(32)
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def create_jwt_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        return jwt.encode(to_encode, self.secret_key, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.PyJWTError:
            return None
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def validate_file_upload(self, filename: str, content: bytes) -> bool:
        """Validate uploaded file"""
        # Check file size
        if len(content) > settings.MAX_UPLOAD_SIZE:
            return False
        
        # Check file extension
        ext = '.' + filename.split('.')[-1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            return False
        
        # Check for malicious content
        if self._contains_malicious_code(content):
            return False
        
        return True
    
    def _contains_malicious_code(self, content: bytes) -> bool:
        """Check for malicious code in file content"""
        dangerous_patterns = [
            b'<?php', b'<script', b'javascript:', b'eval(',
            b'exec(', b'system(', b'passthru(', b'shell_exec(',
            b'<?=', b'<%', b'<%=', b'${', b'{{'
        ]
        
        content_lower = content.lower()
        for pattern in dangerous_patterns:
            if pattern in content_lower:
                return True
        return False

security_manager = SecurityManager()