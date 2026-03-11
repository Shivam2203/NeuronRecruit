# api_service.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, List
import asyncio
import uuid
from datetime import datetime
import os

from config import settings
from security import security_manager
from database import db_manager
from models import (
    UserCreate, UserLogin, APIKeyCreate, EvaluationRequest,
    BatchEvaluationRequest, EvaluationResponse
)
from agents import hiring_graph, HiringState
from utils import text_extractor, report_generator, data_validator

app = FastAPI(title="HireGenAI Pro API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for API key auth
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    api_key_hash = security_manager.hash_api_key(x_api_key)
    user_id = db_manager.validate_api_key(api_key_hash)
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return user_id

@app.get("/")
async def root():
    return {"message": "HireGenAI Pro API", "version": "2.0.0"}

@app.post("/auth/register")
async def register(user: UserCreate):
    """Register a new user"""
    # Check if user exists
    existing = db_manager.get_user(username=user.username) or db_manager.get_user(email=user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    
    # Hash password
    password_hash = security_manager.hash_password(user.password)
    
    # Create user
    user_id = db_manager.create_user(user.username, user.email, password_hash)
    
    if not user_id:
        raise HTTPException(status_code=500, detail="Failed to create user")
    
    # Log activity
    db_manager.log_activity(user_id, "user_registered", f"User registered: {user.username}")
    
    return {"message": "User created successfully", "user_id": user_id}

@app.post("/auth/login")
async def login(user: UserLogin):
    """Login user"""
    db_user = db_manager.get_user(username=user.username)
    
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not security_manager.verify_password(user.password, db_user['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Update last login
    db_manager.update_last_login(db_user['id'])
    
    # Generate token
    token = security_manager.create_jwt_token(str(db_user['id']))
    
    # Log activity
    db_manager.log_activity(db_user['id'], "user_login", f"User logged in: {user.username}")
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": db_user['id'],
        "username": db_user['username']
    }

@app.post("/api-keys/create")
async def create_api_key(key_data: APIKeyCreate, user_id: int = Depends(verify_api_key)):
    """Create a new API key"""
    # Generate API key
    api_key = security_manager.generate_api_key()
    api_key_hash = security_manager.hash_api_key(api_key)
    
    # Set expiration
    expires_at = None
    if key_data.expires_in_days:
        from datetime import timedelta
        expires_at = (datetime.now() + timedelta(days=key_data.expires_in_days)).isoformat()
    
    # Save to database
    key_id = db_manager.save_api_key(user_id, api_key_hash, key_data.name, expires_at)
    
    # Log activity
    db_manager.log_activity(user_id, "api_key_created", f"API key created: {key_data.name}")
    
    return {
        "id": key_id,
        "name": key_data.name,
        "api_key": api_key,  # Only returned once
        "expires_at": expires_at
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_candidate(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    job_title: Optional[str] = Form(None),
    company_name: Optional[str] = Form(None),
    user_id: int = Depends(verify_api_key)
):
    """Evaluate a single candidate"""
    try:
        # Read file content
        content = await resume.read()
        
        # Validate file
        if not security_manager.validate_file_upload(resume.filename, content):
            raise HTTPException(status_code=400, detail="Invalid file")
        
        # Extract text from resume
        file_ext = resume.filename.split('.')[-1].lower()
        resume_text = text_extractor.extract(content, file_ext)
        
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from resume")
        
        # Create initial state
        state = HiringState(
            resume_text=resume_text,
            jd_text=job_description,
            job_title=job_title or "",
            company_name=company_name or "",
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
        
        # Run through graph
        result = hiring_graph.invoke(state)
        
        if result.get('error'):
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Save to database
        candidate_id = db_manager.save_candidate(
            user_id,
            result['resume_data'].get('name', 'Unknown'),
            result['resume_data'].get('email', ''),
            result['resume_data'].get('phone', ''),
            resume_text,
            result['resume_data']
        )
        
        job_id = db_manager.save_job(
            user_id,
            job_title or "Unknown",
            company_name or "Unknown",
            job_description,
            result['jd_data']
        )
        
        evaluation_id = db_manager.save_evaluation(
            user_id,
            candidate_id,
            job_id,
            result['match_result'],
            result['interview_questions'],
            result['feedback_report'],
            result['match_result']['overall_score']
        )
        
        # Generate report URL
        report_url = f"/reports/{evaluation_id}"
        
        # Log activity
        db_manager.log_activity(
            user_id,
            "evaluation_completed",
            f"Evaluated candidate: {result['resume_data'].get('name', 'Unknown')}"
        )
        
        return EvaluationResponse(
            evaluation_id=evaluation_id,
            candidate_name=result['resume_data'].get('name', 'Unknown'),
            job_title=job_title or "Unknown",
            match_score=result['match_result']['overall_score'],
            summary=f"Candidate matches {result['match_result']['overall_score']}% of requirements",
            report_url=report_url,
            created_at=datetime.now()
        )
        
    except Exception as e:
        db_manager.log_activity(user_id, "evaluation_error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/batch")
async def evaluate_batch(
    request: BatchEvaluationRequest,
    user_id: int = Depends(verify_api_key)
):
    """Evaluate multiple candidates in batch"""
    batch_id = str(uuid.uuid4())
    
    # Start async processing
    asyncio.create_task(process_batch(batch_id, request, user_id))
    
    return {
        "batch_id": batch_id,
        "total_candidates": len(request.resume_files),
        "estimated_time_seconds": len(request.resume_files) * 30,
        "status": "processing",
        "results_url": f"/batch-results/{batch_id}"
    }

async def process_batch(batch_id: str, request: BatchEvaluationRequest, user_id: int):
    """Process batch evaluation"""
    results = []
    
    for idx, resume_content in enumerate(request.resume_files):
        try:
            # Process each candidate
            # (Implementation similar to single evaluation)
            pass
        except Exception as e:
            results.append({"error": str(e)})
    
    # Save results
    # Store in database or file
    pass

@app.get("/reports/{evaluation_id}")
async def get_report(evaluation_id: int, user_id: int = Depends(verify_api_key)):
    """Get evaluation report"""
    # Get evaluation from database
    # Generate HTML report
    # Return file
    
    return FileResponse(f"reports/report_{evaluation_id}.html")

@app.get("/analytics")
async def get_analytics(user_id: int = Depends(verify_api_key)):
    """Get user analytics"""
    analytics = db_manager.get_analytics(user_id)
    return analytics

@app.get("/activity-logs")
async def get_activity_logs(limit: int = 100, user_id: int = Depends(verify_api_key)):
    """Get activity logs"""
    logs = db_manager.get_activity_logs(user_id, limit)
    return {"logs": logs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)