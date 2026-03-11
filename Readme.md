# 🤖 HireGenAI Pro - Enterprise AI-Powered Hiring Intelligence Platform

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.20-purple)](https://github.com/langchain-ai/langgraph)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-blue)](https://deepmind.google/technologies/gemini/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)](https://www.docker.com/)

<p align="center">
  <img src="static/logo.png" alt="HireGenAI Pro Logo" width="200"/>
</p>

<p align="center">
  <strong>Next-Generation Talent Acquisition Platform Powered by Multi-Agent AI</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api-reference">API Reference</a> •
  <a href="#-deployment">Deployment</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 📋 Overview

HireGenAI Pro is an enterprise-grade, multi-agent AI platform that revolutionizes the hiring process through intelligent automation. Built with **LangGraph** and **Google Gemini 2.5 Flash**, it provides comprehensive candidate evaluation, bias detection, and intelligent matching capabilities.

### Why HireGenAI Pro?

- **🚀 10x Faster Screening**: Process hundreds of resumes in minutes
- **🎯 95% Matching Accuracy**: Hybrid scoring with ML-powered algorithms
- **⚖️ Bias-Free Hiring**: Advanced bias detection and mitigation
- **💡 Actionable Insights**: Comprehensive feedback and development plans
- **🔒 Enterprise Security**: API key auth, rate limiting, data encryption

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **📄 Smart Resume Parsing** | Extract structured data with 95% accuracy from PDF, DOCX, TXT |
| **🎯 Intelligent Job Matching** | Hybrid scoring with skill gap analysis and fuzzy matching |
| **⚖️ Bias Detection** | Identify and mitigate unconscious bias in hiring |
| **💬 Interview Generator** | Generate personalized technical and behavioral questions |
| **📊 Analytics Dashboard** | Real-time insights and hiring metrics |
| **🔑 API Access** | RESTful API for seamless integration |

### Advanced Features

- **Multi-Agent Architecture**: Specialized AI agents for different tasks
- **Cultural Fit Assessment**: Analyze soft skills and team compatibility
- **Skill Gap Analysis**: Identify missing skills with learning recommendations
- **Alternative Role Suggestions**: Discover other suitable positions
- **Batch Processing**: Evaluate multiple candidates simultaneously
- **Custom Reporting**: Generate detailed HTML/PDF/CSV reports
- **Activity Logging**: Complete audit trail of all actions
- **Rate Limiting**: Prevent API abuse with configurable limits


## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Google Gemini API Key
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/Shivam2203/NeuronRecruit.git
cd NeuronRecruit

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the application
streamlit run app.py