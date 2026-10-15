@echo off
cd /d "%~dp0"
title IoT Threat Detection - Backend API Server
echo ============================================================
echo Starting Backend API Server (FastAPI + Groq + Detectors)
echo ============================================================
if exist "..\..\.venv\Scripts\python.exe" (
    ..\..\.venv\Scripts\python.exe main.py
) else (
    python main.py
)
pause