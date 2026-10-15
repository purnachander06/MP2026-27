@echo off
cd /d "%~dp0"
title IoT Threat Detection - Backend API Server (Supabase Cloud)
echo ============================================================
echo Starting Backend API Server with Supabase Cloud Database
echo ============================================================
if exist "..\..\.venv\Scripts\python.exe" (
    ..\..\.venv\Scripts\python.exe main.py --supabase
) else (
    python main.py --supabase
)
pause