@echo off
cd /d "%~dp0"
title IoT Threat Detection - Machine Learning Baselines
echo ============================================================
echo Running Machine Learning Training (Random Forest + Logistic Regression)
echo ============================================================
if exist "..\..\.venv\Scripts\python.exe" (
    ..\..\.venv\Scripts\python.exe train.py
) else (
    python train.py
)
pause