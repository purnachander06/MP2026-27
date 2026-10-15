@echo off
cd /d "%~dp0"
title IoT Threat Detection - Federated Learning (GSFS)
echo ============================================================
echo Running GSFS Federated Learning Training (Basepaper Proposed Algorithm)
echo ============================================================
if exist "..\..\.venv\Scripts\python.exe" (
    ..\..\.venv\Scripts\python.exe train.py
) else (
    python train.py
)
pause