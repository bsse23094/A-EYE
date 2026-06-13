@echo off
title JARVIS
where py >nul 2>nul
if %errorlevel%==0 (
    py jarvis.py --server %*
) else (
    python jarvis.py --server %*
)
pause
