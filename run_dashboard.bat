@echo off
title VIT ML Mini-Project Dashboard (MACSE504)
cd /d "%~dp0"

echo ======================================================================
echo    VIT M.Tech Machine Learning Lab (MACSE504) - Mini-Project
echo    Interactive Streamlit Web Dashboard Launcher
echo ======================================================================
echo.
echo  Starting Streamlit application...
echo  Your default web browser will open automatically at http://localhost:8501
echo.
echo  [NOTE] Keep this window open while using the dashboard.
echo         To stop the server, simply close this window or press Ctrl+C.
echo ======================================================================
echo.

python -m streamlit run app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ======================================================================
    echo  [ERROR] Failed to start Streamlit.
    echo  Please ensure Python and Streamlit are installed in your environment.
    echo ======================================================================
    echo.
    pause
)
