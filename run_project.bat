@echo off
TITLE CodeCluster AI Proctoring System - Full Stack Launcher
COLOR 0A

echo =======================================================
echo    CodeCluster AI Proctoring System Launcher
echo =======================================================
echo.
echo  Starting 3 Services:
echo   1. ML gRPC Service      (Port 50051)
echo   2. Backend FastAPI      (Port 8000)
echo   3. Frontend React App   (Port 5173 / 3000)
echo.
echo =======================================================

echo [1/3] Launching ML gRPC Server...
start "CodeCluster - ML Server (Port 50051)" cmd /k "cd /d %~dp0CodeClusterML && venv\Scripts\python.exe grpc_service\ml_server.py"

echo Waiting 3 seconds for ML gRPC server initialization...
timeout /t 3 /nobreak >nul

echo [2/3] Launching Backend FastAPI Server...
start "CodeCluster - Backend Server (Port 8000)" cmd /k "cd /d %~dp0backend && venv\Scripts\python.exe app.py"

echo Waiting 2 seconds for Backend server initialization...
timeout /t 2 /nobreak >nul

echo [3/3] Launching Frontend Web Application...
start "CodeCluster - Frontend Web App" cmd /k "cd /d %~dp0CodeCluster && npm run dev"

echo.
echo =======================================================
echo  All 3 Services Fired Successfully!
echo  Keep the opened terminal windows active while testing.
echo =======================================================
echo.
pause
