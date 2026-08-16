@echo off
TITLE CodeCluster Docker Launcher (Ai_Worker)
COLOR 0B

echo =======================================================
echo    CodeCluster Docker Container Launcher (Ai_Worker)
echo =======================================================
echo.
echo  Starting Docker Services:
echo   - Container Name: Ai_Worker
echo   - ML gRPC Port:   50051
echo   - Backend Port:   8000
echo.
echo =======================================================

echo Building and starting Docker container Ai_Worker...
docker compose up --build

echo.
pause
