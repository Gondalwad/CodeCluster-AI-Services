# CodeCluster AI Proctoring System — Step-by-Step Setup & Execution Guide

This document contains step-by-step commands to set up, build, and run the project from scratch on local machines (Windows/Linux/macOS) and inside Docker containers.

---

## Table of Contents
1. [Backend Setup & Local Execution](#1-backend-setup--local-execution)
2. [ML Service Setup & Local Execution](#2-ml-service-setup--local-execution)
3. [Creating Docker Image & Container](#3-creating-docker-image--container)
4. [Running & Managing the Docker Container](#4-running--managing-the-docker-container)
5. [Running the `start_services.sh` Script](#5-running-the-start_servicessh-script)

---

## 1. Backend Setup & Local Execution

Follow these steps to set up and run the FastAPI Backend server independently:

### Step A: Open Terminal & Navigate to `backend`
```bash
cd backend
```

### Step B: Create Virtual Environment (`venv`)
```bash
python -m venv venv
```

### Step C: Activate Virtual Environment
* **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt)**:
  ```cmd
  venv\Scripts\activate.bat
  ```
* **Linux / macOS (Bash/Zsh)**:
  ```bash
  source venv/bin/activate
  ```

### Step D: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step E: Run Backend Server
```bash
python app.py
```
> **Output**: FastAPI server running live on `http://127.0.0.1:8000` (WebSockets listening at `ws://127.0.0.1:8000/ws/proctor/{candidate_id}`).

---

## 2. ML Service Setup & Local Execution

Follow these steps to set up and run the Machine Learning gRPC server independently:

### Step A: Open Terminal & Navigate to `CodeClusterML`
```bash
cd CodeClusterML
```

### Step B: Create Virtual Environment (`venv`)
```bash
python -m venv venv
```

### Step C: Activate Virtual Environment
* **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt)**:
  ```cmd
  venv\Scripts\activate.bat
  ```
* **Linux / macOS (Bash/Zsh)**:
  ```bash
  source venv/bin/activate
  ```

### Step D: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step E: Run ML gRPC Server
```bash
python grpc_service/ml_server.py
```
> **Output**: ML gRPC Server running live on `127.0.0.1:50051` (Ready to receive frames from Backend).

---

## 3. Creating Docker Image & Container

You can build the unified Docker image containing both Backend and ML Service using Docker CLI or Docker Compose.

### Option A: Using Docker Compose (Recommended)
From the project root directory (`c:\PIYUSH\MyProjects\CodeCluster`):

```powershell
# Build image with tag Ai_Worker
docker compose build
```

### Option B: Using Standard Docker CLI
From the project root directory:

```powershell
# Build image and tag as Ai_Worker
docker build -t Ai_Worker .
```

---

## 4. Running & Managing the Docker Container

### Option A: Run via Docker Compose (Recommended)

* **Build and Start Container in Background (Detached Mode)**:
  ```powershell
  docker compose up -d --build
  ```

* **View Live Container Logs**:
  ```powershell
  docker compose logs -f
  ```

* **Stop Container**:
  ```powershell
  docker compose down
  ```

### Option B: Run via Standard Docker CLI

* **Start Container Named `Ai_Worker`**:
  ```powershell
  docker run -d --name Ai_Worker -p 8000:8000 -p 50051:50051 Ai_Worker
  ```

* **View Live Container Logs**:
  ```powershell
  docker logs -f Ai_Worker
  ```

* **Stop Container**:
  ```powershell
  docker stop Ai_Worker
  ```

* **Remove Container**:
  ```powershell
  docker rm Ai_Worker
  ```

---

## 5. Running the `start_services.sh` Script

The [`start_services.sh`](file:///c:/PIYUSH/MyProjects/CodeCluster/start_services.sh) script is an automated Linux shell runner that starts `ml_server.py` first, waits 3 seconds, and then starts `app.py`.

### How It Runs Inside Docker:
When the Docker container starts, `Dockerfile` automatically executes `start_services.sh` as the main container entrypoint:
```dockerfile
ENTRYPOINT ["/app/start_services.sh"]
```

### How to Run Manually (Linux / macOS / Git Bash / WSL):
If you want to run `start_services.sh` directly on Linux or Git Bash:

1. **Make Script Executable**:
   ```bash
   chmod +x start_services.sh
   ```

2. **Execute Script**:
   ```bash
   ./start_services.sh
   ```

3. **Stop Script Services**:
   Press `Ctrl + C` in the terminal to trigger signal cleanup and terminate both background services cleanly.

---

## 6. One-Click Windows `.bat` Launchers

Two automated Windows batch files are provided in the project root for 1-click execution:

### 1. `run_project.bat` (Local Full-Stack Launcher)
* **Location**: [`run_project.bat`](file:///c:/PIYUSH/MyProjects/CodeCluster/run_project.bat)
* **What it does**: Opens 3 separate labelled terminal windows and fires:
  1. ML gRPC Server on port `50051`.
  2. Backend FastAPI Server on port `8000`.
  3. Frontend React Web App (`npm run dev`).
* **Usage**: Double-click [`run_project.bat`](file:///c:/PIYUSH/MyProjects/CodeCluster/run_project.bat) in File Explorer or run `.\run_project.bat` in PowerShell.

### 2. `run_docker.bat` (Docker Container Launcher)
* **Location**: [`run_docker.bat`](file:///c:/PIYUSH/MyProjects/CodeCluster/run_docker.bat)
* **What it does**: Builds and starts container `Ai_Worker` using `docker compose up --build`.
* **Usage**: Double-click [`run_docker.bat`](file:///c:/PIYUSH/MyProjects/CodeCluster/run_docker.bat) in File Explorer or run `.\run_docker.bat` in PowerShell.

