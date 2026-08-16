#!/bin/bash
set -e

echo "======================================================="
echo "  Starting CodeCluster ML gRPC Service (Port 50051)..."
echo "======================================================="
python /app/CodeClusterML/grpc_service/ml_server.py &
ML_PID=$!

# Wait briefly for gRPC server socket initialization
sleep 3

echo "======================================================="
echo "  Starting CodeCluster Backend FastAPI (Port 8000)..."
echo "======================================================="
python /app/backend/app.py &
BACKEND_PID=$!

# Catch container termination signals and pass them to processes
trap "kill -TERM $ML_PID $BACKEND_PID" SIGINT SIGTERM

wait -n $ML_PID $BACKEND_PID
