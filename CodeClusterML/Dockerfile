# Stage 1 — download InsightFace and Silero at build time so the runtime
# image never needs outbound internet access.
FROM python:3.12-slim AS model-downloader

WORKDIR /download

RUN pip install --no-cache-dir \
    insightface==0.7.3 \
    onnxruntime==1.19.2 \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    numpy==1.26.4

RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', root='/models/insightface', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=0, det_size=(640,640))"

RUN python -c "\
import torch; \
torch.hub.load('snakers4/silero-vad', 'silero_vad', source='github', trust_repo=True)"


# Stage 2 — lean runtime image
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=model-downloader /root/.insightface  /root/.insightface
COPY --from=model-downloader /root/.cache        /root/.cache

COPY config.py          .
COPY utils/             utils/
COPY models/            models/
COPY pipeline/          pipeline/
COPY grpc_service/      grpc_service/
COPY weights/           weights/

RUN python -m grpc_tools.protoc \
    --proto_path=grpc_service \
    --python_out=grpc_service \
    --grpc_python_out=grpc_service \
    grpc_service/proctor.proto

RUN useradd -m -u 1000 mlworker && chown -R mlworker:mlworker /app /root/.insightface /root/.cache
USER mlworker

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import grpc; ch = grpc.insecure_channel('localhost:50051'); grpc.channel_ready_future(ch).result(timeout=5)" \
    || exit 1

ENTRYPOINT ["python", "grpc_service/ml_server.py"]
