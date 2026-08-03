import os
import sys
import logging
import subprocess
from concurrent import futures

import grpc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from config import GRPC_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml_server")

_PROTO_FILE = os.path.join(os.path.dirname(__file__), "proctor.proto")
_STUB_FILE  = os.path.join(os.path.dirname(__file__), "proctor_pb2.py")
_GRPC_DIR   = os.path.dirname(__file__)


def _compile_proto():
    if os.path.exists(_STUB_FILE):
        return

    logger.info("Compiling proctor.proto stubs...")
    result = subprocess.run(
        [
            sys.executable, "-m", "grpc_tools.protoc",
            f"--proto_path={_GRPC_DIR}",
            f"--python_out={_GRPC_DIR}",
            f"--grpc_python_out={_GRPC_DIR}",
            _PROTO_FILE,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("protoc failed:\n%s", result.stderr)
        sys.exit(1)
    logger.info("Proto stubs compiled successfully.")


def serve():
    _compile_proto()

    import proctor_pb2_grpc
    from servicer import ProctoringServicer

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_receive_message_length", 10 * 1024 * 1024),
            ("grpc.max_send_message_length",    10 * 1024 * 1024),
        ],
    )

    proctor_pb2_grpc.add_ProctoringServiceServicer_to_server(ProctoringServicer(), server)
    bind_target = f"127.0.0.1:{GRPC_PORT}"
    server.add_insecure_port(bind_target)
    server.start()

    logger.info("=" * 55)
    logger.info("  CodeCluster ML gRPC Server started")
    logger.info("  Listening on %s", bind_target)
    logger.info("  Ready to accept connections from Java backend")
    logger.info("=" * 55)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(grace=5)


if __name__ == "__main__":
    serve()
