import logging
import os
from datetime import datetime
from fastapi import FastAPI

from config.settings import settings
from api.websocket import router as websocket_router

# File logging setup
_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, f"backend_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_fmt)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logging.getLogger("uvicorn.access").handlers = [_file_handler, _console_handler]

app = FastAPI(title=settings.APP_NAME)
app.include_router(websocket_router)


@app.get("/")
async def health_check():
    return {"status": "healthy", "service": settings.APP_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )

