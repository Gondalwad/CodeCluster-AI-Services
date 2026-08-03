from fastapi import FastAPI

from config.settings import settings
from api.websocket import router as websocket_router

app = FastAPI(
    title=settings.APP_NAME,
)

app.include_router(websocket_router)


@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
    }
