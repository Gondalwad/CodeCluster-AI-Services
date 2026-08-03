from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AI Exam Proctoring Backend"

    HOST: str = "0.0.0.0"
    PORT: int = 8080

    AI_WORKER_URL: str = "http://localhost:8001"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
