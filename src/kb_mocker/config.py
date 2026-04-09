from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openrouter_base_url: str
    openrouter_api_key: str
    model_name: str
    knowledge_base_path: str
    max_iterations: int = 30

    class Config:
        env_file = ".env"

settings = Settings()
