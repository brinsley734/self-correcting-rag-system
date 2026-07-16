from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Kubernetes RAG Agent API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Add your new collection name
    COLLECTION_NAME: str = "ukjobs_docs"

    # Infrastructure connection settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Link to your .env file here
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()