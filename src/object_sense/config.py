"""Configuration settings for ObjectSense."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/object_sense"
    database_echo: bool = False

    # Sparkstation LLM Gateway
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "dummy-key"

    # Model names
    model_chat: str = "qwen3-vl-4b"  # Vision + chat
    model_reasoning: str = "gpt-oss-20b"  # Reasoning with traces
    model_text_embedding: str = "bge-large"  # 1024-dim text embeddings
    model_image_embedding: str = "clip-vit"  # 768-dim image/text embeddings

    # Embedding dimensions (must match models)
    dim_text_embedding: int = 1024  # bge-large for rich text semantics
    dim_image_embedding: int = 768  # clip-vit for visual features
    dim_clip_text_embedding: int = 768  # clip-vit for cross-modal text queries


settings = Settings()
