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

    # ── Type Promotion Thresholds (design_v2_corrections.md §7) ──────────────
    # Promotion happens when ANY of these conditions are met:

    # Evidence threshold: promote when evidence_count >= this value
    # "5 entities OR 20 observations" - we use 5 as default
    type_promotion_min_evidence: int = 5

    # Coherence threshold: promote when cluster tightness > this value
    # Set to None for v0 - coherence requires stable entity clusters (Phase 4).
    # Computed and logged, but not used for gating until entity resolution exists.
    type_promotion_min_coherence: float | None = None

    # Time window: candidate must survive this many days without merge/contradiction
    type_promotion_survival_days: int = 7

    # ── TypeCandidate Cleanup Thresholds ─────────────────────────────────────
    # Reject candidates with evidence_count == 1 after this many days
    type_candidate_reject_after_days: int = 30

    # GC (garbage collect) rejected candidates after this many days
    type_candidate_gc_after_days: int = 365

    # ── Entity Resolution Thresholds (design_v2_corrections.md §5) ───────────
    # T_link: High confidence, link to existing entity
    entity_resolution_t_link: float = 0.85

    # T_new: Below this, create new proto-entity
    entity_resolution_t_new: float = 0.60

    # T_margin: If best_score - second_best < this, flag for review
    entity_resolution_t_margin: float = 0.10


settings = Settings()
