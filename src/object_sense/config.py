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

    # ── Provider Configuration ────────────────────────────────────────────────
    # LLM provider for chat/reasoning: "openai" or "sparkstation"
    llm_provider: str = "sparkstation"

    # Embedding provider: "openai" or "sparkstation"
    # Note: OpenAI doesn't offer image embeddings, so image embeddings always use Sparkstation
    embedding_provider: str = "sparkstation"

    # ── OpenAI Configuration ──────────────────────────────────────────────────
    openai_api_key: str = ""  # Set via OPENAI_API_KEY env var

    # Chat/Reasoning models
    openai_model_chat: str = "gpt-5.2"  # GPT-5.2 for chat/vision
    openai_model_reasoning: str = "gpt-5.2"  # GPT-5.2 for type inference

    # Text embedding model (for embedding_provider=openai)
    openai_model_text_embedding: str = "text-embedding-3-large"  # 3072-dim
    openai_dim_text_embedding: int = 3072  # text-embedding-3-large dimensions

    # ── Sparkstation Configuration ────────────────────────────────────────────
    sparkstation_base_url: str = "http://localhost:8000/v1"
    sparkstation_api_key: str = "dummy-key"

    # Chat/Reasoning models
    sparkstation_model_chat: str = "qwen3-vl-4b"  # Vision + chat
    sparkstation_model_reasoning: str = "gpt-oss-20b"  # Reasoning with traces

    # Embedding models
    sparkstation_model_text_embedding: str = "bge-large"  # 1024-dim
    sparkstation_model_image_embedding: str = "clip-vit"  # 768-dim
    sparkstation_dim_text_embedding: int = 1024  # bge-large dimensions
    sparkstation_dim_image_embedding: int = 768  # clip-vit dimensions

    # ── Dynamic Property Accessors ────────────────────────────────────────────
    @property
    def llm_base_url(self) -> str:
        """Get base URL for LLM provider."""
        if self.llm_provider == "openai":
            return "https://api.openai.com/v1"
        return self.sparkstation_base_url

    @property
    def llm_api_key(self) -> str:
        """Get API key for LLM provider."""
        if self.llm_provider == "openai":
            return self.openai_api_key
        return self.sparkstation_api_key

    @property
    def model_chat(self) -> str:
        """Get chat model for current provider."""
        if self.llm_provider == "openai":
            return self.openai_model_chat
        return self.sparkstation_model_chat

    @property
    def model_reasoning(self) -> str:
        """Get reasoning model for current provider."""
        if self.llm_provider == "openai":
            return self.openai_model_reasoning
        return self.sparkstation_model_reasoning

    @property
    def embedding_base_url(self) -> str:
        """Get base URL for embedding provider."""
        if self.embedding_provider == "openai":
            return "https://api.openai.com/v1"
        return self.sparkstation_base_url

    @property
    def embedding_api_key(self) -> str:
        """Get API key for embedding provider."""
        if self.embedding_provider == "openai":
            return self.openai_api_key
        return self.sparkstation_api_key

    @property
    def model_text_embedding(self) -> str:
        """Get text embedding model for current provider."""
        if self.embedding_provider == "openai":
            return self.openai_model_text_embedding
        return self.sparkstation_model_text_embedding

    @property
    def model_image_embedding(self) -> str:
        """Get image embedding model (always Sparkstation - cloud providers don't support)."""
        return self.sparkstation_model_image_embedding

    @property
    def dim_text_embedding(self) -> int:
        """Get text embedding dimensions for current provider."""
        if self.embedding_provider == "openai":
            return self.openai_dim_text_embedding
        return self.sparkstation_dim_text_embedding

    @property
    def dim_image_embedding(self) -> int:
        """Get image embedding dimensions (always Sparkstation)."""
        return self.sparkstation_dim_image_embedding

    @property
    def dim_clip_text_embedding(self) -> int:
        """Get CLIP text embedding dimensions (always 768 for cross-modal search)."""
        return 768  # CLIP always uses 768-dim for text/image cross-modal

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

    # ── INDIVIDUAL Image Guardrails ───────────────────────────────────────────
    # These guardrails prevent false merges for INDIVIDUAL entities when only
    # image embedding is available (sparse signal regime).
    #
    # T_img_min: Minimum raw image cosine similarity for ANY merge decision.
    # Even if composite score is high, reject if image similarity is below this.
    # This prevents merges based on weak visual evidence.
    entity_resolution_t_img_min: float = 0.75

    # T_margin_img: Minimum margin between top1 and top2 candidates for image-only.
    # Larger margin = more confidence that top1 is correct.
    # If margin is too small, flag for review instead of auto-linking.
    entity_resolution_t_margin_img: float = 0.08

    # ── Namespace Control (bead object-sense-bk9) ────────────────────────────
    # Namespaces are stable system-of-record scopes, NOT arbitrary LLM labels.
    # LLM-invented namespaces cause duplicate entities (SKU "123" in catalog_sku
    # vs product_sku become different entities even though they're the same).
    #
    # Valid namespace patterns:
    # - source:<dataset> — Explicitly provided via --id-namespace or derived from stable source ID
    # - global:<authority> — Globally unique ID systems (upc, isbn, iucn, gtin, asin)
    # - geo:wgs84 — GPS coordinates in WGS84 datum
    # - user:<tenant> — User/tenant-provided IDs
    #
    # Enforcement: If LLM proposes invalid namespace, engine overrides to context
    # namespace and writes Evidence with predicate="namespace_overridden".

    # Namespace patterns (regex). LLM namespaces must match one of these.
    namespace_patterns: list[str] = [
        r"^source:[a-z0-9_-]+$",  # source:<dataset> (explicit or derived)
        r"^global:[a-z0-9_-]+$",  # global:<authority> (upc, isbn, gtin, asin, etc.)
        r"^geo:wgs84$",  # GPS coordinates
        r"^user:[a-z0-9_-]+$",  # user:<tenant>
    ]

    # ID type → namespace mappings (engine-controlled, not LLM-controlled)
    # When extraction produces these id_types, engine assigns the namespace.
    # This ensures consistent namespacing regardless of what LLM proposes.
    id_type_namespace_map: dict[str, str] = {
        "gps": "geo:wgs84",
        "upc": "global:upc",
        "ean": "global:ean",
        "gtin": "global:gtin",
        "isbn": "global:isbn",
        "asin": "global:asin",
        "iucn_species_id": "global:iucn",
    }


settings = Settings()
