"""Enumerations for ObjectSense data model."""

from enum import Enum


class Medium(str, Enum):
    """How information is encoded. Determines affordances, not meaning."""

    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    JSON = "json"
    AUDIO = "audio"
    BINARY = "binary"


class ObservationStatus(str, Enum):
    """Lifecycle status of an Observation."""

    ACTIVE = "active"
    MERGED = "merged"
    DEPRECATED = "deprecated"


class TypeStatus(str, Enum):
    """Lifecycle status of a Type."""

    PROVISIONAL = "provisional"  # Newly proposed, few entities
    STABLE = "stable"  # Validated through recurrence
    DEPRECATED = "deprecated"  # No longer used
    MERGED_INTO = "merged_into"  # Merged into another type


class TypeCreatedVia(str, Enum):
    """How a Type was created."""

    LLM_PROPOSED = "llm_proposed"
    CLUSTER_EMERGED = "cluster_emerged"
    USER_CONFIRMED = "user_confirmed"


class TypeCandidateStatus(str, Enum):
    """Lifecycle status of a TypeCandidate.

    TypeCandidates start PROPOSED, then either:
    - PROMOTED → became a stable Type
    - MERGED → folded into another candidate
    - REJECTED → pruned (hallucination, never referenced)
    """

    PROPOSED = "proposed"  # Initial state
    PROMOTED = "promoted"  # Became a stable Type
    MERGED = "merged"  # Merged into another candidate
    REJECTED = "rejected"  # Pruned / invalid


class EntityNature(str, Enum):
    """What kind of thing an entity represents (Correction #6).

    This affects signal weighting and resolution behavior, NOT the type system.
    """

    INDIVIDUAL = "individual"  # A single real-world instance (e.g., "Marula Leopard")
    CLASS = "class"  # A category/species (e.g., "Leopard" the species)
    GROUP = "group"  # A collection (e.g., "Lion Pride", "Safari Trip 2024")
    EVENT = "event"  # A time-bounded occurrence (e.g., "Hunt on Jan 15")


class EntityStatus(str, Enum):
    """Lifecycle status of an Entity."""

    PROTO = "proto"  # Early candidate, low confidence
    CANDIDATE = "candidate"  # Growing evidence
    STABLE = "stable"  # Well-established
    DEPRECATED = "deprecated"  # No longer valid


class EntityEvolutionKind(str, Enum):
    """What kind of entity evolution decision was made (Correction #5)."""

    LINK = "link"  # Observation linked to entity
    MERGE = "merge"  # Entities merged
    SPLIT = "split"  # Entity split
    REJECT = "reject"  # Link/merge rejected (logged for debugging)


class IdentityConflictStatus(str, Enum):
    """Resolution status of an identity conflict (Correction #8)."""

    PENDING = "pending"  # Needs resolution
    RESOLVED = "resolved"  # Has been resolved
    IGNORED = "ignored"  # Explicitly marked as not requiring resolution


class SubjectKind(str, Enum):
    """What kind of thing the evidence is about."""

    OBSERVATION = "observation"
    ENTITY = "entity"
    OBSERVATION_ENTITY_LINK = "observation_entity_link"
    ENTITY_MERGE = "entity_merge"


class EvidenceSource(str, Enum):
    """Where the evidence came from."""

    LLM = "llm"
    VISION_MODEL = "vision_model"
    HEURISTIC = "heuristic"
    USER = "user"
    SYSTEM = "system"  # Engine-controlled decisions (namespace override, etc.)


class TypeEvolutionKind(str, Enum):
    """What kind of type evolution occurred."""

    ALIAS = "alias"  # New name added
    MERGE = "merge"  # Types combined
    SPLIT = "split"  # Type divided
    DEPRECATE = "deprecate"  # Type retired


class LinkStatus(str, Enum):
    """Status of an observation-entity link (Correction #5).

    - hard: Deterministic ID match, posterior=1.0, high confidence
    - soft: Similarity-based link, confidence above T_link threshold
    - candidate: Uncertain, in the "don't know" band [T_new, T_link]
    """

    HARD = "hard"  # Deterministic ID match
    SOFT = "soft"  # High-confidence similarity
    CANDIDATE = "candidate"  # Uncertain, needs validation


class LinkRole(str, Enum):
    """Role of an entity in an observation (v0: subject + context only).

    - subject: The observation is primarily about this entity
    - context: The entity is present/supporting (location, species, group)
    """

    SUBJECT = "subject"
    CONTEXT = "context"


class Affordance(str, Enum):
    """Capabilities enabled by a medium.

    Affordances determine what operations are possible on an object
    based on its medium. The system uses these to decide which
    extractors and processors to run.
    """

    # Image affordances
    CAN_EMBED_IMAGE = "can_embed_image"
    CAN_DETECT_OBJECTS = "can_detect_objects"
    CAN_EXTRACT_EXIF = "can_extract_exif"
    CAN_CAPTION = "can_caption"

    # Video affordances
    CAN_SAMPLE_FRAMES = "can_sample_frames"
    CAN_SEGMENT = "can_segment"
    CAN_EXTRACT_AUDIO = "can_extract_audio"

    # Text affordances
    CAN_CHUNK = "can_chunk"
    CAN_EMBED_TEXT = "can_embed_text"
    CAN_PARSE_TEXT = "can_parse_text"

    # JSON affordances
    CAN_PARSE_KEYS = "can_parse_keys"
    CAN_INFER_SCHEMA = "can_infer_schema"

    # Audio affordances
    CAN_TRANSCRIBE = "can_transcribe"
    CAN_EMBED_AUDIO = "can_embed_audio"

    # Binary affordances
    CAN_ENTROPY_ANALYZE = "can_entropy_analyze"
    CAN_MAGIC_SNIFF = "can_magic_sniff"
