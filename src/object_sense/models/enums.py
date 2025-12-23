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


class ObjectStatus(str, Enum):
    """Lifecycle status of an Object."""

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


class EntityStatus(str, Enum):
    """Lifecycle status of an Entity."""

    PROTO = "proto"  # Early candidate, low confidence
    CANDIDATE = "candidate"  # Growing evidence
    STABLE = "stable"  # Well-established
    DEPRECATED = "deprecated"  # No longer valid


class SubjectKind(str, Enum):
    """What kind of thing the evidence is about."""

    OBJECT = "object"
    ENTITY = "entity"
    OBJECT_ENTITY_LINK = "object_entity_link"
    ENTITY_MERGE = "entity_merge"


class EvidenceSource(str, Enum):
    """Where the evidence came from."""

    LLM = "llm"
    VISION_MODEL = "vision_model"
    HEURISTIC = "heuristic"
    USER = "user"


class TypeEvolutionKind(str, Enum):
    """What kind of type evolution occurred."""

    ALIAS = "alias"  # New name added
    MERGE = "merge"  # Types combined
    SPLIT = "split"  # Type divided
    DEPRECATE = "deprecate"  # Type retired
