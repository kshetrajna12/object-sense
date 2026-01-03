"""Shared pytest fixtures for ObjectSense tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from object_sense.config import settings
from object_sense.models import (
    Base,
    Entity,
    EntityNature,
    EntityStatus,
    Medium,
    Observation,
    Signature,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


# Use a separate test database to avoid polluting development data
TEST_DATABASE_URL = settings.database_url.replace(
    "/object_sense", "/object_sense_test"
)


@pytest.fixture
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create a test database engine for the session.

    This creates all tables at the start and drops them at the end.
    """
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
    )

    async with engine.begin() as conn:
        # Enable pgvector extension (required for VECTOR columns)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup: drop all tables after tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def db_session(
    test_engine: AsyncEngine,
) -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session with transaction rollback.

    Each test gets its own transaction that is rolled back at the end,
    ensuring test isolation without needing to recreate tables.
    """
    async_session_factory = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        async with session.begin():
            yield session
            # Rollback to ensure test isolation
            await session.rollback()


# Type aliases for factory fixtures
MakeObservation = Callable[..., Observation]
MakeEntity = Callable[..., Entity]
MakeSignature = Callable[..., Signature]


@pytest.fixture
def make_observation() -> MakeObservation:
    """Factory fixture for creating Observation instances."""

    def _make(
        *,
        observation_id: UUID | None = None,
        medium: Medium = Medium.IMAGE,
        source_id: str | None = None,
        observation_kind: str = "test_image",
        facets: dict[str, Any] | None = None,
        deterministic_ids: list[dict[str, Any]] | None = None,
    ) -> Observation:
        return Observation(
            observation_id=observation_id or uuid4(),
            medium=medium,
            source_id=source_id or f"test://source/{uuid4()}",
            observation_kind=observation_kind,
            facets=facets or {},
            deterministic_ids=deterministic_ids or [],
        )

    return _make


@pytest.fixture
def make_entity() -> MakeEntity:
    """Factory fixture for creating Entity instances."""

    def _make(
        *,
        entity_id: UUID | None = None,
        entity_nature: EntityNature = EntityNature.INDIVIDUAL,
        name: str | None = None,
        canonical_entity_id: UUID | None = None,
        status: EntityStatus = EntityStatus.PROTO,
        confidence: float = 0.8,
    ) -> Entity:
        return Entity(
            entity_id=entity_id or uuid4(),
            entity_nature=entity_nature,
            name=name,
            canonical_entity_id=canonical_entity_id,
            status=status,
            confidence=confidence,
            slots={},
            prototype_count=0,
        )

    return _make


@pytest.fixture
def make_signature() -> MakeSignature:
    """Factory fixture for creating Signature instances."""

    def _make(
        *,
        signature_id: UUID | None = None,
        observation_id: UUID,
        signature_type: str = "primary",
        image_embedding: list[float] | None = None,
        text_embedding: list[float] | None = None,
    ) -> Signature:
        return Signature(
            signature_id=signature_id or uuid4(),
            observation_id=observation_id,
            signature_type=signature_type,
            image_embedding=image_embedding,
            text_embedding=text_embedding,
        )

    return _make
