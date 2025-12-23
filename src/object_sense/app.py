"""FastAPI application for ObjectSense."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from object_sense import __version__
from object_sense.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler."""
    await init_db()
    yield


app = FastAPI(
    title="ObjectSense",
    description="Semantic substrate for persistent object identity and type awareness",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}
