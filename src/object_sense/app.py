"""FastAPI application for ObjectSense."""

from fastapi import FastAPI

from object_sense import __version__

app = FastAPI(
    title="ObjectSense",
    description="Semantic substrate for persistent object identity and type awareness",
    version=__version__,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}
