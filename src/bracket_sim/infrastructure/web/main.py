"""FastAPI entrypoint for the local web/API surface."""

from __future__ import annotations

import argparse

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from bracket_sim import __version__
from bracket_sim.application.product_foundation import (
    build_product_foundation,
    preview_cache_key,
)
from bracket_sim.domain.product_models import (
    CacheKeyPreview,
    CacheKeyPreviewRequest,
    ProductFoundation,
)
from bracket_sim.infrastructure.web.shell import build_frontend_shell


def create_app() -> FastAPI:
    """Build the FastAPI application for local development."""

    app = FastAPI(
        title="Bracket Pool Simulator",
        version=__version__,
        summary="Phase-0 web/API foundation for bracket tools.",
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the minimal frontend shell."""

        return build_frontend_shell()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        """Return a simple API health probe."""

        return {"status": "ok", "version": __version__}

    @app.get("/api/foundation", response_model=ProductFoundation)
    def foundation() -> ProductFoundation:
        """Return phase-0 product metadata for the web shell."""

        return build_product_foundation()

    @app.post("/api/cache-key", response_model=CacheKeyPreview)
    def cache_key_preview(request: CacheKeyPreviewRequest) -> CacheKeyPreview:
        """Preview the shared cache-key strategy for future workflows."""

        return preview_cache_key(request)

    return app


app = create_app()


def run_server(*, host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the local web/API server."""

    import uvicorn

    uvicorn.run(
        "bracket_sim.infrastructure.web.main:app",
        host=host,
        port=port,
        reload=reload,
        factory=False,
    )


def main() -> None:
    """Console script entrypoint for the local web server."""

    parser = argparse.ArgumentParser(description="Run the Bracket Pool Simulator web/API server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="TCP port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
