"""FastAPI entrypoint for the integrated Bracket Lab and Pool Tracker app."""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from bracket_sim import __version__
from bracket_sim.application.analyze_bracket import BracketLabService
from bracket_sim.application.product_foundation import (
    build_product_foundation,
    preview_cache_key,
)
from bracket_sim.domain.product_models import (
    AnalyzeBracketRequest,
    BracketAnalysis,
    BracketCompletionResult,
    BracketLabBootstrap,
    CacheKeyPreview,
    CacheKeyPreviewRequest,
    CompleteBracketRequest,
    OptimizationResult,
    OptimizeBracketRequest,
    ProductFoundation,
    SaveBracketRequest,
    SavedBracket,
    SavedBracketList,
)
from bracket_sim.infrastructure.storage.saved_brackets import (
    list_saved_brackets,
    load_saved_bracket,
    save_bracket,
)
from bracket_sim.infrastructure.web.app import PoolScheduler
from bracket_sim.infrastructure.web.config import PoolProfile, load_pool_registry
from bracket_sim.infrastructure.web.layout import build_bracket_lab_editor_layout
from bracket_sim.infrastructure.web.service import (
    REPORT_ARTIFACT_FILENAMES,
    LatestReport,
    PoolRunBusyError,
    PoolService,
    UnknownPoolError,
)

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


def create_app(
    *,
    config_path: Path | None = None,
    service: PoolService | None = None,
    bracket_lab_input: Path | None = None,
    bracket_lab_service: BracketLabService | None = None,
    bracket_store_dir: Path | None = None,
    enable_scheduler: bool = True,
    scheduler_poll_seconds: float = 60.0,
) -> FastAPI:
    """Build the integrated local web/API application."""

    pool_service = service
    if pool_service is None and config_path is not None:
        pool_service = PoolService(load_pool_registry(config_path))

    analyzer_service = bracket_lab_service
    if analyzer_service is None and bracket_lab_input is not None:
        analyzer_service = BracketLabService(bracket_lab_input)
    saved_brackets_dir = _resolve_saved_brackets_dir(
        bracket_lab_input=bracket_lab_input,
        bracket_lab_service=analyzer_service,
        bracket_store_dir=bracket_store_dir,
    )

    foundation = build_product_foundation(
        bracket_lab_enabled=analyzer_service is not None,
        tracker_enabled=pool_service is not None,
    )
    scheduler = (
        PoolScheduler(pool_service, poll_seconds=scheduler_poll_seconds)
        if pool_service is not None
        else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.product_foundation = foundation
        app.state.pool_service = pool_service
        app.state.bracket_lab_service = analyzer_service
        app.state.saved_brackets_dir = saved_brackets_dir
        app.state.pool_scheduler = scheduler
        if enable_scheduler and scheduler is not None:
            scheduler.start()
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.stop()

    app = FastAPI(
        title="Bracket Pool Simulator",
        version=__version__,
        summary="Integrated Bracket Lab and Pool Tracker surface.",
        lifespan=lifespan,
    )
    app.state.product_foundation = foundation
    app.state.pool_service = pool_service
    app.state.bracket_lab_service = analyzer_service
    app.state.saved_brackets_dir = saved_brackets_dir
    app.state.pool_scheduler = scheduler

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        """Render the integrated product shell."""

        return _render_home(request, status_code=200)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        """Return a simple API health probe."""

        return {"status": "ok", "version": __version__}

    @app.get("/api/foundation", response_model=ProductFoundation)
    def foundation_api(request: Request) -> ProductFoundation:
        """Return integrated product metadata for the shared app shell."""

        return _product_foundation(request)

    @app.post("/api/cache-key", response_model=CacheKeyPreview)
    def cache_key_preview(request: CacheKeyPreviewRequest) -> CacheKeyPreview:
        """Preview the shared cache-key strategy for future workflows."""

        return preview_cache_key(request)

    @app.get("/api/bracket-lab/bootstrap", response_model=BracketLabBootstrap)
    def bracket_lab_bootstrap_api(request: Request) -> BracketLabBootstrap:
        """Return prepared Bracket Lab graph data for the browser editor."""

        return _bracket_lab_service_or_503(request).build_bootstrap()

    @app.post("/api/bracket-lab/analyze", response_model=BracketAnalysis)
    def analyze_bracket_api(
        payload: AnalyzeBracketRequest,
        request: Request,
    ) -> BracketAnalysis:
        """Analyze one full user bracket against sampled public opponents."""

        try:
            return _bracket_lab_service_or_503(request).analyze_bracket(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/bracket-lab/complete", response_model=BracketCompletionResult)
    def complete_bracket_api(
        payload: CompleteBracketRequest,
        request: Request,
    ) -> BracketCompletionResult:
        """Auto-complete one partial user bracket without mutating locked picks."""

        try:
            return _bracket_lab_service_or_503(request).complete_bracket(payload)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/bracket-lab/optimize", response_model=OptimizationResult)
    def optimize_bracket_api(
        payload: OptimizeBracketRequest,
        request: Request,
    ) -> OptimizationResult:
        """Optimize one complete user bracket against the sampled public field."""

        try:
            return _bracket_lab_service_or_503(request).optimize_bracket(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/bracket-lab/saved-brackets", response_model=SavedBracketList)
    def list_saved_brackets_api(request: Request) -> SavedBracketList:
        """List saved bracket drafts for the active prepared Bracket Lab dataset."""

        service = _bracket_lab_service_or_503(request)
        storage_dir = _saved_brackets_dir_or_503(request)
        return SavedBracketList(
            brackets=list_saved_brackets(
                storage_dir=storage_dir,
                dataset_hash=service.dataset_hash,
            )
        )

    @app.get("/api/bracket-lab/saved-brackets/{bracket_id}", response_model=SavedBracket)
    def load_saved_bracket_api(bracket_id: str, request: Request) -> SavedBracket:
        """Load one saved bracket draft by id."""

        service = _bracket_lab_service_or_503(request)
        storage_dir = _saved_brackets_dir_or_503(request)
        try:
            saved = load_saved_bracket(storage_dir=storage_dir, bracket_id=bracket_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if saved.dataset_hash != service.dataset_hash:
            raise HTTPException(
                status_code=409,
                detail="Saved bracket belongs to a different Bracket Lab dataset",
            )
        return saved

    @app.post("/api/bracket-lab/saved-brackets", response_model=SavedBracket)
    def save_bracket_api(payload: SaveBracketRequest, request: Request) -> SavedBracket:
        """Save one bracket draft to local disk and return the persisted record."""

        service = _bracket_lab_service_or_503(request)
        storage_dir = _saved_brackets_dir_or_503(request)
        try:
            return save_bracket(
                storage_dir=storage_dir,
                dataset_hash=service.dataset_hash,
                request=payload,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/pools")
    def list_pools_api(request: Request) -> dict[str, object]:
        """Return configured tracker pools, or an empty list when tracking is not configured."""

        return {"pools": _serialized_pools(request)}

    @app.post("/pools/{pool_id}/run", response_class=HTMLResponse)
    def run_pool_html(pool_id: str, request: Request) -> HTMLResponse:
        """Run one tracker pool and re-render the integrated shell."""

        service = _pool_service_or_404(request, pool_id)
        try:
            pool = service.get_pool(pool_id)
            service.run_pool(pool_id)
        except UnknownPoolError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown pool {pool_id!r}") from exc
        except PoolRunBusyError as exc:
            return _render_home(request, error=str(exc), status_code=409)
        except ValueError as exc:
            return _render_home(request, error=str(exc), status_code=400)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            return _render_home(request, error=f"Unexpected error: {exc}", status_code=500)

        return _render_home(
            request,
            message=f"Ran {pool.name}.",
            status_code=200,
        )

    @app.post("/api/pools/{pool_id}/run")
    def run_pool_api(pool_id: str, request: Request) -> dict[str, object]:
        """Run one configured tracker pool and return its latest report metadata."""

        service = _pool_service_or_404(request, pool_id)
        try:
            pool = service.get_pool(pool_id)
            result = service.run_pool(pool_id)
        except UnknownPoolError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown pool {pool_id!r}") from exc
        except PoolRunBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        latest_report = service.get_latest_report(pool_id)
        return {
            "pool": _serialize_pool(request, pool=pool, latest_report=latest_report),
            "report_dir": str(result.report_dir),
        }

    @app.get("/api/pools/{pool_id}/latest-report")
    def latest_report_api(pool_id: str, request: Request) -> dict[str, object]:
        """Return the newest tracker report bundle for one configured pool."""

        service = _pool_service_or_404(request, pool_id)
        try:
            pool = service.get_pool(pool_id)
        except UnknownPoolError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown pool {pool_id!r}") from exc

        latest_report = service.get_latest_report(pool_id)
        if latest_report is None:
            raise HTTPException(
                status_code=404,
                detail=f"No report bundle found for pool {pool_id!r}",
            )

        return _serialize_pool(request, pool=pool, latest_report=latest_report)

    @app.get(
        "/pools/{pool_id}/artifacts/{artifact_name}",
        name="download_latest_artifact",
    )
    def download_latest_artifact(
        pool_id: str,
        artifact_name: str,
        request: Request,
    ) -> FileResponse:
        """Download one artifact from the newest report bundle for a configured pool."""

        if artifact_name not in REPORT_ARTIFACT_FILENAMES:
            raise HTTPException(status_code=404, detail=f"Unknown artifact {artifact_name!r}")

        service = _pool_service_or_404(request, pool_id)
        try:
            service.get_pool(pool_id)
        except UnknownPoolError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown pool {pool_id!r}") from exc

        latest_report = service.get_latest_report(pool_id)
        if latest_report is None:
            raise HTTPException(
                status_code=404,
                detail=f"No report bundle found for pool {pool_id!r}",
            )

        artifact_path = latest_report.artifact_paths.get(artifact_name)
        if artifact_path is None or not artifact_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Artifact {artifact_name!r} is not available for pool {pool_id!r}",
            )

        return FileResponse(path=artifact_path, filename=artifact_name)

    return app

def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    config_path: Path | None = None,
    bracket_lab_input: Path | None = None,
) -> None:
    """Run the integrated local web/API server."""

    import uvicorn

    if config_path is not None or bracket_lab_input is not None:
        if reload:
            msg = "--reload is not supported when --config or --bracket-lab-input is set"
            raise ValueError(msg)

        uvicorn.run(
            create_app(
                config_path=config_path,
                bracket_lab_input=bracket_lab_input,
            ),
            host=host,
            port=port,
            reload=False,
        )
        return

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
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional pool config TOML to enable live pool tracking data",
    )
    parser.add_argument(
        "--bracket-lab-input",
        type=Path,
        help="Optional prepared Bracket Lab directory to enable analyzer workflows",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="TCP port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()
    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        config_path=args.config,
        bracket_lab_input=args.bracket_lab_input,
    )


def _product_foundation(request: Request) -> ProductFoundation:
    return cast(ProductFoundation, request.app.state.product_foundation)


def _pool_service(request: Request) -> PoolService | None:
    return cast(PoolService | None, request.app.state.pool_service)


def _bracket_lab_service(request: Request) -> BracketLabService | None:
    return cast(BracketLabService | None, request.app.state.bracket_lab_service)


def _saved_brackets_dir(request: Request) -> Path | None:
    return cast(Path | None, request.app.state.saved_brackets_dir)


def _pool_service_or_404(request: Request, pool_id: str) -> PoolService:
    service = _pool_service(request)
    if service is None:
        raise HTTPException(status_code=404, detail=f"Unknown pool {pool_id!r}")
    return service


def _bracket_lab_service_or_503(request: Request) -> BracketLabService:
    service = _bracket_lab_service(request)
    if service is None:
        raise HTTPException(status_code=503, detail="Bracket Lab is not configured")
    return service


def _saved_brackets_dir_or_503(request: Request) -> Path:
    storage_dir = _saved_brackets_dir(request)
    if storage_dir is None:
        raise HTTPException(status_code=503, detail="Bracket Lab is not configured")
    return storage_dir


def _resolve_saved_brackets_dir(
    *,
    bracket_lab_input: Path | None,
    bracket_lab_service: BracketLabService | None,
    bracket_store_dir: Path | None,
) -> Path | None:
    if bracket_store_dir is not None:
        return bracket_store_dir
    if bracket_lab_input is not None:
        return bracket_lab_input / "saved-brackets"
    if bracket_lab_service is not None:
        return bracket_lab_service.input_dir / "saved-brackets"
    return None


def _render_home(
    request: Request,
    *,
    message: str | None = None,
    error: str | None = None,
    status_code: int,
) -> HTMLResponse:
    foundation = _product_foundation(request).model_dump(mode="json")
    workflow_by_key = {workflow["key"]: workflow for workflow in foundation["workflows"]}
    tracker_service = _pool_service(request)
    bracket_lab_service = _bracket_lab_service(request)
    bracket_lab_bootstrap: dict[str, object] | None = None
    bracket_lab_editor_layout: dict[str, object] | None = None
    if bracket_lab_service is not None:
        bootstrap = bracket_lab_service.build_bootstrap()
        bracket_lab_bootstrap = bootstrap.model_dump(mode="json")
        bracket_lab_editor_layout = asdict(
            build_bracket_lab_editor_layout(
                teams=bootstrap.teams,
                games=bootstrap.games,
            )
        )

    context = {
        "request": request,
        "version": __version__,
        "foundation": foundation,
        "workflow_by_key": workflow_by_key,
        "message": message,
        "error": error,
        "busy": tracker_service.is_busy() if tracker_service is not None else False,
        "bracket_lab_configured": bracket_lab_service is not None,
        "bracket_lab_bootstrap": bracket_lab_bootstrap,
        "bracket_lab_editor_layout": bracket_lab_editor_layout,
        "tracker_configured": tracker_service is not None,
        "tracker_setup_path": "config/pools.example.toml",
        "pools": _serialized_pools(request),
    }
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="dashboard.html",
        context=context,
        status_code=status_code,
    )


def _serialized_pools(request: Request) -> list[dict[str, object]]:
    service = _pool_service(request)
    if service is None:
        return []

    return [
        _serialize_pool(
            request,
            pool=pool,
            latest_report=service.get_latest_report(pool.id),
        )
        for pool in service.list_pools()
    ]


def _serialize_pool(
    request: Request,
    *,
    pool: PoolProfile,
    latest_report: LatestReport | None,
) -> dict[str, object]:
    return {
        "id": pool.id,
        "name": pool.name,
        "group_url": pool.group_url,
        "latest_report": _serialize_latest_report(request, latest_report),
    }


def _serialize_latest_report(
    request: Request,
    latest_report: LatestReport | None,
) -> dict[str, object] | None:
    if latest_report is None:
        return None

    return {
        "report_dir": str(latest_report.report_dir),
        "summary": latest_report.summary.model_dump(mode="json"),
        "artifacts": {
            filename: {
                "name": filename,
                "url": str(
                    request.url_for(
                        "download_latest_artifact",
                        pool_id=latest_report.pool_id,
                        artifact_name=filename,
                    )
                ),
            }
            for filename in latest_report.artifact_paths
        },
    }


app = create_app()


if __name__ == "__main__":
    main()
