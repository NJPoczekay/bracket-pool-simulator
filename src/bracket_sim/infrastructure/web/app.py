"""FastAPI app for the minimal single-user multi-pool web wrapper."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from bracket_sim.infrastructure.web.config import PoolProfile, load_pool_registry
from bracket_sim.infrastructure.web.service import (
    REPORT_ARTIFACT_FILENAMES,
    LatestReport,
    PoolRunBusyError,
    PoolService,
    UnknownPoolError,
)

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


class PoolScheduler:
    """Small background loop that runs due scheduled pools once per minute."""

    def __init__(
        self,
        service: PoolService,
        *,
        poll_seconds: float = 60.0,
    ) -> None:
        self._service = service
        self._poll_seconds = poll_seconds
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        """Start the scheduler background thread if it is not already running."""

        if self._thread is not None:
            return

        self._thread = Thread(target=self._run, name="pool-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler thread and wait briefly for it to exit."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_seconds + 1.0)
            self._thread = None

    def run_pending(self, *, now: datetime | None = None) -> list[str]:
        """Run any pools due at the current tick."""

        return self._service.run_due_pools(now=now)

    def _run(self) -> None:
        while not self._stop_event.wait(self._poll_seconds):
            try:
                self.run_pending(now=datetime.now(UTC))
            except Exception:
                continue


def create_app(
    *,
    config_path: Path,
    service: PoolService | None = None,
    enable_scheduler: bool = True,
    scheduler_poll_seconds: float = 60.0,
) -> FastAPI:
    """Build the local multi-pool web app."""

    pool_service = service or PoolService(load_pool_registry(config_path))
    scheduler = PoolScheduler(pool_service, poll_seconds=scheduler_poll_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pool_service = pool_service
        app.state.pool_scheduler = scheduler
        if enable_scheduler:
            scheduler.start()
        try:
            yield
        finally:
            scheduler.stop()

    app = FastAPI(title="Bracket Pool Simulator", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        return _render_dashboard(request, status_code=200)

    @app.post("/pools/{pool_id}/run", response_class=HTMLResponse)
    def run_pool_html(pool_id: str, request: Request) -> HTMLResponse:
        service = _pool_service(request)
        try:
            pool = service.get_pool(pool_id)
            service.run_pool(pool_id)
        except UnknownPoolError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown pool {pool_id!r}") from exc
        except PoolRunBusyError as exc:
            return _render_dashboard(request, error=str(exc), status_code=409)
        except ValueError as exc:
            return _render_dashboard(request, error=str(exc), status_code=400)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            return _render_dashboard(request, error=f"Unexpected error: {exc}", status_code=500)

        return _render_dashboard(
            request,
            message=f"Ran {pool.name}.",
            status_code=200,
        )

    @app.get("/api/pools")
    def list_pools_api(request: Request) -> dict[str, object]:
        service = _pool_service(request)
        return {
            "pools": [
                _serialize_pool(
                    request,
                    pool=pool,
                    latest_report=service.get_latest_report(pool.id),
                )
                for pool in service.list_pools()
            ]
        }

    @app.post("/api/pools/{pool_id}/run")
    def run_pool_api(pool_id: str, request: Request) -> dict[str, object]:
        service = _pool_service(request)
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
        service = _pool_service(request)
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
        if artifact_name not in REPORT_ARTIFACT_FILENAMES:
            raise HTTPException(status_code=404, detail=f"Unknown artifact {artifact_name!r}")

        service = _pool_service(request)
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


def serve_web_app(
    *,
    config_path: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Start the local web wrapper with uvicorn."""

    import uvicorn

    uvicorn.run(create_app(config_path=config_path), host=host, port=port)


def _pool_service(request: Request) -> PoolService:
    return cast(PoolService, request.app.state.pool_service)


def _render_dashboard(
    request: Request,
    *,
    message: str | None = None,
    error: str | None = None,
    status_code: int,
) -> HTMLResponse:
    service = _pool_service(request)
    context = {
        "request": request,
        "message": message,
        "error": error,
        "busy": service.is_busy(),
        "pools": [
            _serialize_pool(
                request,
                pool=pool,
                latest_report=service.get_latest_report(pool.id),
            )
            for pool in service.list_pools()
        ],
    }
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="dashboard.html",
        context=context,
        status_code=status_code,
    )


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
