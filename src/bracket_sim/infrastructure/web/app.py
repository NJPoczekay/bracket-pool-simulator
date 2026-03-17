"""Shared scheduler helpers for the integrated web app."""

from __future__ import annotations

from datetime import UTC, datetime
from threading import Event, Thread

from bracket_sim.infrastructure.web.service import PoolService


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
