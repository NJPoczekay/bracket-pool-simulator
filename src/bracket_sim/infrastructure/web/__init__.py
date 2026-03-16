"""Web/API adapters for local product surfaces."""

from bracket_sim.infrastructure.web.app import (
    create_app as create_pool_app,
)
from bracket_sim.infrastructure.web.app import (
    serve_web_app,
)
from bracket_sim.infrastructure.web.main import app, create_app, run_server

__all__ = [
    "app",
    "create_app",
    "create_pool_app",
    "run_server",
    "serve_web_app",
]
