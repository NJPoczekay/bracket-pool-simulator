"""Web adapter package for the local multi-pool control panel."""

from bracket_sim.infrastructure.web.app import create_app, serve_web_app

__all__ = ["create_app", "serve_web_app"]
