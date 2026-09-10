"""Compatibility entrypoint; the production application lives in agent_platform.app."""
from agent_platform.app import app

__all__ = ["app"]
