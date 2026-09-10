from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
from typing import Any


class WorkspaceTools:
    """Bounded workspace tools for an execution worker."""

    ALLOWED_COMMANDS = {"python", "pytest", "pip", "npm", "node", "git", "uv", "ruff"}
    BLOCKED_ARGS = {"--system", "--global", "--user", "--break-system-packages"}
    SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "PRIVATE_KEY", "AUTH")
    MAX_FILE_BYTES = 2_000_000
    MAX_COMMAND_OUTPUT = 20_000

    def __init__(self, workspace: str, command_timeout: float = 300):
        self.root = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.command_timeout = max(1.0, float(command_timeout))

    def _path(self, relative: str) -> Path:
        raw = str(relative or "")
        if not raw or "\x00" in raw:
            raise ValueError("invalid workspace path")
        candidate = (self.root / raw).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("path escapes workspace")
        return candidate

    def read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(args["path"]))
        if not path.is_file():
            raise FileNotFoundError(str(path.relative_to(self.root)))
        if path.stat().st_size > self.MAX_FILE_BYTES:
            raise ValueError("file exceeds workspace read limit")
        return {"path": str(path.relative_to(self.root)), "content": path.read_text(encoding="utf-8")}

    def write_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(args["path"]))
        content = str(args.get("content", ""))
        encoded = content.encode("utf-8")
        if len(encoded) > self.MAX_FILE_BYTES:
            raise ValueError("file exceeds workspace write limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
        return {"path": str(path.relative_to(self.root)), "bytes": len(encoded)}

    def _argv(self, command: Any) -> list[str]:
        if isinstance(command, list):
            argv = [str(x) for x in command]
        else:
            argv = shlex.split(str(command or ""), posix=os.name != "nt")
        if not argv or Path(argv[0]).name.lower() not in self.ALLOWED_COMMANDS:
            raise ValueError("command is not allowlisted")
        if any(arg in self.BLOCKED_ARGS for arg in argv[1:]):
            raise ValueError("command contains a blocked package-management option")
        return argv

    @classmethod
    def _safe_environment(cls) -> dict[str, str]:
        """Pass only non-secret environment variables to model-generated commands."""
        safe: dict[str, str] = {}
        for key, value in os.environ.items():
            upper = key.upper()
            if any(marker in upper for marker in cls.SECRET_ENV_MARKERS):
                continue
            safe[key] = value
        return safe

    async def run_command(self, args: dict[str, Any]) -> dict[str, Any]:
        argv = self._argv(args.get("command"))
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=self.root, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env=self._safe_environment(),
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=self.command_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {"exit_code": -1, "output": "command timed out"}
        return {"exit_code": proc.returncode, "output": out.decode("utf-8", errors="replace")[-self.MAX_COMMAND_OUTPUT:]}

    async def git_diff(self, args: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.run_command({"command": ["git", "diff", "--no-ext-diff", "--"]})

    def as_tools(self) -> dict[str, Any]:
        return {"read_file": self.read_file, "write_file": self.write_file, "run_command": self.run_command, "git_diff": self.git_diff}

    @staticmethod
    def specs() -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file inside the workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Create or replace a UTF-8 text file inside the workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "run_command", "description": "Run an allowlisted development command in the workspace without inheriting secret environment variables.", "parameters": {"type": "object", "properties": {"command": {"type": ["string", "array"]}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "git_diff", "description": "Inspect current git diff.", "parameters": {"type": "object", "properties": {}}}},
        ]
