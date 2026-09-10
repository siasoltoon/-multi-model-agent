from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any


class WorkspaceTools:
    """Safe local workspace tools for an execution worker.

    Every path is resolved below the configured workspace. Shell execution is
    allowlisted to common development commands and always runs inside it.
    """
    ALLOWED_COMMANDS = {"python", "pytest", "pip", "npm", "node", "git", "uv", "ruff"}

    def __init__(self, workspace: str, command_timeout: float = 300):
        self.root = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.command_timeout = command_timeout

    def _path(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("path escapes workspace")
        return candidate

    def read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(args["path"]))
        return {"path": str(path.relative_to(self.root)), "content": path.read_text(encoding="utf-8")}

    def write_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(args["path"]))
        content = str(args.get("content", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"path": str(path.relative_to(self.root)), "bytes": len(content.encode("utf-8"))}

    async def run_command(self, args: dict[str, Any]) -> dict[str, Any]:
        command = args.get("command")
        if isinstance(command, list):
            argv = [str(x) for x in command]
        else:
            import shlex
            argv = shlex.split(str(command or ""), posix=os.name != "nt")
        if not argv or Path(argv[0]).name.lower() not in self.ALLOWED_COMMANDS:
            raise ValueError("command is not allowlisted")
        proc = await asyncio.create_subprocess_exec(*argv, cwd=self.root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=self.command_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {"exit_code": -1, "output": "command timed out"}
        return {"exit_code": proc.returncode, "output": out.decode("utf-8", errors="replace")[-20000:]}

    async def git_diff(self, args: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.run_command({"command": ["git", "diff", "--no-ext-diff", "--"]})

    def as_tools(self) -> dict[str, Any]:
        return {"read_file": self.read_file, "write_file": self.write_file, "run_command": self.run_command, "git_diff": self.git_diff}

    @staticmethod
    def specs() -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file inside the workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Create or replace a UTF-8 text file inside the workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "run_command", "description": "Run an allowlisted development command in the workspace.", "parameters": {"type": "object", "properties": {"command": {"type": ["string", "array"]}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "git_diff", "description": "Inspect current git diff.", "parameters": {"type": "object", "properties": {}}}},
        ]
