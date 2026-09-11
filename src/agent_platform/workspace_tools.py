from __future__ import annotations

import asyncio
import os
import re
import shlex
from pathlib import Path
from typing import Any


class WorkspaceTools:
    """Bounded workspace tools for an execution worker."""

    ALLOWED_COMMANDS = {
        "python", "pytest", "pip", "npm", "node", "git", "uv", "ruff",
        "pwd", "ls", "find", "cat", "head", "tail", "grep", "rg", "sed",
        "awk", "wc", "sort", "diff", "file",
    }
    BLOCKED_ARGS = {"--system", "--global", "--user", "--break-system-packages"}
    SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "PRIVATE_KEY", "AUTH")
    SHELL_OPERATORS = (";", ">", "<", "`", "$(", "${", "\n", "\r")
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

    @classmethod
    def _split_chain(cls, command: Any) -> list[list[list[str]]]:
        """Parse safe command groups supporting && and allowlisted pipelines."""
        if isinstance(command, list):
            argv = [str(x) for x in command]
            if not argv:
                raise ValueError("empty command")
            return [[argv]]
        raw = str(command or "")
        if not raw.strip():
            raise ValueError("empty command")
        if any(operator in raw for operator in cls.SHELL_OPERATORS):
            raise ValueError("command contains a shell operator")
        # Pipelines are executed with direct subprocess pipes, never a shell.
        raw = re.sub(r"\s*\|\s*", "|", raw)
        groups: list[list[list[str]]] = []
        for group_text in raw.split("&&"):
            group_text = group_text.strip()
            if not group_text:
                raise ValueError("invalid command chain")
            pipeline: list[list[str]] = []
            for segment in group_text.split("|"):
                segment = segment.strip()
                if not segment:
                    raise ValueError("invalid command pipeline")
                try:
                    argv = shlex.split(segment, posix=True)
                except ValueError as exc:
                    raise ValueError("invalid command syntax") from exc
                if not argv:
                    raise ValueError("invalid command pipeline")
                pipeline.append(argv)
            groups.append(pipeline)
        return groups

    @classmethod
    def _argv_chain(cls, command: Any) -> list[list[list[str]]]:
        groups = cls._split_chain(command)
        for pipeline in groups:
            for argv in pipeline:
                if not argv or Path(argv[0]).name.lower() not in cls.ALLOWED_COMMANDS:
                    raise ValueError("command is not allowlisted")
                if any(arg in cls.BLOCKED_ARGS for arg in argv[1:]):
                    raise ValueError("command contains a blocked package-management option")
        return groups

    @classmethod
    def _argv(cls, command: Any) -> list[str]:
        groups = cls._argv_chain(command)
        if len(groups) != 1 or len(groups[0]) != 1:
            raise ValueError("command chain is not valid for single-command validation")
        return groups[0][0]

    @classmethod
    def _safe_environment(cls) -> dict[str, str]:
        safe: dict[str, str] = {}
        for key, value in os.environ.items():
            upper = key.upper()
            if any(marker in upper for marker in cls.SECRET_ENV_MARKERS):
                continue
            safe[key] = value
        return safe

    async def _run_pipeline(self, pipeline: list[list[str]]) -> tuple[int, str]:
        processes: list[asyncio.subprocess.Process] = []
        previous_stdout = None
        try:
            for argv in pipeline:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=self.root,
                    stdin=previous_stdout if previous_stdout is not None else asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=self._safe_environment(),
                )
                processes.append(proc)
                previous_stdout = proc.stdout
            last = processes[-1]
            try:
                out, _ = await asyncio.wait_for(last.communicate(), timeout=self.command_timeout)
                upstream_codes = await asyncio.wait_for(
                    asyncio.gather(*(proc.wait() for proc in processes[:-1])),
                    timeout=self.command_timeout,
                )
            except asyncio.TimeoutError:
                for proc in processes:
                    if proc.returncode is None:
                        proc.kill()
                await asyncio.gather(*(proc.wait() for proc in processes), return_exceptions=True)
                return -1, "command timed out"
            codes = [*upstream_codes, last.returncode]
            code = next((item for item in reversed(codes) if item != 0), 0)
            return int(code), out.decode("utf-8", errors="replace")
        finally:
            for proc in processes:
                if proc.returncode is None:
                    proc.kill()

    async def run_command(self, args: dict[str, Any]) -> dict[str, Any]:
        groups = self._argv_chain(args.get("command"))
        chunks: list[str] = []
        exit_code = 0
        for pipeline in groups:
            code, output = await self._run_pipeline(pipeline)
            exit_code = code
            if output:
                chunks.append(output)
            if code != 0:
                break
        return {"exit_code": exit_code, "output": "".join(chunks)[-self.MAX_COMMAND_OUTPUT:]}

    async def git_diff(self, args: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.run_command({"command": ["git", "diff", "--no-ext-diff", "--"]})

    def as_tools(self) -> dict[str, Any]:
        return {"read_file": self.read_file, "write_file": self.write_file, "run_command": self.run_command, "git_diff": self.git_diff}

    @staticmethod
    def specs() -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file inside the workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Create or replace a UTF-8 text file inside the workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "run_command", "description": "Run safe allowlisted development commands in the workspace. Supports && chains and allowlisted pipelines, never a shell.", "parameters": {"type": "object", "properties": {"command": {"type": ["string", "array"]}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "git_diff", "description": "Inspect current git diff.", "parameters": {"type": "object", "properties": {}}}},
        ]
