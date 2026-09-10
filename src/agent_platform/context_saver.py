from __future__ import annotations

import hashlib
import os
import re
from typing import Any


_DEFAULT_TARGET = 0.85
_DEFAULT_KEEP_RECENT = 8
_DEFAULT_MAX_TOOL_CHARS = 6000


class ContextSaver:
    """Loss-aware context compactor for long coding-agent runs.

    It never removes the task/system instructions or recent turns. Older tool
    results are compacted by keeping high-signal lines (errors, test failures,
    file paths and command summaries), de-duplicating repeated results, and
    applying a bounded size. The target is an 85% reduction of *historical tool
    output*, not a false promise about total provider billing tokens.
    """

    def __init__(
        self,
        *,
        target_reduction: float | None = None,
        keep_recent: int | None = None,
        max_tool_chars: int | None = None,
    ) -> None:
        self.target_reduction = max(
            0.80,
            min(0.90, float(target_reduction if target_reduction is not None else os.getenv("AGENT_CONTEXT_TARGET_REDUCTION", _DEFAULT_TARGET))),
        )
        self.keep_recent = max(4, int(keep_recent if keep_recent is not None else os.getenv("AGENT_CONTEXT_KEEP_RECENT", _DEFAULT_KEEP_RECENT)))
        self.max_tool_chars = max(1500, int(max_tool_chars if max_tool_chars is not None else os.getenv("AGENT_CONTEXT_MAX_TOOL_CHARS", _DEFAULT_MAX_TOOL_CHARS)))

    @staticmethod
    def _is_tool(message: dict[str, Any]) -> bool:
        return message.get("role") == "tool"

    @staticmethod
    def _compact_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        lines = text.splitlines()
        if not lines:
            return text[:limit]

        important = re.compile(
            r"(error|exception|traceback|failed|failure|test|assert|warning|todo|fixme|fatal|panic|\bpass\b|\bpath\b|/|\\)",
            re.IGNORECASE,
        )
        selected: list[str] = []
        seen: set[str] = set()
        for line in lines:
            normalized = " ".join(line.split())
            if not normalized or normalized in seen:
                continue
            if important.search(normalized):
                selected.append(normalized)
                seen.add(normalized)

        head = [line.strip() for line in lines[:8] if line.strip()]
        tail = [line.strip() for line in lines[-8:] if line.strip()]
        parts: list[str] = []
        for line in head + selected[:80] + tail:
            if line and line not in parts:
                parts.append(line)
        compact = "\n".join(parts)
        if len(compact) > limit:
            compact = compact[: max(0, limit - 64)].rstrip() + "\n[…context compacted…]"
        return compact

    def compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(messages) <= self.keep_recent:
            return list(messages)

        split = len(messages) - self.keep_recent
        prefix = list(messages[:split])
        recent = list(messages[split:])
        seen_hashes: set[str] = set()
        result: list[dict[str, Any]] = []

        for message in prefix:
            item = dict(message)
            if self._is_tool(item):
                content = str(item.get("content", ""))
                digest = hashlib.sha256(content.encode("utf-8", "ignore")).hexdigest()[:12]
                if digest in seen_hashes:
                    item["content"] = f"[repeated tool result omitted; hash={digest}]"
                else:
                    seen_hashes.add(digest)
                    item["content"] = self._compact_text(content, self.max_tool_chars)
            result.append(item)

        # Keep system/task messages intact even when they are old.
        for i, item in enumerate(result):
            if item.get("role") in {"system", "user"}:
                result[i] = dict(item)

        return result + recent

    @staticmethod
    def estimate_chars(messages: list[dict[str, Any]]) -> int:
        return sum(len(str(m.get("content", ""))) for m in messages)

    def savings_ratio(self, before: list[dict[str, Any]], after: list[dict[str, Any]]) -> float:
        old = self.estimate_chars(before)
        if old <= 0:
            return 0.0
        return max(0.0, 1.0 - self.estimate_chars(after) / old)
