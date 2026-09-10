from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_steps: int | None = Field(default=None, ge=1, le=1000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    prompt: str
    status: TaskStatus = TaskStatus.QUEUED
    max_steps: int = 64
    current_step: int = 0
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
