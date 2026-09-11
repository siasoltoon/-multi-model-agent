from dataclasses import dataclass
import os

from .adapters import OpenAICompatibleAdapter
from .dag import DagNode, TaskDag
from .discovery import env_api_key
from .models import Task
from .provider_api_catalog import get_api_contract
from .provider_connections import build_provider_adapter
from .provider_registry import get_provider
from .router import SmartRouter
from .subtask_dag import SubtaskDag, SubtaskPlanner


ROLE_PLAN = (
    ("analysis", "analyst", set()),
    ("architecture", "architect", {"analysis"}),
    ("coding", "coder", {"architecture"}),
    ("testing", "tester", {"coding"}),
    ("review", "reviewer", {"testing"}),
    ("security", "security_reviewer", {"review"}),
    ("repair", "repairer", {"security"}),
    ("verification", "verifier", {"repair"}),
)


@dataclass
class Plan:
    dag: TaskDag


class Planner:
    """Build the explicit role DAG consumed by the multi-model executor."""

    def plan(self, task: Task) -> Plan:
        dag = TaskDag()
        for node_id, role, dependencies in ROLE_PLAN:
            dag.add(DagNode(node_id, role, set(dependencies)))
        return Plan(dag)


class AgentEngine:
    def __init__(self, router: SmartRouter):
        self.router = router
        self.planner = Planner()

    def plan(self, task: Task) -> Plan:
        return self.planner.plan(task)

    def select_endpoint(self, *, min_context: int = 0, tools: bool = True):
        return self.router.choose(min_context=min_context, tools=tools)

    def select_role_endpoint(self, role: str, *, min_context: int = 0, tools: bool = True, task_type: str = "coding"):
        return self.router.choose(min_context=min_context, tools=tools, task_type=task_type, role=role)

    def _planner_adapter(self):
        endpoint = self.router.choose(min_context=4096, tools=False, task_type="coding", role="analysis")
        api_key = env_api_key(getattr(endpoint, "api_key_env", None)) or os.getenv(f"{endpoint.provider.upper()}_API_KEY", "") or os.getenv("AGENT_API_KEY", "")
        definition = get_provider(endpoint.provider)
        if definition.adapter in {"openai", "anthropic", "gemini"} or get_api_contract(endpoint.provider) is not None:
            return build_provider_adapter(endpoint.provider, model=endpoint.model, timeout=180, api_key=api_key), endpoint
        return OpenAICompatibleAdapter(endpoint.base_url, api_key, endpoint.model, timeout=180), endpoint

    async def decompose(self, task: Task, context: str = "") -> SubtaskDag:
        """Create a strict, validated subtask DAG using only a zero-cost routed endpoint."""
        adapter, _endpoint = self._planner_adapter()
        return await SubtaskPlanner(adapter).plan(task.prompt, context)
