import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Optional, TypedDict


class SearchResult(TypedDict):
    """One retrived passage from a lesson-plan document"""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    version: str
    source: str


class ToolResult(TypedDict):
    ok: bool
    data: any
    error: str | None


class Usage(TypedDict):
    """LLM usage for one evaluated question"""
    calls: int
    prompt_tokens: int
    completion_tokens: int
    max_prompt: int


class AgentResult(TypedDict):
    """standard result produced by an agent"""
    answer: str
    status: Literal[
        "answered",
        "no_answered",
        "ifficient_data",
        "error",
    ]
    sources: list[SearchResult]
    usage: Usage


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                 # JSON Schema
    fn: Optional[Callable] = None

    def schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}


@dataclass
class Action:
    type: str                        # "tool" hoặc "final"
    tool: Optional[str] = None
    args: dict = field(default_factory=dict)
    answer: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_json_block(raw: str) -> str:
    """Strip ```json fences that models often add."""
    return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def parse_action(text: str) -> Optional[Action]:
    try:
        data = json.loads(parse_json_block(text))
        return Action(**data)
    except Exception:
        return None                  # thiếu ngoặc cũng không làm chết run


if __name__ == "__main__":
    usage: Usage = {
        "calls": 1,
        "prompt_tokens": 150,
        "completion_tokens": 45,
        "max_prompt": 150,
    }
    assert usage["calls"] == 1

    tool = Tool(
        name="search_lesson_plan",
        description="Tìm kiếm trong lesson plan",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    assert tool.schema()["function"]["name"] == "search_lesson_plan"

    action = parse_action('{"type": "tool", "tool": "search_lesson_plan", "args": {"query": "AIPR"}}')
    assert action is not None
    assert action.type == "tool"
    assert action.tool == "search_lesson_plan"

    final_action = parse_action('```json\n{"type": "final", "answer": "Done"}\n```')
    assert final_action is not None
    assert final_action.type == "final"
    assert final_action.answer == "Done"

    assert parse_action("invalid json") is None

    res: ToolResult = {"ok": True, "data": "OK: added id=1", "error": None}
    assert res["ok"] is True
