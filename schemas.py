import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Optional, TypedDict


class SearchResult(TypedDict):
    """One retrieved passage from a lesson-plan document"""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    version: str
    source: str


@dataclass
class Chunk:
    """Document chunk with metadata"""
    content: str
    chunk_id: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


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
    """Standard result produced by an agent"""
    answer: str
    status: Literal["answered", "no_answered", "insufficient_data", "error"]
    sources: list[SearchResult]
    usage: Usage


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Optional[Callable] = None

    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Action:
    type: str
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
        return None
