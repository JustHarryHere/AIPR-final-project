from typing import Any, Literal, TypedDict

class SearchResult(TypedDict):
    """One retrived passage from a lesson-plan document"""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    version: str
    source: str

class ToolResult(TypedDict):
    ok : bool
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