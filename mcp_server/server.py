import random
import sys
import os
import json
from pathlib import Path
import uuid #tạo ID riêng cho mỗi mục học

DATA_DIR=Path(__file__).resolve().parents[1] / "data"
COURSE_CATALOG_FILE = DATA_DIR / "course_catalog.json"
STUDY_PLAN_FILE = DATA_DIR / "study_plan.json"

from mcp.server.mcpserver import MCPServer
from mcp import StdioServerParameters, Client

mcp = MCPServer("study-assistant",
    description="MCP Server for a lesson-plan assistant",
)

@mcp.tool()
def calc_required_score(
    midterm: float,
    w_mid: float,
    w_final: float,
    target: float,
)->dict:
    """Calculate the final-exam scores needed to reach a target score grade.
     Scores are on a 0-10 scale.
     Weights are proportions, e.g 0.4 and 0.6
    """
    if not all(
        0 <= value <=10
        for value in(midterm, target)
    ):
        return {"ok": False, "error": "Scores must be between 0 and 10."}
    
    if w_mid <0 or w_final <=0:
        return {"ok": False, "error": "Weights must be non-negative and final weight must be positive"}

    if abs(w_mid + w_final -1.0) > 1e-6:
        return {"ok": False, "error":" Weights must add up to 1."}

    required = (target - midterm * w_mid) / w_final

    return {
        "ok": True,
        "data": {
            "required_final_score": round(required, 2),
            "target": target,
            "reachable": 0 <= required <=10,
        },
    }

@mcp.tool()
def calc_workload(
    hours_per_week:float,
    from_week: int,
    to_week:int,
)-> dict:
    """Calculate total study hours over an inclusive range of weeks."""
    if hours_per_week < 0:
        return {"ok": False, "error": "hours_per_week cannot be negative"}

    if from_week < 1 or to_week <from_week:
        return{"ok": False, "error": "Invalid week range."}

    number_of_weeks = to_week - from_week + 1
    total_hours =  hours_per_week * number_of_weeks

    return{
        "ok": True,
        "data": {
            "from_week": from_week,
            "to_week": to_week,
            "total_hours": round(total_hours, 2),
        }
    }
@mcp.tool()
def get_course_info(course_id: str)-> dict:
    """Get course metadata and grading weights by course ID """

    if not COURSE_CATALOG_FILE.exists():
        return {"error": "Course catalog file not found"}

    with open(COURSE_CATALOG_FILE,"r", encoding="utf-8")as file:
        catalog =json.load(file)
    course = catalog.get(course_id)

    if course is None:
        return {"error": f"Course not found: {course_id}"}

    return course


@mcp.tool()
def create_study_plan_item(
    student_id: str,
    course: str,
    week: int,
    topic: str,
    hours: float,
) -> dict:
    """Create a study plan item and verify that it was saved."""

    if not student_id.strip() or not course.strip() or not topic.strip():
        return {"ok": False, "error": "Student, course, and topic must not be empty."}

    if week < 1:
        return {"ok": False, "error": "Week must be at least 1."}

    if hours <= 0:
        return {"ok": False, "error": "Hours must be greater than 0."}

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if STUDY_PLAN_FILE.exists():
        with open(STUDY_PLAN_FILE, "r", encoding="utf-8") as file:
            study_plan = json.load(file)
    else:
        study_plan = []

    if not isinstance(study_plan, list):
        return {"ok": False, "error": "Invalid study plan format."}

    for item in study_plan:
        if (
            item.get("student_id") == student_id
            and item.get("course") == course
            and item.get("week") == week
            and item.get("topic") == topic
        ):
            return {"ok": False, "error": "Study plan item already exists."}

    new_item = {
        "id": str(uuid.uuid4()),
        "student_id": student_id,
        "course": course,
        "week": week,
        "topic": topic,
        "hours": hours,
    }

    study_plan.append(new_item)

    with open(STUDY_PLAN_FILE, "w", encoding="utf-8") as file:
        json.dump(study_plan, file, ensure_ascii=False, indent=4)

    # Read the file again to verify the saved item.
    with open(STUDY_PLAN_FILE, "r", encoding="utf-8") as file:
        saved_plan = json.load(file)

    for item in saved_plan:
        if item.get("id") == new_item["id"]:
            return {
                "ok": True,
                "data": item,
                "verified": True,
            }

    return {
        "ok": False,
        "error": "Study plan item could not be verified after saving.",
    }

@mcp.tool()
def search_lesson_plan(query: str, mode: str = "hybrid") -> dict:
    """Search lesson-plan documents using dense or hybrid retrieval."""

    if not query.strip():
        return {
            "ok": False,
            "error": "Search query must not be empty.",
        }

    if mode not in ("dense", "hybrid"):
        return {
            "ok": False,
            "error": "Invalid mode. Use 'dense' or 'hybrid'.",
        }

    try:
        from rag.retriever import search_lesson_plan as _search
        results = _search(query=query, mode=mode)
        return {
            "ok": True,
            "query": query,
            "mode": mode,
            "count": len(results),
            "results": results,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Retrieval failed: {exc}",
            "results": [],
        }

@mcp.tool()
def list_study_plan(student_id: str) -> dict:
    """List study plan items for a student."""

    if not student_id.strip():
        return {"ok": False, "error": "student_id must not be empty."}

    if not STUDY_PLAN_FILE.exists():
        return {"ok": True, "data": [], "count": 0}

    with open(STUDY_PLAN_FILE, "r", encoding="utf-8") as file:
        study_plan = json.load(file)

    if not isinstance(study_plan, list):
        return {"ok": False, "error": "Invalid study plan format."}

    student_items = [
        item for item in study_plan
        if item.get("student_id") == student_id
    ]

    return {
        "ok": True,
        "data": student_items,
        "count": len(student_items),
    }


if __name__ == "__main__":
    mcp.run()  