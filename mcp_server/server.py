from mcp.server.mcpserver import MCPServer

mcp = MCPServer(
    "study-assistant",
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

if __name__ == "__main__":
    mcp.run()  