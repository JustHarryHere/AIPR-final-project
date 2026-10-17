"""Calculation tools for lesson plan study assistant."""


def calc_required_score(
    midterm: float,
    w_mid: float,
    w_final: float,
    target: float,
) -> dict:
    """Calculate the final-exam score needed to reach a target course grade."""
    if not (0 <= midterm <= 10 and 0 <= target <= 10):
        return {"ok": False, "error": "Scores must be between 0 and 10."}

    if w_mid < 0 or w_final <= 0:
        return {"ok": False, "error": "Weights must be non-negative and w_final > 0."}

    if abs(w_mid + w_final - 1.0) > 1e-4:
        return {"ok": False, "error": "Weights must sum to 1.0."}

    req = (target - midterm * w_mid) / w_final
    req_round = round(req, 2)

    return {
        "ok": True,
        "data": {
            "required_final_score": req_round,
            "target": target,
            "reachable": 0 <= req <= 10,
        },
    }


def calc_workload(
    hours_per_week: float,
    from_week: int,
    to_week: int,
) -> dict:
    """Calculate total self-study hours over an inclusive range of weeks."""
    if hours_per_week < 0:
        return {"ok": False, "error": "hours_per_week cannot be negative."}

    if from_week < 1 or to_week < from_week:
        return {"ok": False, "error": "Invalid week range: from_week must be >= 1 and <= to_week."}

    num_weeks = to_week - from_week + 1
    total_hours = round(hours_per_week * num_weeks, 2)

    return {
        "ok": True,
        "data": {
            "from_week": from_week,
            "to_week": to_week,
            "total_hours": total_hours,
        },
    }
