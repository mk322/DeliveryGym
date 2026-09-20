"""
Utility functions for DeliveryBench environment.
"""

import re
import json
from typing import Dict, Any, List, Optional


def parse_response(
    response: str,
    action_sep: str = ",",
    max_actions: int = 1,
    prompt_format: str = "free_think",
) -> Dict[str, Any]:
    """
    Parse the model's response to extract actions.

    Args:
        response: The raw model response string
        action_sep: Separator between actions (not used for DeliveryBench)
        max_actions: Maximum actions allowed (not used for DeliveryBench)
        prompt_format: "free_think" or "wm" format

    Returns:
        Dict containing:
            - action: The parsed action string
            - format_correct: Whether the format was correct
            - reasoning: The reasoning/thinking portion (if any)
            - plan: The future plan (if any)
    """
    result = {
        "action": None,
        "format_correct": False,
        "reasoning": None,
        "plan": None,
    }

    if not response:
        return result

    response = response.strip()

    # Try parsing as JSON first (free_think format)
    try:
        obj = json.loads(response)
        if _fill_result_from_action_json(result, obj):
            return result
    except json.JSONDecodeError:
        pass

    # Try parsing with <think> and <answer> tags (wm format)
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)

    if answer_match:
        result["action"] = answer_match.group(1).strip()
        result["format_correct"] = True
        if think_match:
            result["reasoning"] = think_match.group(1).strip()
        return result

    # Qwen-style free-think output may prepend prose / </think> before the JSON
    # object. Prefer the embedded JSON action over regex-scanning the prose,
    # otherwise words in reasoning/future_plan can be mistaken for the action.
    embedded_obj = _extract_embedded_action_json(response)
    if _fill_result_from_action_json(result, embedded_obj):
        return result

    # Try to extract action from code fence
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            obj = json.loads(inner)
            if _fill_result_from_action_json(result, obj):
                return result
        except json.JSONDecodeError:
            # Not JSON, treat as raw action
            if _looks_like_action(inner):
                result["action"] = inner
                result["format_correct"] = True
                return result

    # Fallback: try to find a function call pattern in the response
    action = _extract_action_call(response)
    if action:
        result["action"] = action
        result["format_correct"] = True
        return result

    # Last resort: use the first line
    first_line = response.split("\n")[0].strip()
    if _looks_like_action(first_line):
        result["action"] = first_line
        result["format_correct"] = True

    return result


def _fill_result_from_action_json(result: Dict[str, Any], obj: Any) -> bool:
    """Populate parse result from a JSON object with an action field."""
    if not isinstance(obj, dict):
        return False

    action = obj.get("action", "")
    if isinstance(action, dict):
        action = action.get("action_call", "")
    if not isinstance(action, str) or not action.strip():
        return False

    result["action"] = action.strip()
    result["format_correct"] = True
    result["reasoning"] = obj.get("reasoning_and_reflection", "")
    result["plan"] = obj.get("future_plan", "")
    return True


def _extract_embedded_action_json(text: str) -> Optional[Dict[str, Any]]:
    """Find a trailing/embedded JSON object that contains the action field."""
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", text or ""))):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action" in obj:
            return obj
    return None


def _looks_like_action(text: str) -> bool:
    """Check if text looks like a DeliveryBench action."""
    action_pattern = r"^[A-Z_]+\s*\("
    return bool(re.match(action_pattern, text.strip()))


def _extract_action_call(text: str) -> Optional[str]:
    """
    Extract a function call pattern from text.
    Matches patterns like: ACTION_NAME(...) or ACTION_NAME()
    """
    # Pattern for function calls with balanced parentheses
    patterns = [
        r"(VIEW_ORDERS\s*\([^)]*\))",
        r"(VIEW_BAG\s*\([^)]*\))",
        r"(ACCEPT_ORDER\s*\([^)]*\))",
        r"(NAVIGATE\s*\([^)]*\))",
        r"(MOVE\s*\([^)]*\))",
        r"(PICKUP\s*\([^)]*\))",
        r"(PLACE_FOOD_IN_BAG\s*\([^)]*\))",
        r"(CHARGE\s*\([^)]*\))",
        r"(WAIT\s*\([^)]*\))",
        r"(REST\s*\([^)]*\))",
        r"(BUY\s*\([^)]*\))",
        r"(USE_BATTERY_PACK\s*\([^)]*\))",
        r"(USE_ENERGY_DRINK\s*\([^)]*\))",
        r"(USE_ICE_PACK\s*\([^)]*\))",
        r"(USE_HEAT_PACK\s*\([^)]*\))",
        r"(SWITCH\s*\([^)]*\))",
        r"(RENT_CAR\s*\([^)]*\))",
        r"(RETURN_CAR\s*\([^)]*\))",
        r"(DROP_OFF\s*\([^)]*\))",
        r"(BOARD_BUS\s*\([^)]*\))",
        r"(VIEW_BUS_SCHEDULE\s*\([^)]*\))",
        r"(TURN_AROUND\s*\([^)]*\))",
        r"(STEP_FORWARD\s*\([^)]*\))",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    # Generic pattern for any function call
    generic_match = re.search(r"([A-Z][A-Z_]*\s*\([^)]*\))", text)
    if generic_match:
        return generic_match.group(1)

    return None
