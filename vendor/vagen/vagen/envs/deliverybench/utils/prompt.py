"""
Prompt templates for DeliveryBench environment.
"""

from typing import Optional


def system_prompt() -> str:
    """
    Get the base system prompt for DeliveryBench.
    This describes the environment and available actions.
    """
    return """You are a food-delivery courier in a simulated city. Your primary goal is to earn as much money as possible through delivering food to customers. Do your best to complete each delivery and don't let your energy drop to 0.

**Action space:**
COMMANDS (UPPERCASE):
- VIEW_ORDERS()     # view all available orders
- VIEW_BAG()        # view your bag
- ACCEPT_ORDER(order_id) or ACCEPT_ORDER([order_id, ...])
- MOVE(x, y)  # x,y in meters with 'm' suffix
- MOVE(x, y, pace="accel"|"normal"|"decel")
- PICKUP(orders=[12, 18])
- PLACE_FOOD_IN_BAG(bag_cmd="order 12: 1,2 -> A; 3 -> B")
- CHARGE(target_pct=100)
- WAIT(seconds=NN) or WAIT("charge_done")
- REST(target_pct=100)
- BUY(item_id="energy_drink", qty=1)
- USE_BATTERY_PACK()
- USE_ENERGY_DRINK()
- USE_ICE_PACK(comp="A")
- USE_HEAT_PACK(comp="B")
- SWITCH(to="walk"|"e-scooter"|"car"|"drag_scooter")
- RENT_CAR()
- RETURN_CAR()
- DROP_OFF(oid=<int>, method="leave_at_door|knock|call|hand_to_customer")
- BOARD_BUS(bus_id="bus_id", target_stop_id="target_stop_id")
- VIEW_BUS_SCHEDULE()
- TURN_AROUND(angle=60, direction="left"|"right")
- STEP_FORWARD()

**Rules:**
- Prioritize viewing and accepting new orders when you don't have any active orders.
- PICKUP can only happen at the store's pickup door and only when food is ready.
- DROP_OFF your orders only when you are at the dropoff address.
- Movement is the most important part. Use MOVE to navigate between locations.
- Movement consumes energy. REST at rest areas to restore energy.
- E-scooter consumes battery. CHARGE at charging stations.
- Walking speed is 2 m/s, e-scooter is 6 m/s, car is 12 m/s."""


def format_prompt(
    max_actions_per_step: int = 1,
    action_sep: str = ",",
    add_example: bool = True,
    prompt_format: str = "free_think",
) -> str:
    """
    Get the format instruction for the model's response.

    Args:
        max_actions_per_step: Maximum number of actions per response
        action_sep: Separator between multiple actions
        add_example: Whether to include example in the prompt
        prompt_format: "free_think" or "wm" format

    Returns:
        Format instruction string
    """
    if prompt_format == "wm":
        format_str = """**Output Format:**
Return your response in the following format:
<think>Your reasoning about the current situation and what action to take</think>
<answer>YOUR_ACTION_HERE</answer>

Example actions: VIEW_ORDERS(), MOVE(102.3m, 885.5m), ACCEPT_ORDER(12), PICKUP(orders=[12])"""
    else:
        format_str = """**Output Format:**
Return ONLY a valid JSON object with the following keys:
{
"reasoning_and_reflection": "<100 tokens: summarize situation and reasoning>",
"action": "Your next action as a single-line function call",
"future_plan": "<100 tokens: your plan for subsequent steps>"
}

Example actions: VIEW_ORDERS(), MOVE(102.3m, 885.5m), ACCEPT_ORDER(12), PICKUP(orders=[12])"""

    return format_str


def init_observation_template(obs_text: str) -> str:
    """
    Format the initial observation after reset.

    Args:
        obs_text: The observation text from the environment

    Returns:
        Formatted initial observation string
    """
    return f"""**Initial State:**
{obs_text}

What is your first action?"""


def action_template(valid_actions: list, obs_text: str) -> str:
    """
    Format the observation after an action step.

    Args:
        valid_actions: List of actions that were executed
        obs_text: The observation text from the environment

    Returns:
        Formatted step observation string
    """
    if valid_actions:
        actions_str = ", ".join(valid_actions) if isinstance(valid_actions, list) else str(valid_actions)
        return f"""**Previous Action:** {actions_str}

**Current State:**
{obs_text}

What is your next action?"""
    else:
        return f"""**Current State:**
{obs_text}

What is your next action?"""
