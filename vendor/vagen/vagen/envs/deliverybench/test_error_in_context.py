"""
Test script to verify that error messages appear in the observation context.

This script simulates the training flow:
1. Reset environment
2. Execute an invalid action to trigger an error
3. Use "XXX" as a placeholder action (no actual model call)
4. Check if the error appears in the next observation's context

Run with: 
  python -m vagen.envs.deliverybench.test_error_in_context
  or
  cd /home/lingjun/SimWorld-RL/VAGEN-new && python -m vagen.envs.deliverybench.test_error_in_context
"""

import asyncio
import sys
from pathlib import Path

# Add VAGEN-new to path so we can import vagen
_vagen_new_dir = Path(__file__).parent.parent.parent.parent
if str(_vagen_new_dir) not in sys.path:
    sys.path.insert(0, str(_vagen_new_dir))

from vagen.envs.deliverybench import DeliveryBench


async def test_error_in_context():
    """Test that error messages appear in observation context after invalid actions."""
    
    print("=" * 80)
    print("Testing Error Message in Context")
    print("=" * 80)
    
    # Initialize environment (same config as training)
    cfg = {
        "render_mode": "text",
        "max_steps": 100,
        "prompt_format": "free_think",
        "map_name": "medium-city-22",
    }
    
    env = DeliveryBench(cfg)
    
    # Step 1: Reset environment
    print("\n[Step 1] Resetting environment...")
    obs, info = await env.reset(seed=42)
    print("✓ Environment reset")
    
    # Step 2: Get initial observation
    print("\n[Step 2] Initial observation:")
    print("-" * 80)
    initial_obs = obs["obs_str"]
    print(initial_obs)  # Print full observation, no truncation
    print("-" * 80)
    
    # Check if initial observation has error section
    has_error_initially = "### recent_error" in initial_obs or "recent_error" in initial_obs.lower()
    print(f"Initial observation contains error section: {has_error_initially}")
    
    # Step 3: Execute a valid action first (to establish baseline)
    print("\n[Step 3] Executing a valid action first (VIEW_ORDERS)...")
    valid_action = '{"action": "VIEW_ORDERS()"}'
    obs_valid, reward_valid, done_valid, info_valid = await env.step(valid_action)
    print(f"  Reward: {reward_valid}, Action error: {info_valid.get('action_error', 'None')}")
    
    # Step 4: Execute an invalid action to trigger an error
    print("\n[Step 4] Executing invalid action to trigger error...")
    
    # Test case 1: Invalid action format (will trigger parse error)
    print("\n--- Test Case 1: Invalid format (XXX) ---")
    invalid_action_1 = "XXX"  # This should trigger "No valid action parsed from response"
    print(f"Action: {invalid_action_1}")
    
    obs_after_error, reward, done, info_after_error = await env.step(invalid_action_1)
    
    # Check info for error
    action_error = info_after_error.get("action_error")
    print(f"\n[After invalid action]")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  Action error in info: {action_error}")
    
    # Step 5: Use "XXX" as placeholder action (simulating model output)
    print(f"\n[Step 5] Using 'XXX' as placeholder action (simulating model output)...")
    placeholder_action = "XXX"
    
    obs_with_error, reward2, done2, info2 = await env.step(placeholder_action)
    
    # Step 6: Check if error appears in observation context
    print(f"\n[Step 6] Checking if error appears in observation context...")
    print("-" * 80)
    obs_text = obs_with_error["obs_str"]
    
    # Print full observation (no truncation)
    print("Full observation text:")
    print(obs_text)
    print("-" * 80)
    
    # Check for error indicators
    has_recent_error_section = "### recent_error" in obs_text
    has_error_text = "error" in obs_text.lower() or "failed" in obs_text.lower()
    has_action_error = "action" in obs_text.lower() and "error" in obs_text.lower()
    
    print(f"\n[Results]")
    print(f"  Contains '### recent_error' section: {has_recent_error_section}")
    print(f"  Contains error-related text: {has_error_text}")
    print(f"  Contains action error mention: {has_action_error}")
    
    # Extract error section if present
    if "### recent_error" in obs_text:
        error_start = obs_text.find("### recent_error")
        error_end = obs_text.find("###", error_start + 1)
        if error_end == -1:
            error_end = len(obs_text)
        error_section = obs_text[error_start:error_end]
        print(f"\n[Error Section Found]")
        print("-" * 80)
        print(error_section)
        print("-" * 80)
    
    # Final verdict for test case 1
    if has_recent_error_section or (has_error_text and action_error):
        print(f"\n✓ SUCCESS: Error message appears in observation context!")
    else:
        print(f"\n✗ FAILED: Error message NOT found in observation context")
        print(f"   Expected error: {action_error}")
    
    print("\n" + "=" * 80)
    
    # Test case 2: Valid format but invalid action (execution error)
    print("\n--- Test Case 2: Valid format but invalid action (DROP_OFF without customer) ---")
    print("Resetting environment for test case 2...")
    obs, info = await env.reset(seed=42)
    
    invalid_action_2 = '{"action": "DROP_OFF(method=\\"hand_to_customer\\")"}'
    print(f"Action: {invalid_action_2}")
    
    obs_after_error2, reward2, done2, info_after_error2 = await env.step(invalid_action_2)
    action_error2 = info_after_error2.get("action_error")
    print(f"  Action error in info: {action_error2}")
    
    # Use "XXX" as placeholder
    print(f"\nUsing 'XXX' as placeholder action...")
    obs_with_error2, reward3, done3, info3 = await env.step("XXX")
    
    obs_text2 = obs_with_error2["obs_str"]
    has_recent_error_section2 = "### recent_error" in obs_text2
    
    # Print full observation (no truncation)
    print(f"\n[Full observation text for Test Case 2]")
    print("-" * 80)
    print(obs_text2)
    print("-" * 80)
    
    print(f"\n[Results for Test Case 2]")
    print(f"  Contains '### recent_error' section: {has_recent_error_section2}")
    
    if "### recent_error" in obs_text2:
        error_start = obs_text2.find("### recent_error")
        error_end = obs_text2.find("###", error_start + 1)
        if error_end == -1:
            error_end = len(obs_text2)
        error_section = obs_text2[error_start:error_end]
        print(f"\n[Error Section Found]")
        print("-" * 80)
        print(error_section)
        print("-" * 80)
        print(f"\n✓ SUCCESS: Error message appears in observation context!")
    else:
        print(f"\n✗ FAILED: Error message NOT found in observation context")
        print(f"   Expected error: {action_error2}")
    
    # Cleanup
    await env.close()
    print("\n✓ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_error_in_context())
