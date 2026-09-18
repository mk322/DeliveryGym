"""
Test cases for DeliveryBench environment integration.

Run with: python -m vagen.envs.deliverybench.test_deliverybench
"""

import asyncio
import sys
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        from vagen.envs.deliverybench import DeliveryBench, DeliveryBenchEnvConfig
        print("  [PASS] DeliveryBench imports successful")
    except ImportError as e:
        print(f"  [FAIL] DeliveryBench import failed: {e}")
        return False

    try:
        from vagen.envs.deliverybench.utils.prompt import (
            system_prompt,
            format_prompt,
            init_observation_template,
            action_template,
        )
        print("  [PASS] Prompt utils imports successful")
    except ImportError as e:
        print(f"  [FAIL] Prompt utils import failed: {e}")
        return False

    try:
        from vagen.envs.deliverybench.utils.utils import parse_response
        print("  [PASS] Utils imports successful")
    except ImportError as e:
        print(f"  [FAIL] Utils import failed: {e}")
        return False

    return True


def test_config():
    """Test configuration dataclass."""
    print("\nTesting configuration...")

    from vagen.envs.deliverybench import DeliveryBenchEnvConfig

    # Test default config
    config = DeliveryBenchEnvConfig()
    assert config.max_steps == 100, f"Expected max_steps=100, got {config.max_steps}"
    assert config.render_mode == "text", f"Expected render_mode='text', got {config.render_mode}"
    print("  [PASS] Default config works")

    # Test custom config
    config = DeliveryBenchEnvConfig(
        max_steps=50,
        render_mode="vision",
        map_name="small-city",
    )
    assert config.max_steps == 50
    assert config.render_mode == "vision"
    assert config.map_name == "small-city"
    print("  [PASS] Custom config works")

    return True


def test_parse_response():
    """Test response parsing."""
    print("\nTesting response parsing...")

    from vagen.envs.deliverybench.utils.utils import parse_response

    # Test JSON format
    response = '{"action": "VIEW_ORDERS()", "reasoning_and_reflection": "checking orders", "future_plan": "accept one"}'
    result = parse_response(response)
    assert result["action"] == "VIEW_ORDERS()", f"Expected VIEW_ORDERS(), got {result['action']}"
    assert result["format_correct"] == True
    print("  [PASS] JSON format parsing works")

    # Test <think><answer> format
    response = "<think>I should view orders first</think><answer>VIEW_ORDERS()</answer>"
    result = parse_response(response)
    assert result["action"] == "VIEW_ORDERS()", f"Expected VIEW_ORDERS(), got {result['action']}"
    assert result["format_correct"] == True
    print("  [PASS] Think/answer format parsing works")

    # Test raw action
    response = 'MOVE(direction="forward")'
    result = parse_response(response)
    assert result["action"] == 'MOVE(direction="forward")', f"Expected MOVE(...), got {result['action']}"
    assert result["format_correct"] == True
    print("  [PASS] Raw action parsing works")

    # Test ACCEPT_ORDER
    response = "ACCEPT_ORDER(12)"
    result = parse_response(response)
    assert "ACCEPT_ORDER" in result["action"], f"Expected ACCEPT_ORDER, got {result['action']}"
    print("  [PASS] ACCEPT_ORDER parsing works")

    # Test empty response
    response = ""
    result = parse_response(response)
    assert result["action"] is None
    assert result["format_correct"] == False
    print("  [PASS] Empty response handling works")

    return True


def test_prompt_templates():
    """Test prompt template functions."""
    print("\nTesting prompt templates...")

    from vagen.envs.deliverybench.utils.prompt import (
        system_prompt,
        format_prompt,
        init_observation_template,
        action_template,
    )

    # Test system prompt
    sys_prompt = system_prompt()
    assert "food-delivery courier" in sys_prompt
    assert "MOVE" in sys_prompt
    assert "VIEW_ORDERS" in sys_prompt
    print("  [PASS] System prompt contains expected content")

    # Test format prompt
    fmt_prompt = format_prompt(prompt_format="free_think")
    assert "JSON" in fmt_prompt
    print("  [PASS] Format prompt (free_think) works")

    fmt_prompt = format_prompt(prompt_format="wm")
    assert "<think>" in fmt_prompt
    assert "<answer>" in fmt_prompt
    print("  [PASS] Format prompt (wm) works")

    # Test init observation template
    obs = init_observation_template("Test observation")
    assert "Test observation" in obs
    assert "Initial" in obs or "first" in obs.lower()
    print("  [PASS] Init observation template works")

    # Test action template
    obs = action_template(["VIEW_ORDERS()"], "Current state text")
    assert "VIEW_ORDERS()" in obs
    assert "Current state text" in obs
    print("  [PASS] Action template works")

    return True


def test_deliverybench_base_dir():
    """Test that DeliveryBench base directory exists."""
    print("\nTesting DeliveryBench base directory...")

    from vagen.envs.deliverybench import DeliveryBenchEnvConfig

    config = DeliveryBenchEnvConfig()
    base_dir = Path(config.base_dir)

    print(f"  [INFO] Default base_dir: {base_dir}")

    if not base_dir.exists():
        print(f"  [FAIL] Base directory does not exist: {base_dir}")
        return False

    # Check for required subdirectories
    required_paths = [
        "vlm_delivery",
        "maps",
    ]

    all_found = True
    for subpath in required_paths:
        if not (base_dir / subpath).exists():
            print(f"  [FAIL] Missing subdirectory: {subpath}")
            all_found = False
        else:
            print(f"  [PASS] Found: {subpath}")

    if all_found:
        print(f"  [PASS] Base directory has required structure")
    return all_found


async def test_env_creation():
    """Test environment creation."""
    print("\nTesting environment creation...")

    from vagen.envs.deliverybench import DeliveryBench, DeliveryBenchEnvConfig

    config = DeliveryBenchEnvConfig()
    base_dir = Path(config.base_dir)

    if not base_dir.exists():
        print("  [SKIP] Skipping env creation test - base_dir not found")
        return True

    try:
        env = DeliveryBench({
            "base_dir": str(base_dir),
            "max_steps": 10,
            "render_mode": "text",
        })
        print("  [PASS] Environment created successfully")
    except Exception as e:
        print(f"  [FAIL] Environment creation failed: {e}")
        return False

    return True


async def test_env_reset():
    """Test environment reset."""
    print("\nTesting environment reset...")

    from vagen.envs.deliverybench import DeliveryBench, DeliveryBenchEnvConfig

    config = DeliveryBenchEnvConfig()
    base_dir = Path(config.base_dir)

    if not base_dir.exists():
        print("  [SKIP] Skipping reset test - base_dir not found")
        return True

    try:
        env = DeliveryBench({
            "base_dir": str(base_dir),
            "max_steps": 10,
            "render_mode": "text",
        })

        obs, info = await env.reset(seed=42)

        assert "obs_str" in obs, "Missing obs_str in observation"
        assert isinstance(obs["obs_str"], str), "obs_str should be a string"
        assert len(obs["obs_str"]) > 0, "obs_str should not be empty"

        print(f"  [PASS] Environment reset successful")
        print(f"         Observation length: {len(obs['obs_str'])} chars")

        await env.close()
        print("  [PASS] Environment closed successfully")

    except Exception as e:
        print(f"  [FAIL] Environment reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


async def test_env_step():
    """Test environment step."""
    print("\nTesting environment step...")

    from vagen.envs.deliverybench import DeliveryBench, DeliveryBenchEnvConfig

    config = DeliveryBenchEnvConfig()
    base_dir = Path(config.base_dir)

    if not base_dir.exists():
        print("  [SKIP] Skipping step test - base_dir not found")
        return True

    try:
        env = DeliveryBench({
            "base_dir": str(base_dir),
            "max_steps": 10,
            "render_mode": "text",
        })

        obs, info = await env.reset(seed=42)
        print("  [INFO] Environment reset done")

        # Test VIEW_ORDERS action
        action = '{"action": "VIEW_ORDERS()"}'
        obs, reward, done, info = await env.step(action)

        assert "obs_str" in obs, "Missing obs_str in observation"
        assert isinstance(reward, (int, float)), "Reward should be a number"
        assert isinstance(done, bool), "Done should be a boolean"
        assert isinstance(info, dict), "Info should be a dict"

        print(f"  [PASS] Step with VIEW_ORDERS() successful")
        print(f"         Reward: {reward}, Done: {done}")

        # Test another action
        action = '{"action": "VIEW_BAG()"}'
        obs, reward, done, info = await env.step(action)
        print(f"  [PASS] Step with VIEW_BAG() successful")

        await env.close()

    except Exception as e:
        print(f"  [FAIL] Environment step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


async def test_system_prompt():
    """Test system prompt method."""
    print("\nTesting system_prompt method...")

    from vagen.envs.deliverybench import DeliveryBench, DeliveryBenchEnvConfig

    config = DeliveryBenchEnvConfig()
    base_dir = Path(config.base_dir)

    try:
        env = DeliveryBench({
            "base_dir": str(base_dir),
            "max_steps": 10,
            "render_mode": "text",
        })

        sys_prompt = await env.system_prompt()

        assert "obs_str" in sys_prompt, "Missing obs_str in system prompt"
        assert isinstance(sys_prompt["obs_str"], str)
        assert len(sys_prompt["obs_str"]) > 0
        assert "MOVE" in sys_prompt["obs_str"]

        print(f"  [PASS] System prompt method works")
        print(f"         System prompt length: {len(sys_prompt['obs_str'])} chars")

    except Exception as e:
        print(f"  [FAIL] System prompt test failed: {e}")
        return False

    return True


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("DeliveryBench Integration Tests")
    print("=" * 60)

    results = []

    # Sync tests
    results.append(("imports", test_imports()))
    results.append(("config", test_config()))
    results.append(("parse_response", test_parse_response()))
    results.append(("prompt_templates", test_prompt_templates()))
    results.append(("base_dir", test_deliverybench_base_dir()))

    # Async tests
    results.append(("env_creation", await test_env_creation()))
    results.append(("system_prompt", await test_system_prompt()))
    results.append(("env_reset", await test_env_reset()))
    results.append(("env_step", await test_env_step()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")

    return all(r for _, r in results)


def main():
    """Main entry point."""
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
