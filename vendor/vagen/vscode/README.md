# Debugging VAGEN with VSCode

This directory contains VSCode debugging configurations for the VAGEN project.

## Overview

The project uses a two-step debugging approach:
1. **Launch Configuration**: Starts the main training process
2. **Attach Configuration**: Connects to Ray remote workers (AgentLoopWorker) running in separate processes

## How to Debug

### Step0: Add Port Listener
in `verl/experimental/agent_loop/agent_loop.py`, add the following code to the `AgentLoopWorker` class:
'''python
@ray.remote
class AgentLoopWorker(AgentLoopWorkerBase):
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        """Initialize agent loop manager.
        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_router_address (str): reward router address.
        """
        # Debug: attach debugpy listener for AgentLoopWorker (only first worker succeeds)
        import debugpy
        if os.environ.get("VERL_DEBUG_ATTACH"):
            port = int(os.environ.get("VERL_DEBUG_PORT", "5679"))
            debugpy.listen(("0.0.0.0", port))
            print(f"AgentLoopWorker: Waiting for debugger on port {port}...")
            debugpy.wait_for_client()
            print(f"AgentLoopWorker: Debugger attached on port {port}!")
        else:
            print("AgentLoopWorker: Debugpy skipped (VERL_DEBUG_ATTACH not set)")
        super().__init__(config, server_handles, reward_router_address)

### Step 1: Launch the Main Process

1. Open VSCode in the project root directory
2. Go to the Debug panel (Ctrl+Shift+D or Cmd+Shift+D)
3. Select **"Debug: train_grpo_qwen25vl3b"** from the dropdown
4. Click the green play button or press F5

This will:
- Start the main training process with `VERL_DEBUG_ATTACH=1` environment variable
- Allow you to set breakpoints in the main process code
- The main process will continue until Ray workers are spawned

### Step 2: Attach to AgentLoopWorker

When the `AgentLoopWorker` is spawned by Ray, it will:
1. Start a debugpy listener on port **5679** (configurable via `VERL_DEBUG_PORT`)
2. Print: `"AgentLoopWorker: Waiting for debugger on port 5679..."`
3. **Pause and wait** for a debugger to attach

To attach the debugger:

1. Keep the main process running (from Step 1)
2. In the Debug panel, switch to **"Attach: AgentLoopWorker (async)"**
3. Click the green play button or press F5
4. The worker will print: `"AgentLoopWorker: Debugger attached on port 5679!"`
5. Your breakpoints in `AgentLoopWorker` code will now work!

### Setting Breakpoints

- **Main Process**: Set breakpoints anywhere in the main training loop
- **AgentLoopWorker**: Set breakpoints in `/vagen/agent_loop/agent_loop_no_concat.py` or related files
  - The worker code runs in a separate Ray process, so you must use the "Attach" configuration