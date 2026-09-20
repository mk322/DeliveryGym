# VAGEN Code Walkthrough

Assumes familiarity with basic RL (rollout, compute reward, compute advantage, update actor). This focuses on how environments are integrated into the RL loop and how the verl agent loop / Ray worker architecture works.

---

## 1. VAGEN GRPO Overview (multi-turn, environment-based)

```
1. Sample env specs from AgenticDataset (env_name, seed, config, max_turns)
   No text prompts! Instead, env specs that generate prompts dynamically.
   
2. For each env spec, generate N trajectories via agent loop:
   a. Create env instance -> env.reset(seed)  
   b. Get system_prompt + initial_obs -> tokenize as prompt
   c. LOOP (multi-turn):
      - LLM generates response (action)
      - env.step(action) -> next_obs, reward, done
      - Append obs as new user message, continue generating
   d. Collect: full token sequence, per-turn rewards, response_mask
   KEY difference from standard GRPO: rollout involves env interaction!
   
3. Sum env rewards per trajectory -> scalar reward score
4. Group trajectories by prompt (same env spec), compute GRPO advantage
5. Recompute log_probs under current policy
6. Compute PPO clipped loss
7. Backprop and update model
```

---

## 2. Class Architecture: Who Does What

Three layers, from top to bottom:

```
AgentLoopManager          [lives in Driver process, plain Python object]
  - Owns vLLM server replicas (on GPUs) and AgentLoopWorker actors (on CPUs)
  - Called by RayPPOTrainer.fit() to do rollout
  - Splits batch across workers, collects results
  - Controls wake/sleep cycle (FSDP <-> vLLM weight transfer)

AgentLoopWorker           [Ray remote actor, CPU process, one per node]
  - Holds its own tokenizer + processor + AsyncLLMServerManager
  - Receives a chunk of samples, runs ALL of them as concurrent asyncio tasks
  - Pads variable-length outputs to fixed tensor shapes
  - Returns DataProto back to Manager

GymAgentLoop              [transient, one instance per sample]
  - Inherits from AgentLoopBase (ABC)
  - The actual PENDING -> GENERATING -> INTERACTING state machine
  - Creates env, resets, calls server_manager.generate(), calls env.step()
  - Returns AgentLoopOutput (variable-length Python lists, not yet padded)
```

### How they connect:

```
vagen/configs/agent.yaml:
  - name: gym_agent
    _target_: vagen.agent_loop.gym_agent_loop.GymAgentLoop

vagen/gym_agent_dataset.py:
  Each dataset item has agent_name: "gym_agent"
  -> Worker looks up _agent_loop_registry["gym_agent"]
  -> hydra.utils.instantiate() creates a GymAgentLoop
```

---

## 3. Full Call Stack (one training step, rollout phase)

```
RayPPOTrainer.fit()                              [Driver process, CPU]
|
+- gen_batch = batch.repeat(n=4)                  # 32 items -> 128 (4 per prompt for GRPO)
|
+- self.async_rollout_manager.generate_sequences(gen_batch)
   |                                              [AgentLoopManager]
   |
   +- self.wake_up()                              # Tell vLLM to load latest weights
   |   +- replica.wake_up() for each replica      # NCCL weight transfer: FSDP -> vLLM
   |
   +- chunks = gen_batch.chunk(4)                  # Split: 128 items -> 4 chunks of 32
   |
   +- ray.get([worker.generate_sequences.remote(chunk) for ...])
   |   |                                          [4 AgentLoopWorker Ray actors]
   |   |
   |   |  (Inside each worker, e.g. worker_0 with 32 items):
   |   |
   |   +- sampling_params = {temperature: 0.7, top_p: 0.9, logprobs: True}
   |   |
   |   +- for i in range(32):                      # Create 32 asyncio tasks
   |   |     tasks.append(asyncio.create_task(
   |   |         self._run_agent_loop(sampling_params, ...,
   |   |             agent_name="gym_agent", env_name="Sokoban", seed=4231, ...)))
   |   |
   |   +- outputs = await asyncio.gather(*tasks)   # Run all 32 concurrently
   |   |   |
   |   |   |  (Inside each _run_agent_loop):
   |   |   |
   |   |   +- agent_loop_config = _agent_loop_registry["gym_agent"]
   |   |   |   # = {"_target_": "vagen.agent_loop.gym_agent_loop.GymAgentLoop"}
   |   |   |
   |   |   +- agent_loop = hydra.utils.instantiate(config, ...)
   |   |   |   # Creates a NEW GymAgentLoop instance for this one sample
   |   |   |   # GymAgentLoop.__init__() -> calls init_class() (once) to load env registry
   |   |   |
   |   |   +- output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
   |   |   |   |                                  [GymAgentLoop, per-sample state machine]
   |   |   |   |
   |   |   |   +- env = Sokoban(config)            # Create env
   |   |   |   +- obs = await env.reset(seed=4231) # Reset
   |   |   |   +- sys = await env.system_prompt()  # Get system message
   |   |   |   |
   |   |   |   +- State: PENDING
   |   |   |   |   +- Tokenize [system, user_obs] -> prompt_ids
   |   |   |   |
   |   |   |   +- State: GENERATING
   |   |   |   |   +- await self.server_manager.generate(prompt_ids=..., ...)
   |   |   |   |       |                          [AsyncLLMServerManager]
   |   |   |   |       +- server = _choose_server(request_id)  # sticky session
   |   |   |   |       +- await server.generate.remote(...)     # -> vLLM on GPU
   |   |   |   |           |                      [vLLM server, GPU process]
   |   |   |   |           +- returns TokenOutput(token_ids=[...], log_probs=[...])
   |   |   |   |
   |   |   |   +- State: INTERACTING
   |   |   |   |   +- obs, reward, done, info = await env.step(action_text)
   |   |   |   |   +- Not done -> tokenize new obs, append with mask=0
   |   |   |   |   +- -> back to GENERATING (or TERMINATED if done)
   |   |   |   |
   |   |   |   +- Returns AgentLoopOutput(prompt_ids=[...], response_ids=[...],
   |   |   |                              response_mask=[1,1,0,0,1,1,...], reward_score=1.3)
   |   |   |
   |   |   +- PAD to fixed lengths -> _InternalAgentLoopOutput (tensors [1, seq_len])
   |   |
   |   +- self._postprocess(outputs) -> DataProto    # Stack [32, seq_len] tensors
   |       +- Place reward at last real token position in rm_scores
   |
   +- output = DataProto.concat(outputs)            # [4 x 32 = 128, seq_len]
   |
   +- self.sleep()                                  # Tell vLLM to release KV cache
   |
   +- return output                                 # Back to RayPPOTrainer.fit()
```

Parallelism is two-level:
- **Ray-level**: Manager dispatches to N workers in parallel (`ray.get([worker.remote(...) for ...])`)
- **asyncio-level**: Within each worker, all samples run as concurrent coroutines (`asyncio.gather(*tasks)`)

---

## 4. AgentLoopManager Details

```python
# vagen/agent_loop/agent_loop_no_concat.py (or verl/verl/experimental/agent_loop/agent_loop.py)

class AgentLoopManager:
    # Lives in the Driver process. NOT a Ray actor. Plain Python object.

    def __init__(self, config, worker_group=None, rm_wg=None):
        self.worker_group = worker_group  # ActorRolloutRef RayWorkerGroup (GPU workers)

        # 1. Start vLLM/SGLang inference servers
        self._initialize_llm_servers()
        # 2. Spawn AgentLoopWorker Ray actors on CPUs
        self._init_agent_loop_workers()

    def _initialize_llm_servers(self):
        # How many replicas? e.g. 4 GPUs / 2 TP_size = 2 replicas
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            RolloutReplica(replica_rank=i, config=...) for i in range(num_replicas)
        ]
        # HYBRID mode: vLLM reads weights directly from ActorRolloutRefWorker via NCCL
        if self.worker_group:
            [server.init_hybrid(self.worker_group) for server in self.rollout_replicas]
        else:
            [server.init_standalone() for server in self.rollout_replicas]

        self.server_handles = [server._server_handle for server in self.rollout_replicas]

    def _init_agent_loop_workers(self):
        num_workers = config.actor_rollout_ref.rollout.agent.num_workers
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"]]

        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]  # Round-robin across nodes
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=True),
                ).remote(self.config, self.server_handles, self.reward_router_address)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.wake_up()                              # Load latest weights into vLLM
        chunks = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get([                         # Dispatch to workers in parallel
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks)
        ])
        output = DataProto.concat(outputs)          # Merge results
        self.sleep()                                # Free vLLM KV cache for training
        return output

    def wake_up(self):
        # NCCL weight transfer: FSDP shards -> vLLM tensor-parallel layout
        [replica.wake_up() for replica in self.rollout_replicas]

    def sleep(self):
        # Release KV cache so GPU memory is available for training
        [replica.sleep() for replica in self.rollout_replicas]
```

---

## 5. AgentLoopWorkerBase / AgentLoopWorker Details

```python
# verl/verl/experimental/agent_loop/agent_loop.py:255-636

class AgentLoopWorkerBase:
    # Each worker is a Ray actor on a CPU. It does NOT hold model weights.
    # It holds: tokenizer, processor, connection to vLLM servers.

    def __init__(self, config, server_handles, reward_router_address=None):
        # 1. Load-balanced router to vLLM replicas (sticky sessions for prefix caching)
        self.server_manager = AsyncLLMServerManager(config, server_handles)

        # 2. Each worker loads its own tokenizer/processor (separate CPU process)
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path)
        self.processor = hf_processor(local_path)

        # 3. Load agent loop registry from YAML
        #    e.g. "gym_agent" -> "vagen.agent_loop.gym_agent_loop.GymAgentLoop"
        agent_loop_configs = OmegaConf.load(config.actor_rollout_ref.rollout.agent.agent_loop_config_path)
        for cfg in agent_loop_configs:
            _agent_loop_registry[cfg.name] = cfg

        # 4. Reward manager (Ray actor, pinned to same node)
        self.reward_manager_worker = RewardManagerWorker.remote(self.config, ...)

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        # Called remotely by AgentLoopManager

        # A. Build sampling params
        sampling_params = dict(temperature=..., top_p=..., logprobs=True)

        # B. Launch ALL samples as concurrent asyncio tasks
        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            # kwargs = {env_name: "Sokoban", seed: 4231, config: {...}, agent_name: "gym_agent", ...}
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, ..., **kwargs)))

        # C. Run ALL concurrently (I/O-bound: waiting for vLLM + env.step)
        outputs = await asyncio.gather(*tasks)

        # D. Flatten (no-concat mode returns list-of-lists) and postprocess
        flattened = [item for sublist in outputs for item in sublist]
        return self._postprocess(flattened)

    async def _run_agent_loop(self, sampling_params, trajectory, *, agent_name, **kwargs):
        # 1. Look up class: _agent_loop_registry["gym_agent"]
        #    -> {"_target_": "vagen.agent_loop.gym_agent_loop.GymAgentLoop"}
        agent_loop_config = _agent_loop_registry[agent_name]

        # 2. Instantiate via Hydra (one instance per sample)
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=_DummyConfig(config=self.config),
            server_manager=self.server_manager,  # shared across all loops in this worker
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        # 3. Run the state machine (env interaction happens here!)
        outputs: list[AgentLoopOutput] = await agent_loop.run(sampling_params, **kwargs)
        # Returns variable-length Python lists: prompt_ids, response_ids, response_mask, ...

        # 4. PAD outputs to fixed lengths for batching
        internal_outputs = []
        for output in outputs:
            # Prompt: left-pad to prompt_length
            # [151644, 8948, 256] -> [0, 0, ..., 151644, 8948, 256]
            self.tokenizer.padding_side = "left"
            prompt_padded = self.tokenizer.pad({"input_ids": output.prompt_ids},
                padding="max_length", max_length=config.rollout.prompt_length)

            # Response: right-pad to response_length
            # [128, 456, 789] -> [128, 456, 789, 0, 0, ...]
            self.tokenizer.padding_side = "right"
            response_padded = self.tokenizer.pad({"input_ids": output.response_ids},
                padding="max_length", max_length=config.rollout.response_length)

            # Concat: input_ids = [prompt | response]
            input_ids = torch.cat([prompt_padded["input_ids"], response_padded["input_ids"]], dim=1)
            attention_mask = torch.cat([prompt_padded["attention_mask"], response_padded["attention_mask"]], dim=1)

            # response_mask * attention_mask: only LLM-generated, non-padding tokens are 1
            response_mask = response_mask_padded * response_padded["attention_mask"]

            # Position IDs (handles Qwen2VL vision RoPE if multimodal)
            # ...

            # Optionally compute reward if not already provided by the loop
            if output.reward_score is None:
                result = await self.reward_manager_worker.compute_score.remote(data)
                output.reward_score = result["reward_score"]

            internal_outputs.append(_InternalAgentLoopOutput(...))  # tensors [1, seq_len]
        return internal_outputs

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        # Stack all [1, seq_len] tensors into [bsz, seq_len]
        prompt_ids = torch.cat([x.prompt_ids for x in inputs], dim=0)
        # ... same for response_ids, input_ids, attention_mask, response_mask, position_ids

        # Place scalar reward at the last real response token position
        if all(x.reward_score is not None for x in inputs):
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(bsz), last_real_token_idx] = torch.tensor(scores)
            batch["rm_scores"] = rm_scores

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


# The actual Ray actor: just @ray.remote on top of the base class
@ray.remote
class AgentLoopWorker(AgentLoopWorkerBase):
    def __init__(self, config, server_handles, reward_router_address=None):
        super().__init__(config, server_handles, reward_router_address)
```

---

## 6. AgentLoopBase (ABC) and GymAgentLoop

```python
# verl/verl/experimental/agent_loop/agent_loop.py:179-233

class AgentLoopBase(ABC):
    # One instance per sample. Created by hydra.utils.instantiate() inside _run_agent_loop.

    _class_initialized = False  # Class-level flag, shared across all instances

    def __init__(self, trainer_config, server_manager, tokenizer, processor, **kwargs):
        # init_class() does heavy one-time setup (e.g. loading env registry)
        # Only runs once thanks to _class_initialized flag
        self.init_class(config=trainer_config.config, tokenizer=tokenizer, processor=processor)

        self.config = trainer_config.config
        self.server_manager = server_manager  # shared AsyncLLMServerManager from the worker
        self.tokenizer = tokenizer
        self.processor = processor
        self.loop = asyncio.get_running_loop()  # for offloading CPU work to thread pool

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        # Heavy init shared across all instances. Only called once.
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        # Subclass implements the actual rollout logic
        raise NotImplementedError
```

`GymAgentLoop` is VAGEN's concrete implementation:

```python
# vagen/agent_loop/gym_agent_loop.py:154-267

class GymAgentLoop(AgentLoopBase):

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        cls.tokenizer = tokenizer
        cls.processor = processor

        # Load env registry: {"Sokoban": <class Sokoban>, "FrozenLake": <class FrozenLake>, ...}
        cls.env_registry = {}
        for k, v in config.env_registry.items():
            module_path, class_name = v.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls.env_registry[k] = getattr(module, class_name)

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length

    async def run(self, sampling_params, **kwargs):
        # kwargs = {env_name: "Sokoban", seed: 4231, config: {...}, max_turns: 5}

        # 1. Create and reset environment
        env_cls = self.env_registry[kwargs["env_name"]]
        env = env_cls(env_config=kwargs["config"])
        init_obs, info = await env.reset(seed=kwargs["seed"])
        sys_obs = await env.system_prompt()

        # 2. Build initial messages
        messages = [
            {"role": "system", "content": convert_obs_to_content(sys_obs)},
            {"role": "user", "content": convert_obs_to_content(init_obs)}
        ]
        # init_obs example:
        #   {"obs_str": "Current state:\n<image>\nPush boxes to targets.",
        #    "multi_modal_input": {"<image>": [PIL.Image]}}
        # convert_obs_to_content() splits into:
        #   [{"type":"text","text":"Current state:\n"}, {"type":"image"}, {"type":"text","text":"\nPush..."}]

        agent_data = AgentData(messages, image_data, metrics, env, ...)

        # 3. State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_env_state(agent_data, **kwargs)

        await env.close()
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=response_ids,
            response_mask=agent_data.response_mask,
            reward_score=sum(agent_data.env_rewards),
            num_turns=agent_data.env_turns, ...)
```

### State machine diagram

```
+-----------+     +--------------+     +---------------+
| PENDING   |---->| GENERATING   |---->| INTERACTING   |
|(tokenize) |     | (LLM call)   |     | (env.step)    |
+-----------+     +--------------+     +-------+-------+
                        ^                      |
                        |    not done          |
                        +----------------------+
                               | done/success/turn-limit
                               v
                        +--------------+
                        | TERMINATED   |
                        +--------------+
```

### State: PENDING (tokenize messages into prompt_ids)

```python
# vagen/agent_loop/gym_agent_loop.py:269-312
async def _handle_pending_state(self, agent_data, sampling_params):
    # With multimodal processor (e.g. Qwen2-VL):
    raw_prompt = self.processor.apply_chat_template(
        agent_data.messages, add_generation_prompt=True, tokenize=False
    )
    model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data)
    agent_data.prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
    # prompt_ids = [151644, 8948, ..., 151645, ...]
    return AgentState.GENERATING
```

### State: GENERATING (call LLM to generate response)

```python
# vagen/agent_loop/gym_agent_loop.py:314-347
async def _handle_generating_state(self, agent_data, sampling_params):
    output = await self.server_manager.generate(
        request_id=agent_data.request_id,
        prompt_ids=agent_data.prompt_ids,
        sampling_params={"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 512},
        image_data=agent_data.image_data,
    )
    # output.token_ids = [128, 456, 789, ...]
    # output.log_probs = [-0.5, -1.2, -0.3, ...]

    agent_data.prompt_ids += output.token_ids
    agent_data.response_mask += [1] * len(output.token_ids)  # LLM-generated -> mask=1

    assistant_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
    # "<think>I need to push the box right...</think>\n<answer>move right, push up</answer>"
    agent_data.messages.append({"role": "assistant", "content": assistant_text})
    return AgentState.INTERACTING
```

### State: INTERACTING (step environment)

```python
# vagen/agent_loop/gym_agent_loop.py:349-436
async def _handle_env_state(self, agent_data, **kwargs):
    action_str = agent_data.last_assistant_text

    obs, reward, done, info = await agent_data.env.step(action_str)
    # Inside Sokoban.step():
    #   1. parse_response(action_str) -> extracts "move right, push up"
    #   2. Executes actions in the Sokoban grid
    #   3. Renders new state as image
    #   4. Returns obs, reward=0.1 (format_reward), done=False, info

    agent_data.env_rewards.append(reward)
    agent_data.env_turns += 1

    # Termination checks:
    if done or agent_data.traj_success:           return AgentState.TERMINATED
    if agent_data.env_turns >= self.env_max_turns: return AgentState.TERMINATED
    if len(agent_data.response_mask) >= self.response_length: return AgentState.TERMINATED

    # Not terminated -> append env observation as user message
    user_content = convert_obs_to_content(obs)
    agent_data.messages.append({"role": "user", "content": user_content})

    # Tokenize the new user message (env obs tokens get mask=0)
    response_ids = self.tokenizer.apply_chat_template([{}, user_msg], ...)
    response_ids = response_ids[len(self.system_prompt):]  # strip system prefix
    agent_data.prompt_ids += response_ids
    agent_data.response_mask += [0] * len(response_ids)    # env obs -> mask=0
    agent_data.image_data.extend(new_images)

    return AgentState.GENERATING  # loop back for next turn
```

---

## 7. Multi-Turn Example Timeline

```
Turn 0:
  PENDING:     [sys_prompt + init_obs] -> tokenize -> prompt_ids
  GENERATING:  LLM generates action_0 -> response_mask += [1,1,1,...,1]
  INTERACTING: env.step(action_0) -> reward=0.1, done=False
               Append env_obs_1 -> response_mask += [0,0,0,...,0]

Turn 1:
  GENERATING:  LLM generates action_1 -> response_mask += [1,1,1,...,1]
  INTERACTING: env.step(action_1) -> reward=0.1, done=False
               Append env_obs_2 -> response_mask += [0,0,0,...,0]

Turn 2:
  GENERATING:  LLM generates action_2 -> response_mask += [1,1,1,...,1]
  INTERACTING: env.step(action_2) -> reward=1.1 (format + success!), done=True
               -> TERMINATED

Final output:
  prompt_ids:    [sys + init_obs tokens]
  response:      [action_0 | env_obs_1 | action_1 | env_obs_2 | action_2]
  response_mask: [1,1,..,1 | 0,0,..,0  | 1,1,..,1 | 0,0,..,0  | 1,1,..,1]
  total_reward:  0.1 + 0.1 + 1.1 = 1.3
```

Only tokens where `response_mask=1` contribute to the PPO loss. Environment observation tokens (mask=0) are in the sequence for context but are not trained on.

---

## 8. AsyncLLMServerManager

```python
# verl/verl/experimental/agent_loop/agent_loop.py:52-115

class AsyncLLMServerManager:
    # Shared within each AgentLoopWorker. Routes generate() calls to vLLM replicas.
    # Two features:
    #   - Load balancing: least-requests heap
    #   - Sticky sessions: same request_id always goes to same server (prefix caching)

    def __init__(self, config, server_handles):
        # Min-heap: [request_count, (hash, server_handle)]
        self.weighted_servers = [[0, (hash(s), s)] for s in server_handles]
        heapq.heapify(self.weighted_servers)
        # LRU cache: request_id -> server (sticky session for multi-turn)
        self.request_id_to_server = LRUCache(maxsize=10000)

    def _choose_server(self, request_id):
        # If this request_id was seen before, return the same server (prefix caching)
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]
        # Otherwise, pick the least-loaded server
        server = self.weighted_servers[0][1][1]
        self.weighted_servers[0][0] += 1
        heapq.heapreplace(self.weighted_servers, self.weighted_servers[0])
        self.request_id_to_server[request_id] = server
        return server

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None):
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=request_id, prompt_ids=prompt_ids,
            sampling_params=sampling_params, image_data=image_data,
        )
        return output  # TokenOutput(token_ids=[...], log_probs=[...])
```

Sticky sessions matter for multi-turn: in a 5-turn episode, all 5 `generate()` calls use the same `request_id`, so they hit the same vLLM replica. That replica can reuse the KV cache from previous turns (prefix caching) instead of recomputing from scratch.

---

## 9. Key Files

| File | What |
|------|------|
| `vagen/main_ppo.py` | Entry point. Hydra config -> Ray init -> TaskRunner -> trainer.fit() |
| `vagen/ray_trainer.py` | Extends verl's RayPPOTrainer. The `fit()` training loop |
| `vagen/gym_agent_dataset.py` | AgenticDataset: expands env YAML specs into dataset items |
| `vagen/agent_loop/gym_agent_loop.py` | GymAgentLoop: PENDING/GENERATING/INTERACTING state machine |
| `vagen/agent_loop/agent_loop_no_concat.py` | VAGEN's AgentLoopManager + AgentLoopWorker (no-concat mode) |
| `verl/verl/experimental/agent_loop/agent_loop.py` | verl base: AgentLoopBase, AgentLoopWorkerBase, AgentLoopManager, AsyncLLMServerManager |
| `vagen/envs/gym_image_env.py` | GymImageEnv: abstract base for multimodal envs |
| `vagen/envs/sokoban/sokoban_env.py` | Example env: Sokoban with vision/text modes |
| `vagen/configs/agent.yaml` | Maps "gym_agent" -> GymAgentLoop class |
| `vagen/configs/env_registry.yaml` | Maps env names to Python classes |
| `verl/verl/trainer/ppo/core_algos.py` | GRPO advantage computation |
| `verl/verl/workers/fsdp_workers.py` | ActorRolloutRefWorker: GPU worker for training + inference |
