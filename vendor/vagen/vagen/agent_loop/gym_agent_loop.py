# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

import asyncio
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from PIL import Image
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from ..envs.gym_image_env import GymImageEnv
from omegaconf import OmegaConf
import traceback
import importlib
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _flatten_text_only_content(msg):
    """
    convert message['content'] from multimodal list to plain text
    - only allow type == 'text'
    - concatenate multiple text blocks in order
    """
    content = msg.get("content")

    if isinstance(content, str):
        return msg

    if not isinstance(content, list):
        raise TypeError(f"Unsupported content type: {type(content)}")

    texts = []
    for block in content:
        if not isinstance(block, dict):
            raise TypeError(f"Invalid content block: {block}")

        block_type = block.get("type")
        if block_type != "text":
            raise AssertionError(
                f"Non-text block found in text-only tokenizer path: {block_type}"
            )
        texts.append(block.get("text", ""))

    new_msg = dict(msg)
    new_msg["content"] = "".join(texts)
    return new_msg


# Per-constraint compliance metrics, forwarded verbatim when the environment
# reports them (it only does on runs whose yaml switched the constraint on).
# One list, shared by the extraction below and by gym_agent_loop_no_concat.
CONSTRAINT_METRIC_KEYS = (
    "cold_deliveries", "warm_delivery_rate",
    "melted_deliveries", "intact_delivery_rate",
    "notes_followed", "noted_delivery_rate",
    "phone_battery_left", "phone_alive_rate", "phone_recharges",
    "mean_fee_paid", "rests", "stamina_left",
)
# Inert values for an episode that produced no report (env error before the
# first successful step). ALWAYS emitted: DataProto.concat asserts identical
# key sets across every worker's dicts, so a single key-dropping trajectory
# poisons the whole validation batch -- that is how courier-c2-battery died
# at step 0.
CONSTRAINT_METRIC_DEFAULTS = {
    "cold_deliveries": 0.0, "warm_delivery_rate": 1.0,
    "melted_deliveries": 0.0, "intact_delivery_rate": 1.0,
    "notes_followed": 0.0, "noted_delivery_rate": 0.0,
    "phone_battery_left": 100.0, "phone_alive_rate": 1.0, "phone_recharges": 0.0,
    "mean_fee_paid": 0.0, "rests": 0.0, "stamina_left": 100.0,
}


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    INTERACTING = "interacting"
    TERMINATED = "terminated"


class AgentData:
    """Container for all mutable trajectory state."""
    def __init__(
        self,
        messages: List[Dict[str, Any]],
        image_data: List[Image.Image],
        metrics: Dict[str, Any],
        request_id: str,
        env: GymImageEnv,
        response_limit: int,
        env_name: str,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.env = env
        self.response_limit = response_limit
        self.env_name = env_name

        # Token buffers
        self.prompt_ids: List[int] = []
        self.response_ids: List[int] = []
        self.response_mask: List[int] = []
        self.response_logprobs: List[float] = []

        # Env stats
        self.env_rewards: List[float] = []
        self.traj_success: bool = False
        self.env_turns: int = 0
        # Deliveries this episode, from info["delivered"]. Local addition (see
        # vendor/VENDOR_PINS.md): under an ENDLESS shift success is a floor
        # ("not a zero") and the count is the headline, so it must reach the
        # metrics the same way traj_success does.
        self.env_delivered: float = 0.0
        self.env_earnings: float = 0.0
        self.env_red_crossings: float = 0.0
        self.env_blocked_attempts: float = 0.0
        self.env_earnings_at: dict = {}
        # Compliance metrics for the optional constraints. Starts at the
        # inert defaults and is overwritten by every successful step, so the
        # emitted key set is identical on every trajectory -- including one
        # whose environment errored before reporting anything.
        self.env_constraints: dict = dict(CONSTRAINT_METRIC_DEFAULTS)


        # Cached assistant text to step env
        self.last_assistant_text: Optional[str] = None


# -------------------- MM helpers --------------------

def _normalize_images(imgs: List[Image.Image]) -> List[Image.Image]:
    """Ensure PIL RGB and drop Nones."""
    out: List[Image.Image] = []
    for im in imgs or []:
        if im is None:
            continue
        out.append(im.convert("RGB") if isinstance(im, Image.Image) else im)
    return out

def extract_success(info: Dict[str, Any], success_keys: str = "success|is_success") -> bool:
    """Extract success flag from env info dict."""
    for key in success_keys.split("|"):
        if key in info:
            return bool(info[key])
    return False

def convert_obs_to_content(
    obs: Dict[str, Any],
    obs_text_key: str = "obs_str",
    image_placeholder: str = "<image>",
    video_placeholder: str = "<video>",
    multi_modal_key: str = "multi_modal_input",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Convert obs['obs_str'] containing <image>/<video> into structured content."""
    text = obs[obs_text_key]
    mmi = obs.get(multi_modal_key, {}) or {}

    # Simple strict consistency check
    num_img_tok = text.count(image_placeholder)
    num_vid_tok = text.count(video_placeholder)
    num_imgs = len(mmi.get(image_placeholder, []) or [])
    num_vids = len(mmi.get(video_placeholder, []) or [])
    assert num_img_tok == num_imgs, f"#images ({num_imgs}) != #{image_placeholder} ({num_img_tok})"
    assert num_vid_tok == num_vids, f"#videos ({num_vids}) != #{video_placeholder} ({num_vid_tok})"

    # Split and keep tokens
    pattern = f"({re.escape(image_placeholder)}|{re.escape(video_placeholder)})"
    segments = re.split(pattern, text)

    content: List[Dict[str, Any]] = []
    for seg in segments:
        if not seg:
            continue
        if seg == image_placeholder:
            content.append({"type": "image"})
        elif seg == video_placeholder:
            content.append({"type": "video"})
        else:
            content.append({"type": "text", "text": seg})
    return content


# -------------------- Gym Agent Loop --------------------

class GymAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level GymAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
        
        cls.env_registry = {}

        for k, v in config.env_registry.items():
            module_path, class_name = v.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls.env_registry[k] = getattr(module, class_name)
            
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

    @rollout_trace_op
    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics: Dict[str, Any] = {}
        request_id = uuid4().hex

        # Build env
        env_cls = self.env_registry[kwargs["env_name"]]
        env_config = kwargs["config"]
        seed = kwargs["seed"]
        self.env_max_turns = kwargs.get("max_turns", None)
        env: GymImageEnv = env_cls(env_config=env_config)

        # Bootstrap: reset -> system_prompt (message order: system, then initial user)
        init_obs, info = await env.reset(seed=seed)
        sys_obs = await env.system_prompt()

        messages: List[Dict[str, Any]] = []
        image_data: List[Image.Image] = []

        if sys_obs:
            messages.append({"role": "system", "content": convert_obs_to_content(sys_obs, **kwargs)})
            sys_imgs = sys_obs.get("multi_modal_input", {}).get("<image>", []) or []
            image_data.extend(_normalize_images(sys_imgs))
        if init_obs:
            messages.append({"role": "user", "content": convert_obs_to_content(init_obs, **kwargs)})
            init_imgs = init_obs.get("multi_modal_input", {}).get("<image>", []) or []
            image_data.extend(_normalize_images(init_imgs))

        per_turn_response_limit = int(kwargs.get("response_length_per_turn") or self.response_length)
        per_turn_response_limit = min(per_turn_response_limit, self.response_length)
        if per_turn_response_limit <= 0:
            per_turn_response_limit = 1

        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            env=env,
            response_limit=per_turn_response_limit,
            env_name=kwargs["env_name"],
        )

        # State machine: always GENERATE -> INTERACT, and decide termination inside INTERACT
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_env_state(agent_data, **kwargs)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Close env after loop
        await env.close()

        # Finalize output
        resp_len = len(agent_data.response_mask)
        response_ids = agent_data.prompt_ids[-resp_len:] if resp_len else []
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - resp_len]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data else {}

        if len(prompt_ids) > self.prompt_length:
            logger.warning(
                f"In env:{agent_data.env_name}, prompt_ids length {len(prompt_ids)} exceeds prompt_length {self.prompt_length}",
            )
        if len(response_ids) > self.response_length:
            logger.warning(
                f"In env:{agent_data.env_name}, response_ids length {len(response_ids)} exceeds response_length {self.response_length}",
            )

        # Patched by embodiedbench/training/vagen/patches/: cutting the ids
        # without cutting the images leaves image_grid_thw describing frames
        # that are no longer in the sequence.
        keep_prompt, keep_response, images = self._cut_on_image_boundaries(
            prompt_ids, response_ids, agent_data.image_data)
        multi_modal_data = {"image": images} if images else {}

        output = AgentLoopOutput(
            prompt_ids=prompt_ids[len(prompt_ids) - keep_prompt:],
            response_ids=response_ids[:keep_response],
            response_mask=agent_data.response_mask[:keep_response],
            multi_modal_data=multi_modal_data,
            response_logprobs=(
                agent_data.response_logprobs[:keep_response] if agent_data.response_logprobs else None
            ),
            reward_score=sum(agent_data.env_rewards) if agent_data.env_rewards else 0.0,
            num_turns=agent_data.env_turns,
            metrics=agent_data.metrics,
            extra_fields={ "image_data": agent_data.image_data,"reward_extra_info": {"traj_success": float(agent_data.traj_success), "delivered": float(agent_data.env_delivered), "earnings": float(agent_data.env_earnings), "red_crossings": float(agent_data.env_red_crossings), "blocked_attempts": float(agent_data.env_blocked_attempts), **{f"earnings_at_{h}": float(agent_data.env_earnings_at.get(h, agent_data.env_earnings)) for h in (20, 40, 60, 80, 100)}, **agent_data.env_constraints}},
        )
        return output

    def _cut_on_image_boundaries(self, prompt_ids, response_ids, images):
        """How much of each segment to keep so no image is half in it.

        Returns (prompt_kept, response_kept, images_kept). Falls back to the
        plain lengths whenever the sequence cannot be read as one image block
        per image, which is the text-only case and any processor whose pad
        token is not <|image_pad|>.
        """
        prompt_kept = min(self.prompt_length, len(prompt_ids))
        response_kept = min(self.response_length, len(response_ids))
        if not images or self.processor is None:
            return prompt_kept, response_kept, images
        if prompt_kept == len(prompt_ids) and response_kept == len(response_ids):
            return prompt_kept, response_kept, images

        pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if pad_id is None or pad_id < 0:
            return prompt_kept, response_kept, images
        # An image is written <|vision_start|> <|image_pad|>... <|vision_end|>,
        # and get_rope_index reads the token after every <|vision_start|> it
        # finds. Ending the sequence on one indexes off the end of it, so the
        # opening token goes with the image it opens.
        start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")

        sequence = list(prompt_ids) + list(response_ids)
        blocks, index, length = [], 0, len(sequence)
        while index < length:
            if sequence[index] == pad_id:
                end = index
                while end < length and sequence[end] == pad_id:
                    end += 1
                blocks.append((index, end))
                index = end
            else:
                index += 1
        # One contiguous run of pad tokens per image, or this is not a
        # sequence we know how to cut.
        if len(blocks) != len(images):
            return prompt_kept, response_kept, images

        low = len(prompt_ids) - prompt_kept
        high = len(prompt_ids) + response_kept

        first = 0
        for position, (start, end) in enumerate(blocks):
            if end <= low:
                first = position + 1
            elif start < low:
                low = end          # the cut split this one; drop it whole
                first = position + 1
            else:
                break

        last = len(blocks)
        for position in range(len(blocks) - 1, -1, -1):
            start, end = blocks[position]
            if start >= high:
                last = position
            elif end > high:
                high = start       # same, from the other end
                last = position
            else:
                break
        while high > len(prompt_ids) and sequence[high - 1] == start_id:
            high -= 1

        return (len(prompt_ids) - low, high - len(prompt_ids),
                images[first:last])

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: Dict[str, Any]) -> AgentState:
        """Encode initial (system + first user) messages into prompt_ids."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data or None, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if agent_data.image_data:
                raise ValueError("Environment returned images but `processor` is None.")
            try:
                agent_data.prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        agent_data.messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
            except TypeError as e:
                logger.warning(f"TypeError Warning in apply_chat_template in AgentLoop: {e}, switching to flattened text-only content.")
                # Fallback for text-only tokenizer
                flat_messages = [_flatten_text_only_content(msg) for msg in agent_data.messages]
                agent_data.prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        flat_messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
        
        if len(agent_data.prompt_ids)>self.prompt_length:
            logger.warning(f"In env:{agent_data.env_name}, initial prompt length {len(agent_data.prompt_ids)} exceeds prompt_length {self.prompt_length}")
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: Dict[str, Any]
    ) -> AgentState:
        """Generate assistant output and mark generated tokens with mask=1."""
        sampling_params_for_turn = sampling_params.copy()
        max_new_tokens=sampling_params_for_turn.get("max_new_tokens", None) or agent_data.response_limit
        max_new_tokens = min(max_new_tokens, agent_data.response_limit)
        # Patched by embodiedbench/training/vagen/patches/: the engine serves
        # at most prompt_length + response_length tokens in total and sizes
        # each request as that minus len(prompt_ids). The response-budget rule
        # alone does not imply that bound -- an initial prompt past
        # prompt_length is only warned about -- and an unclamped request at
        # the ceiling asks vLLM for a negative allowance and kills the run.
        engine_remaining = (self.prompt_length + self.response_length
                            - len(agent_data.prompt_ids))
        if engine_remaining < 16:
            return AgentState.TERMINATED
        max_new_tokens = min(max_new_tokens, engine_remaining)
        sampling_params_for_turn["max_new_tokens"] = max_new_tokens
            

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params_for_turn,
                image_data=agent_data.image_data,
            )


        agent_data.response_ids = output.token_ids
        if len(output.token_ids)>agent_data.response_limit:
            logger.warning(f"In env:{agent_data.env_name}, generated response length {len(output.token_ids)} exceeds per-turn response_limit {agent_data.response_limit}")
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Cache assistant text and add assistant message (text-only)
        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        )
        agent_data.last_assistant_text = assistant_message
        agent_data.messages.append({"role": "assistant", "content": assistant_message})
        return AgentState.INTERACTING

    async def _handle_env_state(self, agent_data: AgentData, **kwargs) -> AgentState:
        """
        Step the environment with last assistant action; always collect reward first.
        If terminal (done/success/turn-limit/token-limit), stop WITHOUT appending user suffix,
        so the episode ends on an assistant turn.
        """
        action_str = agent_data.last_assistant_text or ""
        try:
            obs, reward, done, info = await agent_data.env.step(action_str)
            # traceback
        except Exception as exc:
            logger.error(
                "Environment step failed in '%s' with action %r: %s",
                agent_data.env_name,
                action_str,
                exc,
            )
            logger.error("Environment traceback:\n%s", traceback.format_exc())
            obs, reward, done, info = {"obs_str":"Environment Error"}, 0.0, True, {"traj_success": False}

        agent_data.env_rewards.append(float(reward))
        agent_data.traj_success = extract_success(info)
        try:
            agent_data.env_delivered = float(info.get("delivered") or 0)
            agent_data.env_earnings = float(info.get("earnings") or 0.0)
            agent_data.env_red_crossings = float(info.get("red_crossings") or 0)
            agent_data.env_blocked_attempts = float(info.get("blocked_attempts") or 0)
            if isinstance(info.get("earnings_at"), dict):
                agent_data.env_earnings_at = dict(info["earnings_at"])
            # Constraint-compliance counters, present only on flagged runs.
            # Kept as a dict so absent keys stay absent downstream: the
            # aggregator takes the union across trajectories and fills 0.0,
            # and a flags-off run must not grow columns it has no mechanic for.
            agent_data.env_constraints.update({
                key: float(info[key]) for key in CONSTRAINT_METRIC_KEYS
                if info.get(key) is not None
            })
        except (TypeError, ValueError):
            pass
        agent_data.env_turns += 1
        # Termination rule #3: env done or success
        if done or agent_data.traj_success:
            return AgentState.TERMINATED

        # Termination rule #2: env turn-limit (if set)
        if self.env_max_turns is not None and agent_data.env_turns >= int(self.env_max_turns):
            return AgentState.TERMINATED

        # Termination rule #1: response token-limit
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED

        # Not terminal -> append user suffix for next turn
        user_content = convert_obs_to_content(obs, **kwargs)
        user_msg = {"role": "user", "content": user_content}
        agent_data.messages.append(user_msg)

        new_images = obs.get("multi_modal_input", {}).get("<image>", []) or []
        new_images = _normalize_images(new_images)

        if self.processor is not None:
            raw_user_suffix = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    [{},user_msg],
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_suffix], images=new_images or None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if new_images:
                raise ValueError("Environment returned images but `processor` is None.")
            try:
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        [{},user_msg], add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                    ),
                )
            except TypeError as e:
                logger.warning(f"TypeError Warning in apply_chat_template in AgentLoop: {e}, switching to flattened text-only content.")
                # Fallback for text-only tokenizer
                flat_user_msg = _flatten_text_only_content(user_msg) 
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        [{},flat_user_msg],
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
        response_ids = response_ids[len(self.system_prompt) :]
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        if new_images:
            agent_data.image_data.extend(new_images)

        return AgentState.GENERATING
