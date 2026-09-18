# DeliveryBench — RL Methods Plan

A grounded mapping from the RL/agent-RL literature to **DeliveryBench's actual env hooks**.
The organizing claim: the hard part is *not* naive reward sparsity (reward is already
`reward_t = format_bonus + Δearnings`, computed every `env.step`). The hard parts are
credit assignment over a long turn-MDP, context growth, targeted exploration, and
multi-turn stability. Several methods below are **dormant hooks already in the code** that
just need wiring.

> Key dormant hooks: `delivery_reward` (config field, defined and unused); `collisions` /
> `traffic_violations` (counted into `traj_metrics`, never entered into reward);
> `vlm_past_memory` + `vlm_add_memory/clear_memory` (exist, no trainer manages them);
> STAGE_1/2/3 (exist, no auto-progression).

## The 4 real bottlenecks

- **P1 — Credit assignment over a long turn-MDP with a lumpy payoff.** One `step()` = one LLM
  turn (look at FPV+map → reason → emit one action). A delivery is
  `ACCEPT → NAVIGATE → many MOVEs → PICKUP → many MOVEs → DROP_OFF`; `Δearnings` ≈ 0 across the
  approach and spikes only at `DROP_OFF`, so most turns get no gradient.
  → **Curriculum (ScalingInter)** keeps early horizons learnable; **Milestone credit
  (M-TRACK/SALT)** + **nav-progress shaping (LongNav-R1/HAPO/VLN-R1/TDR)** fill dead MOVE steps;
  **StepPO** makes the *turn* the credit unit, not the token.
- **P2 — Context growth.** Observation is rebuilt fresh each turn (Markovian), but the rollout
  transcript is full-history + a 4-way FPV cross + two map images *per turn* → explodes over
  hundreds of turns. → **SUPO summary** + **MGDM dual memory**; `vlm_past_memory` is the insertion point.
- **P3 — Exploration is wasted if uniform.** Decisions that matter are few and identifiable:
  order-pool selection, intersections (`### orientation` shows multiple reachable waypoints),
  charge/rest/switch timing. → **ARPO/AEPO** branch only at high-entropy turns.
- **P4 — Long multi-turn VLM RL is unstable.** → **GSPO/DAPO/REINFORCE++** as the optimizer base;
  **HiPER** hierarchy attacks the same instability by shrinking the effective horizon.

## Core ideas + env support

| Core idea (method) | Why it bites in DeliveryBench | Env hook to support it | Cost |
|---|---|---|---|
| **Horizon curriculum** (ScalingInter) | 2h episodes, success = ≥1 delivery; cold-starting on long multi-order trajectories collapses | STAGE_1/2/3 exist (`deliverybench_env.py:162-207`). Add **auto-promotion** when rolling success / on-time rate crosses a threshold; widen via `deadline_multiplier↓`, `max_orders_in_pool↑`, `num_restaurants/customers↑`, then flip `enable_*` flags | **Low** |
| **Milestone credit** (M-TRACK + SALT) | `Δearnings`≈0 across accept→pickup→approach, spikes only at DROP_OFF → long credit lag | Order FSM `is_accepted/has_picked_up/has_delivered` (`order.py:92-95`) → shaped bonus per transition + on reaching `pickup_node`/`dropoff_node`. **Use the already-unused `delivery_reward` field** (`:65`). SALT = back-assign step-advantage along success/fail traj using these checkpoints | **Low–Med** |
| **Nav progress + safety shaping** (LongNav-R1/HAPO/VLN-R1/TDR) | dead MOVE steps need a gradient; collisions/red-lights happen but cost nothing | Potential-based `Δ(dist→nav_target)` (`dm._nav_target_node` + route already computed); wire `collisions`/`traffic_violations` (`:708-710`, currently log-only) as per-event penalty; time-decay over the move sub-sequence. Keep potential-based → policy-invariant | **Low–Med** |
| **Step-MDP credit** (StepPO/HAPO baseline) | one turn = look+reason+1 action; token-level PPO smears credit | Env already returns a **per-turn reward**; `gym_agent_loop` appends `env_rewards` then sums (`:369/262`). Keep the vector, assign advantage at turn boundaries + a temporal (HAPO) baseline over turns | **Med** |
| **Running summary** (SUPO) | full transcript + FPV+2 maps per turn blows context over long traj | `vlm_past_memory` + `vlm_add_memory/clear_memory` exist (`delivery_man.py:185/481/484`) but unmanaged. Inject structured summary every K turns (order FSM states, visited nodes, energy/battery, risky edges); drop raw old turns | **Med** |
| **Dual memory** (MGDM/LH-VLN) | short-term (recent obstacle/light/junction) vs long-term (landmarks, customer/restaurant locations, failure nodes) | Same `vlm_past_memory` channel, two buckets; long-term keyed by waypoint id / POI. Pairs with SUPO | **Med** |
| **Uncertainty-targeted exploration** (ARPO/AEPO) | pivotal branch points are few; uniform sampling wastes budget | Branch rollouts only on high policy-entropy turns; `### orientation`/`reachable_waypoints` + `order_pool` sections mark exactly those states. Rollout-manager change | **Med–High** |
| **Hierarchy planner/executor** (HiPER/HiMAC) | shrinks effective horizon: high-level = order/route/resource, low-level = waypoint follow + avoid | Action set already splits cleanly — planner = `ACCEPT_ORDER/NAVIGATE(query-only tool)/SWITCH/BUY/CHARGE`, executor = `MOVE/PASSBY/WAIT`. Two prompts/policies. **Defer to phase 2** | **High** |
| **RL optimizer base** (GSPO/DAPO/CISPO/REINFORCE++) | long multi-turn VLM RL is unstable; sequence-level clip + sample filtering tame it | Optimizer choice in the VAGEN trainer (`main_ppo.py`), orthogonal to the env. Start GRPO/DAPO, move to GSPO/sequence-level as trajectories lengthen | **Low cfg / core** |

## Demoted / deferred

- **VLA-RFT (world-model RFT)** — its value is avoiding costly/risky real rollouts, but the sim is
  already cheap, deterministic, and gives a *verifiable* settlement reward (`rollout_scripted.py`
  proves no-API rollouts). A learned world model buys little; just run the real sim at scale.
- **Multi-agent** (SAY / help-board / temp-box) — Stage-3-only; defer until single-agent is solid.

## Suggested wiring order

1. **Curriculum auto-progression + milestone/nav/safety shaping** (rows 1–3). Mostly *activating
   dormant hooks*; biggest stability + credit-assignment win for the least code.
2. **Turn-level credit + optimizer base** (rows 4, 9). Make the turn the credit unit; pick GSPO/DAPO.
3. **Context summary + dual memory** (rows 5–6). Turn on once trajectories get long enough to hurt.
4. **Targeted exploration + hierarchy** (rows 7–8). Bring in when the flat policy plateaus.

---
*Reward today:* `reward_t = config.format_reward + (dm.earnings_total_current - dm.earnings_total_previous)`
(`deliverybench_env.py:1418-1436`). One `env.step` = one LLM action. `done` at `time_limit_hours`
(2h) or stuck-guard (3 repeated failed actions); `success` = ≥1 delivery.
