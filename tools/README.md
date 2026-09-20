# tools/

Two kinds of thing live here: what a user of the benchmark runs, and the
pieces those entry points are built from.

**Entry points**

| file | what it does |
|---|---|
| `adaptive_profile.py` | the curriculum sidecar: turns the episode log into sampling weights (`experiments/README.md`) |
| `case_study.py` | roll a checkpoint and its base model on unseen tasks and render the comparison as one HTML page |
| `run_pixel_goal_front_rear_delivery.sh` / `.py` | the any-point (pixel-goal) evaluation: one certified delivery in the live engine (`docs/PIXEL_GOAL.md`) |
| `build_trusted_pedestrian_order_pool.py`, `validate_pixel_goal_order_pool_live.py`, `audit_trusted_pedestrian_graph_live.py` | the certified order pool: rebuild it from a fresh engine surface audit and re-certify it against a running engine (`docs/PIXEL_GOAL.md`) |

**Library** (imported by the entry points, each with its own tests):
`pixel_goal_order_pool.py` (loading and resolving the pool),
`pixel_goal_courier_backend.py` (the courier world over a SPEAR pawn),
`pixel_goal_capture_preflight.py`, `pixel_goal_launcher_identity.py`,
`pixel_goal_range_metrics.py`, `pixel_goal_1b_poc_spear.yaml`, and the
milestone modules the delivery runner grew out of — `run_pixel_goal_m1a.py`,
`run_pixel_goal_1b_poc.*`, `run_pixel_goal_1b_closed_loop.*`,
`run_pixel_goal_full_delivery.*`, `run_pixel_goal_front_rear_probe.*`.
They are not benchmark entry points.

**Album bakers** (`ue/`): drive a SimWorld UE editor to render a city's
street, pavement, lamp and obstacle albums (the bakers themselves document the
steps). They need the editor (`UE_ROOT`), the CityCore content and a GPU;
nothing else in the repository does.
