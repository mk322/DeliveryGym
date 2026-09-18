# F1: enable_waypoint_marks — marked-waypoint MOVE_TO environment integration

Status: IMPLEMENTED (2026-07-13), all gates passed. F0 probe complete & passed.
Owner context: continuation anchor after context compaction (2026-07-13).

## Implementation status (2026-07-13, final)
- Steps 1–6 all landed; flag-off byte-identical behaviour regression-tested.
- New/changed: vlm_delivery/base/defs.py (DMActionKind.MOVE_TO — name reused
  from the deleted legacy coordinate move), vlm_delivery/actions/move.py
  (enumerate_candidates + _execute_edge_step + handle_move_to; handle_move
  refactored onto the shared step executor), vlm_delivery/entities/
  delivery_man.py (dispatch), gameplay/action_space.py (spec/parse/examples/
  action_to_text, _MECHANIC_ACTIONS coupling), deliverybench_env.py (flag +
  plumbing + ### waypoint_marks obs text + FPV marker overlay + system-prompt
  note + reset album audit + 3-level image resolution), fpv_marks.py (NEW
  shared projection/marker module; tools/fpv_waypoint_marks re-exports),
  tools/generate_visual_sft_data.py (+marks oracle, gt_rel_direction, dm_x/y),
  tools/build_balanced_visual_sft_data.py (--waypoint-marks,
  --fpv-dir-overrides, gt-rel buckets), tools/render_fingerprint.py
  (+ render_fingerprint_baseline.json, 13 channels, --write/--check).
- Gates (scratchpad f1_gate1/4/5.py, all green):
  * action chain: 33 checks — 185-waypoint enumerate idempotence, ALL 414
    edges stepped to the right node with facing=edge bearing, invalid-mark
    error lists valid marks, blocked-chosen-edge fails in place, flag-off
    regression, MOVE/MOVE_TO coexistence.
  * observation: text count == markers drawn == candidates over 30 random
    MOVE_TO steps; album audit passes; no compass words; renders eyeballed
    (crossroad + straight on city-15, plus city-11 from the mini dataset).
  * datagen: MOVE oracle ≡ MOVE_TO oracle node sequences on 3 feasible seeds
    (by construction: same edge via available_moves, expressed as mark index);
    mini build balanced 3/3/3/3 from raw 65/8/18/17 with ZERO repeats
    (gt_rel_direction bucketing replaces the old backward upsampling);
    per-row checks (placeholders==images, target mark listed in obs, no
    next_move leak) all pass.
- Album census 2026-07-13: ALL 9 SFT maps pass the marks+FPV audit with
  default paths (scratchpad album_census.py). Root cause of the initial
  small-city-11 failure: the shared-disk dataset `small-city-11-rerun` was
  renamed onto the map root, leaving the 665-row dataset-subdir manifest
  pointing at dead absolute paths — fixed by the env's 3-level image
  resolution (local images/ → manifest image_path → map-root images/), and
  the audit caught it exactly as designed.
- Found on the way: int_41 on small-city-15 has TWO bearing-90 edges
  (dock_101 2.3 m + dock_103 15.4 m) — dock_101 is unreachable via MOVE,
  reachable via MOVE_TO (the Y-fork defect, live on a grid map).
- Full-size dataset when needed (P4 still frozen):
  `PYTHONPATH=. python -m vagen.envs.deliverybench.tools.\
  build_balanced_visual_sft_data --waypoint-marks --enable-fpv \
  --output-dir outputs/deliverybench_sft/visual_move_to_multicity_balanced`
  (vagen conda env — deliverybench env lacks pyarrow).
- Fingerprint discipline: run `tools/render_fingerprint.py --check` before any
  datagen/eval; baseline hashes are host-content-dependent (photos + DejaVu
  font), re-write deliberately after intended render changes.
- NOT done: any training (P4 frozen), F2 non-grid synthesis, BYPASS_TO,
  frontier-VLM probe.

## Goal
One config switch `enable_waypoint_marks` (default **False**) that, when on:
numbered glowing markers for one-hop reachable waypoints in the FPV cross +
per-step ephemeral candidate text + `MOVE_TO(k)` action. When off: env is
byte-identical to today (old MOVE untouched, coexists — user decision ③).

## Locked decisions (user-approved)
- ① vFOV for future F2 synthesis: uniform crop ~56° (not relevant to F1).
- ② Validate on existing GRID maps first (non-grid maps = external dependency,
  collaborators' procgen + UE bake). MOVE_TO ≡ MOVE reachable-set on grids.
- ③ Old MOVE stays, config-gated coexistence (PASSBY/BYPASS gating pattern).
- ④ Numbering: sort candidates by (absolute bearing_deg, dist, id), indices
  1..K, permanently stable per waypoint. Grammar accepts BOTH `MOVE_TO(3)`
  and `MOVE_TO("dock_94")`.
- P4 (any training) stays FROZEN. F2 (non-grid synthesis) separate, later.

## F0 constraints that MUST be honored (learned the hard way)
1. **Never put compass bearings in observation text** — the graph bearing axis
   is FLIPPED vs the rendered map's compass rose (probe smoke went 0/3 on this;
   fixed by wording). Use panel language ("in your FRONT VIEW") when FPV is on,
   MOVE-direction language ("forward of you") when map-only.
2. Markers must be drawn on the FINAL composed canvas (fixed pixel size,
   legible in ¼-scale panels) — implemented in tools/fpv_waypoint_marks.py.
3. Marker clamp margin for very-near candidates: raise 12px → ~30px
   (dead-end case showed clipping at panel bottom).
4. FPV album coverage gaps must FAIL LOUDLY at reset (hazard-sidecar precedent);
   F0 silently skipped 3/76 points with missing photos.
5. Obstacle interaction: MOVE_TO onto a blocked edge must fail-in-place +
   collision_count++ like MOVE-forward does (extend obstacle check in
   handle_move from forward-only to the chosen edge). BYPASS stays
   forward-bound; BYPASS_TO generalization is OUT OF SCOPE (documented item).

## Key conventions (verified, do not re-derive)
- FPV fetch: stored_yaw = (fpv_yaw_offset_deg − compass_dir) % 360, off=90
  (REFLECTION; deliverybench_env.py ~line 171 + _build_fpv_cross ~1897).
- Panel mapping WITH swap: front=facing+0, right=facing+270, back=+180,
  left=+90 (deliverybench_env.py _build_fpv_cross panels_dir).
- Projection: f = (W/2)/tan(45°) = 320px @640×480, cam height 160cm,
  ground point y = cy + f·h/(d·cosρ). `mirror` param UNCALIBRATED (invisible
  on grid maps — all candidates project to panel centers; calibrate in F2).
- `city_map.adjacents(node)` returns id/name/kind/dist_m/bearing_deg/compass/
  road_name/node. Node coords via node.position.x/.y.
- Env's map render: teleporting works (set dm.x/dm.y/facing_deg then
  `await env._get_map_images()`).
- FPV manifests: deliverybench_fpv/<map>/main_base_floor_road_full_1280x960_clean_floor/
  manifest.jsonl (small-city-11: no `_clean_floor` suffix). image_path entries
  are ABSOLUTE into the original VAGEN checkout; the env's own
  _load_fpv_lookup builds LOCAL root/images/... paths → local images/ dirs are
  MISSING in our checkout for small-city-15 etc. F1 must handle: either
  symlink images/ or make lookup fall back to manifest image_path.
- Obstacles: obstacles.json sidecar (7 edges on small-city-15), vision-only,
  MOVE(forward) into block → "blocked, unable to proceed", collision++, no move
  (move.py ~120-131); BYPASS() passes at 1.5×. Failed MOVE does NOT change facing.

## Work plan (order, with gates)
1. Config flag + plumbing (deliverybench_env.py, hazard-flag pattern).
2. Promote `enumerate_candidates()` from tools/fpv_waypoint_marks.py into
   vlm_delivery/actions/move.py beside available_moves (single source of truth
   for renderer/text/validator). Deterministic sort per decision ④.
3. MOVE_TO action: action_space.py spec+parse (gated wording) →
   DMActionKind.MOVE_TO → delivery_man.py dispatch (~line 345) →
   _handle_move_to (validate k or id ∈ candidates; error feedback lists valid
   marks; reuse handle_move step mechanics; facing := edge bearing_deg;
   obstacle check on chosen edge).
   GATE: unit round-trip — every waypoint: enumerate×2 identical; MOVE_TO(i)
   reaches the right node ∀i; invalid k errors with valid list; facing updated.
4. Observation: FPV marker overlay on composed cross (reuse
   tools/fpv_waypoint_marks drawing fns; move shared parts if needed);
   ephemeral text block (panel/move-dir language per F0 rule 1); system-prompt
   MOVE_TO spec (gated); reset-time album-coverage audit (fail loudly).
   GATE: text candidate count == rendered marker count; env-path renders match
   F0 gallery renders by eye (2–3 scenes).
5. Datagen: generate_visual_sft_data oracle labels = next-route-node → mark
   index → MOVE_TO(k); build_balanced_visual_sft_data MOVE_ACTION_RE → MOVE_TO
   regex; rebalance over GT relative-direction buckets (kills the backward
   upsampling problem); leak check: all markers rendered identically.
   GATE: oracle completes a full delivery on a grid map using ONLY MOVE_TO,
   node sequence identical to the MOVE oracle run; dataset generates & passes
   checks.
6. Fingerprint: add marked-FPV + obs-text hash to the render-fingerprint check.

Estimates (calibrated on session history): step 1–3 ≈ 1 day; 4 ≈ 0.5–1 day
(obs-builder internals = main unknown); 5–6 ≈ 0.5–1 day. Total 2–3 days.

## Out of scope for F1
F2 non-grid synthesis (panorama re-crop, mirror calibration, 56° vFOV);
any training (P4 frozen); BYPASS_TO; frontier-VLM probe (separate, needs
OPENROUTER_API_KEY; repo pattern in tools/obstacle_avoidance_eval.py).

## Assets & artifacts (already built, F0)
- tools/fpv_waypoint_marks.py — projection/markers/cross composition (reuse!).
- tools/fpv_move_to_probe.py — collect/eval/percept phases.
- Probe data: scratchpad probe_f0/ {decisions.json (76 pts), results_A/E/B.json,
  results_percept.json, img/}. Key numbers: format compliance 228/228 = 100%;
  percept mark-reading 93.9% exact, 0 missed; decision floor (zero-shot 9B):
  A .370 / E .342 / B .684 vs majority-class .645, turn-acc ≤.28 all arms.
- Reports (outputs/deliverybench_sft/, served :8137, restart cmd: setsid
  python -m http.server 8137 in that dir): f0_move_to_probe_report.html,
  f0_render_gallery.html (8 scene types incl. obstacle pair),
  fpv_panorama_literature_review.html (+ copy in "literature review/"),
  env_change_impact.html, env_migration_report.html.
- Gallery source imgs: scratchpad gallery/ (marked crosses, envmaps, obst pairs).

## Wider project context (one line each)
- MOVE-only LoRA (ckpt-315, full_all_linear_sdpa_4gpu_bs32_5ep_cu124_curenv)
  = current-env realigned model: 0.95 per-step, ~56% scaffolded rollout.
- Env renderer drift (July) broke the old model 89%→22%; render-fingerprint
  discipline exists because of that (check before any eval).
- dev/zenith == origin/dev/env_v2 @ 71f7e81 (synced 7/12; renderer untouched,
  fingerprint verified 31d669884b).
- GPU box shared; GPU0 has ECC errors (avoid); long jobs MUST be launched
  `setsid nohup ... & disown` (session teardown SIGTERM killed a training once).
- Model loading for local eval: qwen_env python, cudnn disabled, sdpa; loaders
  in tools/qwen35_lora_rollout_full.py (_load_base_model) / rollout_eval.
