"""Canonical valid and invalid fixtures for every contract (the design plan M1).

``valid_fixtures()`` returns one well-formed instance per versioned envelope, so
the round-trip, unknown-field, and version tests can enumerate the registry
rather than keep a hand-written list that drifts as contracts are added.
``COVERAGE`` asserts that enumeration is complete.

``invalid_fixtures()`` is the more interesting half: each entry is a payload
that *must* be rejected, paired with the the design plan rule it would otherwise
violate. These are the cases where a permissive schema would let a real defect
through — a synthetic connector certified without NavMesh evidence, a step that
both terminates and truncates, a loss mask containing observation tokens.
"""

from __future__ import annotations

from typing import Any, Iterator, NamedTuple

from embodiedbench.schemas.base import SchemaModel, schema_registry
from embodiedbench.schemas.env_spec import (
    AffordanceInventory,
    EnvSpec,
    GraphRepairSummary,
    GraphSummary,
    NavigationStyle,
    ObservationSupport,
    QualityFlag,
    SolvabilityEvidence,
)
from embodiedbench.schemas.embodiment import (
    ControllerOutcome,
    ControllerResult,
    DistanceSource,
    EmbodimentCapabilities,
    EmbodimentProfile,
    NavigationRequest,
)
from embodiedbench.schemas.environment import (
    AffordanceRequirement,
    Certificate,
    CertificationGrade,
    EnvironmentBundle,
    NavigationMode,
    OverlayPlacement,
    OverlaySpec,
    RemovalManifest,
    SensorSpec,
)
from embodiedbench.schemas.episode import (
    Budgets,
    CourierProfile,
    EpisodeSpec,
    ScheduledOrder,
    TaskConfigRef,
)
from embodiedbench.schemas.geometry import (
    CameraIntrinsics,
    CoordinateFrame,
    FrameName,
    Pose,
    Vec3,
)
from embodiedbench.schemas.runtime import (
    ActionEnvelope,
    ActionResult,
    ActionStatus,
    Event,
    ImagePoint,
    MarkCandidate,
    MediaRef,
    NavPoint3DAction,
    NavWaypointAction,
    Observation,
    ResetInfo,
    RuntimeCapabilities,
    RuntimeMode,
    StepResult,
    TerminationReason,
)
from embodiedbench.schemas.trajectory import (
    CostAccounting,
    MetricValue,
    ScoreReport,
    TokenAccounting,
    Trajectory,
    TrajectoryTurn,
)
from embodiedbench.schemas.world import (
    EdgeProvenance,
    EdgeStatus,
    InteractionSite,
    LabelProvenance,
    LocomotionMode,
    NavEdge,
    NavGraph,
    NavMeshEvidence,
    NavNode,
    NodeKind,
    SemanticEntity,
    SourceProvenance,
    WorldBundle,
)

SHA = "a" * 64
EPISODE = "ep_0001"


def _nav_graph() -> NavGraph:
    nodes = [
        NavNode(node_id="int_1", kind=NodeKind.JUNCTION, position=Vec3(x_cm=0, y_cm=0),
                road_name="Rue de Rivoli"),
        NavNode(node_id="int_2", kind=NodeKind.JUNCTION, position=Vec3(x_cm=1850, y_cm=0),
                road_name="Rue de Rivoli"),
        NavNode(node_id="dock_1", kind=NodeKind.DOCK, position=Vec3(x_cm=1850, y_cm=400)),
    ]
    edges = [
        NavEdge(
            edge_id="e_1",
            from_node="int_1",
            to_node="int_2",
            polyline=[Vec3(x_cm=0, y_cm=0), Vec3(x_cm=1850, y_cm=0)],
            length_cm=1850.0,
            provenance=EdgeProvenance.AUTHORED_CENTERLINE,
            status=EdgeStatus.CERTIFIED,
            allowed_modes=[LocomotionMode.WALK, LocomotionMode.SCOOTER],
            navmesh_evidence=NavMeshEvidence(path_found=True, path_length_cm=1852.0, waypoint_count=3),
        ),
        NavEdge(
            edge_id="e_2",
            from_node="int_2",
            to_node="dock_1",
            polyline=[Vec3(x_cm=1850, y_cm=0), Vec3(x_cm=1850, y_cm=400)],
            length_cm=400.0,
            provenance=EdgeProvenance.AUTHORED_CENTERLINE,
            status=EdgeStatus.CERTIFIED,
            allowed_modes=[LocomotionMode.WALK],
            navmesh_evidence=NavMeshEvidence(path_found=True, path_length_cm=401.0, waypoint_count=2),
        ),
        NavEdge(
            edge_id="e_3_synthetic",
            from_node="int_1",
            to_node="dock_1",
            polyline=[Vec3(x_cm=0, y_cm=0), Vec3(x_cm=1850, y_cm=400)],
            length_cm=4528.0,
            provenance=EdgeProvenance.SYNTHETIC_CONNECTOR,
            status=EdgeStatus.EXCLUDED,
            exclusion_reason="unvalidated 45.3 m connector (design plan §3.2); may cross non-road space",
        ),
    ]
    return NavGraph(nodes=nodes, edges=edges, components=[["int_1", "int_2", "dock_1"]])


def _world_bundle() -> WorldBundle:
    return WorldBundle(
        world_id="citycore-paris",
        version="0.1.0",
        frames=[
            CoordinateFrame(name=FrameName.UE_WORLD, units="cm", description="UE authored frame"),
            CoordinateFrame(name=FrameName.BUNDLE_WORLD, units="cm"),
        ],
        nav_graph=_nav_graph(),
        entities=[
            SemanticEntity(
                entity_id="rest_1",
                entity_type="restaurant",
                position=Vec3(x_cm=1900, y_cm=420),
                provenance=LabelProvenance.AUTHORED_METADATA,
                display_name="Le Bistro",
            ),
            SemanticEntity(
                entity_id="shop_1",
                entity_type="store",
                position=Vec3(x_cm=100, y_cm=50),
                provenance=LabelProvenance.VLM_SUGGESTION,
                confidence=0.72,
            ),
        ],
        interaction_sites=[
            InteractionSite(
                site_id="dock_site_1",
                site_type="customer_dock",
                position=Vec3(x_cm=1850, y_cm=400),
                nearest_node="dock_1",
                entity_id="rest_1",
                reachable_modes=[LocomotionMode.WALK],
            )
        ],
        source=SourceProvenance(
            ue_project="SimWorld",
            engine_version="5.8.0-0+UE5",
            level_package="/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints",
            compiler_version="0.1.0",
            actor_count=3290,
        ),
        previews=["previews/topdown.webp"],
    )


def _overlay() -> OverlaySpec:
    return OverlaySpec(
        overlay_id="deliverybench-paris",
        version="0.1.0",
        task_plugin="delivery@1",
        base_world_id="citycore-paris",
        base_world_version="0.1.0",
        requires_affordances={"scooter_spawn": AffordanceRequirement(min=1)},
        placements=[
            OverlayPlacement(
                placement_id="p_1",
                affordance="scooter_spawn",
                position=Vec3(x_cm=1800, y_cm=100),
                spawned_asset_path="/Game/TaskAssets/Delivery/BP_ScooterDock",
                data_layer="DL_DeliveryOverlay",
                rule="nearest_certified_node_with_clearance",
                seed=0,
                nearest_node="int_2",
            ),
            OverlayPlacement(
                placement_id="p_2",
                affordance="restaurant",
                position=Vec3(x_cm=1900, y_cm=420),
                reused_entity_id="rest_1",
                rule="reuse_authored_restaurant",
                seed=0,
            ),
        ],
        removal=RemovalManifest(data_layers=["DL_DeliveryOverlay"], source_umap_sha256=SHA),
    )


def _environment() -> EnvironmentBundle:
    return EnvironmentBundle(
        environment_id="citycore-paris-delivery",
        version="0.1.0",
        base_world_id="citycore-paris",
        base_world_version="0.1.0",
        base_world_sha256=SHA,
        overlay_id="deliverybench-paris",
        overlay_version="0.1.0",
        supported_embodiment_profiles=["abstract_courier_v1"],
        supported_navigation_modes=[NavigationMode.NAV_WAYPOINT],
        supported_runtimes=["text", "cached"],
        sensors=SensorSpec(channels=["rgb", "depth"], intrinsics=_intrinsics()),
        certificates=[_certificate()],
    )


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width_px=640, height_px=480, fx_px=320.0, fy_px=320.0,
        cx_px=320.0, cy_px=240.0, near_cm=10.0, far_cm=100000.0,
    )


def _certificate() -> Certificate:
    return Certificate(
        environment_id="citycore-paris-delivery",
        environment_version="0.1.0",
        task_plugin="delivery",
        task_plugin_version="0.1.0",
        embodiment_profile="abstract_courier_v1",
        navigation_mode=NavigationMode.NAV_WAYPOINT,
        runtime="text",
        grade=CertificationGrade.B,
        checks_passed=["schema", "graph_topology"],
        limitations=["no NavMesh evidence for the shipped map (M0-F6)"],
        certified_road_coverage=0.91,
    )


def _capabilities() -> EmbodimentCapabilities:
    return EmbodimentCapabilities(
        profile_id="abstract_courier_v1",
        locomotion_modes=[LocomotionMode.WALK, LocomotionMode.SCOOTER],
        navigation_modes=[NavigationMode.NAV_WAYPOINT],
        max_range_m=18.0,
        max_speed_m_s=6.0,
        radius_cm=35.0,
        height_cm=180.0,
    )


def _observation() -> Observation:
    return Observation(
        episode_id=EPISODE,
        step_index=0,
        text="You are at Rue de Rivoli facing north.",
        media=[MediaRef(channel="rgb", path="observations/rgb/int_1_090.jpg", sha256=SHA,
                        width_px=640, height_px=480)],
        marks=[MarkCandidate(mark=0, node_id="int_2", bearing_deg=90.0, distance_m=18.5)],
        agent_pose=Pose(frame=FrameName.BUNDLE_WORLD, position=Vec3(x_cm=0, y_cm=0), yaw_deg=0.0),
        available_actions=["nav_waypoint", "task"],
        budgets_remaining={"steps": 400.0},
    )


def _action_envelope() -> ActionEnvelope:
    return ActionEnvelope(
        episode_id=EPISODE,
        step_index=0,
        action=NavWaypointAction(target_node="int_2"),
        subgoal="head toward the restaurant",
    )


def _event() -> Event:
    return Event(
        event_id="ev_0001",
        episode_id=EPISODE,
        step_index=0,
        kind="moved",
        sim_time_s=12.5,
        payload={"from": "int_1", "to": "int_2"},
    )


def _step_result() -> StepResult:
    return StepResult(
        observation=_observation(),
        reward=1.5,
        reward_components={"progress": 1.0, "time_cost": 0.5},
        terminated=False,
        truncated=False,
        events=[_event()],
        action_result=ActionResult(status=ActionStatus.ACCEPTED, resolved_target_node="int_2"),
        metrics_delta={"distance_cm": 1850.0},
        privileged_state_ref="private://ep_0001/step_0",
    )


def _courier() -> CourierProfile:
    return CourierProfile(
        profile_id="scooter_standard",
        transport_modes=["walk", "scooter"],
        carrying_capacity=3,
        owns_scooter=True,
        battery_enabled=True,
        outcome_relevant_fields=[
            "transport_modes", "carrying_capacity", "owns_scooter", "battery_enabled",
        ],
    )


def _episode_spec() -> EpisodeSpec:
    return EpisodeSpec(
        instance_id="dbam_v1_paris_000123",
        environment_id="citycore-paris-delivery",
        environment_version="0.1.0",
        environment_sha256=SHA,
        task=TaskConfigRef(plugin="delivery", version="0.1.0", config_sha256=SHA),
        embodiment_profile="abstract_courier_v1",
        courier_profile=_courier(),
        navigation_mode=NavigationMode.NAV_WAYPOINT,
        runtime_track=RuntimeMode.TEXT,
        seed=4711,
        spawn=Pose(frame=FrameName.BUNDLE_WORLD, position=Vec3(x_cm=0, y_cm=0), yaw_deg=90.0),
        order_schedule=[
            ScheduledOrder(order_id="o_0", available_from_sim_time_s=0.0,
                           pickup_site="dock_site_1", dropoff_site="dock_site_1"),
            ScheduledOrder(order_id="o_1", available_from_sim_time_s=300.0,
                           pickup_site="dock_site_1", dropoff_site="dock_site_1"),
        ],
        observable_instruction="Deliver as much value as you can before your shift ends.",
        budgets=Budgets(steps=400, tool_calls=80, sim_s=7200.0, output_tokens=120000),
        evaluator_id="delivery_score",
        evaluator_version="0.1.0",
    )


def _trajectory() -> Trajectory:
    turn = TrajectoryTurn(
        step_index=0,
        observation=_observation(),
        observation_text="You are at Rue de Rivoli facing north.",
        raw_model_output='{"action": {"type": "nav_waypoint", "target_node": "int_2"}}',
        action=_action_envelope(),
        action_result=ActionResult(status=ActionStatus.ACCEPTED),
        events=[_event()],
        reward=1.5,
        reward_components={"progress": 1.0, "time_cost": 0.5},
        tokens=TokenAccounting(
            prompt_tokens=900,
            response_tokens=24,
            response_loss_mask=list(range(24)),
            observation_token_indices=[],
            image_token_indices=[],
        ),
        cost=CostAccounting(model_latency_s=0.8, environment_latency_s=0.002),
    )
    return Trajectory(
        episode_id=EPISODE,
        instance_id="dbam_v1_paris_000123",
        environment_id="citycore-paris-delivery",
        environment_version="0.1.0",
        task_plugin="delivery",
        task_plugin_version="0.1.0",
        model_id="scripted-oracle",
        harness_version="0.1.0",
        seed=4711,
        turns=[turn],
        total_reward=1.5,
    )


def _score_report() -> ScoreReport:
    return ScoreReport(
        instance_id="dbam_v1_paris_000123",
        episode_id=EPISODE,
        evaluator_id="delivery_score",
        evaluator_version="0.1.0",
        success=True,
        normalized_utility_vs_upper_bound=0.42,
        metrics={"deliveries": MetricValue(value=3.0, ci_low=2.1, ci_high=3.9)},
        costs={"output_tokens": 4200.0},
    )


def _env_spec() -> EnvSpec:
    """A compiled environment, shaped like real pipeline output for Paris."""
    return EnvSpec(
        env_id="citycore-paris-env",
        map_name="citycore-paris",
        navigation_style=NavigationStyle.GRAPH,
        navigation_modes=[NavigationMode.NAV_WAYPOINT],
        enabled_actions=["VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF", "WAIT", "MOVE_TO"],
        navigation_rationale="only 23.1% of edges are near-cardinal (< 60%)",
        graph=GraphSummary(
            node_count=1162, edge_count=5649, mean_degree=9.72, max_degree=35,
            dock_nodes=443, junction_nodes=719, cardinal_fraction=0.231,
            largest_component_fraction=1.0, component_count=1,
            median_edge_m=25.0, longest_edge_m=313.0, long_edge_threshold_m=150.1,
        ),
        graph_repair=GraphRepairSummary(
            applied=True, converged=True, passes_note="reached a fixpoint after 7 pass(es)",
            edges_before=7230, edges_after=5649, edges_split=2838,
            skipped_nodes_recovered=4540, mean_degree_before=12.44, mean_degree_after=9.72,
            longest_edge_before_m=640.0, longest_edge_after_m=313.0,
        ),
        affordances=AffordanceInventory(counts={"building": 414, "restaurant": 18, "store": 11}),
        observation=ObservationSupport(channels=["text"]),
        grade=CertificationGrade.B,
        quality_flags=[
            QualityFlag(code="over_connected_nodes", count=635, stage="graph"),
            QualityFlag(code="degenerate_edges", count=2, stage="source"),
        ],
        solvability=SolvabilityEvidence(
            episodes=5, delivered_episodes=5, solvability_rate=1.0, mean_steps=21.8
        ),
    )


def valid_fixtures() -> dict[str, SchemaModel]:
    """One valid instance per versioned envelope schema."""
    return {
        "embodiedbench/env_spec": _env_spec(),
        "embodiedbench/world_bundle": _world_bundle(),
        "embodiedbench/nav_graph": _nav_graph(),
        "embodiedbench/overlay_spec": _overlay(),
        "embodiedbench/environment_bundle": _environment(),
        "embodiedbench/certificate": _certificate(),
        "embodiedbench/observation": _observation(),
        "embodiedbench/action_envelope": _action_envelope(),
        "embodiedbench/event": _event(),
        "embodiedbench/step_result": _step_result(),
        "embodiedbench/reset_info": ResetInfo(
            episode_id=EPISODE,
            seed=4711,
            environment_id="citycore-paris-delivery",
            environment_version="0.1.0",
            runtime_mode=RuntimeMode.TEXT,
        ),
        "embodiedbench/runtime_capabilities": RuntimeCapabilities(
            mode=RuntimeMode.TEXT,
            observation_channels=["text"],
            navigation_modes=[NavigationMode.NAV_WAYPOINT],
            supports_snapshot=True,
        ),
        "embodiedbench/embodiment_capabilities": _capabilities(),
        "embodiedbench/embodiment_profile": EmbodimentProfile(
            profile_id="abstract_courier_v1", capabilities=_capabilities()
        ),
        "embodiedbench/navigation_request": NavigationRequest(
            request_id="req_1",
            mode=NavigationMode.NAV_WAYPOINT,
            distance_source=DistanceSource.GRAPH_EDGE,
            target_node="int_2",
            max_range_m=18.0,
        ),
        "embodiedbench/controller_result": ControllerResult(
            request_id="req_1",
            outcome=ControllerOutcome.ACCEPTED,
            final_pose=Pose(frame=FrameName.BUNDLE_WORLD, position=Vec3(x_cm=1850, y_cm=0)),
            elapsed_sim_s=3.1,
            distance_travelled_cm=1850.0,
        ),
        "embodiedbench/courier_profile": _courier(),
        "embodiedbench/episode_spec": _episode_spec(),
        "embodiedbench/trajectory": _trajectory(),
        "embodiedbench/score_report": _score_report(),
    }


class InvalidFixture(NamedTuple):
    """A payload that must be rejected, and the rule it would violate."""

    name: str
    model: type[SchemaModel]
    payload: dict[str, Any]
    rule: str


def _mutate(base: SchemaModel, **overrides: Any) -> dict[str, Any]:
    payload = base.to_dict()
    payload.update(overrides)
    return payload


def invalid_fixtures() -> list[InvalidFixture]:
    """Payloads that must fail, each tied to the the design plan rule it breaks."""
    graph = _nav_graph()
    step = _step_result()
    trajectory = _trajectory()
    return [
        InvalidFixture(
            "synthetic_edge_certified_without_navmesh_evidence",
            NavGraph,
            _mutate(
                graph,
                edges=[
                    {
                        **graph.edges[2].to_dict(),
                        "status": "certified",
                        "exclusion_reason": "",
                    }
                ]
                + [e.to_dict() for e in graph.edges[:2]],
            ),
            "design plan §6.1 P2 / 17: synthetic links are unsafe until a NavMesh path confirms them",
        ),
        InvalidFixture(
            "excluded_edge_without_reason",
            NavGraph,
            _mutate(
                graph,
                edges=[{**graph.edges[2].to_dict(), "exclusion_reason": ""}]
                + [e.to_dict() for e in graph.edges[:2]],
            ),
            "the design plan M2: every excluded edge must record why",
        ),
        InvalidFixture(
            "edge_references_unknown_node",
            NavGraph,
            _mutate(
                graph,
                edges=[{**graph.edges[0].to_dict(), "to_node": "does_not_exist"}],
            ),
            "design plan §5.1: stable IDs with referential integrity",
        ),
        InvalidFixture(
            "vlm_label_without_confidence",
            WorldBundle,
            _mutate(
                _world_bundle(),
                entities=[
                    {
                        "entity_id": "shop_1",
                        "entity_type": "store",
                        "position": {"x_cm": 100.0, "y_cm": 50.0, "z_cm": 0.0},
                        "provenance": "vlm_suggestion",
                    }
                ],
            ),
            "design plan §6.1 P3: VLM proposals persist with confidence for review",
        ),
        InvalidFixture(
            "absolute_path_in_bundle",
            WorldBundle,
            _mutate(_world_bundle(), previews=["/srv/previews/topdown.webp"]),
            "design plan §5.1: no absolute machine-specific paths in portable artifacts",
        ),
        InvalidFixture(
            "escaping_relative_path",
            WorldBundle,
            _mutate(_world_bundle(), previews=["../../etc/passwd"]),
            "design plan §5.1: bundle paths must stay inside the bundle",
        ),
        InvalidFixture(
            "placement_spawns_without_data_layer",
            OverlaySpec,
            _mutate(
                _overlay(),
                placements=[
                    {
                        "placement_id": "p_1",
                        "affordance": "scooter_spawn",
                        "position": {"x_cm": 1800.0, "y_cm": 100.0, "z_cm": 0.0},
                        "spawned_asset_path": "/Game/TaskAssets/Delivery/BP_ScooterDock",
                        "rule": "r",
                        "seed": 0,
                    }
                ],
            ),
            "design plan §5.2.3: spawn or modify only through a named UE Data Layer",
        ),
        InvalidFixture(
            "overlay_misses_its_own_requirement",
            OverlaySpec,
            _mutate(_overlay(), placements=[]),
            "design plan §5.2: an overlay must satisfy the affordance minimums it declares",
        ),
        InvalidFixture(
            "grade_a_with_declared_limitations",
            Certificate,
            _mutate(_certificate(), grade="A"),
            "design plan §6.2: grade A requires all checks to pass with nothing unreviewed",
        ),
        InvalidFixture(
            "point_mode_without_pose_lattice",
            EnvironmentBundle,
            _mutate(_environment(), supported_navigation_modes=["nav_point_3d"]),
            "design plan §9.5: point modes need a runtime-independent pose lattice",
        ),
        InvalidFixture(
            "terminated_and_truncated_together",
            StepResult,
            _mutate(step, terminated=True, truncated=True, termination_reason="task_success"),
            "design plan §7: task termination and budget truncation are distinct",
        ),
        InvalidFixture(
            "truncated_with_a_termination_cause",
            StepResult,
            _mutate(step, terminated=False, truncated=True, termination_reason="task_success"),
            "the design plan M1: terminated and truncated have separate causes",
        ),
        InvalidFixture(
            "ended_without_a_reason",
            StepResult,
            _mutate(step, terminated=True),
            "design plan §7: termination must state its reason",
        ),
        InvalidFixture(
            "reward_components_do_not_sum",
            StepResult,
            _mutate(step, reward=99.0),
            "design plan §7: named reward components must account for the reward",
        ),
        InvalidFixture(
            "rejected_action_without_error_code",
            StepResult,
            _mutate(step, action_result={"status": "rejected", "message": "no"}),
            "the design plan M6: every invalid action receives typed feedback",
        ),
        InvalidFixture(
            "nav_point_2d_carrying_a_model_distance",
            ActionEnvelope,
            _mutate(
                _action_envelope(),
                action={
                    "type": "nav_point_2d_depth",
                    "frame": "camera",
                    "target": {"u_norm": 0.6, "v_norm": 0.7, "distance_m": 6.4},
                },
            ),
            "design plan §9.3: range comes from the external depth provider, not the model",
        ),
        InvalidFixture(
            "nav_waypoint_with_both_node_and_mark",
            ActionEnvelope,
            _mutate(
                _action_envelope(),
                action={"type": "nav_waypoint", "target_node": "int_2", "target_mark": 0},
            ),
            "design plan §9.1: a mark resolves to a node before the action is recorded",
        ),
        InvalidFixture(
            "unknown_action_type",
            ActionEnvelope,
            _mutate(_action_envelope(), action={"type": "teleport", "target_node": "int_2"}),
            "design plan §10.2: actions are schema-validated",
        ),
        InvalidFixture(
            "observation_tokens_inside_the_loss_mask",
            Trajectory,
            _mutate(
                trajectory,
                turns=[
                    {
                        **trajectory.turns[0].to_dict(),
                        "tokens": {
                            "prompt_tokens": 900,
                            "response_tokens": 24,
                            "response_loss_mask": list(range(24)),
                            "observation_token_indices": [5, 6],
                            "image_token_indices": [],
                        },
                    }
                ],
            ),
            "design plan §2.1 / 11.1 / M6: observation and tool tokens are excluded from policy loss",
        ),
        InvalidFixture(
            "duplicate_step_index_from_a_retry",
            Trajectory,
            _mutate(
                trajectory,
                turns=[trajectory.turns[0].to_dict(), trajectory.turns[0].to_dict()],
                total_reward=3.0,
            ),
            "design plan §12.6: retrying a step cannot duplicate reward or events",
        ),
        InvalidFixture(
            "total_reward_disagrees_with_turns",
            Trajectory,
            _mutate(trajectory, total_reward=99.0),
            "design plan §11.1: total reward is preserved exactly once",
        ),
        InvalidFixture(
            "courier_hides_an_outcome_relevant_extra",
            CourierProfile,
            _mutate(_courier(), extra={"secret_speed_bonus": 1.25}),
            "design plan §17: hidden courier attributes make instances unfair and uninterpretable",
        ),
        InvalidFixture(
            "order_schedule_out_of_time_order",
            EpisodeSpec,
            _mutate(
                _episode_spec(),
                order_schedule=[
                    {"order_id": "o_1", "available_from_sim_time_s": 300.0,
                     "pickup_site": "dock_site_1", "dropoff_site": "dock_site_1"},
                    {"order_id": "o_0", "available_from_sim_time_s": 0.0,
                     "pickup_site": "dock_site_1", "dropoff_site": "dock_site_1"},
                ],
            ),
            "design plan §12.6: divergent policies must observe the same order schedule",
        ),
        InvalidFixture(
            "accepted_controller_result_without_final_pose",
            ControllerResult,
            {
                "schema": "embodiedbench/controller_result",
                "schema_version": ControllerResult.SCHEMA_VERSION,
                "request_id": "req_1",
                "outcome": "accepted",
                "elapsed_sim_s": 1.0,
            },
            "design plan §9.6: ControllerResult reports the final pose it reached",
        ),
        InvalidFixture(
            "point_runtime_without_declared_range",
            RuntimeCapabilities,
            {
                "schema": "embodiedbench/runtime_capabilities",
                "schema_version": RuntimeCapabilities.SCHEMA_VERSION,
                "mode": "cached",
                "navigation_modes": ["nav_point_3d"],
            },
            "design plan §9.4: point modes must declare max_range_m",
        ),
        InvalidFixture(
            "nav_request_mode_and_distance_source_disagree",
            NavigationRequest,
            {
                "schema": "embodiedbench/navigation_request",
                "schema_version": NavigationRequest.SCHEMA_VERSION,
                "request_id": "req_1",
                "mode": "nav_point_2d_depth",
                "distance_source": "model_prediction",
                "source_image_point": {"u_norm": 0.5, "v_norm": 0.5},
                "max_range_m": 18.0,
            },
            "design plan §9.3: external depth determines range in nav_point_2d_depth",
        ),
    ]


COVERAGE_EXEMPT: frozenset[str] = frozenset()


def uncovered_schemas() -> set[str]:
    """Versioned envelopes with no valid fixture — must be empty (the design plan M1)."""
    registered = {
        schema_id for schema_id, model in schema_registry().items() if model.VERSIONED_ENVELOPE
    }
    return registered - set(valid_fixtures()) - COVERAGE_EXEMPT


def iter_valid() -> Iterator[tuple[str, SchemaModel]]:
    yield from sorted(valid_fixtures().items())
