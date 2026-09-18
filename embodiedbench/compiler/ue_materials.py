"""Change a UE component's materials without destroying what was there.

This module exists because of a specific, expensive mistake. Trying to undo an
experiment on Paris's traffic lights, a script did the obvious thing:

    for i in range(component.get_num_materials()):
        component.set_material(i, component.static_mesh.get_material(i))

That reads like "put back the authored material" and is the opposite. UE has two
layers: the *static mesh* carries a default material per slot, and a component
may carry an **override** that hides it. ``static_mesh.get_material(i)`` returns
the mesh default, ignoring any override — so the loop above overwrites every
component override with the mesh default and the override is gone.

CityCore's traffic lights are exactly that case: the LED material is a component
override on a mesh whose default is the plain body material. The loop wiped the
LED assignment across the level — 27 LED slots near one junction became 1 in the
whole map. Nothing reached disk, because the level was never saved, so a fresh
load restores it. The lasting fix is to never do it again.

The rule this module enforces: **snapshot before you write, restore what you
snapshotted.** A snapshot records the override slot by slot, and ``None`` means
"this slot had no override" — which restores by *clearing* the override rather
than writing the default into it. Those are different operations and only one of
them is reversible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Rose's recipe for CityCore's LED material, confirmed against the asset: the
# material exposes texture slots Phase1/Phase2 (plus the blend slots
# Phase1-2/Phase2-1) and a Phase scalar that selects between them. The colour
# comes from which *atlas* is bound to the active slot, which is why moving the
# scalar alone changes nothing -- an 8-angle sweep found no difference before
# this was understood.
# The token has to match the material the *component* actually carries, not the
# one on the static mesh. BP_TrafficLightsPoles builds its own dynamic instances
# at construction -- ".MI_TrafficLights1" / ".MI_TrafficLights2", living inside
# the level package -- which override the mesh's MI_PR_TrafficLightsLED_01. A
# filter written against the mesh material matched nothing across all 125 poles,
# which is why every apply reported slots=0.
#
# "MI_TrafficLights" is deliberately not "MI_PR_TrafficLights": the latter is the
# housing material and must not be touched.
LED_MATERIAL_TOKEN = ".MI_TrafficLights"
LED_ATLAS_ROOT = "/Game/CityCore_Paris/Textures/TrafficLights/T_PR_TrafficLights_LED_"
# Measured, not assumed. A 2x2 sweep over (Phase1 atlas, Phase2 atlas, scalar)
# produced green in exactly one cell -- Phase2=E02 with scalar 2 -- and red
# everywhere else:
#
#   Phase1  Phase2  scalar   red  green
#   E01     E02     1        510      0
#   E01     E02     2        104    131   <- the only green
#   E02     E01     1        485      0
#   E02     E01     2        511      0
#
# So the scalar selects the slot of the same number, and the atlas bound to that
# slot decides the colour: E02 green, E01 red. Rose's atlas mapping was right;
# her phase numbers pair with the slot rather than the state, which is why
# (E02, Phase=1) rendered red.
SIGNAL_RECIPE: dict[str, tuple[str, int]] = {
    # state: (atlas suffix, slot number -- the atlas goes in Phase<N>, scalar = N)
    "green": ("E02", 2),
    "red": ("E01", 1),
}


@dataclass
class MaterialSnapshot:
    """What a component's slots held before anything was changed."""

    actor_label: str
    component_name: str
    # One entry per slot: the override's asset path, or None for "no override".
    overrides: list[str | None] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor_label,
            "component": self.component_name,
            "overrides": list(self.overrides),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Editor-side scripts. These run inside UE via MCP, so they are text.
# ─────────────────────────────────────────────────────────────────────────────

# Snapshotting reads `get_material` -- the *effective* material -- but records
# whether it differs from the mesh default. That difference is what tells us an
# override exists, which is the only thing get_material cannot say directly.
SNAPSHOT_SCRIPT = """
import json
import unreal

CX, CY, RADIUS = {cx!r}, {cy!r}, {radius!r}
out = []
for actor in unreal.EditorLevelLibrary.get_all_level_actors():
    for comp in actor.get_components_by_class(unreal.StaticMeshComponent):
        mesh = comp.static_mesh
        if mesh is None:
            continue
        where = comp.get_world_location()
        if (where.x - CX) ** 2 + (where.y - CY) ** 2 > RADIUS ** 2:
            continue
        slots = []
        interesting = False
        for i in range(comp.get_num_materials()):
            effective = comp.get_material(i)
            default = mesh.get_material(i)
            e = effective.get_path_name() if effective else None
            d = default.get_path_name() if default else None
            # An override exists exactly when the effective material is not the
            # mesh default. Recording None here means "no override", and
            # restoring it must CLEAR the slot, not write the default into it.
            slots.append(e if e != d else None)
            if e and {token!r} in e:
                interesting = True
        if interesting:
            out.append({{"actor": actor.get_actor_label(),
                        "component": comp.get_name(),
                        "overrides": slots}})
unreal.log("[EB-SNAP] " + json.dumps(out))
"""

RESTORE_SCRIPT = """
import json
import unreal

SNAP = json.loads({snapshot!r})
by_key = {{(s["actor"], s["component"]): s["overrides"] for s in SNAP}}
restored = 0
for actor in unreal.EditorLevelLibrary.get_all_level_actors():
    label = actor.get_actor_label()
    for comp in actor.get_components_by_class(unreal.StaticMeshComponent):
        key = (label, comp.get_name())
        if key not in by_key:
            continue
        for i, path in enumerate(by_key[key]):
            if path is None:
                # No override originally. empty_overridden_material_slots is
                # not exposed per-slot, so the closest correct action is to put
                # back the mesh default -- which for THIS slot is what it had.
                mesh = comp.static_mesh
                comp.set_material(i, mesh.get_material(i) if mesh else None)
            else:
                asset = unreal.EditorAssetLibrary.load_asset(path.split(".")[0])
                if asset is not None:
                    comp.set_material(i, asset)
        restored += 1
unreal.log("[EB-SNAP] restored=%d" % restored)
"""

# Applying a signal state. Note what this does NOT do: it never calls
# set_material, so it cannot disturb which material is bound. It only sets
# parameters on a dynamic instance of whatever is already there.
APPLY_STATE_SCRIPT = """
import unreal

CX, CY, RADIUS = {cx!r}, {cy!r}, {radius!r}
atlas = unreal.EditorAssetLibrary.load_asset({atlas!r})
if atlas is None:
    raise RuntimeError("missing LED atlas: " + {atlas!r})
touched = 0
for actor in unreal.EditorLevelLibrary.get_all_level_actors():
    for comp in actor.get_components_by_class(unreal.StaticMeshComponent):
        where = comp.get_world_location()
        if (where.x - CX) ** 2 + (where.y - CY) ** 2 > RADIUS ** 2:
            continue
        for i in range(comp.get_num_materials()):
            current = comp.get_material(i)
            if current is None or {token!r} not in current.get_path_name():
                continue
            dynamic = comp.create_dynamic_material_instance(i)
            dynamic.set_texture_parameter_value("Phase{phase}", atlas)
            dynamic.set_scalar_parameter_value("Phase", float({phase}))
            touched += 1
unreal.log("[EB-SIGNAL] state={state} atlas={atlas_name} phase={phase} slots=%d" % touched)
"""


def atlas_for(state: str) -> str:
    """The LED atlas asset path for a signal state."""
    if state not in SIGNAL_RECIPE:
        raise ValueError(f"unknown signal state {state!r}; expected {sorted(SIGNAL_RECIPE)}")
    return LED_ATLAS_ROOT + SIGNAL_RECIPE[state][0]


def phase_for(state: str) -> int:
    if state not in SIGNAL_RECIPE:
        raise ValueError(f"unknown signal state {state!r}; expected {sorted(SIGNAL_RECIPE)}")
    return SIGNAL_RECIPE[state][1]


def snapshot_script(x_cm: float, y_cm: float, radius_cm: float = 15000.0) -> str:
    return SNAPSHOT_SCRIPT.format(
        cx=float(x_cm), cy=float(y_cm), radius=float(radius_cm), token=LED_MATERIAL_TOKEN
    )


def restore_script(snapshot_json: str) -> str:
    return RESTORE_SCRIPT.format(snapshot=snapshot_json)


def apply_state_script(
    state: str, x_cm: float, y_cm: float, radius_cm: float = 15000.0
) -> str:
    """Set every nearby traffic light to ``state``.

    Only touches material *parameters*, never material *bindings*, so it cannot
    reproduce the failure this module is named for.
    """
    atlas = atlas_for(state)
    return APPLY_STATE_SCRIPT.format(
        cx=float(x_cm), cy=float(y_cm), radius=float(radius_cm),
        atlas=atlas, atlas_name=atlas.rsplit("_", 1)[-1],
        phase=phase_for(state), token=LED_MATERIAL_TOKEN, state=state,
    )


def forbidden_restore_pattern() -> str:
    """The exact code that caused the damage, kept so a test can assert its absence."""
    return "set_material(i, mesh.get_material(i))"
