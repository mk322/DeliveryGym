"""Guards on UE material handling, after a restore loop wiped a level's LEDs.

The mistake was subtle and expensive: `component.set_material(i,
static_mesh.get_material(i))` reads like "put the authored material back" and
actually destroys the component override, because the mesh default and the
component override are different layers. It took CityCore's traffic lights from
27 LED slots near a junction to 1 in the whole map. Nothing reached disk, but
nothing stopped it happening either — these tests do.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from embodiedbench.compiler.ue_materials import (
    LED_MATERIAL_TOKEN,
    SIGNAL_RECIPE,
    apply_state_script,
    atlas_for,
    forbidden_restore_pattern,
    phase_for,
    snapshot_script,
)

REPO = Path(__file__).resolve().parents[1]


class TestSignalRecipe:
    """The recipe, pinned to what was *measured*, not to what was proposed.

    The LED material exposes texture slots Phase1/Phase2 and a Phase scalar, and
    the colour comes from which atlas is bound — which is why moving the scalar
    alone did nothing across an 8-angle sweep. These two tests pinned the
    original guess, green→Phase1 and red→Phase2, and kept failing after the 2x2
    sweep recorded in ``ue_materials.SIGNAL_RECIPE`` overturned it: green appears
    in exactly one cell, Phase2=E02 with the scalar at 2. The scalar selects the
    slot of the same number, so the phase number pairs with the slot rather than
    with the state, and (E02, Phase=1) renders red.

    A test that pins a hypothesis its own module has since disproved is not a
    guard, it is a second copy of the bug.
    """

    def test_green_is_atlas_e02_in_phase_slot_2(self):
        assert atlas_for("green").endswith("E02")
        assert phase_for("green") == 2

    def test_red_is_atlas_e01_in_phase_slot_1(self):
        assert atlas_for("red").endswith("E01")
        assert phase_for("red") == 1

    def test_the_scalar_always_names_the_slot_it_selects(self):
        """The finding the sweep actually produced, stated once: whichever slot
        the atlas goes in, the scalar carries that slot's number."""
        for state in ("green", "red"):
            atlas, phase = SIGNAL_RECIPE[state]
            script = apply_state_script(state, 0.0, 0.0)
            assert f'"Phase{phase}"' in script
            assert f'float({phase})' in script

    def test_states_use_different_atlases(self):
        """If both states bound the same atlas the lamp could not change, which
        is exactly the bug that hid this for hours."""
        assert atlas_for("green") != atlas_for("red")
        assert phase_for("green") != phase_for("red")

    def test_an_unknown_state_is_refused(self):
        for bad in ("amber", "off", ""):
            with pytest.raises(ValueError, match="unknown signal state"):
                atlas_for(bad)

    def test_the_atlas_path_points_at_citycore_textures(self):
        assert atlas_for("red").startswith("/Game/CityCore_Paris/Textures/TrafficLights/")


class TestApplyStateIsNonDestructive:
    def test_it_never_rebinds_a_material(self):
        """The whole safety argument. Setting parameters on a dynamic instance
        cannot lose the binding; calling set_material can."""
        script = apply_state_script("green", 0.0, 0.0)
        assert "set_material" not in script
        assert "create_dynamic_material_instance" in script

    def test_it_only_touches_led_materials(self):
        script = apply_state_script("red", 0.0, 0.0)
        assert LED_MATERIAL_TOKEN in script

    def test_it_sets_both_the_atlas_and_the_scalar(self):
        script = apply_state_script("green", 0.0, 0.0)
        assert "set_texture_parameter_value" in script
        assert "set_scalar_parameter_value" in script
        assert '"Phase2"' in script

    def test_red_and_green_scripts_differ_in_atlas_and_phase(self):
        green, red = apply_state_script("green", 0, 0), apply_state_script("red", 0, 0)
        assert "E02" in green and "E01" in red
        assert '"Phase2"' in green and '"Phase1"' in red


class TestSnapshotSemantics:
    def test_a_slot_with_no_override_is_recorded_as_none(self):
        """None means "clear", not "write the default in". Conflating the two is
        precisely how the override was destroyed."""
        script = snapshot_script(0.0, 0.0)
        assert "mesh.get_material(i)" in script
        assert "e if e != d else None" in script

    def test_the_snapshot_compares_effective_against_default(self):
        script = snapshot_script(0.0, 0.0)
        assert "comp.get_material(i)" in script and "mesh.get_material(i)" in script


class TestTheDestructivePatternStaysOut:
    def test_no_committed_code_restores_from_the_mesh_default(self):
        """The exact loop that caused the damage, banned repo-wide.

        It only ever lived in throwaway scripts, which is why nothing caught it.
        """
        pattern = re.compile(r"set_material\(\s*\w+\s*,\s*\w*mesh\w*\.get_material\(")
        offenders = []
        for path in list((REPO / "embodiedbench").rglob("*.py")) + list((REPO / "tools").rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if pattern.search(line) and "RESTORE_SCRIPT" not in text[:200]:
                    # ue_materials.py contains it inside the documented restore
                    # path, where it is correct: it only runs for slots the
                    # snapshot recorded as having had no override.
                    if path.name != "ue_materials.py":
                        offenders.append(f"{path}: {line.strip()}")
        assert not offenders, offenders

    def test_the_forbidden_pattern_is_documented(self):
        assert "set_material" in forbidden_restore_pattern()
