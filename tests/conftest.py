"""Suite-wide guards.

Nothing here changes what any test asserts. It removes a way for one test to
change the answer another test gets.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _restore_working_directory():
    """Put the working directory back after every test.

    The vendored engine cannot be imported from an arbitrary directory --
    ``vlm_delivery/utils/global_logger.py`` calls ``os.makedirs("../../log")``
    at module scope -- so ``compat.bootstrap_vendor_import`` changes directory
    around the import and changes back. That is careful, and it is still one
    ``os.chdir`` in a long-lived process: anything that leaves the cwd moved
    silently redirects every relative path in every test that runs afterwards.

    It cost seven hours here. ``test_map_image`` resolved the Paris map by a
    relative path, its module-scoped fixture happened to be built during a
    window when the cwd had moved, and ``build_road_network`` returned an empty
    city with the note "roads.json is missing". Twelve tests failed with
    assertion errors about street counts and captions -- a set of symptoms
    pointing nowhere near the cause. The tests passed alone, passed in pairs,
    and passed against the whole alphabetical prefix ahead of them, because in
    every one of those runs the cwd happened to be right.

    The individual fix was to resolve the path from ``__file__``, which every
    other test module here already did. This is the general one: whatever a test
    does to the working directory ends with the test.
    """
    before = os.getcwd()
    try:
        yield
    finally:
        if os.getcwd() != before:
            os.chdir(before)
