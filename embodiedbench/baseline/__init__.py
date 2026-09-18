"""Compatibility shims for the vendored DeliveryBench engine.

Three modules, all imported by the compiler and the text runtime:

``compat``        map-compatibility patches applied before the engine loads a city
``determinism``   the stable node ordering and seeded patches the replay relies on
``replay``        loading the vendored env module and its state policy

The M0 baseline-freeze tooling (manifests, replay CLIs, the scripted
baseline policy) that used to live beside them was retired before the
public release; the pins it recorded are in vendor/VENDOR_PINS.md.
"""
