# nav-render/v0 golden fixtures

Byte-exact serialisations of the wire examples, one per JSON shape in
`embodiedbench/runtime/live/protocol.py`. Serialisation convention: JSON with 2-space indent, sorted keys,
one trailing newline (`embodiedbench.runtime.live.protocol.dumps`). The
`sha256` in `render_response.json` is the SHA-256 of the empty string -- a
recognisable placeholder, since the spec's example elides the value.

The `track_b_*.json` fixtures pin section 3b (embodied episodes) the same
way: one per Track B message, written by the same `dumps`. The walk
response's numbers come from the reference fixed-dt integrator (140 cm/s,
dt 0.0333, arrive 50 cm) so the exemplar is kinematically coherent, and the
`sha256` in `track_b_observe_result.json` is the empty-string placeholder
again. The observe result's `key` is `""` on the wire: the caller owns album
naming, the service only echoes the shape.

These same files are copied verbatim into the SimWorld2 repo
(`feat/nav-live-rollouts`), so both sides of the protocol are pinned to one
set of bytes. Change them in both places or not at all.
