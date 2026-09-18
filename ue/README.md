# The engine side of the any-point track

The any-point track runs in Unreal Engine. The engine-side code it needs is
here: the `SpPixelGoalSubsystem` (capture a photograph with the exact camera
state it was taken with, resolve a pixel of that photograph to a point on the
NavMesh, judge and walk the straight controller path to it), its automation
tests, and the three SimWorld classes it drives -- the humanoid pawn, the
camera capture pool and the NavMesh helper.

Everything in this directory comes from
[**SimWorld_SPEAR**](https://github.com/SimWorld-AI/SimWorld_SPEAR) (UE 5.8 on
the SPEAR runtime fork) and is MIT-licensed by that project; `ue/LICENSE` is
its licence, and it governs these files. Only the files the any-point track
actually needs were copied: the rest of that module (the pedestrian, vehicle
and robot agent bases, the cluster beacon and handoff state, the Mass
processors, the city-sample integration, the traffic plugin) belongs to
SimWorld and is not reproduced here.

```
ue/
  LICENSE                                              SimWorld_SPEAR's MIT licence
  Source/SimWorld/Public/SpPixelGoalSubsystem.h         the subsystem           334 lines   new
  Source/SimWorld/Private/SpPixelGoalSubsystem.cpp                             4009 lines   new
  Source/SimWorld/Private/Tests/SpPixelGoalSubsystemTest.cpp  25 tests         2469 lines   new
  Source/SimWorld/Public/SpHumanoidAgent.h              the pawn it drives      448 lines   +15 -0
  Source/SimWorld/Private/SpHumanoidAgent.cpp                                   877 lines   +77 -12
  Source/SimWorld/Public/SpCameraCapturePool.h          the capture path        337 lines   +8 -0
  Source/SimWorld/Private/SpCameraCapturePool.cpp                              4024 lines   +101 -4
  Source/SimWorld/Public/SpNavMeshHelper.h              NavMesh queries         147 lines   +15 -0
  Source/SimWorld/Private/SpNavMeshHelper.cpp                                   311 lines   +49 -0
  simworld_spear-pixel-goal.patch                      the six files' changes, 21 hunks
```

Three files are new; the other six existed in SimWorld_SPEAR and carry the
any-point changes (the four camera views the harness turns through, a camera
snapshot recorded with every capture, and the NavMesh queries the subsystem
resolves a pixel against). The patch is those changes and nothing else,
against SimWorld_SPEAR `main` at commit
`d3d1dadc20643f9d15a21404e05124b1bca8aaa3` (2026-08-19), where the six files
were blobs `14c62a39f6` and `e4654deb36` (`SpHumanoidAgent.h`, `.cpp`),
`92d9901fad` and `fb4b68eebf` (`SpCameraCapturePool.h`, `.cpp`), and
`81ffd7db0e` and `5250bb6d64` (`SpNavMeshHelper.h`, `.cpp`). Applying it to
those blobs reproduces the sources here byte for byte, which is how it was
checked.

## Building it

These files are not a module of their own: they are the any-point part of
SimWorld's `SimWorld` module and are built inside it. Drop them into a
checkout of SimWorld_SPEAR and build the Linux editor as that project's
guides describe (a UE 5.8 source build, the SPEAR plugins under
`Plugins/spear`, a Development Editor target):

```bash
cd SimWorld_SPEAR                        # at main d3d1dad, or rebase the patch
cp -r <this repo>/ue/Source/. Source/    # the nine files, in place
```

They compile against four things this repository does not carry:

- **SimWorld headers the copied files include** and that belong to the rest of
  that module: `SpActionDefinition.h` and `SpAgentActionRegistry.h` (the pawn's
  action registry) and `SpPedestrianAgentBase.h` and `SpVehicleAgentBase.h`
  (the other agent kinds the capture pool serves). They are in the checkout.
- **SPEAR plugin headers**: `SpCore/Log.h`, `SpCore/Unreal.h`,
  `SpCore/UnrealUtils.h`, `SpUnrealTypes/SpSceneCaptureComponent2D.h`,
  `SpUnrealTypes/SpMeshProxyComponentManager.h`.
- The module's `SimWorld.Build.cs`, the `.uproject`, and the engine itself.
- The **CityCore Paris** scene, which is Epic marketplace content.

SimWorld_SPEAR is private and stays private, so these instructions are for
whoever holds that project; a build cannot be made from this repository alone,
and the waypoint track is the one that runs anywhere. The build the numbers in
this repository were measured on is branch `pixel-goal-1b-poc`, commit
`7a1131a`, of that project, which carries exactly these sources; ask the
authors for the editor bundle rather than rebuilding it.

These files are here to be read and checked: the engine's half of the
any-point contract -- what a resolved pixel is, what makes a straight path
legal, what a camera snapshot promises -- is in them, so the rules the Python
side depends on can be audited against the code that enforces them.

The automation tests run in the editor as `SimWorld.PixelGoal.*` (for example
`-ExecCmds="Automation RunTests SimWorld.PixelGoal"`) and cover the camera-view
geometry, the deprojection and the snapshot rules without the Paris scene.

## Using the build

The launcher (`tools/run_pixel_goal_front_rear_delivery.sh`) expects the bundle
at `.simworld-ue/` in the repository root (a symlink is fine), reads
`Binaries/Linux/SimWorldEditor`, and refuses any build other than the one it
was audited on: `tools/pixel_goal_launcher_identity.py` pins seven files of
`Binaries/Linux/` by SHA-256 (`APPROVED_BUNDLE_SHA256`) and checks the running
process against them. A fresh build has different hashes, so after building,
put the new digests there deliberately:

```bash
cd .simworld-ue/Binaries/Linux
sha256sum SimWorldEditor SimWorldEditor.debug SimWorldEditor.modules SimWorldEditor.target \
          SimWorldEditor.version libSimWorldEditor-SimWorld.so libSimWorldEditor-SimWorld.debug
python tools/pixel_goal_launcher_identity.py validate-bundle \
    --expected-editor .simworld-ue/Binaries/Linux/SimWorldEditor \
    --selected-editor .simworld-ue/Binaries/Linux/SimWorldEditor
```

Copy `Saved/DerivedDataCache` with the bundle (about 3 GB) or the first Paris
load runs past the launcher's 30-minute readiness cap. The rest of the
requirements -- the CityCore Paris content, the SPEAR Python client, the
interpreter, the two GPUs -- are the table in `docs/PIXEL_GOAL.md`.

## What the subsystem does

The Python side (`embodiedbench/runtime/pixel_goal.py` and the clients under
`tools/`) speaks to the subsystem through SPEAR's RPC service. One call
captures a photograph (or a front/rear pair) from the agent's camera and
records, under a snapshot id, the camera state it was taken with. Another
resolves a pixel of a named snapshot against that exact state -- the ray, the
NavMesh projection, and the verdict on the straight controller path (a marked
crossing, or nothing but pavement) -- and walks it, or, given an acceptance
radius wider than the map, only judges it. A status call reports the agent's
feet and the world clock. The four-view harness turns the agent between
captures and resolves each pixel only while the agent stands as it did for
that snapshot; the subsystem refuses a pixel whose snapshot is stale.
