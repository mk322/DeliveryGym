// SpNavMeshHelper.h
//
// Runtime navmesh generation for SimWorld maps that ship without
// pre-baked navigation data.  agent_test (the SPEAR migration test
// map) is one such map -- without a RecastNavMesh actor, every
// AAIController::MoveToLocation call returns Failed and the agent
// can only use the straight-line fallback in SpHumanoidAgent::MoveTo.
//
// This helper spawns:
//   1. ANavMeshBoundsVolume   -- defines the volume to be covered
//   2. (NavigationSystemV1 auto-creates ARecastNavMesh on first
//       registration of a bounds volume, then rebuilds.)
//
// Usage from Python:
//   uclass  = service.load_class(uclass="AActor",
//                                name="/Script/SimWorld.SpNavMeshHelper")
//   helper  = service.spawn_actor(uclass=uclass)
//   helper.SpawnNavMeshBounds(Center={"X":0,"Y":0,"Z":100},
//                              Extent={"X":5000,"Y":5000,"Z":500})
//   helper.BuildNavMesh()
//   # then step a few seconds while NavData generates
//   instance.step(num_frames=60)
//   # query
//   ready = helper.IsNavMeshReady()
//
// After this, ASpHumanoidAgent::MoveTo's primary path (navmesh) will
// actually succeed.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "NavMesh/NavMeshBoundsVolume.h"   // for ANavMeshBoundsVolume base
#include "SpNavMeshHelper.generated.h"

// Subclass of ANavMeshBoundsVolume that overrides the bounds query to
// return an explicit FBox we set at construction time.  ANavMeshBoundsVolume's
// default GetComponentsBoundingBox traverses the brush component, but
// brush UModel rebuild is editor-only -- at runtime via SpawnActor the
// model bounds stay 0 even after SetActorScale3D.  This subclass
// short-circuits with our explicit box, so NavSys.OnNavigationBoundsAdded
// registers the correct AreaBox.
UCLASS()
class SIMWORLD_API ASpExplicitNavMeshBoundsVolume : public ANavMeshBoundsVolume
{
    GENERATED_BODY()
public:
    // World-space FBox for the navmesh bounds.  Set this BEFORE
    // NavSys->OnNavigationBoundsAdded sees the volume.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="SpNavMesh")
    FVector ExplicitMin = FVector::ZeroVector;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="SpNavMesh")
    FVector ExplicitMax = FVector::ZeroVector;

    virtual FBox GetComponentsBoundingBox(bool bNonColliding = false, bool bIncludeFromChildActors = false) const override
    {
        return FBox(ExplicitMin, ExplicitMax);
    }
};

UCLASS(BlueprintType, Blueprintable)
class SIMWORLD_API ASpNavMeshHelper : public AActor
{
    GENERATED_BODY()

public:
    ASpNavMeshHelper();

    // Spawn a NavMeshBoundsVolume at Center with the requested half-extent
    // (cm).  Internal brush is 200x200x200 cube (half-size = 100); we
    // scale to (Extent / 100) to get the requested final size.
    UFUNCTION(BlueprintCallable, Category="SpNavMesh")
    bool SpawnNavMeshBounds(FVector Center, FVector Extent);

    // Trigger a synchronous navmesh build.  The actual generation is
    // async (multi-threaded recast tiles); IsNavMeshReady() polls.
    UFUNCTION(BlueprintCallable, Category="SpNavMesh")
    void BuildNavMesh();

    // Returns true once the NavigationSystem has at least one registered
    // NavData and it's not currently rebuilding.
    UFUNCTION(BlueprintCallable, BlueprintPure, Category="SpNavMesh")
    bool IsNavMeshReady() const;

    // Number of NavData actors currently registered (1 = RecastNavMesh
    // present and ready in the simplest case).
    UFUNCTION(BlueprintCallable, BlueprintPure, Category="SpNavMesh")
    int32 GetNumRegisteredNavData() const;

    // Diagnostic: how many navigation bounds is NavSys aware of right
    // now?  Useful to verify our OnNavigationBoundsAdded actually
    // registered the volume.
    UFUNCTION(BlueprintCallable, BlueprintPure, Category="SpNavMesh")
    int32 GetNumNavBounds() const;

    // Dump everything that's relevant to navmesh build state.  Returns
    // a multi-line string for inspection from Python.
    UFUNCTION(BlueprintCallable, Category="SpNavMesh")
    FString DumpNavState();

    // Force the default RecastNavMesh's RuntimeGeneration mode to
    // Dynamic.  Useful when the project ini says Dynamic but the
    // auto-created instance came in as Static.
    UFUNCTION(BlueprintCallable, Category="SpNavMesh")
    bool ForceDynamicRuntimeGeneration();

    // Runtime proof used by the live Paris Pixel Goal PoC.  These query the
    // registered default NavData after BuildNavMesh() rather than assuming
    // that the project config was applied to a serialized/runtime instance.
    UFUNCTION(BlueprintCallable, BlueprintPure, Category="SpNavMesh")
    bool IsDynamicRuntimeGeneration() const;

    UFUNCTION(BlueprintCallable, BlueprintPure, Category="SpNavMesh")
    bool SupportsRuntimeNavGeneration() const;

    UFUNCTION(BlueprintCallable, BlueprintPure, Category="SpNavMesh")
    bool HasValidNavMeshData() const;

    UFUNCTION(BlueprintCallable, BlueprintPure, Category="SpNavMesh")
    int32 GetNumActiveNavMeshTiles() const;

    // Diagnostic: force a rebuild on the NavData directly (bypassing
    // NavSys-level logic).  Returns true if rebuild was dispatched.
    UFUNCTION(BlueprintCallable, Category="SpNavMesh")
    bool ForceNavDataRebuild();

    // Spawn a guaranteed-walkable static mesh floor plate inside the
    // nav bounds.  agent_test (and other UE5 World Partition maps) hide
    // their ground collision in streamed cells that aren't loaded at
    // navmesh build time -- recast then generates 0 tiles even with
    // bounds in place ("ProcessTileTasks build time: 0.00s").
    // Spawning a static mesh in the PERSISTENT level guarantees recast
    // finds at least one walkable surface inside the bounds.
    //
    // Center: world-space floor center (XY mid, Z = top of plate).
    // Extent: full size (cm) of the plate.  The plate is 20cm thick.
    UFUNCTION(BlueprintCallable, Category="SpNavMesh")
    bool SpawnFloorPlate(FVector Center, FVector Extent);

    // Bounds volume we spawned (or nullptr).
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="SpNavMesh")
    AActor* BoundsVolume = nullptr;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="SpNavMesh")
    AActor* FloorPlate = nullptr;
};
