// SpHumanoidAgent.h
//
// Minimal C++ ACharacter with first-class movement + camera + pickup,
// built so a Python LLM loop can drive it through reflection without
// depending on any Blueprint graph.
//
// Why this exists:
//   The user's Base_User_Agent_SPEAR BP exposes MoveForward / Rotate /
//   PickUp UFunctions that return None but don't actually translate the
//   Character -- they appear to depend on UE_Manager.Tick to dispatch
//   AddMovementInput, and our Python tests never reliably saw motion.
//
//   This class is BP-graph-free: every action drives CharacterMovement
//   directly in C++, ticked every frame regardless of any external
//   manager actor.  Spawn it from Python, call UFunctions via SPEAR
//   reflection, and the character actually walks.
//
// Components:
//   - CapsuleComponent (inherited root from ACharacter, Movable by ctor).
//   - CharacterMovementComponent (inherited, default settings).
//   - Mesh (inherited; user can SetSkeletalMesh from BP if desired).
//   - SpringArm + USpSceneCaptureComponent2D for 3rd-person observation.
//
// Per-frame Tick:
//   if (bMovingForward) AddMovementInput(GetActorForwardVector(), 1.0).
//   if (RemainingYawDeg != 0) rotate this frame's slice toward target.
//
// Spawn from Python:
//   uclass = service.load_class(uclass="AActor",
//                              name="/Script/SimWorld.SpHumanoidAgent")
//   agent = service.spawn_actor(uclass=uclass)
//   agent.MoveForward()        # walks until StopAgent()
//   agent.Rotate(45.0, "left") # turns 45deg over the next ~0.5s
//   agent.PickUp("Box_1")      # attaches actor named Box_1 to hand socket
//   agent.Drop()

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "GameFramework/CharacterMovementComponent.h"   // for inline GetMaxSpeed accessor
#include "SpHumanoidAgent.generated.h"

class USpringArmComponent;
class USpSceneCaptureComponent2D;
class UCameraComponent;

namespace SpPixelGoal
{
    SIMWORLD_API bool IsSupportedCameraView(const FString& ViewId);
    SIMWORLD_API FRotator CameraViewRelativeRotation(const FString& ViewId);
}

UCLASS(BlueprintType, Blueprintable)
class SIMWORLD_API ASpHumanoidAgent : public ACharacter
{
    GENERATED_BODY()

public:
    ASpHumanoidAgent();

    // ---- Replication (server-authoritative architecture, Path A) ----
    //
    // When run in a dedicated-server / listen-server build, the SERVER
    // is authoritative for agent state.  Clients receive replicated
    // location + animation state automatically (via ACharacter base
    // class), but our custom flags (`bMovingForward`, `RemainingYawDeg`,
    // `HeldActor`) need explicit Replicated UPROPERTY + lifetime
    // registration to propagate.
    //
    // Standalone / editor mode: Replicated UPROPERTY is a no-op (no
    // network), so this is backward-compatible with all earlier Tier C
    // tests.
    virtual void GetLifetimeReplicatedProps(TArray<class FLifetimeProperty>& OutLifetimeProps) const override;

    // Stable cross-process identifier. Matches the pedestrian/vehicle/robot
    // AgentTag contract so batch clients, routers, and cluster handoff can
    // address humanoids without relying on transient actor names.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, ReplicatedUsing=OnRep_AgentTag, Category="SpControllableAgent")
    FName AgentTag;

    UFUNCTION()
    void OnRep_AgentTag();

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    void Agent_SetAgentTag(FName InAgentTag);

    UFUNCTION(BlueprintCallable, BlueprintPure, Category="SpControllableAgent")
    FString Agent_GetTagString() const;

    // Server-only RPCs used by client → server input forwarding.  In
    // standalone these collapse to direct calls.  In dedicated server
    // mode, `Server_*` RPCs send the request over the wire; the server
    // executes the underlying logic; result replicates back via
    // replicated UPROPERTY (e.g. bMovingForward).
    UFUNCTION(Server, Reliable, BlueprintCallable, Category="SpControllableAgent")
    void Server_Agent_MoveForward();

    UFUNCTION(Server, Reliable, BlueprintCallable, Category="SpControllableAgent")
    void Server_Agent_StopAgent();

    UFUNCTION(Server, Reliable, BlueprintCallable, Category="SpControllableAgent")
    void Server_Agent_Rotate(float AngleDeg, const FString& Direction);

    UFUNCTION(Server, Reliable, BlueprintCallable, Category="SpControllableAgent")
    void Server_Agent_MoveTo(FVector Target);

    UFUNCTION(Server, Reliable, BlueprintCallable, Category="SpControllableAgent")
    void Server_Agent_PickUp(const FString& ObjectName);

    UFUNCTION(Server, Reliable, BlueprintCallable, Category="SpControllableAgent")
    void Server_Agent_Drop();

    UFUNCTION(Server, Reliable, BlueprintCallable, Category="SpControllableAgent")
    void Server_Agent_SetMaxSpeed(float CmPerSec);

    UFUNCTION(Server, Reliable, BlueprintCallable, Category="SpControllableAgent")
    void Server_Agent_SetPath(const TArray<FVector>& InWaypoints);

    // ---- SpControllableAgent contract (`Agent_*` UFunctions) ----
    //
    // These are the methods that SpearClient / external Python code is
    // expected to call.  The naming convention `Agent_*` is the
    // stable contract -- any actor class that wants to be controllable
    // by the same Python code MUST expose UFunctions with these names
    // and signatures.  We initially tried to express this as a UE
    // UINTERFACE (`ISpControllableAgent`) so BPs could declare conformance,
    // but interface UFunctions don't appear in the implementing class's
    // `unqualified_function_descs` table that SPEAR's reflection uses,
    // so calls via SPEAR resolved them as properties (UnrealObject not
    // callable).  Until that's solved, declare the contract methods
    // directly on the class as plain UFunctions and treat the
    // interface header as a documentation-only artifact.
    //
    // Authority routing: SPEAR Python clients connect to the SERVER, so
    // HasAuthority()==true on server and these wrappers call local methods
    // directly.  If invoked on a rendering client (HasAuthority()==false),
    // we forward via the Server_* RPC so the action executes authoritatively.
    // This fixes AUDIT_TODO P1: "client-side humanoid RPC wrappers".

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    void Agent_MoveForward()
    {
        if (!HasAuthority()) { Server_Agent_MoveForward(); return; }
        MoveForward();
    }

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    void Agent_StopAgent()
    {
        if (!HasAuthority()) { Server_Agent_StopAgent(); return; }
        StopAgent();
    }

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    void Agent_Rotate(float AngleDeg, const FString& Direction)
    {
        if (!HasAuthority()) { Server_Agent_Rotate(AngleDeg, Direction); return; }
        Rotate(AngleDeg, Direction);
    }

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    void Agent_SetMaxSpeed(float CmPerSec)
    {
        if (!HasAuthority()) { Server_Agent_SetMaxSpeed(CmPerSec); return; }
        SetMaxSpeed(CmPerSec);
    }

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    bool Agent_IsRotating() { return CheckIfRotating(); }

    // NOTE: Agent_PickUp returns bool for local feedback, but Server_Agent_PickUp
    // is void (RPCs cannot return values).  On a client, the RPC is forwarded and
    // true is returned immediately (optimistic); actual success/failure is
    // server-authoritative and reflected in replicated state.
    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    bool Agent_PickUp(const FString& ObjectName)
    {
        if (!HasAuthority()) { Server_Agent_PickUp(ObjectName); return true; }
        return PickUp(ObjectName);
    }

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    void Agent_Drop()
    {
        if (!HasAuthority()) { Server_Agent_Drop(); return; }
        Drop();
    }

    // ---- Runtime-extensible actions (data-driven, pak-loadable) ----
    //
    // The released agent ships with only built-in navigation.  Richer actions
    // (wave, sit, ...) are NOT hardcoded: they are USpActionDefinition assets
    // (animation montage + metadata) that can be loaded AT RUNTIME from a
    // mounted .pak in a packaged build.  Agent_PlayAction resolves the action
    // by name from USpAgentActionRegistry (which discovers definitions in
    // mounted paks) and plays its montage on every net instance.  Add a new
    // action with NO C++ rebuild: ship montage + USpActionDefinition in a pak,
    // mount it (-SpDynamicPak), and call Agent_PlayAction("<id>") via SPEAR.
    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    bool Agent_PlayAction(FName ActionId);

    // Play an animation montage by asset path directly (ad-hoc, no registry).
    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    bool Agent_PlayMontageByPath(const FString& MontagePath, float PlayRate = 1.0f);

    // Stop the currently playing action montage.
    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    void Agent_StopAction();

    // Server RPCs — client→server forwarding (montage play is authoritative,
    // then multicast to all clients), matching the other Agent_* actions.
    UFUNCTION(Server, Reliable)
    void Server_Agent_PlayMontage(const FString& MontagePath, float PlayRate);
    UFUNCTION(Server, Reliable)
    void Server_Agent_StopAction();

    // Multicast so the montage plays on every net instance that renders the
    // agent (all clients; the server too if it isn't -nullrhi).  In standalone
    // the body just runs locally.
    UFUNCTION(NetMulticast, Reliable)
    void Multicast_PlayMontage(const FString& MontagePath, float PlayRate);
    UFUNCTION(NetMulticast, Reliable)
    void Multicast_StopMontage();

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    FVector Agent_GetLocation() { return GetActorLocation(); }

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    float Agent_GetYaw() { return GetActorRotation().Yaw; }

    // Tier C #7: navmesh-routed move to a world-space target.  Returns
    // true if the AAIController accepted the request; movement completes
    // asynchronously in subsequent ticks.  Caller polls Agent_IsMoving()
    // / Agent_GetLocation() to detect arrival.
    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    bool Agent_MoveTo(FVector Target)
    {
        if (!HasAuthority()) { Server_Agent_MoveTo(Target); return true; }
        return MoveTo(Target);
    }

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    bool Agent_IsMoving() { return IsMoving(); }

    // Tier C #5: native path serialization.  Pass a TArray<FVector>
    // of world-space waypoints; the agent walks them in order via
    // MoveTo (navmesh-routed or straight-line fallback).  Returns true
    // if the path was accepted; agent advances waypoint-by-waypoint
    // in Tick() so caller polls Agent_IsMoving / Agent_GetLocation.
    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    bool Agent_SetPath(const TArray<FVector>& InWaypoints)
    {
        if (!HasAuthority()) { Server_Agent_SetPath(InWaypoints); return true; }
        return SetPath(InWaypoints);
    }

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    int32 Agent_GetCurrentWaypointIndex() { return CurrentWaypointIdx; }

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    int32 Agent_GetPathLength() { return Waypoints.Num(); }

    // ------------------ Cross-shard handoff state ------------------
    //
    // Cross-shard handoff (CoordinationServer) currently transfers
    // (loc, yaw) only.  Without these, an agent crossing a region
    // boundary loses its "currently walking forward" state — it stops
    // until the server re-issues MoveForward.  These extra getters /
    // setters let the server preserve behavior continuity across the
    // handoff: read the source agent's full state, spawn the target,
    // restore the state.

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    bool Agent_GetMovingForward() { return bMovingForward; }

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    float Agent_GetRemainingYawDeg() { return RemainingYawDeg; }

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    float Agent_GetMaxSpeed() { return GetCharacterMovement() ? GetCharacterMovement()->MaxWalkSpeed : 0.f; }

    // Restore a previously-captured state.  Pass bMovingForward=true to
    // resume walking; remainingYaw != 0 to resume an in-progress rotate.
    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    void Agent_RestoreState(bool bMoving, float RemainingYaw, float MaxSpeed)
    {
        bMovingForward = bMoving;
        RemainingYawDeg = RemainingYaw;
        if (MaxSpeed > 0.f && GetCharacterMovement())
        {
            GetCharacterMovement()->MaxWalkSpeed = MaxSpeed;
        }
    }

    // Held-actor introspection: used by the server during handoff to
    // also transfer the held box (or whatever) to the target client.
    // Returns the held actor's class path (e.g. /Game/Foo/BP_Box.BP_Box_C)
    // and label, or empty strings if not holding anything.
    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    FString Agent_GetHeldActorClassPath()
    {
        if (!HeldActor) return FString();
        return HeldActor->GetClass() ? HeldActor->GetClass()->GetPathName() : FString();
    }

    UFUNCTION(BlueprintCallable, Category="SpControllableAgent")
    FString Agent_GetHeldActorName()
    {
        // GetActorLabel() is editor-only (WITH_EDITOR=0 on dedicated server).
        // GetActorNameOrLabel() works everywhere: returns label in editor,
        // falls back to GetName() in shipping/server builds.
        return HeldActor ? HeldActor->GetActorNameOrLabel() : FString();
    }

    // ------------------ Components ------------------
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="SpHumanoidAgent")
    USpringArmComponent* SpringArm = nullptr;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="SpHumanoidAgent")
    UCameraComponent* ViewCamera = nullptr;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="SpHumanoidAgent")
    USpSceneCaptureComponent2D* SceneCapture = nullptr;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="SpObservation")
    TObjectPtr<USpSceneCaptureComponent2D> RearSceneCapture = nullptr;

    USpSceneCaptureComponent2D* GetObservationCamera(const FString& ViewId) const;
    bool ConfigureObservationCamera(
        const FString& ViewId, int32 InWidth, int32 InHeight, float InFovDegrees);
    void InitializeObservationCamera(const FString& ViewId);
    void TerminateObservationCamera(const FString& ViewId);

    // ------------------ Action UFunctions ------------------
    // Begin/continue walking forward (CharacterMovement->AddMovementInput).
    UFUNCTION(BlueprintCallable, CallInEditor, Category="SpHumanoidAgent")
    void MoveForward();

    // Stop walking.
    UFUNCTION(BlueprintCallable, CallInEditor, Category="SpHumanoidAgent")
    void StopAgent();

    // Begin rotating Direction (0=left, 1=right) by AngleDeg degrees over
    // the next ~RotateRateDegPerSec / AngleDeg seconds.  Tick interpolates.
    // Direction can be "left" / "right" string or int 0/1 -- we accept both.
    UFUNCTION(BlueprintCallable, CallInEditor, Category="SpHumanoidAgent")
    void Rotate(float AngleDeg, const FString& Direction);

    // True while a rotation is in progress.
    UFUNCTION(BlueprintCallable, BlueprintPure, Category="SpHumanoidAgent")
    bool CheckIfRotating() const { return RemainingYawDeg != 0.f; }

    // Sets desired walking speed cm/s.
    UFUNCTION(BlueprintCallable, CallInEditor, Category="SpHumanoidAgent")
    void SetMaxSpeed(float CmPerSec);

    // Try to attach actor by in-world name to a "right hand" socket.
    // Returns true on success.  Searches GetWorld() for an actor named
    // ObjectName, AttachActorToComponent on this character's mesh socket
    // (or root if no socket).
    UFUNCTION(BlueprintCallable, CallInEditor, Category="SpHumanoidAgent")
    bool PickUp(const FString& ObjectName);

    // Drop the currently-held actor (if any).
    UFUNCTION(BlueprintCallable, CallInEditor, Category="SpHumanoidAgent")
    void Drop();

    // Configure the SpSceneCapture's pixel size + FOV.  Must be called
    // BEFORE InitializeCamera() so the underlying resources allocate at
    // the requested size.
    UFUNCTION(BlueprintCallable, Category="SpHumanoidAgent")
    bool ConfigureCamera(int32 InWidth, int32 InHeight, float InFovDegrees = 90.f);

    // Bind the read_pixels SpFunc.  Call AFTER ConfigureCamera + after
    // the world has begun play.
    UFUNCTION(BlueprintCallable, Category="SpHumanoidAgent")
    void InitializeCamera();

    UFUNCTION(BlueprintCallable, Category="SpHumanoidAgent")
    void TerminateCamera();

    // Navmesh-routed move (Tier C #7).  Delegates to the auto-spawned
    // AAIController's MoveToLocation(...) so the agent path-finds along
    // the level's RecastNavMesh.  Returns true if the request was
    // accepted by the controller (not necessarily that it succeeded;
    // poll IsMoving() for completion).
    UFUNCTION(BlueprintCallable, CallInEditor, Category="SpHumanoidAgent")
    bool MoveTo(FVector Target);

    // True iff the AIController currently has an active MoveTo request.
    UFUNCTION(BlueprintCallable, BlueprintPure, Category="SpHumanoidAgent")
    bool IsMoving() const;

    // Set + start following a path of world-space waypoints.  Returns
    // false if Waypoints is empty.  Calls MoveTo on each waypoint in
    // order, advancing in Tick when AcceptanceRadius2D is reached.
    UFUNCTION(BlueprintCallable, CallInEditor, Category="SpHumanoidAgent")
    bool SetPath(const TArray<FVector>& InWaypoints);

    // Stop path following and clear the path.
    UFUNCTION(BlueprintCallable, CallInEditor, Category="SpHumanoidAgent")
    void ClearPath();

    // ------------------ State (BP-readable + Replicated) ------------------
    // All state UPROPERTYs are marked Replicated so server-authority
    // mode can replicate them to clients automatically.  In standalone
    // mode the Replicated specifier is a no-op.
    UPROPERTY(Replicated, VisibleAnywhere, BlueprintReadOnly, Category="SpHumanoidAgent")
    bool bMovingForward = false;

    UPROPERTY(Replicated, VisibleAnywhere, BlueprintReadOnly, Category="SpHumanoidAgent")
    float RemainingYawDeg = 0.f;

    // Positive = yaw to the right (clockwise from above), negative = left.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="SpHumanoidAgent")
    float RotateRateDegPerSec = 180.f;

    // Currently picked-up actor (or nullptr).  Replicated so clients
    // see the box attached to the agent's hand.
    UPROPERTY(Replicated, VisibleAnywhere, BlueprintReadOnly, Category="SpHumanoidAgent")
    AActor* HeldActor = nullptr;

    // Tier C #5: native path-following state.
    UPROPERTY(Replicated, VisibleAnywhere, BlueprintReadOnly, Category="SpHumanoidAgent")
    TArray<FVector> Waypoints;

    UPROPERTY(Replicated, VisibleAnywhere, BlueprintReadOnly, Category="SpHumanoidAgent")
    int32 CurrentWaypointIdx = -1;

    // Waypoint reached when within this XY distance.  Triggers advance
    // to next waypoint via MoveTo.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="SpHumanoidAgent")
    float WaypointAcceptanceRadiusCm = 80.f;

    // Debug: enable per-agent tick logging (default off to avoid log spam with many agents).
    // Set to true in the editor or via blueprint to diagnose individual agents.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="SpHumanoidAgent|Debug")
    bool bDebugTickLog = false;

    // Per-instance tick log counter (not static — each agent counts independently).
    int32 TickLogCounter = 0;

    // AActor overrides
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void Tick(float DeltaSeconds) override;
};
