// SpHumanoidAgent.cpp

#include "SpHumanoidAgent.h"

#include "AIController.h"
#include "Animation/AnimBlueprint.h"
#include "Animation/AnimInstance.h"
#include "Animation/AnimMontage.h"
#include "SpActionDefinition.h"
#include "SpAgentActionRegistry.h"
#include "Navigation/PathFollowingComponent.h"
#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "Engine/EngineTypes.h"
#include "Engine/Scene.h"                 // EDynamicGlobalIlluminationMethod, EReflectionMethod
#include "Engine/SkeletalMesh.h"
#include "Components/SkeletalMeshComponent.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/Controller.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/SpringArmComponent.h"
#include "Kismet/KismetMathLibrary.h"
#include "Net/UnrealNetwork.h"            // for DOREPLIFETIME (replication)
#include "UObject/ConstructorHelpers.h"

#include "SpCore/Log.h"
#include "SpCore/Unreal.h"

#include "SpUnrealTypes/SpSceneCaptureComponent2D.h"

bool SpPixelGoal::IsSupportedCameraView(const FString& ViewId)
{
    return ViewId == TEXT("front") || ViewId == TEXT("rear");
}

FRotator SpPixelGoal::CameraViewRelativeRotation(const FString& ViewId)
{
    return ViewId == TEXT("rear")
        ? FRotator(0.f, 180.f, 0.f)
        : FRotator::ZeroRotator;
}

ASpHumanoidAgent::ASpHumanoidAgent()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.bStartWithTickEnabled = true;
    PrimaryActorTick.bTickEvenWhenPaused = false;

    // Path A (server-authoritative): mark Replicated.  In standalone /
    // editor mode this is a no-op.  In dedicated-server mode the
    // server's instance of this actor replicates to clients via
    // UNetDriver; the inline-marked UPROPERTY(Replicated) members
    // travel automatically.  Default replication is spatial-relevancy
    // based — clients only see actors in their relevant set.
    bReplicates = true;
    // bAlwaysRelevant = true: every rendering client always receives every
    // agent, regardless of the client's viewpoint / spatial relevancy.
    // Rationale: SimWorld is a multi-agent simulation platform where rendering
    // clients are expected to observe the full set of agents (each client
    // renders the scene or an assigned region).  With bAlwaysRelevant=false,
    // a fresh client whose spectator pawn is near the PlayerStart never enters
    // the relevant set of agents spawned elsewhere, so they never replicate
    // and the client sees zero agents.  Large-scale spatial culling is the
    // job of USpReplicationGraph's spatial/ghost nodes, NOT actor-level
    // bAlwaysRelevant — so making the agent always-relevant here is correct.
    bAlwaysRelevant = true;
    SetReplicateMovement(true);     // ACharacter location auto-replicates

    // Force root mobility = Movable.  ACharacter already does this by
    // default for its CapsuleComponent root, but we set it explicitly
    // for clarity.
    if (UCapsuleComponent* Caps = GetCapsuleComponent())
    {
        Caps->SetMobility(EComponentMobility::Movable);
        Caps->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        // Block WorldStatic (walls / buildings) so the agent can't walk
        // through real geometry, but Overlap Pawn / PhysicsBody so the
        // agent can pass through the dense CitySample pedestrian crowd
        // and parked vehicles on agent_test.  This is the right
        // compromise for "real city" + "walking actually works".
        Caps->SetCollisionResponseToAllChannels(ECR_Block);
        Caps->SetCollisionResponseToChannel(ECC_Pawn, ECR_Overlap);
        Caps->SetCollisionResponseToChannel(ECC_PhysicsBody, ECR_Overlap);
        Caps->SetCollisionResponseToChannel(ECC_Destructible, ECR_Overlap);
    }

    // Set up the mesh component for default ACharacter offsets.  Actual
    // SkeletalMesh + AnimInstance are now applied at BeginPlay via
    // USpVisualProfileBase::TrySpawnVisual() (see ASpPedestrianAgentBase
    // refactor, 2026-05-27).  No ConstructorHelpers asset lookups here —
    // those caused cook-time CDO errors when the hard-coded Manny paths
    // didn't exist in the project's Content/, and the resulting mesh
    // would be overwritten by VisualProfile at BeginPlay anyway.
    if (USkeletalMeshComponent* MeshComp = GetMesh())
    {
        MeshComp->SetMobility(EComponentMobility::Movable);
        // Standard ACharacter offsets: feet at -capsule_half_height, face -90.
        MeshComp->SetRelativeLocation(FVector(0.f, 0.f, -89.f));
        MeshComp->SetRelativeRotation(FRotator(0.f, -90.f, 0.f));
    }

    // SpringArm for 3rd-person camera.
    SpringArm = CreateDefaultSubobject<USpringArmComponent>(TEXT("SpringArm"));
    SpringArm->SetupAttachment(GetCapsuleComponent());
    SpringArm->TargetArmLength = 300.f;
    SpringArm->bUsePawnControlRotation = false;
    SpringArm->SetRelativeLocation(FVector(0.f, 0.f, 80.f));
    SpringArm->SetRelativeRotation(FRotator(-15.f, 0.f, 0.f));
    SpringArm->bDoCollisionTest = false;

    // Active camera for PlayerController/PixelStreaming view targets. The
    // SceneCapture below remains the SPEAR observation surface.
    ViewCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("ViewCamera"));
    ViewCamera->SetupAttachment(SpringArm, USpringArmComponent::SocketName);
    ViewCamera->bUsePawnControlRotation = false;
    ViewCamera->bAutoActivate = true;
    ViewCamera->SetActive(true);

    // SpSceneCaptureComponent2D mounted on the spring arm.
    //
    // Camera config mirrors SPEAR's canonical BP_CameraSensor "final_tone_curve_hdr_"
    // component (spear-sim-spear/editor/create_asset_bp_camera_sensor.py).
    // The CRITICAL settings that make read_pixels() return a real visible image
    // instead of noise/black:
    //   - CaptureSource = SCS_FinalToneCurveHDR: full post-process + tone curve
    //     in sRGB gamut. The default SCS_SceneColorHDR is raw HDR float — when
    //     read as uint8 it looks like noise, and without tone mapping it can
    //     read as black. FinalToneCurveHDR is the "what the eye sees" output.
    //   - bOverrideTextureRenderTargetFormat=true + RTF_RGBA8_SRGB: the auto
    //     format for an HDR capture source is a 16f float target; reading that
    //     as uint8 is garbage. An explicit RGBA8_SRGB 8-bit target matches the
    //     uint8 read path (NumChannelsPerPixel=4, ChannelDataType=UInt8).
    //   - Lumen GI + reflections: matches the viewport's lit appearance.
    SceneCapture = CreateDefaultSubobject<USpSceneCaptureComponent2D>(TEXT("SceneCapture"));
    SceneCapture->SetupAttachment(SpringArm);
    SceneCapture->bUseSharedMemory = false;
    SceneCapture->BufferingMode = ESpBufferingMode::SingleBuffered;
    SceneCapture->bCaptureEveryFrame = true;
    SceneCapture->bCaptureOnMovement = false;
    SceneCapture->bReadPixelsEveryFrame = false;
    // Inherited USceneCaptureComponent2D field — the key fix for noise/black.
    SceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalToneCurveHDR;
    // Match the uint8 read path with an 8-bit sRGB render target.
    SceneCapture->bOverrideTextureRenderTargetFormat = true;
    SceneCapture->TextureRenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8_SRGB;
    SceneCapture->NumChannelsPerPixel = 4;
    SceneCapture->ChannelDataType = ESpArrayDataType::UInt8;
    // Lumen GI + reflections so the captured image matches the lit viewport.
    SceneCapture->PostProcessSettings.bOverride_DynamicGlobalIlluminationMethod = true;
    SceneCapture->PostProcessSettings.DynamicGlobalIlluminationMethod = EDynamicGlobalIlluminationMethod::Lumen;
    SceneCapture->PostProcessSettings.bOverride_ReflectionMethod = true;
    SceneCapture->PostProcessSettings.ReflectionMethod = EReflectionMethod::Lumen;
    SceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
    SceneCapture->Width = 640;
    SceneCapture->Height = 360;
    SceneCapture->FOVAngle = 90.f;
    SceneCapture->ProjectionType = ECameraProjectionMode::Perspective;
    SceneCapture->PrimitiveRenderMode =
        ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;

    RearSceneCapture =
        CreateDefaultSubobject<USpSceneCaptureComponent2D>(TEXT("RearSceneCapture"));
    RearSceneCapture->SetupAttachment(SpringArm);
    RearSceneCapture->bUseSharedMemory = false;
    RearSceneCapture->BufferingMode = ESpBufferingMode::SingleBuffered;
    RearSceneCapture->bCaptureEveryFrame = true;
    RearSceneCapture->bCaptureOnMovement = false;
    RearSceneCapture->bReadPixelsEveryFrame = false;
    RearSceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalToneCurveHDR;
    RearSceneCapture->bOverrideTextureRenderTargetFormat = true;
    RearSceneCapture->TextureRenderTargetFormat =
        ETextureRenderTargetFormat::RTF_RGBA8_SRGB;
    RearSceneCapture->NumChannelsPerPixel = 4;
    RearSceneCapture->ChannelDataType = ESpArrayDataType::UInt8;
    RearSceneCapture->PostProcessSettings.bOverride_DynamicGlobalIlluminationMethod = true;
    RearSceneCapture->PostProcessSettings.DynamicGlobalIlluminationMethod =
        EDynamicGlobalIlluminationMethod::Lumen;
    RearSceneCapture->PostProcessSettings.bOverride_ReflectionMethod = true;
    RearSceneCapture->PostProcessSettings.ReflectionMethod = EReflectionMethod::Lumen;
    RearSceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
    RearSceneCapture->Width = 640;
    RearSceneCapture->Height = 360;
    RearSceneCapture->FOVAngle = 90.f;
    RearSceneCapture->ProjectionType = ECameraProjectionMode::Perspective;
    RearSceneCapture->PrimitiveRenderMode =
        ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
    RearSceneCapture->SetRelativeRotation(FRotator(0.f, 180.f, 0.f));

    // Walking-mode defaults.  Accept slopes up to 89 deg so the agent_test
    // map's slightly-tilted street surfaces still classify as walkable.
    if (UCharacterMovementComponent* CM = GetCharacterMovement())
    {
        CM->MaxWalkSpeed = 200.f;
        CM->BrakingDecelerationWalking = 1024.f;
        CM->SetWalkableFloorAngle(89.f);
        CM->AirControl = 1.0f;  // full input authority while briefly in air
        CM->GravityScale = 1.0f;
    }

    // Default controller so MoveForward (AddMovementInput) actually
    // routes through CharacterMovement.  Without an owning Controller
    // ACharacter::AddMovementInput is a no-op.
    AutoPossessAI = EAutoPossessAI::Spawned;
    AIControllerClass = AAIController::StaticClass();
}

void ASpHumanoidAgent::BeginPlay()
{
    Super::BeginPlay();
    if (!AgentTag.IsNone())
    {
        Tags.AddUnique(AgentTag);
    }

    // Belt + suspenders: re-spawn default controller if one wasn't
    // already auto-possessed by EAutoPossessAI::Spawned above.  SPEAR's
    // runtime spawn_actor() path doesn't always trigger AutoPossessAI
    // since the actor is added to the world programmatically rather
    // than via the standard Spawn flow.
    if (GetController() == nullptr)
    {
        SP_LOG("SpHumanoidAgent::BeginPlay -- no controller, calling SpawnDefaultController");
        SpawnDefaultController();
    }
    if (GetController() == nullptr)
    {
        // Last resort: spawn an AIController explicitly and Possess.
        if (UWorld* World = GetWorld())
        {
            // IMPORTANT: do NOT set Params.Owner = this.  AController::Possess()
            // calls APawn::PossessedBy(), which calls SetOwner(Controller) to
            // make the *Controller* the owner of the *Pawn*.  If we had already
            // made the Pawn the owner of the Controller (Params.Owner = this),
            // SetOwner would detect a cycle (Controller owns Pawn owns Controller)
            // and fail with "would cause an Owner loop".  Leaving Owner unset is
            // the correct UE lifecycle: spawn the controller un-owned, then let
            // Possess() establish the Controller→Pawn ownership.
            FActorSpawnParameters Params;
            Params.SpawnCollisionHandlingOverride =
                ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
            AAIController* NewCtrl =
                World->SpawnActor<AAIController>(AAIController::StaticClass(),
                                                 GetActorLocation(),
                                                 GetActorRotation(),
                                                 Params);
            if (NewCtrl)
            {
                NewCtrl->Possess(this);
                SP_LOG("SpHumanoidAgent::BeginPlay -- manually spawned + possessed AIController");
            }
        }
    }
    SP_LOG_CURRENT_FUNCTION();
    if (AController* Ctrl = GetController())
    {
        SP_LOG("  controller class: ", Unreal::toStdString(Ctrl->GetClass()->GetName()));
    }
    else
    {
        SP_LOG("  WARNING: still no controller after fallback!");
    }
    if (UCharacterMovementComponent* CM = GetCharacterMovement())
    {
        SP_LOG("  movement mode: ", (int)CM->MovementMode);
        SP_LOG("  max walk speed: ", CM->MaxWalkSpeed);
    }
}

void ASpHumanoidAgent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (bDebugTickLog || EndPlayReason != EEndPlayReason::Quit)
    {
#if WITH_EDITORONLY_DATA
        const int32 bSpatiallyLoadedForLog = GetIsSpatiallyLoaded() ? 1 : 0;
#else
        const int32 bSpatiallyLoadedForLog = 0;
#endif
        UE_LOG(LogTemp, Warning,
               TEXT("SpHumanoidAgent EndPlay name=%s tag=%s reason=%d loc=%s spatial=%d"),
               *GetName(),
               *AgentTag.ToString(),
               static_cast<int32>(EndPlayReason),
               *GetActorLocation().ToString(),
               bSpatiallyLoadedForLog);
    }
    Super::EndPlay(EndPlayReason);
}

void ASpHumanoidAgent::Agent_SetAgentTag(FName InAgentTag)
{
    if (InAgentTag.IsNone())
    {
        return;
    }
    AgentTag = InAgentTag;
    Tags.AddUnique(InAgentTag);
    if (HasAuthority())
    {
        ForceNetUpdate();
    }
}

void ASpHumanoidAgent::OnRep_AgentTag()
{
    if (!AgentTag.IsNone())
    {
        Tags.AddUnique(AgentTag);
    }
}

FString ASpHumanoidAgent::Agent_GetTagString() const
{
    return AgentTag.ToString();
}

void ASpHumanoidAgent::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);

    // CRITICAL: do NOT manually force MOVE_Falling -> MOVE_Walking
    // here.  An earlier version had this guard, intended to recover
    // from LaunchCharacter side-effects, but on first-spawn it
    // triggered on frame 1 (vel.Z is still ~-32 because gravity has
    // only had one tick to act) and snapped the agent into Walking
    // mode while still at spawn altitude (z=200) -- the agent then
    // floated forever and the very-first observation frame showed
    // the Falling pose because the anim graph hadn't transitioned.
    //
    // Just let CharacterMovement's PhysFalling do its standard
    // landing detection: when the capsule hits the floor, the engine
    // calls SetMovementMode(MOVE_Walking) itself.  No manual override.
    UCharacterMovementComponent* CM = GetCharacterMovement();

    if (CM && bMovingForward)
    {
        const FVector Forward = GetActorForwardVector();
        // Add movement input (the normal CharacterMovement input pathway).
        AddMovementInput(Forward, 1.f, /*bForce=*/true);
        // Also directly seed Velocity so the agent starts moving even if
        // ApplyControlInputToVelocity's acceleration math takes time to
        // ramp up.  Walking mode integrates Velocity into position via
        // MoveAlongFloor each tick.
        if (CM->MovementMode == EMovementMode::MOVE_Walking)
        {
            FVector NewVel = Forward * CM->MaxWalkSpeed;
            NewVel.Z = CM->Velocity.Z;
            CM->Velocity = NewVel;
        }
    }
    else if (CM && !bMovingForward && CM->MovementMode == EMovementMode::MOVE_Walking)
    {
        CM->Velocity.X = 0.f;
        CM->Velocity.Y = 0.f;
    }

    // Periodic state log every 30 frames — gated by bDebugTickLog (default off).
    // The counter is a member variable so each agent instance counts independently
    // (previously was `static`, which meant all agents shared one counter and
    // produced unpredictable interleaved output in multi-agent scenarios).
    if (bDebugTickLog && CM && (++TickLogCounter % 30 == 0))
    {
        const FVector Input = GetPendingMovementInputVector();
        SP_LOG("SpHumanoidAgent tick: mode=", (int)CM->MovementMode,
               " vel=(", CM->Velocity.X, ",", CM->Velocity.Y, ",", CM->Velocity.Z, ")",
               " input=(", Input.X, ",", Input.Y, ",", Input.Z, ")",
               " pos=(", GetActorLocation().X, ",", GetActorLocation().Y, ",", GetActorLocation().Z, ")",
               " bMoveFwd=", bMovingForward ? 1 : 0);
    }

    if (RemainingYawDeg != 0.f)
    {
        const float Sign = (RemainingYawDeg > 0.f) ? 1.f : -1.f;
        float DeltaYaw = Sign * RotateRateDegPerSec * DeltaSeconds;
        // Don't overshoot.
        if (FMath::Abs(DeltaYaw) > FMath::Abs(RemainingYawDeg))
        {
            DeltaYaw = RemainingYawDeg;
        }
        AddActorWorldRotation(FRotator(0.f, DeltaYaw, 0.f));
        RemainingYawDeg -= DeltaYaw;
        if (FMath::IsNearlyZero(RemainingYawDeg, 0.01f))
        {
            RemainingYawDeg = 0.f;
        }
    }

    // Path-follow tick (Tier C #5).  If a path is active and we're close
    // enough to the current waypoint, advance to the next one via
    // MoveTo.  When we run out of waypoints, stop.
    if (CurrentWaypointIdx >= 0 && CurrentWaypointIdx < Waypoints.Num())
    {
        const FVector& WP = Waypoints[CurrentWaypointIdx];
        const FVector Delta = WP - GetActorLocation();
        const float Dist2D = Delta.Size2D();
        if (Dist2D < WaypointAcceptanceRadiusCm)
        {
            CurrentWaypointIdx++;
            if (CurrentWaypointIdx < Waypoints.Num())
            {
                SP_LOG("Path: reached wp ", CurrentWaypointIdx - 1,
                       "; advancing to wp ", CurrentWaypointIdx);
                MoveTo(Waypoints[CurrentWaypointIdx]);
            }
            else
            {
                SP_LOG("Path: completed all ", Waypoints.Num(), " waypoints");
                bMovingForward = false;
                CurrentWaypointIdx = -1;
            }
        }
    }
}

void ASpHumanoidAgent::MoveForward()
{
    bMovingForward = true;
}

void ASpHumanoidAgent::StopAgent()
{
    bMovingForward = false;
}

void ASpHumanoidAgent::Rotate(float AngleDeg, const FString& Direction)
{
    const FString D = Direction.ToLower();
    const float Sign = (D == TEXT("left") || D == TEXT("0")) ? -1.f : 1.f;
    RemainingYawDeg = Sign * FMath::Abs(AngleDeg);
}

void ASpHumanoidAgent::SetMaxSpeed(float CmPerSec)
{
    if (GetCharacterMovement())
    {
        GetCharacterMovement()->MaxWalkSpeed = FMath::Max(0.f, CmPerSec);
    }
}

bool ASpHumanoidAgent::PickUp(const FString& ObjectName)
{
    // Idempotent: if we already hold something, that IS success.
    if (HeldActor != nullptr)
    {
        return true;
    }

    UWorld* World = GetWorld();
    if (!World) return false;

    const FVector MyLoc = GetActorLocation();
    AActor* BestNearest = nullptr;
    float BestDist2 = TNumericLimits<float>::Max();

    SP_LOG("PickUp called: searching for '", Unreal::toStdString(ObjectName),
           "' from agent at (", MyLoc.X, ",", MyLoc.Y, ",", MyLoc.Z, ")");

    for (TActorIterator<AActor> It(World); It; ++It)
    {
        AActor* Candidate = *It;
        if (!Candidate || Candidate == this) continue;

        const FString Name = Candidate->GetName();
        const FString Label = Candidate->GetActorNameOrLabel();
        const float Dist = FVector::Dist(Candidate->GetActorLocation(), MyLoc);

        // Log candidates within 500 cm.
        if (Dist < 500.f)
        {
            SP_LOG("  candidate: name=", Unreal::toStdString(Name),
                   " label=", Unreal::toStdString(Label),
                   " class=", Unreal::toStdString(Candidate->GetClass()->GetName()),
                   " dist=", Dist);
        }

        // (a) exact match
        if (Name == ObjectName || Label == ObjectName)
        {
            BestNearest = Candidate;
            break;
        }

        // (b/c) interactable/box-class match within 200 cm, take closest
        if (Dist < 200.f)
        {
            const bool bLooksLikeTarget =
                Name.Contains(ObjectName) || Label.Contains(ObjectName) ||
                Name.Contains(TEXT("Box")) || Name.Contains(TEXT("Interactable")) ||
                Candidate->GetClass()->GetName().Contains(TEXT("Box")) ||
                Candidate->GetClass()->GetName().Contains(TEXT("Interactable"));
            if (bLooksLikeTarget && Dist * Dist < BestDist2)
            {
                BestDist2 = Dist * Dist;
                BestNearest = Candidate;
            }
        }
    }

    if (!BestNearest)
    {
        SP_LOG("PickUp: no matching actor found");
        return false;
    }

    SP_LOG("PickUp: attaching ", Unreal::toStdString(BestNearest->GetName()),
           " (label=", Unreal::toStdString(BestNearest->GetActorNameOrLabel()), ")");

    // CRITICAL: disable physics + collision on the picked-up actor so it
    // doesn't fight the attachment.  Without this, BP_Interactable_Box
    // has SimulatePhysics=true; attaching it to the agent root creates a
    // constraint that physics tries to resolve, sending both actors
    // flying (-127km in one run!).
    TArray<UPrimitiveComponent*> Prims;
    BestNearest->GetComponents<UPrimitiveComponent>(Prims);
    for (UPrimitiveComponent* P : Prims)
    {
        if (P && P->IsSimulatingPhysics())
        {
            P->SetSimulatePhysics(false);
        }
        if (P)
        {
            P->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        }
    }

    const FName Socket(TEXT("hand_r"));
    USkeletalMeshComponent* MeshComp = GetMesh();
    if (MeshComp && MeshComp->DoesSocketExist(Socket))
    {
        BestNearest->AttachToComponent(
            MeshComp, FAttachmentTransformRules::SnapToTargetIncludingScale, Socket);
    }
    else
    {
        // No skeletal mesh socket -- attach to root with an offset so the
        // box visibly sits in front of the agent rather than overlapping.
        BestNearest->AttachToActor(
            this, FAttachmentTransformRules::KeepWorldTransform);
        BestNearest->SetActorRelativeLocation(FVector(50.f, 0.f, 50.f));
    }
    HeldActor = BestNearest;
    // Picking up = task done, halt forward motion so agent doesn't drift.
    bMovingForward = false;
    return true;
}

void ASpHumanoidAgent::Drop()
{
    if (HeldActor)
    {
        HeldActor->DetachFromActor(FDetachmentTransformRules::KeepWorldTransform);
        HeldActor = nullptr;
    }
}

USpSceneCaptureComponent2D* ASpHumanoidAgent::GetObservationCamera(
    const FString& ViewId) const
{
    if (ViewId == TEXT("front")) return SceneCapture;
    if (ViewId == TEXT("rear")) return RearSceneCapture;
    return nullptr;
}

bool ASpHumanoidAgent::ConfigureObservationCamera(
    const FString& ViewId, int32 InWidth, int32 InHeight, float InFovDegrees)
{
    USpSceneCaptureComponent2D* Capture = GetObservationCamera(ViewId);
    if (!Capture) return false;
    Capture->Width = InWidth;
    Capture->Height = InHeight;
    Capture->FOVAngle = InFovDegrees;
    Capture->ProjectionType = ECameraProjectionMode::Perspective;
    return true;
}

void ASpHumanoidAgent::InitializeObservationCamera(const FString& ViewId)
{
    if (USpSceneCaptureComponent2D* Capture = GetObservationCamera(ViewId))
    {
        Capture->Initialize();
    }
}

void ASpHumanoidAgent::TerminateObservationCamera(const FString& ViewId)
{
    if (USpSceneCaptureComponent2D* Capture = GetObservationCamera(ViewId))
    {
        Capture->Terminate();
    }
}

bool ASpHumanoidAgent::ConfigureCamera(int32 InWidth, int32 InHeight, float InFovDegrees)
{
    return ConfigureObservationCamera(TEXT("front"), InWidth, InHeight, InFovDegrees);
}

void ASpHumanoidAgent::InitializeCamera()
{
    InitializeObservationCamera(TEXT("front"));
}

void ASpHumanoidAgent::TerminateCamera()
{
    TerminateObservationCamera(TEXT("front"));
}

bool ASpHumanoidAgent::MoveTo(FVector Target)
{
    // Tier C #7: drive the auto-spawned AAIController to path-find along
    // the level's RecastNavMesh.  We DO NOT compute the path ourselves --
    // UE's AIController + NavigationSystem handles that and steers
    // CharacterMovement via the standard pathfollowing component.
    //
    // GRACEFUL FALLBACK: if the level has no built navmesh covering the
    // current location (common in third-party maps like agent_test that
    // weren't authored for AI), MoveToLocation() returns Failed.  In that
    // case we fall through to a straight-line "face + walk forward"
    // policy so the API always makes the agent move; callers can still
    // distinguish navmesh-routed from straight-line by checking
    // IsMoving() pattern (path-follower IsMoving==True vs fallback
    // bMovingForward==True).
    AController* Ctrl = GetController();
    AAIController* AICtrl = Cast<AAIController>(Ctrl);
    if (AICtrl)
    {
        // Cancel any pending forward-walk so we don't fight the path follower.
        bMovingForward = false;

        EPathFollowingRequestResult::Type Req = AICtrl->MoveToLocation(
            Target,
            /*AcceptanceRadius=*/50.f,
            /*bStopOnOverlap=*/false,
            /*bUsePathfinding=*/true,
            /*bProjectDestinationToNavigation=*/true,
            /*bCanStrafe=*/false,
            /*FilterClass=*/nullptr,
            /*bAllowPartialPath=*/true);
        SP_LOG("MoveTo (navmesh) target=(", Target.X, ",", Target.Y, ",", Target.Z,
               ") request_result=", static_cast<int32>(Req));
        if (Req == EPathFollowingRequestResult::RequestSuccessful ||
            Req == EPathFollowingRequestResult::AlreadyAtGoal)
        {
            return true;
        }
        SP_LOG("MoveTo: navmesh path failed (no nav data?); falling back to straight-line");
    }
    else if (!Ctrl)
    {
        SP_LOG("MoveTo: no controller; using straight-line fallback");
    }

    // Straight-line fallback: face the target then enable bMovingForward.
    // Tick() will translate via AddMovementInput + Velocity write just
    // like MoveForward() does, so the agent walks directly toward Target
    // until it overshoots or the caller calls StopAgent().
    const FVector Delta = Target - GetActorLocation();
    if (Delta.SizeSquared2D() < 100.f * 100.f)
    {
        SP_LOG("MoveTo straight-line: already within 100cm of target; no-op");
        return true;
    }
    const float TargetYaw = FMath::RadiansToDegrees(FMath::Atan2(Delta.Y, Delta.X));
    SetActorRotation(FRotator(0.f, TargetYaw, 0.f));
    RemainingYawDeg = 0.f;        // cancel any in-progress rotate
    bMovingForward = true;
    SP_LOG("MoveTo straight-line: facing yaw=", TargetYaw, " walking forward");
    return true;
}

bool ASpHumanoidAgent::IsMoving() const
{
    if (const AAIController* AICtrl = Cast<AAIController>(GetController()))
    {
        const UPathFollowingComponent* PFC = AICtrl->GetPathFollowingComponent();
        if (PFC && PFC->GetStatus() == EPathFollowingStatus::Moving)
        {
            return true;
        }
    }
    // Fall back to bMovingForward so callers can still poll a meaningful
    // "is the agent actively translating" flag during pure MoveForward use.
    return bMovingForward;
}

bool ASpHumanoidAgent::SetPath(const TArray<FVector>& InWaypoints)
{
    if (InWaypoints.Num() == 0)
    {
        SP_LOG("SetPath: empty waypoint list");
        return false;
    }
    Waypoints = InWaypoints;
    CurrentWaypointIdx = 0;
    SP_LOG("SetPath: starting follow of ", Waypoints.Num(), " waypoints");
    MoveTo(Waypoints[0]);
    return true;
}

void ASpHumanoidAgent::ClearPath()
{
    Waypoints.Empty();
    CurrentWaypointIdx = -1;
    bMovingForward = false;
}

// =================================================================
// Replication + Server RPCs (Path A: UE dedicated server architecture)
// =================================================================
//
// In standalone editor mode (current Tier C tests), Replicated UPROPERTYs
// are no-ops (no network).  In dedicated-server / listen-server mode,
// these get propagated server→clients automatically.
//
// Server_* RPCs route client input to the server (where the agent
// lives, authoritative).  In standalone mode they execute locally.

void ASpHumanoidAgent::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const
{
    Super::GetLifetimeReplicatedProps(OutLifetimeProps);
    DOREPLIFETIME(ASpHumanoidAgent, AgentTag);
    DOREPLIFETIME(ASpHumanoidAgent, bMovingForward);
    DOREPLIFETIME(ASpHumanoidAgent, RemainingYawDeg);
    DOREPLIFETIME(ASpHumanoidAgent, HeldActor);
    DOREPLIFETIME(ASpHumanoidAgent, Waypoints);
    DOREPLIFETIME(ASpHumanoidAgent, CurrentWaypointIdx);
}

// All Server_* RPC implementations forward to the local impl.  In
// dedicated-server mode UE routes the call to the server actor; the
// server runs MoveForward() etc, which mutates Replicated state, which
// then propagates back to clients.
//
// In standalone mode, NetMode == NM_Standalone, the call is dispatched
// directly without networking — same behavior as calling the underlying
// method directly.

void ASpHumanoidAgent::Server_Agent_MoveForward_Implementation()
{
    MoveForward();
}

void ASpHumanoidAgent::Server_Agent_StopAgent_Implementation()
{
    StopAgent();
}

void ASpHumanoidAgent::Server_Agent_Rotate_Implementation(float AngleDeg, const FString& Direction)
{
    Rotate(AngleDeg, Direction);
}

void ASpHumanoidAgent::Server_Agent_MoveTo_Implementation(FVector Target)
{
    MoveTo(Target);
}

void ASpHumanoidAgent::Server_Agent_PickUp_Implementation(const FString& ObjectName)
{
    PickUp(ObjectName);
}

void ASpHumanoidAgent::Server_Agent_Drop_Implementation()
{
    Drop();
}

void ASpHumanoidAgent::Server_Agent_SetMaxSpeed_Implementation(float CmPerSec)
{
    SetMaxSpeed(CmPerSec);
}

void ASpHumanoidAgent::Server_Agent_SetPath_Implementation(const TArray<FVector>& InWaypoints)
{
    SetPath(InWaypoints);
}

// ---- Runtime-extensible actions -------------------------------------------

bool ASpHumanoidAgent::Agent_PlayAction(FName ActionId)
{
    // Resolve the action by name from the registry (which discovers
    // USpActionDefinition assets, including ones from a runtime-mounted pak).
    USpAgentActionRegistry* Registry = USpAgentActionRegistry::Get(this);
    if (!Registry) {
        UE_LOG(LogTemp, Warning, TEXT("Agent_PlayAction: no action registry available."));
        return false;
    }
    USpActionDefinition* Def = Registry->FindAction(ActionId);
    if (!Def) {
        UE_LOG(LogTemp, Warning, TEXT("Agent_PlayAction: action not registered: %s"), *ActionId.ToString());
        return false;
    }

    // Optional custom handler (native or Blueprint UFunction on this agent),
    // invoked via reflection so a pak-shipped Blueprint can supply arbitrary
    // action logic with no engine rebuild.
    if (!Def->HandlerFunctionName.IsNone()) {
        if (UFunction* Fn = FindFunction(Def->HandlerFunctionName)) {
            ProcessEvent(Fn, nullptr);
        } else {
            UE_LOG(LogTemp, Warning, TEXT("Agent_PlayAction: handler function not found: %s"),
                   *Def->HandlerFunctionName.ToString());
        }
    }

    const FString MontagePath = Def->Montage.ToSoftObjectPath().ToString();
    if (MontagePath.IsEmpty()) {
        // Handler-only action (no montage): success iff a handler actually ran.
        return !Def->HandlerFunctionName.IsNone();
    }
    return Agent_PlayMontageByPath(MontagePath, Def->PlayRate);
}

bool ASpHumanoidAgent::Agent_PlayMontageByPath(const FString& MontagePath, float PlayRate)
{
    if (MontagePath.IsEmpty()) {
        return false;
    }
    // Authoritative on the server; forward from a non-authoritative client.
    if (!HasAuthority()) {
        Server_Agent_PlayMontage(MontagePath, PlayRate);
        return true;
    }
    // Multicast so every net instance that renders the agent plays it.
    Multicast_PlayMontage(MontagePath, PlayRate);
    return true;
}

void ASpHumanoidAgent::Agent_StopAction()
{
    if (!HasAuthority()) {
        Server_Agent_StopAction();
        return;
    }
    Multicast_StopMontage();
}

void ASpHumanoidAgent::Server_Agent_PlayMontage_Implementation(const FString& MontagePath, float PlayRate)
{
    Multicast_PlayMontage(MontagePath, PlayRate);
}

void ASpHumanoidAgent::Server_Agent_StopAction_Implementation()
{
    Multicast_StopMontage();
}

void ASpHumanoidAgent::Multicast_PlayMontage_Implementation(const FString& MontagePath, float PlayRate)
{
    // Runs on every net instance.  Load the montage (it lives in content,
    // possibly a runtime-mounted pak) and play it on the agent's mesh.
    UAnimMontage* Montage = Cast<UAnimMontage>(
        StaticLoadObject(UAnimMontage::StaticClass(), nullptr, *MontagePath));
    if (!Montage) {
        UE_LOG(LogTemp, Warning, TEXT("Multicast_PlayMontage: could not load montage: %s"), *MontagePath);
        return;
    }
    USkeletalMeshComponent* MeshComp = GetMesh();
    UAnimInstance* Anim = MeshComp ? MeshComp->GetAnimInstance() : nullptr;
    if (!Anim) {
        // A -nullrhi dedicated server has no AnimInstance; expected — the
        // rendering clients are the ones that play the montage.  Not an error.
        return;
    }
    Anim->Montage_Play(Montage, PlayRate <= 0.f ? 1.f : PlayRate);
}

void ASpHumanoidAgent::Multicast_StopMontage_Implementation()
{
    USkeletalMeshComponent* MeshComp = GetMesh();
    UAnimInstance* Anim = MeshComp ? MeshComp->GetAnimInstance() : nullptr;
    if (Anim) {
        Anim->Montage_Stop(0.2f);
    }
}
