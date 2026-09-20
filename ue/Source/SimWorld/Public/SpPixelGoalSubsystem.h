// Copyright (c) 2026 The SimWorld Development Team. Licensed under the MIT License.

#pragma once

#include "CoreMinimal.h"
#include "AITypes.h"
#include "Navigation/PathFollowingComponent.h"
#include "Subsystems/WorldSubsystem.h"

#include "SpPixelGoalSubsystem.generated.h"

class AAIController;
class ASpHumanoidAgent;
class ASpNavMeshHelper;
class USceneCaptureComponent2D;
class USkyAtmosphereComponent;
class USpSceneCaptureComponent2D;
class FJsonObject;

/** Immutable UE view state captured with one rendered RGB frame. */
struct SIMWORLD_API FSpPixelGoalCameraSnapshot
{
    FString SnapshotId;
    FString IntrinsicsId;
    FIntPoint RenderSize = FIntPoint::ZeroValue;
    FMatrix InvViewMatrix = FMatrix::Identity;
    FMatrix InvProjectionMatrix = FMatrix::Identity;
    FVector CameraLocation = FVector::ZeroVector;
    FRotator CameraRotation = FRotator::ZeroRotator;
    FVector AgentLocation = FVector::ZeroVector;
    FRotator AgentRotation = FRotator::ZeroRotator;
    float HorizontalFovDegrees = 90.f;
    double CapturedWorldTimeSeconds = 0.0;
    FString ViewId = TEXT("front");
    FString CaptureGroupId;
    TWeakObjectPtr<USceneCaptureComponent2D> CaptureComponent;
    TWeakObjectPtr<ASpHumanoidAgent> Agent;

    bool Deproject(
        const FVector2D& NormalizedUV,
        FVector& OutOrigin,
        FVector& OutDirection) const;

#if WITH_DEV_AUTOMATION_TESTS
    static FSpPixelGoalCameraSnapshot PerspectiveForTest(
        const FVector& Location,
        const FRotator& Rotation,
        int32 Width,
        int32 Height,
        float HorizontalFovDegrees);
#endif
};

UENUM()
enum class ESpPixelGoalHitClass : uint8
{
    WalkableGround,
    NotWalkableGround,
};

/** Strict, internal-only setup for one local CityCore_Paris PoC region. */
struct SIMWORLD_API FSpPixelGoalParisPocSetup
{
    FString RegionName;
    FVector AgentSpawn = FVector::ZeroVector;
    float AgentYawDegrees = 0.f;
    FVector NavBoundsCenter = FVector::ZeroVector;
    FVector NavBoundsExtent = FVector::ZeroVector;
    bool bEnableRearCamera = false;
};

/** Strictly parsed request fields for the Task 4 atomic pair RPC. */
struct SIMWORLD_API FSpPixelGoalViewPairCaptureRequest
{
    FString AgentTag;
    int32 Width = 640;
    int32 Height = 360;
    float HorizontalFovDegrees = 90.f;
    int32 JpegQuality = 90;
    bool bValidate = true;
    /** Preview captures return RGB but do not leave resolvable action snapshots. */
    bool bCommitSnapshots = true;
};

namespace SpPixelGoal
{
    SIMWORLD_API ESpPixelGoalHitClass ClassifyFirstHit(
        const FVector& ImpactNormal,
        float MaxGroundSlopeDegrees);

    SIMWORLD_API bool WithinNavAdjustment(
        const FVector& RawHit,
        const FVector& ProjectedTarget,
        float MaxAdjustmentCm);

    SIMWORLD_API float PlanarErrorCm(const FVector& A, const FVector& B);

    /** Sum one controller polyline in the same planar metric used by walking. */
    SIMWORLD_API float ControllerPathLengthCm(const TArray<FVector>& Points);

    /** A visible pixel goal must not silently become a materially longer detour. */
    SIMWORLD_API bool ControllerPathDetourExceeded(
        float PathLengthCm,
        float DirectDistanceCm,
        float MaxStretchRatio,
        float DetourAllowanceCm);

    SIMWORLD_API FString SnapshotBindingError(
        const FString& SnapshotViewId,
        const FString& SnapshotCaptureGroupId,
        const FString& RequestedViewId,
        const FString& RequestedCaptureGroupId);

    /** Parse only exact raw JSON types accepted by the atomic pair RPC. */
    SIMWORLD_API FString ParseViewPairCaptureRequest(
        const TSharedPtr<FJsonObject>& Request,
        FSpPixelGoalViewPairCaptureRequest& OutRequest);

    /** Fail closed unless both persistent cameras already match one pair request. */
    SIMWORLD_API FString WarmedViewPairCameraError(
        const ASpHumanoidAgent* Agent,
        int32 Width,
        int32 Height,
        float HorizontalFovDegrees);

    /** Validate image bindings and the retained effective intrinsics as one pair. */
    SIMWORLD_API FString ViewPairSnapshotError(
        const FSpPixelGoalCameraSnapshot& FrontSnapshot,
        const FSpPixelGoalCameraSnapshot& RearSnapshot,
        const FString& FrontImageSnapshotId,
        const FString& RearImageSnapshotId,
        const FString& FrontImageIntrinsicsId,
        const FString& RearImageIntrinsicsId,
        int32 ExpectedWidth,
        int32 ExpectedHeight,
        float ExpectedHorizontalFovDegrees,
        const FString& CaptureGroupId,
        const USceneCaptureComponent2D* FrontComponent,
        const USceneCaptureComponent2D* RearComponent);

    /** Validate one ordered pair image without JSON type coercion. */
    SIMWORLD_API FString ViewPairImageError(
        const TSharedPtr<FJsonObject>& Image,
        const FString& ExpectedViewId,
        const FString& AgentTag,
        const FString& CaptureGroupId,
        int32 ExpectedWidth,
        int32 ExpectedHeight,
        const FVector& ExpectedAgentLocation,
        double ExpectedAgentYawDegrees,
        FString& OutSnapshotId,
        FString& OutIntrinsicsId);

#if WITH_DEV_AUTOMATION_TESTS
    SIMWORLD_API void ResetGeometryTraceEntryCountForTest();
    SIMWORLD_API int32 GeometryTraceEntryCountForTest();
    SIMWORLD_API bool PrepareParisPocCaptureForTest(
        ASpHumanoidAgent* Agent,
        bool bEnableRearCamera);
#endif

    /** Select the dedicated capture only for the live Paris PoC agent. */
    SIMWORLD_API FString CaptureModeForAgentTag(const FString& AgentTag);

    /** Keep the Paris-owned RGB capture rendering between snapshots. */
    SIMWORLD_API bool ConfigurePersistentParisCapture(
        USpSceneCaptureComponent2D* Capture);

    /** Remove the Paris-authored white horizon from the Pixel Goal RGB path. */
    SIMWORLD_API bool ConfigureParisSkyAtmosphere(
        USkyAtmosphereComponent* Atmosphere);

    /** Parse setup without exposing it through the Pixel Goal action. */
    SIMWORLD_API bool ParseParisPocSetupRequest(
        const FString& RequestJson,
        const FString& CurrentMapName,
        FSpPixelGoalParisPocSetup& OutSetup,
        FString& OutError);
}

/**
 * Live-only Pixel Goal geometry/controller service.
 *
 * Camera frames are registered by USpCameraCapturePool after synchronous RGB
 * capture. Resolution and movement remain entirely inside UE; policy callers
 * submit only the snapshot ID returned with that RGB and normalized (u, v).
 */
UCLASS()
class SIMWORLD_API USpPixelGoalSubsystem : public UWorldSubsystem
{
    GENERATED_BODY()

public:
    virtual void Deinitialize() override;

    void RecordCameraSnapshot(
        USceneCaptureComponent2D* CaptureComponent,
        ASpHumanoidAgent* Agent,
        int32 Width,
        int32 Height,
        TSharedPtr<FJsonObject>& ImageJson,
        const FString& ViewId = FString(),
        const FString& CaptureGroupId = FString());

    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_CaptureFrameJson(const FString& RequestJson);

    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_CaptureViewPairJson(const FString& RequestJson);

    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_SpawnCalibrationFixtureJson(const FString& RequestJson);

    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_GetCalibrationStatusJson(const FString& RequestJson);

    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_SetupParisPocJson(const FString& RequestJson);

    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_GetParisPocStatusJson(const FString& RequestJson);

    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_ResetParisPocTrialJson(const FString& RequestJson);

    /** Authoring-only inventory of loaded PR_Crossswalk_* scene assets. */
    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_GetCrosswalkCatalogJson(const FString& RequestJson);

    /** Authoring-only inventory of source building *_Entrance* mesh instances. */
    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_GetBuildingEntranceCatalogJson(const FString& RequestJson);

    /** Batch-project candidate pedestrian points against this run's Recast data. */
    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_AuditPedestrianPointsJson(const FString& RequestJson);

    /** Page through polygons and adjacency from this run's active Recast mesh. */
    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_GetRecastPolygonsJson(const FString& RequestJson);

    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_ResolveAndMoveJson(const FString& RequestJson);

    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_GetMoveStatusJson(const FString& RequestJson);

    UFUNCTION(BlueprintCallable, Category = "SimWorld|PixelGoal")
    FString PixelGoal_CancelMoveJson(const FString& RequestJson);

private:
    struct FMoveAudit
    {
        FString RequestId;
        FString CameraSnapshotId;
        FString CameraIntrinsicsId;
        FString ViewId;
        FString CaptureGroupId;
        bool bHasSnapshotBinding = false;
        FVector2D RequestedUV = FVector2D::ZeroVector;
        FVector RawWorldHit = FVector::ZeroVector;
        FVector AcceptedTarget = FVector::ZeroVector;
        FVector InitialFeet = FVector::ZeroVector;
        FVector FinalFeet = FVector::ZeroVector;
        FVector FinalAgentPosition = FVector::ZeroVector;
        FVector LastSampledFeet = FVector::ZeroVector;
        TArray<FVector> ControllerPathPoints;
        float ControllerPathLengthCm = 0.f;
        float ControllerPathDirectCm = 0.f;
        float ControllerPathStretchRatio = 1.f;
        FAIRequestID ControllerRequestId;
        TWeakObjectPtr<ASpHumanoidAgent> Agent;
        TWeakObjectPtr<AAIController> Controller;
        FString State = TEXT("moving");
        FString ControllerResult = TEXT("moving");
        FString FailureReason;
        float FinalYawDegrees = 0.f;
        float StartWorldTimeSeconds = 0.f;
        float ElapsedSimSeconds = 0.f;
        float DistanceTravelledCm = 0.f;
    };

    UFUNCTION()
    void HandleMoveCompleted(
        FAIRequestID RequestId,
        EPathFollowingResult::Type Result);

    static FVector GetFeetLocation(const ASpHumanoidAgent* Agent);
    void UpdateMoveSample(FMoveAudit& Audit);
    FString SerializeMoveAudit(FMoveAudit& Audit, bool bCancelled = false);
    void EvictOldSnapshots();

    TMap<FString, FSpPixelGoalCameraSnapshot> CameraSnapshots;
    TArray<FString> SnapshotOrder;
    TMap<FString, FMoveAudit> MoveAudits;
    uint64 NextSnapshotNumber = 1;
    uint64 NextCaptureGroupNumber = 1;
    uint64 NextMoveNumber = 1;
    FString ActiveCaptureGroupId;

    UPROPERTY(Transient)
    TObjectPtr<ASpNavMeshHelper> CalibrationHelper;

    UPROPERTY(Transient)
    TObjectPtr<AActor> CalibrationFloor;

    UPROPERTY(Transient)
    TObjectPtr<ASpHumanoidAgent> CalibrationAgent;

    UPROPERTY(Transient)
    TObjectPtr<AActor> CalibrationLight;

    UPROPERTY(Transient)
    TObjectPtr<AActor> CalibrationWall;

    UPROPERTY(Transient)
    TObjectPtr<ASpNavMeshHelper> ParisPocHelper;

    UPROPERTY(Transient)
    TObjectPtr<ASpHumanoidAgent> ParisPocAgent;

    FSpPixelGoalParisPocSetup ParisPocSetup;

#if WITH_DEV_AUTOMATION_TESTS
    friend class FSpPixelGoalViewPairCaptureTest;
    FString ViewPairFailureAfterBatchForTest;
#endif
};

namespace SpPixelGoal
{
    /** Polling must never mutate an audit after its controller is terminal. */
    SIMWORLD_API bool ShouldSampleMoveAudit(const FString& State);
}
