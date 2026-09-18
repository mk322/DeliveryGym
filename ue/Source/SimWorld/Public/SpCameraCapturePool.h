// Copyright (c) 2026 The SimWorld Development Team. Licensed under the MIT License.
//
// Phase 3.b: real USceneCaptureComponent2D pool serving LLM-agent
// camera observations.
//
// **Goal**: serve ~1K LLM-controlled agents' camera observations from a
// fixed-size pool of K capture components.  Each per-cluster server
// owns one pool subsystem; requests are FIFO-drained at the configured
// frame rate.
//
// Quality tiers (`ESpCameraQualityTier`) match
// docs/architecture/ARCHITECTURE_CLUSTER.md §7:
//   Hero    : human/cinematic — full SKM, post-process, shadows, 1024².
//   LLM     : LLM agent observation — cheap shaders, no shadows, 256².
//   Inspect : UI/debug snapshot — minimal, 128².
//
// Pipeline per request:
//   1. EnqueueRequest queues FSpCaptureRequest with sequence number.
//   2. Tick: pop K oldest pending requests (one per pool slot).
//   3. For each: attach pool component to request.AttachTo, set tier
//      RT size + show flags + post-process, CaptureScene().
//   4. Synchronous pixel readback (BGRA8) into FSpCaptureResult.
//   5. Result lands in ResultsBySequence map keyed by SequenceNumber.
//   6. Python/BP polls TryGetCaptureResult(SequenceNumber) until done.
//
// Async GPU read-back (FRHIGPUTextureReadback) is a future optimization;
// the synchronous CaptureScene + ReadPixels path is the baseline.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "Tickable.h"
#include "SpCameraCapturePool.generated.h"

class USceneCaptureComponent2D;
class USpSceneCaptureComponent2D;
class UTextureRenderTarget2D;
class AActor;
class AActor;

namespace SpCameraCapture
{
    SIMWORLD_API bool NormalizeCameraView(
        const FString& Requested,
        FString& OutView,
        FString& OutError);
}

UENUM(BlueprintType)
enum class ESpCameraQualityTier : uint8
{
    Hero    UMETA(DisplayName = "Hero (full quality, 1024^2)"),
    LLM     UMETA(DisplayName = "LLM (256x256, no post)"),
    Inspect UMETA(DisplayName = "Inspect (128x128, debug)"),
};

/**
 * Single capture request enqueued by an external caller (LLM batcher,
 * Python AgentRouter, or BP).  The pool drains the queue at Tick time
 * and binds K of these to its K pool components per frame.
 */
USTRUCT(BlueprintType)
struct SIMWORLD_API FSpCaptureRequest
{
    GENERATED_BODY()

    /** Actor whose mesh we attach the capture component to. */
    UPROPERTY() TObjectPtr<AActor> AttachTo;

    /** Quality tier — determines render target size + post-process. */
    UPROPERTY() ESpCameraQualityTier Tier = ESpCameraQualityTier::LLM;

    /** Stable identifier — caller's reference for matching the result back. */
    UPROPERTY() FName RequestId;

    /** Monotonically increasing per-pool sequence number (set by pool). */
    UPROPERTY() int64 SequenceNumber = 0;
};

/**
 * Result of a completed capture.  Polled by Python/BP via
 * TryGetCaptureResult(SequenceNumber, &Out).  PixelData is BGRA8 row
 * order (top-down), Width*Height*4 bytes.
 */
USTRUCT(BlueprintType)
struct SIMWORLD_API FSpCaptureResult
{
    GENERATED_BODY()

    UPROPERTY() int64 SequenceNumber = 0;
    UPROPERTY() FName RequestId;
    UPROPERTY() int32 Width = 0;
    UPROPERTY() int32 Height = 0;

    /** BGRA8, row-major, top-down.  Empty on failure. */
    UPROPERTY() TArray<uint8> PixelData;

    UPROPERTY() bool bSuccess = false;
    UPROPERTY() FString ErrorMessage;
};

/**
 * World subsystem owning the K-sized USceneCaptureComponent2D pool.
 * One instance per UWorld (= per UE cluster client + server in
 * standalone mode).
 *
 * NOTE: extends FTickableGameObject so we can drain the queue every
 * frame without waiting for Actor Tick.  UWorldSubsystem doesn't
 * provide Tick by default.
 */
UCLASS()
class SIMWORLD_API USpCameraCapturePool : public UWorldSubsystem,
                                          public FTickableGameObject
{
    GENERATED_BODY()

public:
    // ── UWorldSubsystem ────────────────────────────────────────────────────
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;

    // ── FTickableGameObject ────────────────────────────────────────────────
    virtual void Tick(float DeltaTime) override;
    virtual TStatId GetStatId() const override;
    virtual bool IsTickable() const override { return Pool.Num() > 0; }
    virtual bool IsTickableInEditor() const override { return false; }

    // ── Public API ─────────────────────────────────────────────────────────

    /**
     * Reserve K USceneCaptureComponent2D instances under an auto-spawned
     * holder actor in the world.  Idempotent: re-calling with the same
     * size is a no-op; with a different size, releases the old pool and
     * allocates fresh.
     */
    UFUNCTION(BlueprintCallable, Category = "SimWorld|Camera")
    void PreallocatePool(int32 PoolSize);

    /**
     * Submit a capture request.  Returns the assigned SequenceNumber
     * (caller uses it to poll the result via TryGetCaptureResult).
     */
    UFUNCTION(BlueprintCallable, Category = "SimWorld|Camera")
    int64 EnqueueRequest(const FSpCaptureRequest& Request);

    /**
     * Try to fetch the completed capture result for the given sequence.
     * Returns true on success; OutResult is filled with pixel data.
     * Returns false if (a) the sequence isn't known, (b) the capture
     * is still in flight, or (c) the result has already been consumed
     * (results auto-evict on first successful read).
     */
    UFUNCTION(BlueprintCallable, Category = "SimWorld|Camera")
    bool TryGetCaptureResult(int64 SequenceNumber, FSpCaptureResult& OutResult);

    /**
     * Current pool occupancy = number of pre-allocated components.
     * 0 means PreallocatePool hasn't run yet.
     */
    UFUNCTION(BlueprintPure, Category = "SimWorld|Camera")
    int32 GetPoolSize() const { return Pool.Num(); }

    /** Pending requests not yet bound to a pool component. */
    UFUNCTION(BlueprintPure, Category = "SimWorld|Camera")
    int32 GetPendingCount() const { return PendingRequests.Num(); }

    /** Completed-but-unconsumed result count. */
    UFUNCTION(BlueprintPure, Category = "SimWorld|Camera")
    int32 GetCompletedCount() const { return ResultsBySequence.Num(); }

    /**
     * Batch endpoint for production LLM/policy observations.
     *
     * This does not use legacy UnrealCV. It resolves each camera source in the
     * current render-client world, then renders every modality through the same
     * shared SimWorld/SPEAR capture backend. Sources may be agent-attached
     * cameras or standalone world-space cameras.
     *
     * Request JSON:
     * {
     *   "agent_tags": ["Agent_0000"],
     *   "camera_sources": [
     *     {"camera_id": "hero", "type": "agent", "agent_tag": "Agent_0000"},
     *     {
     *       "camera_id": "free_front",
     *       "type": "world",
     *       "location_cm": [0, 0, 220],
     *       "rotation_degrees": [-10, 0, 0]
     *     }
     *   ],
     *   "width": 320,
     *   "height": 180,
     *   "fov_degrees": 90.0,
     *   "jpeg_quality": 75,
     *   "modalities": ["rgb"],
     *   "mask_actor_tags": ["RuntimeCube_001"],
     *   "segmentation_scope": "auto",
     *   "capture_mode": "shared_pool",
     *   "shared_pool_size": 16,
     *   "force_capture": true,
     *   "validate": true,
     *   "initialize_only": false
     * }
     *
     * modalities:
     *   "rgb"          : default; lit colour frame, JPEG data URL.
     *   "depth"        : UE SceneDepth pass, PNG data URL encoded as RGB24
     *                    centimetres plus depth statistics.
     *   "normal"       : UE surface-normal pass, PNG data URL. Not requested
     *                    by default; default remains RGB.
     *   "instance_seg" : SPEAR object-id segmentation for the visible scene
     *                    when mask_actor_tags is empty, PNG data URL containing
     *                    SPEAR raw RGB24 object IDs. Python evidence/clients
     *                    decode this through SPEAR SegmentationService to
     *                    visible_id_image + visible_descs. When mask_actor_tags
     *                    is supplied, this remains the selected-target
     *                    flat-colour mask for compatibility.
     *   "semantic_seg" : class-colour variant of the same selected-actor mask
     *                    path; unselected/background pixels are black.
     *
     * mask_actor_tags:
     *   Optional actor tags/names to render into segmentation modalities. This
     *   is the supported replacement for legacy UnrealCV object_mask probes and
     *   is the path used to verify runtime-spawned actors for #88.
     *
     * segmentation_scope:
     *   "auto"           : default; instance_seg uses scene object IDs unless
     *                      mask_actor_tags are provided.
     *   "scene_object_ids": force SPEAR full-scene instance IDs.
     *   "mask_actor_tags" : force the selected-target mask path.
     *
     * capture_mode:
     *   "shared_pool"  : default; reuse a small client-local component/render
     *                    target pool while sampling each requested camera source.
     *                    Agent-attached and standalone cameras use the same
     *                    modality/render-target/readback code path.
     *   "agent_native" : initialize/read the agent's own SceneCapture target
     *                    (kept for compatibility/debug; high GPU memory; RGB
     *                    debug path only).
     */
    UFUNCTION(BlueprintCallable, Category = "SimWorld|Camera")
    FString Agent_CaptureCamerasJson(const FString& RequestJson);

    /** Stable public camera capture API. Agent_CaptureCamerasJson is kept as a compatibility alias. */
    UFUNCTION(BlueprintCallable, Category = "SimWorld|Camera")
    FString Camera_CaptureCamerasJson(const FString& RequestJson);

    /**
     * Spawn a minimal public #88 observation fixture in the current world.
     *
     * The fixture is intentionally content-light: one ASpHumanoidAgent camera
     * source plus visible runtime static, pedestrian, and vehicle target actors
     * tagged for mask/depth validation. It is a preparation helper for
     * Agent_CaptureCamerasJson evidence, not a replacement for the evidence
     * validator.
     */
    UFUNCTION(BlueprintCallable, Category = "SimWorld|Camera")
    FString Agent_SpawnObservationFixtureJson(const FString& RequestJson);

protected:
    /** Ensure the shared agent-camera batch capture component is ready. */
    USceneCaptureComponent2D* EnsureAgentBatchCapture(
        int32 Width,
        int32 Height,
        FString& OutError);

    /** Ensure N shared agent-camera batch capture components/RTs are ready. */
    bool EnsureAgentBatchCapturePool(
        int32 Width,
        int32 Height,
        int32 PoolSize,
        FString& OutError);

    /** Ensure SPEAR's full-scene object-id proxy manager exists and is initialized. */
    bool EnsureObjectIdsProxyManager(FString& OutError);

    /** Ensure N SPEAR object-id scene captures are ready for full-scene instance masks. */
    bool EnsureAgentBatchObjectIdCapturePool(
        int32 Width,
        int32 Height,
        int32 PoolSize,
        FString& OutError);

    /** Bind tier preset (RT size + show flags) to a pool component. */
    void ApplyTierPreset(USceneCaptureComponent2D* Capture, ESpCameraQualityTier Tier);

    /**
     * Process ONE pending request through one pool component:
     * attach, configure tier, CaptureScene, ReadPixels, store result.
     * Result lands in ResultsBySequence keyed by Request.SequenceNumber.
     */
    void ProcessRequest(const FSpCaptureRequest& Request, USceneCaptureComponent2D* Capture);

    /** Pre-allocated capture components, owned by HolderActor. */
    UPROPERTY()
    TArray<TObjectPtr<USceneCaptureComponent2D>> Pool;

    /** Single hidden actor that owns the capture components. */
    UPROPERTY()
    TObjectPtr<AActor> HolderActor;

    /** Reused by Agent_CaptureCamerasJson shared_pool mode. */
    UPROPERTY()
    TObjectPtr<USceneCaptureComponent2D> AgentBatchCapture;

    /** One render target reused for all agent-camera captures in a batch. */
    UPROPERTY()
    TObjectPtr<UTextureRenderTarget2D> AgentBatchRenderTarget;

    /** Shared-pool batch capture components for pipelined agent-camera capture. */
    UPROPERTY()
    TArray<TObjectPtr<USceneCaptureComponent2D>> AgentBatchCaptures;

    /** Shared-pool render targets, one per AgentBatchCaptures entry. */
    UPROPERTY()
    TArray<TObjectPtr<UTextureRenderTarget2D>> AgentBatchRenderTargets;

    /** SPEAR full-scene object-id proxy manager, hidden from public API. */
    UPROPERTY()
    TObjectPtr<AActor> ObjectIdsProxyManager;

    /** SPEAR object-id captures used when instance_seg covers the full scene. */
    UPROPERTY()
    TArray<TObjectPtr<USpSceneCaptureComponent2D>> AgentBatchObjectIdCaptures;

    /** FIFO of requests not yet bound to a pool component. */
    UPROPERTY()
    TArray<FSpCaptureRequest> PendingRequests;

    /** Completed-but-unconsumed results, keyed by SequenceNumber. */
    UPROPERTY()
    TMap<int64, FSpCaptureResult> ResultsBySequence;

    /** Per-pool monotonic sequence number assigned to each enqueued request. */
    int64 NextSequenceNumber = 1;
};
