// Copyright (c) 2026 The SimWorld Development Team. Licensed under the MIT License.

#include "SpCameraCapturePool.h"

#include "Components/SceneCaptureComponent2D.h"
#include "Components/PointLightComponent.h"
#include "Components/PrimitiveComponent.h"
#include "Components/SceneComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "Engine/PointLight.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"
#include "HAL/PlatformTime.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInterface.h"
#include "Misc/Base64.h"
#include "Modules/ModuleManager.h"
#include "RenderingThread.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "ShowFlags.h"
#include "SpCore/UnrealUtils.h"
#include "SpHumanoidAgent.h"
#include "SpPixelGoalSubsystem.h"
#include "SpPedestrianAgentBase.h"
#include "SpUnrealTypes/SpSceneCaptureComponent2D.h"
#include "SpUnrealTypes/SpMeshProxyComponentManager.h"
#include "SpVehicleAgentBase.h"
#include "Stats/Stats.h"

#include <initializer_list>

bool SpCameraCapture::NormalizeCameraView(
    const FString& Requested,
    FString& OutView,
    FString& OutError)
{
    OutView.Reset();
    OutError.Reset();
    FString Canonical;
    if (Requested.IsEmpty() ||
        Requested.Equals(TEXT("front"), ESearchCase::IgnoreCase))
    {
        Canonical = TEXT("front");
    }
    else if (Requested.Equals(TEXT("rear"), ESearchCase::IgnoreCase))
    {
        Canonical = TEXT("rear");
    }
    if (Canonical.IsEmpty() || !SpPixelGoal::IsSupportedCameraView(Canonical))
    {
        OutError = TEXT("unsupported_camera_view");
        return false;
    }
    OutView = Canonical;
    return true;
}

namespace
{
    // Tier preset table — RT side length + post-process suppression.
    // Values match docs/architecture/ARCHITECTURE_CLUSTER.md §7.  Tune via yaml in
    // a future iteration; hard-coded for Phase 3.b's first real impl.
    struct FTierPreset
    {
        int32 Side;
        bool bDisableBloom;
        bool bDisableMotionBlur;
        bool bDisableToneMapper;
        bool bDisableSSAO;
    };

    constexpr FTierPreset kHeroPreset    = {1024, false, false, false, false};
    constexpr FTierPreset kLLMPreset     = { 256, true,  true,  true,  true };
    constexpr FTierPreset kInspectPreset = { 128, true,  true,  true,  true };
    constexpr const char* kSpearObjectIdsProxyManagerStableName =
        "__SP_OBJECT_IDS_PROXY_COMPONENT_MANAGER__";

    enum class ESpAgentCaptureModality : uint8
    {
        RGB,
        Depth,
        Normal,
        InstanceSeg,
        SemanticSeg,
    };

    enum class ESpSegmentationScope : uint8
    {
        Auto,
        MaskActorTags,
        SceneObjectIds,
    };

    enum class ESpCameraSourceType : uint8
    {
        Agent,
        World,
    };

    struct FSpCameraSourceRequest
    {
        FString CameraId;
        FString AgentTag;
        FString CameraView = TEXT("front");
        FString CameraViewError;
        ESpCameraSourceType SourceType = ESpCameraSourceType::Agent;
        FVector LocationCm = FVector::ZeroVector;
        FRotator RotationDegrees = FRotator::ZeroRotator;
        bool bHasLocation = false;
        bool bHasRotation = false;
    };

    const FTierPreset& GetPreset(ESpCameraQualityTier Tier)
    {
        switch (Tier)
        {
            case ESpCameraQualityTier::Hero:    return kHeroPreset;
            case ESpCameraQualityTier::LLM:     return kLLMPreset;
            case ESpCameraQualityTier::Inspect: return kInspectPreset;
        }
        return kLLMPreset;
    }

    FString ModalityToString(ESpAgentCaptureModality Modality)
    {
        switch (Modality)
        {
            case ESpAgentCaptureModality::RGB:         return TEXT("rgb");
            case ESpAgentCaptureModality::Depth:       return TEXT("depth");
            case ESpAgentCaptureModality::Normal:      return TEXT("normal");
            case ESpAgentCaptureModality::InstanceSeg: return TEXT("instance_seg");
            case ESpAgentCaptureModality::SemanticSeg: return TEXT("semantic_seg");
        }
        return TEXT("rgb");
    }

    FString SegmentationScopeToString(ESpSegmentationScope Scope)
    {
        switch (Scope)
        {
            case ESpSegmentationScope::Auto:           return TEXT("auto");
            case ESpSegmentationScope::MaskActorTags:  return TEXT("mask_actor_tags");
            case ESpSegmentationScope::SceneObjectIds: return TEXT("scene_object_ids");
        }
        return TEXT("auto");
    }

    FString CameraSourceTypeToString(ESpCameraSourceType SourceType)
    {
        switch (SourceType)
        {
            case ESpCameraSourceType::Agent: return TEXT("agent");
            case ESpCameraSourceType::World: return TEXT("world");
        }
        return TEXT("agent");
    }

    bool TryParseModality(const FString& Raw, ESpAgentCaptureModality& OutModality)
    {
        FString Value = Raw;
        Value.TrimStartAndEndInline();
        Value.ToLowerInline();
        if (Value == TEXT("rgb") || Value == TEXT("color") || Value == TEXT("colour"))
        {
            OutModality = ESpAgentCaptureModality::RGB;
            return true;
        }
        if (Value == TEXT("depth") || Value == TEXT("scene_depth"))
        {
            OutModality = ESpAgentCaptureModality::Depth;
            return true;
        }
        if (Value == TEXT("normal") || Value == TEXT("normals") ||
            Value == TEXT("scene_normal") || Value == TEXT("world_normal"))
        {
            OutModality = ESpAgentCaptureModality::Normal;
            return true;
        }
        if (Value == TEXT("instance_seg") || Value == TEXT("instance") ||
            Value == TEXT("instance_segmentation") || Value == TEXT("object_mask") ||
            Value == TEXT("mask"))
        {
            OutModality = ESpAgentCaptureModality::InstanceSeg;
            return true;
        }
        if (Value == TEXT("semantic_seg") || Value == TEXT("semantic") ||
            Value == TEXT("semantic_segmentation"))
        {
            OutModality = ESpAgentCaptureModality::SemanticSeg;
            return true;
        }
        return false;
    }

    TArray<ESpAgentCaptureModality> ParseModalities(const TSharedPtr<FJsonObject>& Root)
    {
        TArray<ESpAgentCaptureModality> Modalities;
        if (!Root.IsValid())
        {
            Modalities.Add(ESpAgentCaptureModality::RGB);
            return Modalities;
        }

        FString SingleModality;
        if (Root->TryGetStringField(TEXT("modality"), SingleModality))
        {
            ESpAgentCaptureModality Parsed;
            if (TryParseModality(SingleModality, Parsed))
            {
                Modalities.AddUnique(Parsed);
            }
        }

        const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
        if (Root->TryGetArrayField(TEXT("modalities"), Values))
        {
            for (const TSharedPtr<FJsonValue>& Value : *Values)
            {
                if (!Value.IsValid())
                {
                    continue;
                }
                ESpAgentCaptureModality Parsed;
                if (TryParseModality(Value->AsString(), Parsed))
                {
                    Modalities.AddUnique(Parsed);
                }
            }
        }

        if (Modalities.IsEmpty())
        {
            Modalities.Add(ESpAgentCaptureModality::RGB);
        }
        return Modalities;
    }

    ESpSegmentationScope ParseSegmentationScope(const TSharedPtr<FJsonObject>& Root)
    {
        FString Value;
        if (!Root.IsValid() ||
            (!Root->TryGetStringField(TEXT("segmentation_scope"), Value) &&
             !Root->TryGetStringField(TEXT("segmentation_mode"), Value)))
        {
            return ESpSegmentationScope::Auto;
        }

        Value.TrimStartAndEndInline();
        Value.ToLowerInline();
        if (Value == TEXT("scene") ||
            Value == TEXT("all") ||
            Value == TEXT("all_visible") ||
            Value == TEXT("object_ids") ||
            Value == TEXT("spear_object_ids") ||
            Value == TEXT("scene_object_ids"))
        {
            return ESpSegmentationScope::SceneObjectIds;
        }
        if (Value == TEXT("mask") ||
            Value == TEXT("target") ||
            Value == TEXT("targets") ||
            Value == TEXT("selected") ||
            Value == TEXT("mask_actor_tags"))
        {
            return ESpSegmentationScope::MaskActorTags;
        }
        return ESpSegmentationScope::Auto;
    }

    bool ShouldUseSceneObjectIds(
        ESpAgentCaptureModality Modality,
        const TArray<FString>& MaskActorTags,
        ESpSegmentationScope Scope)
    {
        if (Modality != ESpAgentCaptureModality::InstanceSeg)
        {
            return false;
        }
        if (Scope == ESpSegmentationScope::SceneObjectIds)
        {
            return true;
        }
        if (Scope == ESpSegmentationScope::MaskActorTags)
        {
            return false;
        }
        return MaskActorTags.IsEmpty();
    }

    TArray<FString> ParseStringArrayAliases(
        const TSharedPtr<FJsonObject>& Root,
        std::initializer_list<const TCHAR*> FieldNames)
    {
        TArray<FString> Strings;
        if (!Root.IsValid())
        {
            return Strings;
        }

        const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
        for (const TCHAR* FieldName : FieldNames)
        {
            if (Root->TryGetArrayField(FieldName, Values))
            {
                break;
            }
        }
        if (!Values)
        {
            return Strings;
        }

        for (const TSharedPtr<FJsonValue>& Value : *Values)
        {
            if (!Value.IsValid())
            {
                continue;
            }
            FString Text = Value->AsString();
            Text.TrimStartAndEndInline();
            if (!Text.IsEmpty())
            {
                Strings.AddUnique(Text);
            }
        }
        return Strings;
    }

    bool TryGetJsonNumberAliases(
        const TSharedPtr<FJsonObject>& Root,
        std::initializer_list<const TCHAR*> FieldNames,
        double& OutValue)
    {
        if (!Root.IsValid())
        {
            return false;
        }
        for (const TCHAR* FieldName : FieldNames)
        {
            if (Root->TryGetNumberField(FieldName, OutValue))
            {
                return true;
            }
        }
        return false;
    }

    bool TryGetJsonVectorAliases(
        const TSharedPtr<FJsonObject>& Root,
        std::initializer_list<const TCHAR*> FieldNames,
        FVector& OutValue)
    {
        if (!Root.IsValid())
        {
            return false;
        }
        for (const TCHAR* FieldName : FieldNames)
        {
            const TArray<TSharedPtr<FJsonValue>>* ArrayValues = nullptr;
            if (Root->TryGetArrayField(FieldName, ArrayValues) &&
                ArrayValues &&
                ArrayValues->Num() >= 3)
            {
                OutValue = FVector(
                    (*ArrayValues)[0]->AsNumber(),
                    (*ArrayValues)[1]->AsNumber(),
                    (*ArrayValues)[2]->AsNumber());
                return true;
            }

            const TSharedPtr<FJsonObject>* ObjectValue = nullptr;
            if (Root->TryGetObjectField(FieldName, ObjectValue) && ObjectValue)
            {
                double X = 0.0;
                double Y = 0.0;
                double Z = 0.0;
                if (TryGetJsonNumberAliases(*ObjectValue, {TEXT("X"), TEXT("x")}, X) &&
                    TryGetJsonNumberAliases(*ObjectValue, {TEXT("Y"), TEXT("y")}, Y) &&
                    TryGetJsonNumberAliases(*ObjectValue, {TEXT("Z"), TEXT("z")}, Z))
                {
                    OutValue = FVector(X, Y, Z);
                    return true;
                }
            }
        }
        return false;
    }

    bool TryGetJsonRotatorAliases(
        const TSharedPtr<FJsonObject>& Root,
        std::initializer_list<const TCHAR*> FieldNames,
        FRotator& OutValue)
    {
        if (!Root.IsValid())
        {
            return false;
        }
        for (const TCHAR* FieldName : FieldNames)
        {
            const TArray<TSharedPtr<FJsonValue>>* ArrayValues = nullptr;
            if (Root->TryGetArrayField(FieldName, ArrayValues) &&
                ArrayValues &&
                ArrayValues->Num() >= 3)
            {
                OutValue = FRotator(
                    (*ArrayValues)[0]->AsNumber(),
                    (*ArrayValues)[1]->AsNumber(),
                    (*ArrayValues)[2]->AsNumber());
                return true;
            }

            const TSharedPtr<FJsonObject>* ObjectValue = nullptr;
            if (Root->TryGetObjectField(FieldName, ObjectValue) && ObjectValue)
            {
                double Pitch = 0.0;
                double Yaw = 0.0;
                double Roll = 0.0;
                if (TryGetJsonNumberAliases(
                        *ObjectValue,
                        {TEXT("Pitch"), TEXT("pitch"), TEXT("P"), TEXT("p")},
                        Pitch) &&
                    TryGetJsonNumberAliases(
                        *ObjectValue,
                        {TEXT("Yaw"), TEXT("yaw"), TEXT("Y"), TEXT("y")},
                        Yaw))
                {
                    TryGetJsonNumberAliases(
                        *ObjectValue,
                        {TEXT("Roll"), TEXT("roll"), TEXT("R"), TEXT("r")},
                        Roll);
                    OutValue = FRotator(Pitch, Yaw, Roll);
                    return true;
                }
            }
        }
        return false;
    }

    bool TryGetJsonStringAliases(
        const TSharedPtr<FJsonObject>& Root,
        std::initializer_list<const TCHAR*> FieldNames,
        FString& OutValue)
    {
        if (!Root.IsValid())
        {
            return false;
        }
        for (const TCHAR* FieldName : FieldNames)
        {
            if (Root->TryGetStringField(FieldName, OutValue))
            {
                OutValue.TrimStartAndEndInline();
                return !OutValue.IsEmpty();
            }
        }
        return false;
    }

    bool IsWorldCameraSourceTypeString(FString Value)
    {
        Value.TrimStartAndEndInline();
        Value.ToLowerInline();
        return Value == TEXT("world") ||
            Value == TEXT("standalone") ||
            Value == TEXT("free") ||
            Value == TEXT("fixed") ||
            Value == TEXT("static");
    }

    bool IsAgentCameraSourceTypeString(FString Value)
    {
        Value.TrimStartAndEndInline();
        Value.ToLowerInline();
        return Value == TEXT("agent") ||
            Value == TEXT("attached") ||
            Value == TEXT("agent_attached");
    }

    TArray<TSharedPtr<FJsonValue>> JsonVectorArray(const FVector& Value)
    {
        TArray<TSharedPtr<FJsonValue>> Out;
        Out.Add(MakeShared<FJsonValueNumber>(Value.X));
        Out.Add(MakeShared<FJsonValueNumber>(Value.Y));
        Out.Add(MakeShared<FJsonValueNumber>(Value.Z));
        return Out;
    }

    TArray<TSharedPtr<FJsonValue>> JsonRotatorArray(const FRotator& Value)
    {
        TArray<TSharedPtr<FJsonValue>> Out;
        Out.Add(MakeShared<FJsonValueNumber>(Value.Pitch));
        Out.Add(MakeShared<FJsonValueNumber>(Value.Yaw));
        Out.Add(MakeShared<FJsonValueNumber>(Value.Roll));
        return Out;
    }

    bool IsMaskModality(ESpAgentCaptureModality Modality)
    {
        return Modality == ESpAgentCaptureModality::InstanceSeg ||
               Modality == ESpAgentCaptureModality::SemanticSeg;
    }

    bool IsSegmentationMeshPrimitive(const UPrimitiveComponent* Component)
    {
        return Component &&
               Component->IsRegistered() &&
               (Component->IsA<UStaticMeshComponent>() ||
                Component->IsA<USkeletalMeshComponent>());
    }

    void DisableDiagnosticPrimitiveShowFlags(FEngineShowFlags& ShowFlags)
    {
        ShowFlags.SetBounds(false);
        ShowFlags.SetBones(false);
        ShowFlags.SetBrushes(false);
        ShowFlags.SetBSPSplit(false);
        ShowFlags.SetCollision(false);
        ShowFlags.SetConstraints(false);
        ShowFlags.SetHitProxies(false);
        ShowFlags.SetInputDebugVisualizer(false);
        ShowFlags.SetLargeVertices(false);
        ShowFlags.SetModeWidgets(false);
        ShowFlags.SetNavigation(false);
        ShowFlags.SetPhysicsField(false);
        ShowFlags.SetSelection(false);
        ShowFlags.SetSelectionOutline(false);
        ShowFlags.SetServerDrawDebug(false);
        ShowFlags.SetSplines(false);
        ShowFlags.SetVisualizeOutOfBoundsPixels(false);
        ShowFlags.SetVisualizeSenses(false);
    }

    FColor MakeDeterministicMaskColor(
        const FString& TagText,
        const AActor* Agent,
        ESpAgentCaptureModality Modality)
    {
        const FString Seed = (Modality == ESpAgentCaptureModality::SemanticSeg && Agent)
            ? Agent->GetClass()->GetPathName()
            : TagText;
        const uint32 Hash = GetTypeHash(Seed);
        const uint8 Hue = static_cast<uint8>(Hash & 0xff);
        const FLinearColor Linear = FLinearColor::MakeFromHSV8(Hue, 220, 255);
        FColor Color = Linear.ToFColor(false);
        Color.A = 255;
        if (Color.R < 24 && Color.G < 24 && Color.B < 24)
        {
            Color.R = 255;
        }
        return Color;
    }

    int32 ApplyFlatMask(
        TArray<FColor>& Bitmap,
        const FColor& MaskColor,
        int32 Threshold = 6)
    {
        int32 MaskPixels = 0;
        for (FColor& Pixel : Bitmap)
        {
            const int32 MaxChannel = FMath::Max<int32>(
                Pixel.R,
                FMath::Max<int32>(Pixel.G, Pixel.B));
            if (MaxChannel > Threshold)
            {
                Pixel = MaskColor;
                ++MaskPixels;
            }
            else
            {
                Pixel = FColor(0, 0, 0, 255);
            }
        }
        return MaskPixels;
    }

    AActor* SpawnHiddenCaptureHolder(UWorld* World)
    {
        if (!World)
        {
            return nullptr;
        }

        FActorSpawnParameters Params;
        Params.SpawnCollisionHandlingOverride =
            ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AActor* Holder = World->SpawnActor<AActor>(
            AActor::StaticClass(), FTransform::Identity, Params);
        if (!Holder)
        {
            return nullptr;
        }

        Holder->SetActorHiddenInGame(true);
        Holder->SetActorEnableCollision(false);
        Holder->SetCanBeDamaged(false);

        if (!Holder->GetRootComponent())
        {
            USceneComponent* Root = NewObject<USceneComponent>(
                Holder,
                USceneComponent::StaticClass(),
                TEXT("SpCameraCapturePoolRoot"));
            if (!Root)
            {
                Holder->Destroy();
                return nullptr;
            }
            Root->CreationMethod = EComponentCreationMethod::Instance;
            Holder->SetRootComponent(Root);
            Root->RegisterComponentWithWorld(World);
        }

        return Holder;
    }

    TArray<FString> ParseAgentTags(const TSharedPtr<FJsonObject>& Root)
    {
        TArray<FString> Tags;
        if (!Root.IsValid())
        {
            return Tags;
        }

        const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
        if (!Root->TryGetArrayField(TEXT("agent_tags"), Values) &&
            !Root->TryGetArrayField(TEXT("tags"), Values))
        {
            return Tags;
        }

        for (const TSharedPtr<FJsonValue>& Value : *Values)
        {
            if (!Value.IsValid())
            {
                continue;
            }
            const FString Tag = Value->AsString();
            if (!Tag.IsEmpty())
            {
                Tags.Add(Tag);
            }
        }
        return Tags;
    }

    TArray<FSpCameraSourceRequest> ParseCameraSourceRequests(
        const TSharedPtr<FJsonObject>& Root,
        const TArray<FString>& LegacyAgentTags,
        bool bHasTopLevelLocation,
        const FVector& TopLevelLocation,
        bool bHasTopLevelRotation,
        const FRotator& TopLevelRotation)
    {
        TArray<FSpCameraSourceRequest> Sources;
        if (!Root.IsValid())
        {
            return Sources;
        }

        const TArray<TSharedPtr<FJsonValue>>* SourceValues = nullptr;
        if (Root->TryGetArrayField(TEXT("camera_sources"), SourceValues) ||
            Root->TryGetArrayField(TEXT("cameras"), SourceValues))
        {
            int32 Index = 0;
            for (const TSharedPtr<FJsonValue>& Value : *SourceValues)
            {
                if (!Value.IsValid())
                {
                    ++Index;
                    continue;
                }

                FSpCameraSourceRequest Source;
                TSharedPtr<FJsonObject> SourceObject = Value->AsObject();
                if (SourceObject.IsValid())
                {
                    FString RequestedCameraView;
                    if (SourceObject->HasField(TEXT("camera_view")) &&
                        !SourceObject->TryGetStringField(
                            TEXT("camera_view"), RequestedCameraView))
                    {
                        Source.CameraView.Reset();
                        Source.CameraViewError = TEXT("unsupported_camera_view");
                    }
                    else
                    {
                        SpCameraCapture::NormalizeCameraView(
                            RequestedCameraView,
                            Source.CameraView,
                            Source.CameraViewError);
                    }
                    FString TypeText;
                    TryGetJsonStringAliases(
                        SourceObject,
                        {TEXT("type"), TEXT("source_type"), TEXT("kind")},
                        TypeText);
                    TryGetJsonStringAliases(
                        SourceObject,
                        {
                            TEXT("agent_tag"),
                            TEXT("attached_to_tag"),
                            TEXT("attached_to"),
                            TEXT("tag")
                        },
                        Source.AgentTag);
                    TryGetJsonStringAliases(
                        SourceObject,
                        {TEXT("camera_id"), TEXT("id"), TEXT("name")},
                        Source.CameraId);
                    Source.bHasLocation = TryGetJsonVectorAliases(
                        SourceObject,
                        {
                            TEXT("camera_location_cm"),
                            TEXT("camera_location"),
                            TEXT("location_cm"),
                            TEXT("location")
                        },
                        Source.LocationCm);
                    Source.bHasRotation = TryGetJsonRotatorAliases(
                        SourceObject,
                        {
                            TEXT("camera_rotation_degrees"),
                            TEXT("camera_rotation"),
                            TEXT("rotation_degrees"),
                            TEXT("rotation")
                        },
                        Source.RotationDegrees);

                    const bool bExplicitWorld = IsWorldCameraSourceTypeString(TypeText);
                    const bool bExplicitAgent = IsAgentCameraSourceTypeString(TypeText);
                    Source.SourceType =
                        (bExplicitWorld ||
                         (!bExplicitAgent && Source.AgentTag.IsEmpty() &&
                          (Source.bHasLocation || Source.bHasRotation)))
                            ? ESpCameraSourceType::World
                            : ESpCameraSourceType::Agent;
                    if (Source.SourceType == ESpCameraSourceType::Agent &&
                        Source.AgentTag.IsEmpty() &&
                        !Source.CameraId.IsEmpty())
                    {
                        Source.AgentTag = Source.CameraId;
                    }
                }
                else
                {
                    Source.AgentTag = Value->AsString();
                    Source.AgentTag.TrimStartAndEndInline();
                    Source.CameraId = Source.AgentTag;
                    Source.SourceType = ESpCameraSourceType::Agent;
                }

                if (Source.SourceType == ESpCameraSourceType::World)
                {
                    if (!Source.bHasLocation && bHasTopLevelLocation)
                    {
                        Source.LocationCm = TopLevelLocation;
                        Source.bHasLocation = true;
                    }
                    if (!Source.bHasRotation && bHasTopLevelRotation)
                    {
                        Source.RotationDegrees = TopLevelRotation;
                        Source.bHasRotation = true;
                    }
                }
                else
                {
                    if (!Source.bHasLocation && bHasTopLevelLocation)
                    {
                        Source.LocationCm = TopLevelLocation;
                        Source.bHasLocation = true;
                    }
                    if (!Source.bHasRotation && bHasTopLevelRotation)
                    {
                        Source.RotationDegrees = TopLevelRotation;
                        Source.bHasRotation = true;
                    }
                }

                if (Source.CameraId.IsEmpty())
                {
                    Source.CameraId = Source.SourceType == ESpCameraSourceType::Agent
                        ? Source.AgentTag
                        : FString::Printf(TEXT("camera_%d"), Index);
                }
                Source.CameraId.TrimStartAndEndInline();
                Source.AgentTag.TrimStartAndEndInline();
                if (!Source.CameraId.IsEmpty())
                {
                    Sources.Add(Source);
                }
                ++Index;
            }
        }

        if (Sources.IsEmpty())
        {
            for (const FString& LegacyTag : LegacyAgentTags)
            {
                FString Tag = LegacyTag;
                Tag.TrimStartAndEndInline();
                if (Tag.IsEmpty())
                {
                    continue;
                }
                FSpCameraSourceRequest Source;
                Source.CameraId = Tag;
                Source.AgentTag = Tag;
                Source.SourceType = ESpCameraSourceType::Agent;
                Source.LocationCm = TopLevelLocation;
                Source.RotationDegrees = TopLevelRotation;
                Source.bHasLocation = bHasTopLevelLocation;
                Source.bHasRotation = bHasTopLevelRotation;
                Sources.Add(Source);
            }
        }

        if (Sources.IsEmpty() && (bHasTopLevelLocation || bHasTopLevelRotation))
        {
            FSpCameraSourceRequest Source;
            TryGetJsonStringAliases(
                Root,
                {TEXT("camera_id"), TEXT("id"), TEXT("name")},
                Source.CameraId);
            if (Source.CameraId.IsEmpty())
            {
                Source.CameraId = TEXT("camera");
            }
            Source.SourceType = ESpCameraSourceType::World;
            Source.LocationCm = TopLevelLocation;
            Source.RotationDegrees = TopLevelRotation;
            Source.bHasLocation = bHasTopLevelLocation;
            Source.bHasRotation = bHasTopLevelRotation;
            Sources.Add(Source);
        }

        return Sources;
    }

    ASpHumanoidAgent* FindHumanoidByTagInWorld(UWorld* World, const FName& AgentTag)
    {
        if (!World || AgentTag.IsNone())
        {
            return nullptr;
        }
        for (TActorIterator<ASpHumanoidAgent> It(World); It; ++It)
        {
            ASpHumanoidAgent* Agent = *It;
            if (!IsValid(Agent))
            {
                continue;
            }
            if (Agent->AgentTag == AgentTag || Agent->Tags.Contains(AgentTag))
            {
                return Agent;
            }
        }
        return nullptr;
    }

    bool ActorMatchesTagText(const AActor* Actor, const FString& TagText)
    {
        if (!Actor || TagText.IsEmpty())
        {
            return false;
        }

        const FName TagName(*TagText);
        return Actor->Tags.Contains(TagName) ||
               Actor->GetFName() == TagName ||
               Actor->GetName().Equals(TagText, ESearchCase::IgnoreCase);
    }

    TArray<TWeakObjectPtr<AActor>> FindActorsByTagTextsInWorld(
        UWorld* World,
        const TArray<FString>& TagTexts)
    {
        TArray<TWeakObjectPtr<AActor>> Actors;
        if (!World || TagTexts.IsEmpty())
        {
            return Actors;
        }

        for (TActorIterator<AActor> It(World); It; ++It)
        {
            AActor* Actor = *It;
            if (!IsValid(Actor))
            {
                continue;
            }
            for (const FString& TagText : TagTexts)
            {
                if (ActorMatchesTagText(Actor, TagText))
                {
                    Actors.AddUnique(Actor);
                    break;
                }
            }
        }
        return Actors;
    }

    AActor* FindFirstActorByTagTextInWorld(UWorld* World, const FString& TagText)
    {
        if (!World || TagText.IsEmpty())
        {
            return nullptr;
        }
        for (TActorIterator<AActor> It(World); It; ++It)
        {
            AActor* Actor = *It;
            if (IsValid(Actor) && ActorMatchesTagText(Actor, TagText))
            {
                return Actor;
            }
        }
        return nullptr;
    }

    FString JsonStringOrDefault(
        const TSharedPtr<FJsonObject>& Request,
        const TCHAR* Field,
        const TCHAR* DefaultValue)
    {
        FString Value;
        if (Request.IsValid() && Request->TryGetStringField(Field, Value))
        {
            Value.TrimStartAndEndInline();
            if (!Value.IsEmpty())
            {
                return Value;
            }
        }
        return FString(DefaultValue);
    }

    void AddTagToActor(AActor* Actor, const FName& Tag)
    {
        if (Actor && !Tag.IsNone())
        {
            Actor->Tags.AddUnique(Tag);
        }
    }

    void MakeActorComponentsMovable(AActor* Actor)
    {
        if (!Actor)
        {
            return;
        }
        TInlineComponentArray<USceneComponent*> SceneComponents(Actor);
        for (USceneComponent* SceneComponent : SceneComponents)
        {
            if (SceneComponent)
            {
                SceneComponent->SetMobility(EComponentMobility::Movable);
            }
        }
    }

    USceneComponent* EnsureSceneRoot(AActor* Actor)
    {
        if (!Actor)
        {
            return nullptr;
        }
        if (USceneComponent* Root = Actor->GetRootComponent())
        {
            return Root;
        }
        USceneComponent* Root = NewObject<USceneComponent>(
            Actor,
            USceneComponent::StaticClass(),
            TEXT("SimWorldObservationFixtureRoot"));
        if (!Root)
        {
            return nullptr;
        }
        Root->CreationMethod = EComponentCreationMethod::Instance;
        Actor->SetRootComponent(Root);
        Actor->AddInstanceComponent(Root);
        Root->RegisterComponent();
        return Root;
    }

    bool EnsureVisibleMeshComponent(
        AActor* Actor,
        UStaticMesh* Mesh,
        UMaterialInterface* Material,
        const FLinearColor& MaterialColor,
        const FName& ComponentName,
        const FVector& RelativeLocation,
        const FVector& RelativeScale,
        FString& OutError)
    {
        if (!Actor)
        {
            OutError = TEXT("missing_actor");
            return false;
        }
        Actor->SetActorHiddenInGame(false);
        Actor->SetActorEnableCollision(false);
        if (!Mesh)
        {
            OutError = TEXT("missing_engine_static_mesh");
            return false;
        }

        UStaticMeshComponent* MeshComponent = nullptr;
        if (AStaticMeshActor* StaticMeshActor = Cast<AStaticMeshActor>(Actor))
        {
            MeshComponent = StaticMeshActor->GetStaticMeshComponent();
        }
        if (!MeshComponent)
        {
            MeshComponent = NewObject<UStaticMeshComponent>(
                Actor,
                UStaticMeshComponent::StaticClass(),
                ComponentName);
            if (!MeshComponent)
            {
                OutError = TEXT("create_static_mesh_component_failed");
                return false;
            }
            MeshComponent->CreationMethod = EComponentCreationMethod::Instance;
            Actor->AddInstanceComponent(MeshComponent);
            if (USceneComponent* Root = EnsureSceneRoot(Actor))
            {
                MeshComponent->AttachToComponent(
                    Root,
                    FAttachmentTransformRules::KeepRelativeTransform);
            }
        }

        MeshComponent->SetMobility(EComponentMobility::Movable);
        MeshComponent->SetStaticMesh(Mesh);
        if (Material)
        {
            UMaterialInstanceDynamic* DynamicMaterial =
                UMaterialInstanceDynamic::Create(Material, Actor);
            if (DynamicMaterial)
            {
                DynamicMaterial->SetVectorParameterValue(
                    TEXT("EmissiveColor"),
                    MaterialColor);
                const int32 MaterialCount = FMath::Max(1, MeshComponent->GetNumMaterials());
                for (int32 Index = 0; Index < MaterialCount; ++Index)
                {
                    MeshComponent->SetMaterial(Index, DynamicMaterial);
                }
            }
        }
        MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        MeshComponent->SetVisibility(true, true);
        MeshComponent->SetHiddenInGame(false, true);
        MeshComponent->SetRelativeLocation(RelativeLocation);
        MeshComponent->SetRelativeScale3D(RelativeScale);
        if (!MeshComponent->IsRegistered())
        {
            MeshComponent->RegisterComponent();
        }
        return true;
    }

    TSharedPtr<FJsonObject> MakeErrorObject(const FString& Tag, const FString& Error)
    {
        TSharedPtr<FJsonObject> Object = MakeShared<FJsonObject>();
        Object->SetStringField(TEXT("agent_tag"), Tag);
        Object->SetBoolField(TEXT("success"), false);
        Object->SetStringField(TEXT("error"), Error);
        return Object;
    }

    void ComputeBitmapStats(
        const TArray<FColor>& Bitmap,
        double& OutMean,
        double& OutStd,
        int32& OutMin,
        int32& OutMax)
    {
        OutMean = 0.0;
        OutStd = 0.0;
        OutMin = 255;
        OutMax = 0;
        const int64 Count = static_cast<int64>(Bitmap.Num()) * 3;
        if (Count <= 0)
        {
            OutMin = 0;
            return;
        }

        double Sum = 0.0;
        double SumSq = 0.0;
        for (const FColor& Pixel : Bitmap)
        {
            const int32 Channels[3] = {Pixel.R, Pixel.G, Pixel.B};
            for (int32 Value : Channels)
            {
                OutMin = FMath::Min(OutMin, Value);
                OutMax = FMath::Max(OutMax, Value);
                Sum += static_cast<double>(Value);
                SumSq += static_cast<double>(Value) * static_cast<double>(Value);
            }
        }
        OutMean = Sum / static_cast<double>(Count);
        const double Variance = FMath::Max(
            0.0,
            (SumSq / static_cast<double>(Count)) - (OutMean * OutMean));
        OutStd = FMath::Sqrt(Variance);
    }

    bool EncodeJpegBase64(
        const TArray<FColor>& Bitmap,
        int32 Width,
        int32 Height,
        int32 Quality,
        FString& OutDataUrl,
        int32& OutBytes,
        FString& OutError)
    {
        OutDataUrl.Reset();
        OutBytes = 0;
        if (Bitmap.Num() != Width * Height)
        {
            OutError = TEXT("bitmap size does not match dimensions");
            return false;
        }

        IImageWrapperModule& ImageWrapperModule =
            FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
        TSharedPtr<IImageWrapper> ImageWrapper =
            ImageWrapperModule.CreateImageWrapper(EImageFormat::JPEG);
        if (!ImageWrapper.IsValid())
        {
            OutError = TEXT("failed to create JPEG image wrapper");
            return false;
        }

        if (!ImageWrapper->SetRaw(
                Bitmap.GetData(),
                Bitmap.Num() * sizeof(FColor),
                Width,
                Height,
                ERGBFormat::BGRA,
                8))
        {
            OutError = TEXT("JPEG SetRaw failed");
            return false;
        }

        const TArray64<uint8>& Encoded =
            ImageWrapper->GetCompressed(FMath::Clamp(Quality, 1, 95));
        if (Encoded.Num() <= 0)
        {
            OutError = TEXT("JPEG compressor returned no bytes");
            return false;
        }

        OutBytes = static_cast<int32>(Encoded.Num());
        OutDataUrl = TEXT("data:image/jpeg;base64,") +
            FBase64::Encode(Encoded.GetData(), static_cast<uint32>(Encoded.Num()));
        return true;
    }

    bool EncodeImageBase64(
        const TArray<FColor>& Bitmap,
        int32 Width,
        int32 Height,
        EImageFormat ImageFormat,
        const FString& MimeType,
        int32 Quality,
        FString& OutDataUrl,
        int32& OutBytes,
        FString& OutError)
    {
        OutDataUrl.Reset();
        OutBytes = 0;
        if (Bitmap.Num() != Width * Height)
        {
            OutError = TEXT("bitmap size does not match dimensions");
            return false;
        }

        IImageWrapperModule& ImageWrapperModule =
            FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
        TSharedPtr<IImageWrapper> ImageWrapper =
            ImageWrapperModule.CreateImageWrapper(ImageFormat);
        if (!ImageWrapper.IsValid())
        {
            OutError = TEXT("failed to create image wrapper");
            return false;
        }

        if (!ImageWrapper->SetRaw(
                Bitmap.GetData(),
                Bitmap.Num() * sizeof(FColor),
                Width,
                Height,
                ERGBFormat::BGRA,
                8))
        {
            OutError = TEXT("image SetRaw failed");
            return false;
        }

        const TArray64<uint8>& Encoded =
            ImageWrapper->GetCompressed(FMath::Clamp(Quality, 1, 100));
        if (Encoded.Num() <= 0)
        {
            OutError = TEXT("image compressor returned no bytes");
            return false;
        }

        OutBytes = static_cast<int32>(Encoded.Num());
        OutDataUrl = TEXT("data:") + MimeType + TEXT(";base64,") +
            FBase64::Encode(Encoded.GetData(), static_cast<uint32>(Encoded.Num()));
        return true;
    }

    uint32 DecodeDepthCmFromRgb24(const FColor& Pixel)
    {
        return (static_cast<uint32>(Pixel.R) << 16) |
               (static_cast<uint32>(Pixel.G) << 8) |
               static_cast<uint32>(Pixel.B);
    }

    FColor EncodeDepthCmToRgb24(float DepthCm)
    {
        if (!FMath::IsFinite(DepthCm) || DepthCm <= 0.0f)
        {
            return FColor(0, 0, 0, 255);
        }
        const uint32 EncodedCm = static_cast<uint32>(
            FMath::Clamp(
                FMath::RoundToInt(DepthCm),
                1,
                0xFFFFFF));
        return FColor(
            static_cast<uint8>((EncodedCm >> 16) & 0xff),
            static_cast<uint8>((EncodedCm >> 8) & 0xff),
            static_cast<uint8>(EncodedCm & 0xff),
            255);
    }

    bool ConfigureRenderTargetForModality(
        USceneCaptureComponent2D* Capture,
        TObjectPtr<UTextureRenderTarget2D>& RenderTargetSlot,
        int32 Width,
        int32 Height,
        ESpAgentCaptureModality Modality,
        FString& OutError)
    {
        OutError.Reset();
        if (!Capture)
        {
            OutError = TEXT("capture_component_missing");
            return false;
        }

        const bool bDepth = Modality == ESpAgentCaptureModality::Depth ||
            IsMaskModality(Modality);
        const bool bLinearColor =
            Modality == ESpAgentCaptureModality::Depth ||
            Modality == ESpAgentCaptureModality::Normal ||
            IsMaskModality(Modality);
        UTextureRenderTarget2D* RenderTarget = RenderTargetSlot.Get();
        const ETextureRenderTargetFormat DesiredFormat = bDepth
            ? ETextureRenderTargetFormat::RTF_R32f
            : (Modality == ESpAgentCaptureModality::Normal
                ? ETextureRenderTargetFormat::RTF_RGBA8
                : ETextureRenderTargetFormat::RTF_RGBA8_SRGB);
        const bool bNeedRenderTarget =
            !RenderTarget ||
            RenderTarget->SizeX != Width ||
            RenderTarget->SizeY != Height ||
            RenderTarget->RenderTargetFormat != DesiredFormat;

        if (bNeedRenderTarget)
        {
            RenderTarget = NewObject<UTextureRenderTarget2D>(
                Capture,
                UTextureRenderTarget2D::StaticClass(),
                NAME_None);
            if (!RenderTarget)
            {
                OutError = TEXT("failed_to_allocate_agent_batch_render_target");
                return false;
            }
            RenderTarget->RenderTargetFormat = DesiredFormat;
            RenderTarget->ClearColor = FLinearColor::Black;
            RenderTarget->bAutoGenerateMips = false;
            if (bDepth)
            {
                RenderTarget->SRGB = false;
                RenderTarget->bForceLinearGamma = true;
                RenderTarget->InitCustomFormat(
                    Width,
                    Height,
                    PF_R32_FLOAT,
                    /*bForceLinearGamma=*/true);
            }
            else
            {
                RenderTarget->SRGB = !bLinearColor;
                RenderTarget->bForceLinearGamma = bLinearColor;
                RenderTarget->InitAutoFormat(Width, Height);
            }
            RenderTarget->UpdateResourceImmediate(true);
            RenderTargetSlot = RenderTarget;
        }

        Capture->TextureTarget = RenderTarget;
        return true;
    }

    void ConfigureSharedAgentCapture(
        USceneCaptureComponent2D* CaptureComponent,
        const USpSceneCaptureComponent2D* SourceCapture,
        float Fov)
    {
        if (!CaptureComponent || !SourceCapture)
        {
            return;
        }

        CaptureComponent->SetWorldLocationAndRotation(
            SourceCapture->GetComponentLocation(),
            SourceCapture->GetComponentRotation());
        CaptureComponent->FOVAngle = Fov;
        CaptureComponent->ProjectionType = SourceCapture->ProjectionType;
        CaptureComponent->CaptureSource = SourceCapture->CaptureSource;
        CaptureComponent->PrimitiveRenderMode =
            SourceCapture->PrimitiveRenderMode;
        CaptureComponent->PostProcessSettings =
            SourceCapture->PostProcessSettings;
        CaptureComponent->PostProcessBlendWeight =
            SourceCapture->PostProcessBlendWeight;
        CaptureComponent->ShowFlags = SourceCapture->ShowFlags;
        CaptureComponent->ShowOnlyActors.Reset();
        CaptureComponent->ShowOnlyComponents.Reset();
        CaptureComponent->HiddenActors.Reset();
        CaptureComponent->HiddenComponents.Reset();
        CaptureComponent->bCaptureEveryFrame = false;
        CaptureComponent->bCaptureOnMovement = false;
        CaptureComponent->bAlwaysPersistRenderingState = false;
        CaptureComponent->UpdateComponentToWorld();
    }

    void ConfigureStandaloneCapture(
        USceneCaptureComponent2D* CaptureComponent,
        const FVector& Location,
        const FRotator& Rotation,
        float Fov)
    {
        if (!CaptureComponent)
        {
            return;
        }

        CaptureComponent->DetachFromComponent(FDetachmentTransformRules::KeepWorldTransform);
        CaptureComponent->SetWorldLocationAndRotation(
            Location,
            Rotation,
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
        CaptureComponent->FOVAngle = Fov;
        CaptureComponent->ProjectionType = ECameraProjectionMode::Perspective;
        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
        CaptureComponent->PrimitiveRenderMode =
            ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
        CaptureComponent->PostProcessSettings = FPostProcessSettings();
        CaptureComponent->PostProcessBlendWeight = 0.0f;
        CaptureComponent->ShowFlags = FEngineShowFlags(EShowFlagInitMode::ESFIM_Game);
        CaptureComponent->ShowOnlyActors.Reset();
        CaptureComponent->ShowOnlyComponents.Reset();
        CaptureComponent->HiddenActors.Reset();
        CaptureComponent->HiddenComponents.Reset();
        CaptureComponent->bCaptureEveryFrame = false;
        CaptureComponent->bCaptureOnMovement = false;
        CaptureComponent->bAlwaysPersistRenderingState = false;
        DisableDiagnosticPrimitiveShowFlags(CaptureComponent->ShowFlags);
        CaptureComponent->UpdateComponentToWorld();
    }

    void ApplyDataCaptureShowFlags(USceneCaptureComponent2D* CaptureComponent)
    {
        if (!CaptureComponent)
        {
            return;
        }

        FEngineShowFlags& ShowFlags = CaptureComponent->ShowFlags;
        ShowFlags.SetAntiAliasing(false);
        ShowFlags.SetTemporalAA(false);
        ShowFlags.SetMotionBlur(false);
        ShowFlags.SetPostProcessing(false);
        ShowFlags.SetTonemapper(false);
        ShowFlags.SetToneCurve(false);
        ShowFlags.SetEyeAdaptation(false);
        ShowFlags.SetLocalExposure(false);
        ShowFlags.SetColorGrading(false);
        ShowFlags.SetBloom(false);
        ShowFlags.SetGrain(false);
        ShowFlags.SetScreenPercentage(false);
        DisableDiagnosticPrimitiveShowFlags(ShowFlags);
        CaptureComponent->PostProcessBlendWeight = 0.0f;
    }

    void ConfigureCaptureForModality(
        USceneCaptureComponent2D* CaptureComponent,
        const USpSceneCaptureComponent2D* SourceCapture,
        const FVector& SourceLocation,
        const FRotator& SourceRotation,
        float Fov,
        ESpAgentCaptureModality Modality)
    {
        if (SourceCapture)
        {
            ConfigureSharedAgentCapture(CaptureComponent, SourceCapture, Fov);
        }
        else
        {
            ConfigureStandaloneCapture(CaptureComponent, SourceLocation, SourceRotation, Fov);
        }
        if (!CaptureComponent)
        {
            return;
        }

        if (Modality == ESpAgentCaptureModality::RGB)
        {
            CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
            CaptureComponent->PrimitiveRenderMode =
                ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
            return;
        }

        if (Modality == ESpAgentCaptureModality::Depth)
        {
            CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_SceneDepth;
            CaptureComponent->PrimitiveRenderMode =
                ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
            ApplyDataCaptureShowFlags(CaptureComponent);
            return;
        }

        if (Modality == ESpAgentCaptureModality::Normal)
        {
            CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_Normal;
            CaptureComponent->PrimitiveRenderMode =
                ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
            ApplyDataCaptureShowFlags(CaptureComponent);
            return;
        }

        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_SceneDepth;
        CaptureComponent->PrimitiveRenderMode =
            ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
        CaptureComponent->ShowFlags.SetAtmosphere(false);
        CaptureComponent->ShowFlags.SetFog(false);
        ApplyDataCaptureShowFlags(CaptureComponent);
    }

    UClass* LoadObjectIdsProxyManagerClass(FString& OutError)
    {
        OutError.Reset();
        UClass* ManagerClass = LoadObject<UClass>(
            nullptr,
            TEXT("/Script/SpUnrealTypes.SpObjectIdsProxyComponentManager"));
        if (!ManagerClass)
        {
            OutError = TEXT("spear_object_ids_proxy_manager_class_missing");
            return nullptr;
        }
        return ManagerClass;
    }

    bool CallNoArgUFunction(UObject* Object, const TCHAR* FunctionName, FString& OutError)
    {
        OutError.Reset();
        if (!Object)
        {
            OutError = TEXT("object_missing");
            return false;
        }
        UFunction* Function = Object->FindFunction(FName(FunctionName));
        if (!Function)
        {
            OutError = FString::Printf(TEXT("function_missing:%s"), FunctionName);
            return false;
        }
        Object->ProcessEvent(Function, nullptr);
        return true;
    }

    void ApplyObjectIdShowFlags(USpSceneCaptureComponent2D* CaptureComponent)
    {
        if (!CaptureComponent)
        {
            return;
        }

        FEngineShowFlags& ShowFlags = CaptureComponent->ShowFlags;
        ShowFlags.SetAmbientOcclusion(false);
        ShowFlags.SetAntiAliasing(false);
        ShowFlags.SetAtmosphere(false);
        ShowFlags.SetBloom(false);
        ShowFlags.SetColorGrading(false);
        ShowFlags.SetDynamicShadows(false);
        ShowFlags.SetEyeAdaptation(false);
        ShowFlags.SetFog(false);
        ShowFlags.SetGrain(false);
        ShowFlags.SetLensFlares(false);
        ShowFlags.SetLocalExposure(false);
        ShowFlags.SetMotionBlur(false);
        ShowFlags.SetPostProcessing(false);
        ShowFlags.SetReflectionEnvironment(false);
        ShowFlags.SetScreenPercentage(false);
        ShowFlags.SetTemporalAA(false);
        ShowFlags.SetToneCurve(false);
        ShowFlags.SetTonemapper(false);
        DisableDiagnosticPrimitiveShowFlags(ShowFlags);
        CaptureComponent->PostProcessBlendWeight = 0.0f;
    }

    bool ConfigureObjectIdCaptureForSource(
        USpSceneCaptureComponent2D* CaptureComponent,
        const USpSceneCaptureComponent2D* SourceCapture,
        const FVector& SourceLocation,
        const FRotator& SourceRotation,
        float Fov,
        FString& OutError)
    {
        OutError.Reset();
        if (!CaptureComponent)
        {
            OutError = TEXT("object_id_capture_missing");
            return false;
        }

        CaptureComponent->SetWorldLocationAndRotation(
            SourceCapture ? SourceCapture->GetComponentLocation() : SourceLocation,
            SourceCapture ? SourceCapture->GetComponentRotation() : SourceRotation,
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
        CaptureComponent->FOVAngle = Fov;
        if (SourceCapture)
        {
            CaptureComponent->ProjectionType = SourceCapture->ProjectionType;
        }
        else
        {
            CaptureComponent->ProjectionType = ECameraProjectionMode::Perspective;
        }
        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_FinalColorHDR;
        CaptureComponent->PrimitiveRenderMode =
            ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
        CaptureComponent->bCaptureEveryFrame = false;
        CaptureComponent->bCaptureOnMovement = false;
        CaptureComponent->bAlwaysPersistRenderingState = true;
        CaptureComponent->UpdateComponentToWorld();
        ApplyObjectIdShowFlags(CaptureComponent);
        return true;
    }

    uint32 DecodeSpearObjectIdColor(const FColor& Pixel)
    {
        return (static_cast<uint32>(Pixel.R) << 16) |
               (static_cast<uint32>(Pixel.G) << 8) |
               static_cast<uint32>(Pixel.B);
    }

    void ApplyWorldCaptureTransformOverride(
        USceneCaptureComponent2D* CaptureComponent,
        bool bHasLocation,
        const FVector& Location,
        bool bHasRotation,
        const FRotator& Rotation)
    {
        if (!CaptureComponent || (!bHasLocation && !bHasRotation))
        {
            return;
        }
        const FVector EffectiveLocation =
            bHasLocation ? Location : CaptureComponent->GetComponentLocation();
        const FRotator EffectiveRotation =
            bHasRotation ? Rotation : CaptureComponent->GetComponentRotation();
        CaptureComponent->DetachFromComponent(FDetachmentTransformRules::KeepWorldTransform);
        CaptureComponent->SetWorldLocationAndRotation(
            EffectiveLocation,
            EffectiveRotation,
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
        CaptureComponent->UpdateComponentToWorld();
    }

    void AddTransformOverrideMetadata(
        TSharedPtr<FJsonObject> ImageJson,
        bool bHasLocation,
        const FVector& Location,
        bool bHasRotation,
        const FRotator& Rotation)
    {
        if (!ImageJson.IsValid() || (!bHasLocation && !bHasRotation))
        {
            return;
        }
        ImageJson->SetBoolField(TEXT("camera_transform_override"), true);
        if (bHasLocation)
        {
            ImageJson->SetArrayField(TEXT("camera_location_cm"), JsonVectorArray(Location));
        }
        if (bHasRotation)
        {
            ImageJson->SetArrayField(TEXT("camera_rotation_degrees"), JsonRotatorArray(Rotation));
            ImageJson->SetNumberField(TEXT("camera_pitch_deg"), Rotation.Pitch);
            ImageJson->SetNumberField(TEXT("camera_yaw_deg"), Rotation.Yaw);
            ImageJson->SetNumberField(TEXT("camera_roll_deg"), Rotation.Roll);
        }
    }

    void AddCameraSourceMetadata(
        TSharedPtr<FJsonObject> ImageJson,
        const FSpCameraSourceRequest& Source)
    {
        if (!ImageJson.IsValid())
        {
            return;
        }
        ImageJson->SetStringField(TEXT("camera_id"), Source.CameraId);
        ImageJson->SetStringField(
            TEXT("source_type"),
            CameraSourceTypeToString(Source.SourceType));
        if (Source.SourceType == ESpCameraSourceType::Agent)
        {
            ImageJson->SetStringField(TEXT("source_agent_tag"), Source.AgentTag);
        }
        ImageJson->SetStringField(TEXT("camera_view"), Source.CameraView);
        AddTransformOverrideMetadata(
            ImageJson,
            Source.bHasLocation,
            Source.LocationCm,
            Source.bHasRotation,
            Source.RotationDegrees);
    }

    bool ReadDepthAsRgb24Cm(
        UTextureRenderTarget2D* RenderTarget,
        float DepthFarCm,
        TArray<FColor>& OutBitmap,
        double& OutDepthMeanCm,
        double& OutDepthMinCm,
        double& OutDepthMaxCm,
        int32& OutValidDepthPixels,
        int32& OutDepthUniqueValueCount,
        FString& OutError)
    {
        OutBitmap.Reset();
        OutDepthMeanCm = 0.0;
        OutDepthMinCm = 0.0;
        OutDepthMaxCm = 0.0;
        OutValidDepthPixels = 0;
        OutDepthUniqueValueCount = 0;
        if (!RenderTarget)
        {
            OutError = TEXT("scene_capture_has_no_texture_target");
            return false;
        }
        FTextureRenderTargetResource* RTResource =
            RenderTarget->GameThread_GetRenderTargetResource();
        if (!RTResource)
        {
            OutError = TEXT("texture_target_has_no_resource");
            return false;
        }

        TArray<FLinearColor> DepthPixels;
        if (!RTResource->ReadLinearColorPixels(DepthPixels) ||
            DepthPixels.Num() != RenderTarget->SizeX * RenderTarget->SizeY)
        {
            OutError = TEXT("read_depth_pixels_failed");
            return false;
        }

        const float FarCm = FMath::Max(1.0f, DepthFarCm);
        OutBitmap.SetNumUninitialized(DepthPixels.Num());
        double SumCm = 0.0;
        double MinCm = TNumericLimits<double>::Max();
        double MaxCm = 0.0;
        TSet<uint32> UniqueDepthValues;
        for (int32 Index = 0; Index < DepthPixels.Num(); ++Index)
        {
            const float DepthCm = DepthPixels[Index].R;
            if (FMath::IsFinite(DepthCm) && DepthCm > 0.0f && DepthCm <= FarCm)
            {
                OutBitmap[Index] = EncodeDepthCmToRgb24(DepthCm);
                UniqueDepthValues.Add(DecodeDepthCmFromRgb24(OutBitmap[Index]));
                SumCm += static_cast<double>(DepthCm);
                MinCm = FMath::Min(MinCm, static_cast<double>(DepthCm));
                MaxCm = FMath::Max(MaxCm, static_cast<double>(DepthCm));
                ++OutValidDepthPixels;
            }
            else
            {
                OutBitmap[Index] = FColor(0, 0, 0, 255);
            }
        }

        if (OutValidDepthPixels > 0)
        {
            OutDepthMeanCm = SumCm / static_cast<double>(OutValidDepthPixels);
            OutDepthMinCm = MinCm;
            OutDepthMaxCm = MaxCm;
        }
        OutDepthUniqueValueCount = UniqueDepthValues.Num();
        return true;
    }

    TSharedPtr<FJsonObject> CaptureOneAgentModalityJson(
        const FString& TagText,
        ASpHumanoidAgent* Agent,
        const TArray<TWeakObjectPtr<AActor>>& MaskActors,
        USceneCaptureComponent2D* CaptureComponent,
        UTextureRenderTarget2D* RenderTarget,
        ESpAgentCaptureModality Modality,
        bool bForceCapture,
        bool bValidate,
        int32 JpegQuality,
        float DepthFarCm,
        double& CaptureReadMs,
        double& EncodeMs,
        FString& OutError)
    {
        OutError.Reset();
        if (!CaptureComponent)
        {
            OutError = TEXT("capture_component_missing");
            return nullptr;
        }

        const double CaptureStart = FPlatformTime::Seconds();
        if (bForceCapture && !IsMaskModality(Modality))
        {
            CaptureComponent->CaptureScene();
        }
        if (!RenderTarget)
        {
            CaptureReadMs += (FPlatformTime::Seconds() - CaptureStart) * 1000.0;
            OutError = TEXT("scene_capture_has_no_texture_target");
            return nullptr;
        }
        FTextureRenderTargetResource* RTResource =
            RenderTarget->GameThread_GetRenderTargetResource();
        if (!RTResource)
        {
            CaptureReadMs += (FPlatformTime::Seconds() - CaptureStart) * 1000.0;
            OutError = TEXT("texture_target_has_no_resource");
            return nullptr;
        }

        TArray<FColor> Bitmap;
        double DepthMeanCm = 0.0;
        double DepthMinCm = 0.0;
        double DepthMaxCm = 0.0;
        int32 ValidDepthPixels = 0;
        int32 DepthUniqueValueCount = 0;
        int32 MaskPixels = 0;
        int32 MaskRenderableComponentCount = 0;
        int32 MaskSkippedPrimitiveCount = 0;
        TArray<FString> MaskTargetNames;
        TArray<TSharedPtr<FJsonValue>> MaskTargetColorValues;
        TSet<uint32> MaskUniqueColors;
        bool bAllowEmptySemanticMask = false;

        if (Modality == ESpAgentCaptureModality::Depth)
        {
            FString DepthError;
            if (!ReadDepthAsRgb24Cm(
                    RenderTarget,
                    DepthFarCm,
                    Bitmap,
                    DepthMeanCm,
                    DepthMinCm,
                    DepthMaxCm,
                    ValidDepthPixels,
                    DepthUniqueValueCount,
                    DepthError))
            {
                CaptureReadMs += (FPlatformTime::Seconds() - CaptureStart) * 1000.0;
                OutError = DepthError;
                return nullptr;
            }
        }
        else if (IsMaskModality(Modality))
        {
            if (!RTResource)
            {
                CaptureReadMs += (FPlatformTime::Seconds() - CaptureStart) * 1000.0;
                OutError = TEXT("texture_target_has_no_resource");
                return nullptr;
            }

            TArray<TWeakObjectPtr<AActor>> EffectiveMaskActors = MaskActors;
            const bool bHasExplicitMaskActors = !MaskActors.IsEmpty();
            if (Modality == ESpAgentCaptureModality::SemanticSeg && !bHasExplicitMaskActors)
            {
                bAllowEmptySemanticMask = true;
            }
            if (EffectiveMaskActors.IsEmpty() && Agent)
            {
                EffectiveMaskActors.Add(Agent);
            }

            Bitmap.Init(FColor(0, 0, 0, 255), RenderTarget->SizeX * RenderTarget->SizeY);
            if (EffectiveMaskActors.IsEmpty())
            {
                if (Modality == ESpAgentCaptureModality::SemanticSeg)
                {
                    bAllowEmptySemanticMask = true;
                }
                else
                {
                    CaptureReadMs += (FPlatformTime::Seconds() - CaptureStart) * 1000.0;
                    OutError = TEXT("segmentation_requested_without_mask_actors");
                    return nullptr;
                }
            }

            for (const TWeakObjectPtr<AActor>& WeakActor : EffectiveMaskActors)
            {
                AActor* TargetActor = WeakActor.Get();
                if (!IsValid(TargetActor))
                {
                    continue;
                }
                TargetActor->SetActorHiddenInGame(false);
                TInlineComponentArray<UPrimitiveComponent*> PrimitiveComponents(TargetActor);
                TArray<UPrimitiveComponent*> SegmentationComponents;
                for (UPrimitiveComponent* PrimitiveComponent : PrimitiveComponents)
                {
                    if (IsSegmentationMeshPrimitive(PrimitiveComponent))
                    {
                        SegmentationComponents.Add(PrimitiveComponent);
                        PrimitiveComponent->SetVisibility(true, true);
                        PrimitiveComponent->SetHiddenInGame(false, true);
                    }
                    else if (PrimitiveComponent)
                    {
                        ++MaskSkippedPrimitiveCount;
                    }
                }
                if (SegmentationComponents.IsEmpty())
                {
                    continue;
                }

                CaptureComponent->ShowOnlyActors.Reset();
                CaptureComponent->ShowOnlyComponents.Reset();
                for (UPrimitiveComponent* PrimitiveComponent : SegmentationComponents)
                {
                    CaptureComponent->ShowOnlyComponents.Add(PrimitiveComponent);
                    ++MaskRenderableComponentCount;
                }
                CaptureComponent->CaptureScene();

                TArray<FLinearColor> ActorDepthPixels;
                if (!RTResource->ReadLinearColorPixels(ActorDepthPixels) ||
                    ActorDepthPixels.Num() != Bitmap.Num())
                {
                    CaptureReadMs += (FPlatformTime::Seconds() - CaptureStart) * 1000.0;
                    OutError = TEXT("read_segmentation_depth_pixels_failed");
                    return nullptr;
                }

                const FString TargetKey = !TargetActor->Tags.IsEmpty()
                    ? TargetActor->Tags[0].ToString()
                    : TargetActor->GetName();
                const FColor MaskColor = MakeDeterministicMaskColor(
                    TargetKey,
                    TargetActor,
                    Modality);
                int32 ActorMaskPixels = 0;
                for (int32 PixelIndex = 0; PixelIndex < ActorDepthPixels.Num(); ++PixelIndex)
                {
                    const float DepthCm = ActorDepthPixels[PixelIndex].R;
                    if (FMath::IsFinite(DepthCm) &&
                        DepthCm > 0.0f &&
                        DepthCm <= DepthFarCm)
                    {
                        Bitmap[PixelIndex] = MaskColor;
                        ++ActorMaskPixels;
                    }
                }

                if (ActorMaskPixels > 0)
                {
                    MaskPixels += ActorMaskPixels;
                    MaskTargetNames.AddUnique(TargetKey);
                    const uint32 ColorKey =
                        (static_cast<uint32>(MaskColor.R) << 16) |
                        (static_cast<uint32>(MaskColor.G) << 8) |
                        static_cast<uint32>(MaskColor.B);
                    MaskUniqueColors.Add(ColorKey);

                    TSharedPtr<FJsonObject> ColorObject = MakeShared<FJsonObject>();
                    ColorObject->SetStringField(TEXT("target"), TargetKey);
                    ColorObject->SetStringField(
                        TEXT("class_name"),
                        TargetActor->GetClass()->GetPathName());
                    ColorObject->SetNumberField(TEXT("r"), MaskColor.R);
                    ColorObject->SetNumberField(TEXT("g"), MaskColor.G);
                    ColorObject->SetNumberField(TEXT("b"), MaskColor.B);
                    ColorObject->SetNumberField(TEXT("pixels"), ActorMaskPixels);
                    ColorObject->SetStringField(
                        TEXT("hex"),
                        FString::Printf(
                            TEXT("#%02X%02X%02X"),
                            MaskColor.R,
                            MaskColor.G,
                            MaskColor.B));
                    MaskTargetColorValues.Add(MakeShared<FJsonValueObject>(ColorObject));
                }
            }

            CaptureComponent->ShowOnlyActors.Reset();
            CaptureComponent->ShowOnlyComponents.Reset();
        }
        else
        {
            FReadSurfaceDataFlags Flags(RCM_UNorm, CubeFace_MAX);
            Flags.SetLinearToGamma(false);
            if (!RTResource->ReadPixels(Bitmap, Flags) || Bitmap.Num() <= 0)
            {
                CaptureReadMs += (FPlatformTime::Seconds() - CaptureStart) * 1000.0;
                OutError = TEXT("read_pixels_failed");
                return nullptr;
            }
            if (Modality == ESpAgentCaptureModality::Normal)
            {
                for (FColor& Pixel : Bitmap)
                {
                    Pixel.A = 255;
                }
            }
        }
        CaptureReadMs += (FPlatformTime::Seconds() - CaptureStart) * 1000.0;

        double Mean = 0.0;
        double Std = 0.0;
        int32 MinValue = 0;
        int32 MaxValue = 0;
        ComputeBitmapStats(Bitmap, Mean, Std, MinValue, MaxValue);
        if (bValidate)
        {
            if (Modality == ESpAgentCaptureModality::RGB && (MaxValue <= 0 || Std < 0.25))
            {
                OutError = FString::Printf(
                    TEXT("invalid_camera_frame max=%d std=%.4f"),
                    MaxValue,
                    Std);
                return nullptr;
            }
            if (Modality == ESpAgentCaptureModality::Depth && ValidDepthPixels <= 0)
            {
                OutError = TEXT("invalid_depth_frame no_valid_depth_pixels");
                return nullptr;
            }
            if (IsMaskModality(Modality) && MaskPixels <= 0 && !bAllowEmptySemanticMask)
            {
                OutError = TEXT("invalid_segmentation_frame no_mask_pixels");
                return nullptr;
            }
        }

        FString DataUrl;
        int32 EncodedBytes = 0;
        FString EncodeError;
        const double EncodeStart = FPlatformTime::Seconds();
        const bool bRgb = Modality == ESpAgentCaptureModality::RGB;
        const bool bEncodeOk = EncodeImageBase64(
            Bitmap,
            RenderTarget->SizeX,
            RenderTarget->SizeY,
            bRgb ? EImageFormat::JPEG : EImageFormat::PNG,
            bRgb ? TEXT("image/jpeg") : TEXT("image/png"),
            bRgb ? JpegQuality : 100,
            DataUrl,
            EncodedBytes,
            EncodeError);
        EncodeMs += (FPlatformTime::Seconds() - EncodeStart) * 1000.0;
        if (!bEncodeOk)
        {
            OutError = EncodeError;
            return nullptr;
        }

        const FVector Loc = Agent ? Agent->GetActorLocation() : FVector::ZeroVector;
        const FRotator Rot = Agent ? Agent->GetActorRotation() : FRotator::ZeroRotator;
        TSharedPtr<FJsonObject> ImageJson = MakeShared<FJsonObject>();
        ImageJson->SetStringField(TEXT("agent_tag"), TagText);
        ImageJson->SetBoolField(TEXT("success"), true);
        ImageJson->SetStringField(TEXT("modality"), ModalityToString(Modality));
        ImageJson->SetNumberField(TEXT("width"), RenderTarget->SizeX);
        ImageJson->SetNumberField(TEXT("height"), RenderTarget->SizeY);
        ImageJson->SetNumberField(TEXT("bytes"), EncodedBytes);
        if (bRgb)
        {
            ImageJson->SetNumberField(TEXT("jpeg_bytes"), EncodedBytes);
        }
        else
        {
            ImageJson->SetNumberField(TEXT("png_bytes"), EncodedBytes);
        }
        ImageJson->SetStringField(TEXT("encoding"), bRgb ? TEXT("jpeg") : TEXT("png"));
        ImageJson->SetStringField(TEXT("mime_type"), bRgb ? TEXT("image/jpeg") : TEXT("image/png"));
        ImageJson->SetNumberField(TEXT("mean"), Mean);
        ImageJson->SetNumberField(TEXT("std"), Std);
        ImageJson->SetNumberField(TEXT("min"), MinValue);
        ImageJson->SetNumberField(TEXT("max"), MaxValue);
        ImageJson->SetStringField(TEXT("data_url"), DataUrl);

        if (Modality == ESpAgentCaptureModality::Depth)
        {
            ImageJson->SetNumberField(TEXT("valid_depth_pixels"), ValidDepthPixels);
            ImageJson->SetNumberField(TEXT("depth_mean_cm"), DepthMeanCm);
            ImageJson->SetNumberField(TEXT("depth_min_cm"), DepthMinCm);
            ImageJson->SetNumberField(TEXT("depth_max_cm"), DepthMaxCm);
            ImageJson->SetNumberField(TEXT("depth_far_cm"), DepthFarCm);
            ImageJson->SetStringField(TEXT("depth_encoding"), TEXT("depth_cm_uint24_rgb"));
            ImageJson->SetStringField(TEXT("depth_unit"), TEXT("cm"));
            ImageJson->SetNumberField(TEXT("depth_precision_cm"), 1.0);
            ImageJson->SetNumberField(TEXT("depth_unique_value_count"), DepthUniqueValueCount);
        }
        if (Modality == ESpAgentCaptureModality::Normal)
        {
            ImageJson->SetStringField(TEXT("normal_encoding"), TEXT("ue_scs_normal_rgb"));
        }
        if (IsMaskModality(Modality))
        {
            ImageJson->SetStringField(TEXT("segmentation_scope"), TEXT("mask_actor_tags"));
            ImageJson->SetNumberField(TEXT("mask_pixels"), MaskPixels);
            ImageJson->SetNumberField(TEXT("mask_actor_count"), MaskTargetNames.Num());
            ImageJson->SetNumberField(TEXT("mask_unique_color_count"), MaskUniqueColors.Num());
            ImageJson->SetBoolField(
                TEXT("semantic_empty_no_mask_actor_tags"),
                bAllowEmptySemanticMask);
            ImageJson->SetNumberField(
                TEXT("mask_renderable_component_count"),
                MaskRenderableComponentCount);
            ImageJson->SetNumberField(
                TEXT("mask_skipped_non_mesh_primitive_count"),
                MaskSkippedPrimitiveCount);
            TArray<TSharedPtr<FJsonValue>> MaskTargetValues;
            for (const FString& MaskTargetName : MaskTargetNames)
            {
                MaskTargetValues.Add(MakeShared<FJsonValueString>(MaskTargetName));
            }
            ImageJson->SetArrayField(TEXT("mask_targets"), MaskTargetValues);
            ImageJson->SetArrayField(TEXT("mask_target_colors"), MaskTargetColorValues);
        }

        TArray<TSharedPtr<FJsonValue>> LocValues;
        LocValues.Add(MakeShared<FJsonValueNumber>(Loc.X));
        LocValues.Add(MakeShared<FJsonValueNumber>(Loc.Y));
        LocValues.Add(MakeShared<FJsonValueNumber>(Loc.Z));
        ImageJson->SetArrayField(TEXT("loc_cm"), LocValues);
        ImageJson->SetNumberField(TEXT("yaw_deg"), Rot.Yaw);
        return ImageJson;
    }

    TSharedPtr<FJsonObject> CaptureSceneObjectIdsJson(
        const FString& TagText,
        ASpHumanoidAgent* Agent,
        USpSceneCaptureComponent2D* CaptureComponent,
        ASpMeshProxyComponentManager* ObjectIdsProxyManager,
        ESpAgentCaptureModality Modality,
        bool bForceCapture,
        bool bValidate,
        int32 JpegQuality,
        double& CaptureReadMs,
        double& EncodeMs,
        FString& OutError)
    {
        OutError.Reset();
        if (Modality != ESpAgentCaptureModality::InstanceSeg)
        {
            OutError = TEXT("scene_object_ids_supports_instance_seg_only");
            return nullptr;
        }
        if (!CaptureComponent || !CaptureComponent->TextureTarget)
        {
            OutError = TEXT("object_id_capture_has_no_texture_target");
            return nullptr;
        }

        const double CaptureStart = FPlatformTime::Seconds();
        if (bForceCapture)
        {
            CaptureComponent->CaptureScene();
        }
        UTextureRenderTarget2D* RenderTarget = CaptureComponent->TextureTarget;
        FTextureRenderTargetResource* RTResource =
            RenderTarget->GameThread_GetRenderTargetResource();
        if (!RTResource)
        {
            CaptureReadMs += (FPlatformTime::Seconds() - CaptureStart) * 1000.0;
            OutError = TEXT("object_id_texture_target_has_no_resource");
            return nullptr;
        }

        TArray<FColor> Bitmap;
        FReadSurfaceDataFlags Flags(RCM_UNorm, CubeFace_MAX);
        Flags.SetLinearToGamma(false);
        if (!RTResource->ReadPixels(Bitmap, Flags) || Bitmap.Num() <= 0)
        {
            CaptureReadMs += (FPlatformTime::Seconds() - CaptureStart) * 1000.0;
            OutError = TEXT("read_object_id_pixels_failed");
            return nullptr;
        }
        CaptureReadMs += (FPlatformTime::Seconds() - CaptureStart) * 1000.0;

        TSet<uint32> ValidObjectIds;
        ValidObjectIds.Add(0);
        if (ObjectIdsProxyManager)
        {
            for (const FMeshProxyGeometryDesc& Desc :
                 ObjectIdsProxyManager->GetMeshProxyGeometryDescs(false))
            {
                if (Desc.RawId >= 0 && Desc.RawId <= 0xffffff)
                {
                    ValidObjectIds.Add(static_cast<uint32>(Desc.RawId));
                }
            }
        }
        const bool bCanCleanInvalidObjectIds = ValidObjectIds.Num() > 1;

        int32 InvalidObjectIdPixelsCleaned = 0;
        TSet<uint32> InvalidObjectIdsCleaned;
        int32 ObjectIdPixels = 0;
        TSet<uint32> VisibleObjectIds;
        TArray<uint32> ObjectIds;
        ObjectIds.SetNumUninitialized(Bitmap.Num());
        for (int32 Index = 0; Index < Bitmap.Num(); ++Index)
        {
            FColor& Pixel = Bitmap[Index];
            Pixel.A = 255;
            uint32 ObjectId = DecodeSpearObjectIdColor(Pixel);
            if (bCanCleanInvalidObjectIds && !ValidObjectIds.Contains(ObjectId))
            {
                InvalidObjectIdsCleaned.Add(ObjectId);
                ++InvalidObjectIdPixelsCleaned;
                Pixel = FColor(0, 0, 0, 255);
                ObjectId = 0;
            }
            ObjectIds[Index] = ObjectId;
            if (ObjectId != 0)
            {
                ++ObjectIdPixels;
                VisibleObjectIds.Add(ObjectId);
            }
        }

        int64 NeighborPairs = 0;
        int64 NeighborSamePairs = 0;
        int32 NonSingletonPixels = 0;
        const int32 Width = RenderTarget->SizeX;
        const int32 Height = RenderTarget->SizeY;
        for (int32 Y = 0; Y < Height; ++Y)
        {
            for (int32 X = 0; X < Width; ++X)
            {
                const int32 Index = Y * Width + X;
                const uint32 Id = ObjectIds[Index];
                if (Id == 0)
                {
                    continue;
                }
                int32 SameLocalNeighbors = 0;
                if (X + 1 < Width)
                {
                    const uint32 Right = ObjectIds[Index + 1];
                    if (Right != 0)
                    {
                        ++NeighborPairs;
                        if (Right == Id)
                        {
                            ++NeighborSamePairs;
                            ++SameLocalNeighbors;
                        }
                    }
                }
                if (Y + 1 < Height)
                {
                    const uint32 Down = ObjectIds[Index + Width];
                    if (Down != 0)
                    {
                        ++NeighborPairs;
                        if (Down == Id)
                        {
                            ++NeighborSamePairs;
                            ++SameLocalNeighbors;
                        }
                    }
                }
                if ((X > 0 && ObjectIds[Index - 1] == Id) ||
                    (Y > 0 && ObjectIds[Index - Width] == Id) ||
                    SameLocalNeighbors > 0)
                {
                    ++NonSingletonPixels;
                }
            }
        }
        const double NeighborSameRatio = NeighborPairs > 0
            ? static_cast<double>(NeighborSamePairs) / static_cast<double>(NeighborPairs)
            : 1.0;
        const double NonSingletonRatio = ObjectIdPixels > 0
            ? static_cast<double>(NonSingletonPixels) / static_cast<double>(ObjectIdPixels)
            : 0.0;

        double Mean = 0.0;
        double Std = 0.0;
        int32 MinValue = 0;
        int32 MaxValue = 0;
        ComputeBitmapStats(Bitmap, Mean, Std, MinValue, MaxValue);
        if (bValidate && ObjectIdPixels <= 0)
        {
            OutError = TEXT("invalid_scene_object_id_frame no_object_id_pixels");
            return nullptr;
        }

        FString DataUrl;
        int32 EncodedBytes = 0;
        FString EncodeError;
        const double EncodeStart = FPlatformTime::Seconds();
        const bool bEncodeOk = EncodeImageBase64(
            Bitmap,
            RenderTarget->SizeX,
            RenderTarget->SizeY,
            EImageFormat::PNG,
            TEXT("image/png"),
            JpegQuality,
            DataUrl,
            EncodedBytes,
            EncodeError);
        EncodeMs += (FPlatformTime::Seconds() - EncodeStart) * 1000.0;
        if (!bEncodeOk)
        {
            OutError = EncodeError;
            return nullptr;
        }

        const FVector Loc = Agent ? Agent->GetActorLocation() : FVector::ZeroVector;
        const FRotator Rot = Agent ? Agent->GetActorRotation() : FRotator::ZeroRotator;
        TSharedPtr<FJsonObject> ImageJson = MakeShared<FJsonObject>();
        ImageJson->SetStringField(TEXT("agent_tag"), TagText);
        ImageJson->SetBoolField(TEXT("success"), true);
        ImageJson->SetStringField(TEXT("modality"), ModalityToString(Modality));
        ImageJson->SetStringField(TEXT("segmentation_scope"), TEXT("scene_object_ids"));
        ImageJson->SetStringField(TEXT("object_id_encoding"), TEXT("spear_rgb24_raw_id"));
        ImageJson->SetNumberField(TEXT("background_object_id"), 0);
        ImageJson->SetNumberField(TEXT("width"), RenderTarget->SizeX);
        ImageJson->SetNumberField(TEXT("height"), RenderTarget->SizeY);
        ImageJson->SetNumberField(TEXT("bytes"), EncodedBytes);
        ImageJson->SetNumberField(TEXT("png_bytes"), EncodedBytes);
        ImageJson->SetStringField(TEXT("encoding"), TEXT("png"));
        ImageJson->SetStringField(TEXT("mime_type"), TEXT("image/png"));
        ImageJson->SetNumberField(TEXT("mean"), Mean);
        ImageJson->SetNumberField(TEXT("std"), Std);
        ImageJson->SetNumberField(TEXT("min"), MinValue);
        ImageJson->SetNumberField(TEXT("max"), MaxValue);
        ImageJson->SetStringField(TEXT("data_url"), DataUrl);
        ImageJson->SetNumberField(TEXT("mask_pixels"), ObjectIdPixels);
        ImageJson->SetNumberField(TEXT("object_id_pixels"), ObjectIdPixels);
        ImageJson->SetNumberField(TEXT("visible_object_id_count"), VisibleObjectIds.Num());
        ImageJson->SetNumberField(TEXT("mask_actor_count"), VisibleObjectIds.Num());
        ImageJson->SetNumberField(TEXT("object_id_valid_raw_id_count"), ValidObjectIds.Num());
        ImageJson->SetNumberField(
            TEXT("invalid_object_id_pixels_cleaned"),
            InvalidObjectIdPixelsCleaned);
        ImageJson->SetNumberField(
            TEXT("invalid_object_id_count_cleaned"),
            InvalidObjectIdsCleaned.Num());
        ImageJson->SetNumberField(TEXT("object_id_neighbor_pairs"), static_cast<double>(NeighborPairs));
        ImageJson->SetNumberField(TEXT("object_id_neighbor_same_pairs"), static_cast<double>(NeighborSamePairs));
        ImageJson->SetNumberField(TEXT("object_id_neighbor_same_ratio"), NeighborSameRatio);
        ImageJson->SetNumberField(TEXT("object_id_non_singleton_pixel_ratio"), NonSingletonRatio);

        TArray<uint32> SortedIds = VisibleObjectIds.Array();
        SortedIds.Sort();
        TArray<TSharedPtr<FJsonValue>> VisibleIdValues;
        TArray<TSharedPtr<FJsonValue>> MaskTargetValues;
        const int32 MaxIdsInResponse = FMath::Min(64, SortedIds.Num());
        for (int32 Index = 0; Index < MaxIdsInResponse; ++Index)
        {
            const uint32 ObjectId = SortedIds[Index];
            VisibleIdValues.Add(MakeShared<FJsonValueNumber>(ObjectId));
            MaskTargetValues.Add(MakeShared<FJsonValueString>(
                FString::Printf(TEXT("object_id:%u"), ObjectId)));
        }
        ImageJson->SetArrayField(TEXT("visible_object_ids_sample"), VisibleIdValues);
        ImageJson->SetArrayField(TEXT("mask_targets"), MaskTargetValues);
        ImageJson->SetNumberField(
            TEXT("visible_object_ids_sample_count"),
            MaxIdsInResponse);

        TArray<uint32> SortedInvalidIds = InvalidObjectIdsCleaned.Array();
        SortedInvalidIds.Sort();
        TArray<TSharedPtr<FJsonValue>> InvalidIdValues;
        const int32 MaxInvalidIdsInResponse = FMath::Min(64, SortedInvalidIds.Num());
        for (int32 Index = 0; Index < MaxInvalidIdsInResponse; ++Index)
        {
            InvalidIdValues.Add(MakeShared<FJsonValueNumber>(SortedInvalidIds[Index]));
        }
        ImageJson->SetArrayField(TEXT("invalid_object_ids_cleaned_sample"), InvalidIdValues);
        ImageJson->SetNumberField(
            TEXT("invalid_object_ids_cleaned_sample_count"),
            MaxInvalidIdsInResponse);

        TArray<TSharedPtr<FJsonValue>> LocValues;
        LocValues.Add(MakeShared<FJsonValueNumber>(Loc.X));
        LocValues.Add(MakeShared<FJsonValueNumber>(Loc.Y));
        LocValues.Add(MakeShared<FJsonValueNumber>(Loc.Z));
        ImageJson->SetArrayField(TEXT("loc_cm"), LocValues);
        ImageJson->SetNumberField(TEXT("yaw_deg"), Rot.Yaw);
        return ImageJson;
    }
}

void USpCameraCapturePool::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);
    UE_LOG(LogTemp, Log, TEXT("USpCameraCapturePool::Initialize — pool subsystem online "
                              "(call PreallocatePool to reserve components)."));
}

void USpCameraCapturePool::Deinitialize()
{
    PendingRequests.Reset();
    ResultsBySequence.Reset();
    Pool.Reset();
    AgentBatchCapture = nullptr;
    AgentBatchRenderTarget = nullptr;
    AgentBatchCaptures.Reset();
    AgentBatchRenderTargets.Reset();
    for (TObjectPtr<USpSceneCaptureComponent2D>& Capture : AgentBatchObjectIdCaptures)
    {
        if (Capture.Get())
        {
            if (Capture->IsInitialized())
            {
                Capture->Terminate();
            }
            Capture->DestroyComponent();
        }
    }
    AgentBatchObjectIdCaptures.Reset();
    if (ObjectIdsProxyManager.Get())
    {
        FString TerminateError;
        CallNoArgUFunction(ObjectIdsProxyManager.Get(), TEXT("Terminate"), TerminateError);
        ObjectIdsProxyManager->Destroy();
        ObjectIdsProxyManager = nullptr;
    }
    if (HolderActor.Get())
    {
        HolderActor->Destroy();
        HolderActor = nullptr;
    }
    UE_LOG(LogTemp, Log, TEXT("USpCameraCapturePool::Deinitialize — pool released."));
    Super::Deinitialize();
}

void USpCameraCapturePool::PreallocatePool(int32 PoolSize)
{
    if (Pool.Num() == PoolSize)
    {
        return;
    }
    if (PoolSize <= 0)
    {
        UE_LOG(LogTemp, Warning,
               TEXT("USpCameraCapturePool::PreallocatePool: invalid size %d, ignoring"),
               PoolSize);
        return;
    }

    UWorld* World = GetWorld();
    if (!World)
    {
        UE_LOG(LogTemp, Error, TEXT("USpCameraCapturePool::PreallocatePool: no world"));
        return;
    }

    // Tear down any prior allocation.
    Pool.Reset();
    if (HolderActor.Get())
    {
        HolderActor->Destroy();
        HolderActor = nullptr;
    }

    // Spawn a hidden holder actor at origin to parent the components to.
    HolderActor = SpawnHiddenCaptureHolder(World);
    if (!HolderActor.Get())
    {
        UE_LOG(LogTemp, Error,
               TEXT("USpCameraCapturePool::PreallocatePool: failed to spawn holder actor"));
        return;
    }
#if WITH_EDITOR
    HolderActor->SetActorLabel(TEXT("SpCameraCapturePoolHolder"));
#endif

    Pool.Reserve(PoolSize);
    for (int32 i = 0; i < PoolSize; ++i)
    {
        USceneCaptureComponent2D* Capture = NewObject<USceneCaptureComponent2D>(
            HolderActor.Get(), USceneCaptureComponent2D::StaticClass(),
            *FString::Printf(TEXT("Capture_%03d"), i));
        if (!Capture)
        {
            UE_LOG(LogTemp, Error,
                   TEXT("USpCameraCapturePool: failed to NewObject capture %d"), i);
            continue;
        }
        Capture->SetupAttachment(HolderActor->GetRootComponent());
        Capture->RegisterComponent();

        // Manual capture only — we drive it from Tick, no per-frame
        // auto-capture overhead.
        Capture->bCaptureEveryFrame = false;
        Capture->bCaptureOnMovement = false;
        Capture->bAlwaysPersistRenderingState = true;
        Capture->CaptureSource = SCS_FinalColorLDR;

        Pool.Add(Capture);
    }
    UE_LOG(LogTemp, Log, TEXT("USpCameraCapturePool::PreallocatePool: reserved %d "
                              "USceneCaptureComponent2D under %s."),
           Pool.Num(),
           HolderActor.Get() ? *HolderActor->GetName() : TEXT("<none>"));
}

int64 USpCameraCapturePool::EnqueueRequest(const FSpCaptureRequest& Request)
{
    FSpCaptureRequest Mutable = Request;
    Mutable.SequenceNumber = NextSequenceNumber++;
    PendingRequests.Add(Mutable);
    return Mutable.SequenceNumber;
}

bool USpCameraCapturePool::TryGetCaptureResult(int64 SequenceNumber, FSpCaptureResult& OutResult)
{
    if (FSpCaptureResult* Found = ResultsBySequence.Find(SequenceNumber))
    {
        OutResult = *Found;
        // Auto-evict on first read.  Caller is expected to handle the
        // result on their side; storing further would just leak.
        ResultsBySequence.Remove(SequenceNumber);
        return true;
    }
    return false;
}

USceneCaptureComponent2D* USpCameraCapturePool::EnsureAgentBatchCapture(
    int32 Width,
    int32 Height,
    FString& OutError)
{
    if (!EnsureAgentBatchCapturePool(Width, Height, 1, OutError))
    {
        return nullptr;
    }
    return AgentBatchCaptures.Num() > 0 ? AgentBatchCaptures[0].Get() : nullptr;
}

bool USpCameraCapturePool::EnsureAgentBatchCapturePool(
    int32 Width,
    int32 Height,
    int32 PoolSize,
    FString& OutError)
{
    OutError.Reset();
    UWorld* World = GetWorld();
    if (!World)
    {
        OutError = TEXT("no_world");
        return false;
    }

    if (!HolderActor.Get())
    {
        HolderActor = SpawnHiddenCaptureHolder(World);
        if (!HolderActor.Get())
        {
            OutError = TEXT("failed_to_spawn_camera_pool_holder");
            return false;
        }
#if WITH_EDITOR
        HolderActor->SetActorLabel(TEXT("SpCameraCapturePoolHolder"));
#endif
    }

    PoolSize = FMath::Clamp(PoolSize, 1, 128);

    while (AgentBatchCaptures.Num() > PoolSize)
    {
        TObjectPtr<USceneCaptureComponent2D> Capture = AgentBatchCaptures.Pop();
        if (Capture.Get())
        {
            Capture->DestroyComponent();
        }
    }

    while (AgentBatchRenderTargets.Num() > PoolSize)
    {
        AgentBatchRenderTargets.Pop();
    }

    while (AgentBatchCaptures.Num() < PoolSize)
    {
        const int32 Index = AgentBatchCaptures.Num();
        USceneCaptureComponent2D* Capture = NewObject<USceneCaptureComponent2D>(
            HolderActor.Get(),
            USceneCaptureComponent2D::StaticClass(),
            NAME_None);
        if (!Capture)
        {
            OutError = TEXT("failed_to_allocate_agent_batch_capture");
            return false;
        }
        Capture->CreationMethod = EComponentCreationMethod::Instance;
        Capture->bCaptureEveryFrame = false;
        Capture->bCaptureOnMovement = false;
        Capture->bAlwaysPersistRenderingState = false;
        Capture->CaptureSource = SCS_FinalColorLDR;
        Capture->PrimitiveRenderMode =
            ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
        Capture->SetupAttachment(HolderActor->GetRootComponent());
        Capture->RegisterComponentWithWorld(World);
        AgentBatchCaptures.Add(Capture);
    }

    AgentBatchRenderTargets.SetNum(PoolSize);
    for (int32 Index = 0; Index < PoolSize; ++Index)
    {
        USceneCaptureComponent2D* Capture = AgentBatchCaptures[Index].Get();
        if (!Capture)
        {
            OutError = TEXT("agent_batch_capture_missing");
            return false;
        }

        FString RenderTargetError;
        if (!ConfigureRenderTargetForModality(
                Capture,
                AgentBatchRenderTargets[Index],
                Width,
                Height,
                ESpAgentCaptureModality::RGB,
                RenderTargetError))
        {
            OutError = RenderTargetError;
            return false;
        }
    }

    AgentBatchCapture = nullptr;
    AgentBatchRenderTarget = nullptr;
    if (AgentBatchCaptures.Num() > 0)
    {
        AgentBatchCapture = AgentBatchCaptures[0].Get();
    }
    if (AgentBatchRenderTargets.Num() > 0)
    {
        AgentBatchRenderTarget = AgentBatchRenderTargets[0].Get();
    }
    return AgentBatchCapture.Get() && AgentBatchRenderTarget.Get();
}

bool USpCameraCapturePool::EnsureObjectIdsProxyManager(FString& OutError)
{
    OutError.Reset();
    UWorld* World = GetWorld();
    if (!World)
    {
        OutError = TEXT("no_world");
        return false;
    }

    FString ClassError;
    UClass* ManagerClass = LoadObjectIdsProxyManagerClass(ClassError);
    if (!ManagerClass)
    {
        OutError = ClassError;
        return false;
    }

    if (!ObjectIdsProxyManager.Get())
    {
        for (TActorIterator<AActor> It(World, ManagerClass); It; ++It)
        {
            AActor* Existing = *It;
            if (IsValid(Existing))
            {
                ObjectIdsProxyManager = Existing;
                UnrealUtils::setStableName(Existing, kSpearObjectIdsProxyManagerStableName);
                break;
            }
        }
    }

    if (!ObjectIdsProxyManager.Get())
    {
        FActorSpawnParameters Params;
        Params.ObjectFlags |= RF_Transient;
        Params.SpawnCollisionHandlingOverride =
            ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AActor* Manager = World->SpawnActor<AActor>(
            ManagerClass,
            FTransform::Identity,
            Params);
        if (!Manager)
        {
            OutError = TEXT("spawn_spear_object_ids_proxy_manager_failed");
            return false;
        }
        Manager->SetActorEnableCollision(false);
#if WITH_EDITOR
        Manager->SetActorLabel(TEXT("SimWorldSpearObjectIdsProxyManager"));
#endif
        UnrealUtils::setStableName(Manager, kSpearObjectIdsProxyManagerStableName);
        ObjectIdsProxyManager = Manager;
    }

    FString InitializeError;
    if (!CallNoArgUFunction(ObjectIdsProxyManager.Get(), TEXT("Initialize"), InitializeError))
    {
        OutError = TEXT("initialize_spear_object_ids_proxy_manager_failed: ") +
            InitializeError;
        return false;
    }
    return true;
}

bool USpCameraCapturePool::EnsureAgentBatchObjectIdCapturePool(
    int32 Width,
    int32 Height,
    int32 PoolSize,
    FString& OutError)
{
    OutError.Reset();
    UWorld* World = GetWorld();
    if (!World)
    {
        OutError = TEXT("no_world");
        return false;
    }

    FString ManagerError;
    if (!EnsureObjectIdsProxyManager(ManagerError))
    {
        OutError = ManagerError;
        return false;
    }

    FString ClassError;
    UClass* ManagerClass = LoadObjectIdsProxyManagerClass(ClassError);
    if (!ManagerClass)
    {
        OutError = ClassError;
        return false;
    }

    if (!HolderActor.Get())
    {
        HolderActor = SpawnHiddenCaptureHolder(World);
        if (!HolderActor.Get())
        {
            OutError = TEXT("failed_to_spawn_camera_pool_holder");
            return false;
        }
#if WITH_EDITOR
        HolderActor->SetActorLabel(TEXT("SpCameraCapturePoolHolder"));
#endif
    }

    PoolSize = FMath::Clamp(PoolSize, 1, 128);
    while (AgentBatchObjectIdCaptures.Num() > PoolSize)
    {
        TObjectPtr<USpSceneCaptureComponent2D> Capture =
            AgentBatchObjectIdCaptures.Pop();
        if (Capture.Get())
        {
            if (Capture->IsInitialized())
            {
                Capture->Terminate();
            }
            Capture->DestroyComponent();
        }
    }

    while (AgentBatchObjectIdCaptures.Num() < PoolSize)
    {
        USpSceneCaptureComponent2D* Capture = NewObject<USpSceneCaptureComponent2D>(
            HolderActor.Get(),
            USpSceneCaptureComponent2D::StaticClass(),
            NAME_None);
        if (!Capture)
        {
            OutError = TEXT("failed_to_allocate_object_id_capture");
            return false;
        }
        Capture->CreationMethod = EComponentCreationMethod::Instance;
        Capture->SetupAttachment(HolderActor->GetRootComponent());
        Capture->RegisterComponentWithWorld(World);
        AgentBatchObjectIdCaptures.Add(Capture);
    }

    for (int32 Index = 0; Index < AgentBatchObjectIdCaptures.Num(); ++Index)
    {
        USpSceneCaptureComponent2D* Capture = AgentBatchObjectIdCaptures[Index].Get();
        if (!Capture)
        {
            OutError = TEXT("object_id_capture_missing");
            return false;
        }

        const bool bNeedsReinitialize =
            Capture->IsInitialized() &&
            (Capture->Width != Width ||
             Capture->Height != Height ||
             Capture->TextureRenderTargetFormat != ETextureRenderTargetFormat::RTF_RGBA8 ||
             Capture->MeshProxyComponentManagerClass.Get() != ManagerClass);
        if (bNeedsReinitialize)
        {
            Capture->Terminate();
        }

        Capture->Width = Width;
        Capture->Height = Height;
        Capture->NumChannelsPerPixel = 4;
        Capture->ChannelDataType = ESpArrayDataType::UInt8;
        Capture->bUseSharedMemory = false;
        Capture->BufferingMode = ESpBufferingMode::SingleBuffered;
        Capture->bReadPixelsEveryFrame = false;
        Capture->bCaptureEveryFrame = false;
        Capture->bCaptureOnMovement = false;
        Capture->bAlwaysPersistRenderingState = true;
        Capture->CaptureSource = ESceneCaptureSource::SCS_FinalColorHDR;
        Capture->bOverrideTextureRenderTargetFormat = true;
        Capture->TextureRenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;
        Capture->bOverrideTextureRenderTargetSRGB = true;
        Capture->bTextureRenderTargetSRGB = false;
        Capture->bOverrideTextureRenderTargetForceLinearGamma = true;
        Capture->bTextureRenderTargetForceLinearGamma = true;
        Capture->bOverrideTextureRenderTargetGamma = true;
        Capture->TextureRenderTargetGamma = 1.0f;
        Capture->MeshProxyComponentManagerClass = ManagerClass;
        ApplyObjectIdShowFlags(Capture);

        if (!Capture->IsInitialized())
        {
            Capture->Initialize();
        }
    }
    FlushRenderingCommands();
    return true;
}

FString USpCameraCapturePool::Agent_CaptureCamerasJson(const FString& RequestJson)
{
    return Camera_CaptureCamerasJson(RequestJson);
}

FString USpCameraCapturePool::Camera_CaptureCamerasJson(const FString& RequestJson)
{
    const double StartSeconds = FPlatformTime::Seconds();
    double LookupMs = 0.0;
    double InitMs = 0.0;
    double CaptureReadMs = 0.0;
    double EncodeMs = 0.0;

    TSharedPtr<FJsonObject> Request;
    const TSharedRef<TJsonReader<>> Reader =
        TJsonReaderFactory<>::Create(RequestJson);
    const bool bValidJson =
        FJsonSerializer::Deserialize(Reader, Request) && Request.IsValid();

    TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    TArray<TSharedPtr<FJsonValue>> ImageValues;
    TArray<TSharedPtr<FJsonValue>> MissingValues;
    TArray<TSharedPtr<FJsonValue>> ErrorValues;

    if (!bValidJson)
    {
        Root->SetStringField(TEXT("error"), TEXT("invalid_request_json"));
    }

    const TArray<FString> LegacyAgentTags = ParseAgentTags(Request);
    double WidthValue = 320.0;
    double HeightValue = 180.0;
    double FovDegrees = 90.0;
    double QualityValue = 75.0;
    double SharedPoolSizeValue = 16.0;
    double DepthFarValue = 10000.0;
    bool bForceCapture = true;
    bool bValidate = true;
    bool bInitializeOnly = false;
    FString CaptureMode = TEXT("shared_pool");
    if (Request.IsValid())
    {
        Request->TryGetNumberField(TEXT("width"), WidthValue);
        Request->TryGetNumberField(TEXT("height"), HeightValue);
        Request->TryGetNumberField(TEXT("fov_degrees"), FovDegrees);
        Request->TryGetNumberField(TEXT("jpeg_quality"), QualityValue);
        Request->TryGetNumberField(TEXT("shared_pool_size"), SharedPoolSizeValue);
        Request->TryGetNumberField(TEXT("depth_far_cm"), DepthFarValue);
        Request->TryGetBoolField(TEXT("force_capture"), bForceCapture);
        Request->TryGetBoolField(TEXT("validate"), bValidate);
        Request->TryGetBoolField(TEXT("initialize_only"), bInitializeOnly);
        Request->TryGetStringField(TEXT("capture_mode"), CaptureMode);
    }
    CaptureMode.TrimStartAndEndInline();
    if (CaptureMode.IsEmpty())
    {
        CaptureMode = TEXT("shared_pool");
    }
    const bool bUseAgentNativeCapture =
        CaptureMode.Equals(TEXT("agent_native"), ESearchCase::IgnoreCase);

    const int32 Width = FMath::Clamp(FMath::RoundToInt(WidthValue), 16, 4096);
    const int32 Height = FMath::Clamp(FMath::RoundToInt(HeightValue), 16, 4096);
    const int32 JpegQuality = FMath::Clamp(FMath::RoundToInt(QualityValue), 1, 95);
    const int32 SharedPoolSize = FMath::Clamp(
        FMath::RoundToInt(SharedPoolSizeValue),
        1,
        128);
    const float Fov = FMath::Clamp(static_cast<float>(FovDegrees), 5.0f, 170.0f);
    const float DepthFarCm = FMath::Clamp(static_cast<float>(DepthFarValue), 1.0f, 10000000.0f);
    const TArray<ESpAgentCaptureModality> Modalities = ParseModalities(Request);
    const TArray<FString> MaskActorTags = ParseStringArrayAliases(
        Request,
        {TEXT("mask_actor_tags"), TEXT("annotation_actor_tags"), TEXT("target_actor_tags")});
    const ESpSegmentationScope SegmentationScope = ParseSegmentationScope(Request);
    bool bRequestedSceneObjectIds = false;
    for (ESpAgentCaptureModality Modality : Modalities)
    {
        if (ShouldUseSceneObjectIds(Modality, MaskActorTags, SegmentationScope))
        {
            bRequestedSceneObjectIds = true;
            break;
        }
    }
    FVector CameraLocationOverride = FVector::ZeroVector;
    FRotator CameraRotationOverride = FRotator::ZeroRotator;
    const bool bHasCameraLocationOverride = TryGetJsonVectorAliases(
        Request,
        {
            TEXT("camera_location_cm"),
            TEXT("camera_location"),
            TEXT("capture_location_cm"),
            TEXT("location_cm"),
            TEXT("location")
        },
        CameraLocationOverride);
    const bool bHasCameraRotationOverride = TryGetJsonRotatorAliases(
        Request,
        {
            TEXT("camera_rotation_degrees"),
            TEXT("camera_rotation"),
            TEXT("capture_rotation_degrees"),
            TEXT("rotation_degrees"),
            TEXT("rotation")
        },
        CameraRotationOverride);
    const bool bUseCameraTransformOverride =
        bHasCameraLocationOverride || bHasCameraRotationOverride;
    const TArray<FSpCameraSourceRequest> CameraSources = ParseCameraSourceRequests(
        Request,
        LegacyAgentTags,
        bHasCameraLocationOverride,
        CameraLocationOverride,
        bHasCameraRotationOverride,
        CameraRotationOverride);
    bool bAnyCameraTransformOverride = bUseCameraTransformOverride;
    bool bHasWorldCameraSource = false;
    for (const FSpCameraSourceRequest& Source : CameraSources)
    {
        bAnyCameraTransformOverride =
            bAnyCameraTransformOverride || Source.bHasLocation || Source.bHasRotation;
        bHasWorldCameraSource =
            bHasWorldCameraSource || Source.SourceType == ESpCameraSourceType::World;
    }

    UWorld* World = GetWorld();
    if (!World)
    {
        for (const FSpCameraSourceRequest& Source : CameraSources)
        {
            if (!Source.CameraViewError.IsEmpty())
            {
                ErrorValues.Add(MakeShared<FJsonValueObject>(
                    MakeErrorObject(Source.CameraId, Source.CameraViewError)));
            }
        }
        Root->SetStringField(TEXT("error"), TEXT("no_world"));
    }
    else if (CameraSources.IsEmpty())
    {
        Root->SetStringField(TEXT("error"), TEXT("missing_camera_sources"));
    }
    else
    {
        const TArray<TWeakObjectPtr<AActor>> MaskActors =
            FindActorsByTagTextsInWorld(World, MaskActorTags);

        struct FSharedCameraJob
        {
            FString TagText;
            FSpCameraSourceRequest Source;
            TWeakObjectPtr<ASpHumanoidAgent> Agent;
            TWeakObjectPtr<USpSceneCaptureComponent2D> SceneCapture;
        };

        TArray<FSharedCameraJob> Jobs;
        Jobs.Reserve(CameraSources.Num());
        for (const FSpCameraSourceRequest& Source : CameraSources)
        {
            if (!Source.CameraViewError.IsEmpty())
            {
                ErrorValues.Add(MakeShared<FJsonValueObject>(
                    MakeErrorObject(Source.CameraId, Source.CameraViewError)));
                continue;
            }
            if (Source.SourceType == ESpCameraSourceType::World)
            {
                if (!Source.bHasLocation)
                {
                    ErrorValues.Add(MakeShared<FJsonValueObject>(
                        MakeErrorObject(
                            Source.CameraId,
                            TEXT("world_camera_requires_location_cm"))));
                    continue;
                }

                FSharedCameraJob Job;
                Job.TagText = Source.CameraId;
                Job.Source = Source;
                Jobs.Add(Job);
                continue;
            }

            if (Source.AgentTag.IsEmpty())
            {
                ErrorValues.Add(MakeShared<FJsonValueObject>(
                    MakeErrorObject(
                        Source.CameraId,
                        TEXT("agent_camera_requires_agent_tag"))));
                continue;
            }

            const double LookupStart = FPlatformTime::Seconds();
            const FName AgentTag(*Source.AgentTag);
            ASpHumanoidAgent* Agent = FindHumanoidByTagInWorld(World, AgentTag);
            LookupMs += (FPlatformTime::Seconds() - LookupStart) * 1000.0;
            if (!Agent)
            {
                MissingValues.Add(MakeShared<FJsonValueString>(Source.CameraId));
                ErrorValues.Add(MakeShared<FJsonValueObject>(
                    MakeErrorObject(Source.CameraId, TEXT("agent_not_visible_on_this_client"))));
                continue;
            }
            USpSceneCaptureComponent2D* SceneCapture =
                Agent->GetObservationCamera(Source.CameraView);
            if (!SceneCapture)
            {
                ErrorValues.Add(MakeShared<FJsonValueObject>(
                    MakeErrorObject(Source.CameraId, TEXT("agent_has_no_scene_capture"))));
                continue;
            }

            FSharedCameraJob Job;
            Job.TagText = Source.CameraId;
            Job.Source = Source;
            Job.Agent = Agent;
            Job.SceneCapture = SceneCapture;
            Jobs.Add(Job);
        }

        const bool bRequestedNonRgb = Modalities.Contains(ESpAgentCaptureModality::Depth) ||
            Modalities.Contains(ESpAgentCaptureModality::Normal) ||
            Modalities.Contains(ESpAgentCaptureModality::InstanceSeg) ||
            Modalities.Contains(ESpAgentCaptureModality::SemanticSeg);
        if (Jobs.IsEmpty())
        {
            // Source-level errors were already appended while resolving jobs.
        }
        else if (bUseAgentNativeCapture && (bAnyCameraTransformOverride || bHasWorldCameraSource))
        {
            for (const FSharedCameraJob& Job : Jobs)
            {
                ErrorValues.Add(MakeShared<FJsonValueObject>(
                    MakeErrorObject(
                        Job.TagText,
                        TEXT("agent_native_capture_requires_agent_source_without_transform_override"))));
            }
        }
        else if (bUseAgentNativeCapture && bRequestedNonRgb)
        {
            for (const FSharedCameraJob& Job : Jobs)
            {
                ErrorValues.Add(MakeShared<FJsonValueObject>(
                    MakeErrorObject(
                        Job.TagText,
                        TEXT("agent_native_capture_supports_rgb_only_use_shared_pool_for_depth_or_seg"))));
            }
        }
        else if (bInitializeOnly)
        {
            FString PoolError;
            const double InitStart = FPlatformTime::Seconds();
            const bool bPoolReady = bUseAgentNativeCapture ||
                EnsureAgentBatchCapturePool(
                    Width,
                    Height,
                    FMath::Clamp(SharedPoolSize, 1, FMath::Max(1, Jobs.Num())),
                    PoolError);
            bool bObjectIdPoolReady = true;
            FString ObjectIdPoolError;
            if (bPoolReady && bRequestedSceneObjectIds)
            {
                bObjectIdPoolReady = EnsureAgentBatchObjectIdCapturePool(
                    Width,
                    Height,
                    FMath::Clamp(SharedPoolSize, 1, FMath::Max(1, Jobs.Num())),
                    ObjectIdPoolError);
            }
            InitMs += (FPlatformTime::Seconds() - InitStart) * 1000.0;

            for (const FSharedCameraJob& Job : Jobs)
            {
                if (!bPoolReady || !bObjectIdPoolReady)
                {
                    ErrorValues.Add(MakeShared<FJsonValueObject>(
                        MakeErrorObject(
                            Job.TagText,
                            !bPoolReady ? PoolError : ObjectIdPoolError)));
                    continue;
                }
                TSharedPtr<FJsonObject> ImageJson = MakeShared<FJsonObject>();
                ImageJson->SetStringField(TEXT("agent_tag"), Job.TagText);
                AddCameraSourceMetadata(ImageJson, Job.Source);
                ImageJson->SetBoolField(TEXT("success"), true);
                ImageJson->SetBoolField(TEXT("initialized_only"), true);
                ImageJson->SetStringField(TEXT("capture_mode"), CaptureMode);
                ImageJson->SetStringField(
                    TEXT("segmentation_scope"),
                    SegmentationScopeToString(SegmentationScope));
                ImageJson->SetBoolField(
                    TEXT("scene_object_ids_requested"),
                    bRequestedSceneObjectIds);
                ImageJson->SetNumberField(TEXT("width"), Width);
                ImageJson->SetNumberField(TEXT("height"), Height);
                TArray<TSharedPtr<FJsonValue>> InitModalities;
                for (ESpAgentCaptureModality Modality : Modalities)
                {
                    InitModalities.Add(
                        MakeShared<FJsonValueString>(ModalityToString(Modality)));
                }
                ImageJson->SetArrayField(TEXT("modalities"), InitModalities);
                ImageValues.Add(MakeShared<FJsonValueObject>(ImageJson));
            }
        }
        else if (bUseAgentNativeCapture)
        {
            for (const FSharedCameraJob& Job : Jobs)
            {
                ASpHumanoidAgent* Agent = Job.Agent.Get();
                USpSceneCaptureComponent2D* SceneCapture = Job.SceneCapture.Get();
                if (!Agent || !SceneCapture)
                {
                    ErrorValues.Add(MakeShared<FJsonValueObject>(
                        MakeErrorObject(Job.TagText, TEXT("agent_native_capture_missing"))));
                    continue;
                }

                const double InitStart = FPlatformTime::Seconds();
                const bool bSizeChanged =
                    SceneCapture->Width != Width ||
                    SceneCapture->Height != Height ||
                    !FMath::IsNearlyEqual(SceneCapture->FOVAngle, Fov, 0.01f);
                if (SceneCapture->IsInitialized() && bSizeChanged)
                {
                    Agent->TerminateObservationCamera(Job.Source.CameraView);
                }
                if (!Agent->ConfigureObservationCamera(
                        Job.Source.CameraView, Width, Height, Fov))
                {
                    ErrorValues.Add(MakeShared<FJsonValueObject>(
                        MakeErrorObject(Job.TagText, TEXT("configure_camera_failed"))));
                    InitMs += (FPlatformTime::Seconds() - InitStart) * 1000.0;
                    continue;
                }
                if (!SceneCapture->IsInitialized())
                {
                    Agent->InitializeObservationCamera(Job.Source.CameraView);
                }
                InitMs += (FPlatformTime::Seconds() - InitStart) * 1000.0;

                FString CaptureError;
                const double CaptureReadBefore = CaptureReadMs;
                const double EncodeBefore = EncodeMs;
                const double WallStart = FPlatformTime::Seconds();
                TSharedPtr<FJsonObject> ImageJson = CaptureOneAgentModalityJson(
                    Job.TagText,
                    Agent,
                    MaskActors,
                    SceneCapture,
                    SceneCapture->TextureTarget,
                    ESpAgentCaptureModality::RGB,
                    bForceCapture,
                    bValidate,
                    JpegQuality,
                    DepthFarCm,
                    CaptureReadMs,
                    EncodeMs,
                    CaptureError);
                const double ImageWallMs =
                    (FPlatformTime::Seconds() - WallStart) * 1000.0;
                if (!ImageJson.IsValid())
                {
                    ErrorValues.Add(MakeShared<FJsonValueObject>(
                        MakeErrorObject(Job.TagText, CaptureError)));
                    continue;
                }
                TSharedPtr<FJsonObject> ImageTiming = MakeShared<FJsonObject>();
                ImageTiming->SetNumberField(
                    TEXT("capture_read_ms"),
                    FMath::Max(0.0, CaptureReadMs - CaptureReadBefore));
                ImageTiming->SetNumberField(
                    TEXT("encode_ms"),
                    FMath::Max(0.0, EncodeMs - EncodeBefore));
                ImageTiming->SetNumberField(
                    TEXT("wall_ms"),
                    FMath::Max(0.0, ImageWallMs));
                ImageJson->SetObjectField(TEXT("timing"), ImageTiming);
                if (USpPixelGoalSubsystem* PixelGoal =
                        World->GetSubsystem<USpPixelGoalSubsystem>())
                {
                    PixelGoal->RecordCameraSnapshot(
                        SceneCapture, Agent, Width, Height, ImageJson);
                }
                ImageJson->SetStringField(TEXT("capture_mode"), CaptureMode);
                AddCameraSourceMetadata(ImageJson, Job.Source);
                ImageValues.Add(MakeShared<FJsonValueObject>(ImageJson));
            }
        }
        else if (!Jobs.IsEmpty())
        {
            FString PoolError;
            const int32 EffectivePoolSize = FMath::Clamp(SharedPoolSize, 1, Jobs.Num());
            const double InitStart = FPlatformTime::Seconds();
            const bool bPoolReady = EnsureAgentBatchCapturePool(
                Width,
                Height,
                EffectivePoolSize,
                PoolError);
            bool bObjectIdPoolReady = true;
            FString ObjectIdPoolError;
            if (bPoolReady && bRequestedSceneObjectIds)
            {
                bObjectIdPoolReady = EnsureAgentBatchObjectIdCapturePool(
                    Width,
                    Height,
                    EffectivePoolSize,
                    ObjectIdPoolError);
            }
            InitMs += (FPlatformTime::Seconds() - InitStart) * 1000.0;

            if (!bPoolReady || !bObjectIdPoolReady)
            {
                for (const FSharedCameraJob& Job : Jobs)
                {
                    ErrorValues.Add(MakeShared<FJsonValueObject>(
                        MakeErrorObject(
                            Job.TagText,
                            !bPoolReady ? PoolError : ObjectIdPoolError)));
                }
            }
            else
            {
                for (int32 StartIndex = 0; StartIndex < Jobs.Num();
                     StartIndex += EffectivePoolSize)
                {
                    const int32 Count = FMath::Min(EffectivePoolSize, Jobs.Num() - StartIndex);
                    for (ESpAgentCaptureModality Modality : Modalities)
                    {
                        for (int32 LocalIndex = 0; LocalIndex < Count; ++LocalIndex)
                        {
                            const FSharedCameraJob& Job =
                                Jobs[StartIndex + LocalIndex];
                            ASpHumanoidAgent* Agent = Job.Agent.Get();
                            USceneCaptureComponent2D* CaptureComponent =
                                AgentBatchCaptures[LocalIndex].Get();
                            USpSceneCaptureComponent2D* SourceCapture =
                                Job.SceneCapture.Get();
                            if (!CaptureComponent ||
                                (Job.Source.SourceType == ESpCameraSourceType::Agent &&
                                 (!Agent || !SourceCapture)))
                            {
                                ErrorValues.Add(MakeShared<FJsonValueObject>(
                                    MakeErrorObject(
                                        Job.TagText,
                                        TEXT("shared_pool_capture_missing"))));
                                continue;
                            }

                            const bool bUseSceneObjectIds = ShouldUseSceneObjectIds(
                                Modality,
                                MaskActorTags,
                                SegmentationScope);
                            if (bUseSceneObjectIds)
                            {
                                USpSceneCaptureComponent2D* ObjectIdCapture =
                                    AgentBatchObjectIdCaptures.IsValidIndex(LocalIndex)
                                        ? AgentBatchObjectIdCaptures[LocalIndex].Get()
                                        : nullptr;
                                if (!ObjectIdCapture)
                                {
                                    ErrorValues.Add(MakeShared<FJsonValueObject>(
                                        MakeErrorObject(
                                            Job.TagText,
                                            TEXT("object_id_capture_missing"))));
                                    continue;
                                }

                                FString ConfigureObjectIdError;
                                if (!ConfigureObjectIdCaptureForSource(
                                        ObjectIdCapture,
                                        SourceCapture,
                                        Job.Source.bHasLocation
                                            ? Job.Source.LocationCm
                                            : FVector::ZeroVector,
                                        Job.Source.bHasRotation
                                            ? Job.Source.RotationDegrees
                                            : FRotator::ZeroRotator,
                                        Fov,
                                        ConfigureObjectIdError))
                                {
                                    ErrorValues.Add(MakeShared<FJsonValueObject>(
                                        MakeErrorObject(
                                            Job.TagText,
                                            ConfigureObjectIdError)));
                                    continue;
                                }
                                ApplyWorldCaptureTransformOverride(
                                    ObjectIdCapture,
                                    Job.Source.bHasLocation,
                                    Job.Source.LocationCm,
                                    Job.Source.bHasRotation,
                                    Job.Source.RotationDegrees);

                                FString CaptureError;
                                TSharedPtr<FJsonObject> ImageJson =
                                    CaptureSceneObjectIdsJson(
                                        Job.TagText,
                                        Agent,
                                        ObjectIdCapture,
                                        Cast<ASpMeshProxyComponentManager>(
                                            ObjectIdsProxyManager.Get()),
                                        Modality,
                                        bForceCapture,
                                        bValidate,
                                        JpegQuality,
                                        CaptureReadMs,
                                        EncodeMs,
                                        CaptureError);
                                if (!ImageJson.IsValid())
                                {
                                    ErrorValues.Add(MakeShared<FJsonValueObject>(
                                        MakeErrorObject(
                                            Job.TagText,
                                            ModalityToString(Modality) + TEXT(": ") + CaptureError)));
                                    continue;
                                }
                                ImageJson->SetStringField(TEXT("capture_mode"), CaptureMode);
                                AddCameraSourceMetadata(ImageJson, Job.Source);
                                ImageValues.Add(MakeShared<FJsonValueObject>(ImageJson));
                                continue;
                            }

                            FString RenderTargetError;
                            if (!ConfigureRenderTargetForModality(
                                    CaptureComponent,
                                    AgentBatchRenderTargets[LocalIndex],
                                    Width,
                                    Height,
                                    Modality,
                                    RenderTargetError))
                            {
                                ErrorValues.Add(MakeShared<FJsonValueObject>(
                                    MakeErrorObject(Job.TagText, RenderTargetError)));
                                continue;
                            }

                            ConfigureCaptureForModality(
                                CaptureComponent,
                                SourceCapture,
                                Job.Source.bHasLocation
                                    ? Job.Source.LocationCm
                                    : FVector::ZeroVector,
                                Job.Source.bHasRotation
                                    ? Job.Source.RotationDegrees
                                    : FRotator::ZeroRotator,
                                Fov,
                                Modality);
                            if (ObjectIdsProxyManager.Get())
                            {
                                CaptureComponent->HiddenActors.AddUnique(
                                    ObjectIdsProxyManager.Get());
                            }
                            ApplyWorldCaptureTransformOverride(
                                CaptureComponent,
                                Job.Source.SourceType == ESpCameraSourceType::Agent &&
                                    Job.Source.bHasLocation,
                                Job.Source.LocationCm,
                                Job.Source.SourceType == ESpCameraSourceType::Agent &&
                                    Job.Source.bHasRotation,
                                Job.Source.RotationDegrees);
                            UTextureRenderTarget2D* RenderTarget =
                                AgentBatchRenderTargets[LocalIndex].Get();
                            FString CaptureError;
                            TSharedPtr<FJsonObject> ImageJson = CaptureOneAgentModalityJson(
                                Job.TagText,
                                Agent,
                                MaskActors,
                                CaptureComponent,
                                RenderTarget,
                                Modality,
                                bForceCapture,
                                bValidate,
                                JpegQuality,
                                DepthFarCm,
                                CaptureReadMs,
                                EncodeMs,
                                CaptureError);
                            if (!ImageJson.IsValid())
                            {
                                ErrorValues.Add(MakeShared<FJsonValueObject>(
                                    MakeErrorObject(
                                        Job.TagText,
                                        ModalityToString(Modality) + TEXT(": ") + CaptureError)));
                                continue;
                            }
                            if (Modality == ESpAgentCaptureModality::RGB)
                            {
                                if (USpPixelGoalSubsystem* PixelGoal =
                                        World->GetSubsystem<USpPixelGoalSubsystem>())
                                {
                                    PixelGoal->RecordCameraSnapshot(
                                        CaptureComponent,
                                        Agent,
                                        Width,
                                        Height,
                                        ImageJson);
                                }
                            }
                            ImageJson->SetStringField(TEXT("capture_mode"), CaptureMode);
                            AddCameraSourceMetadata(ImageJson, Job.Source);
                            ImageValues.Add(MakeShared<FJsonValueObject>(ImageJson));
                        }
                    }
                }
            }
        }
    }

    const double TotalMs = (FPlatformTime::Seconds() - StartSeconds) * 1000.0;
    Root->SetNumberField(TEXT("requested_count"), CameraSources.Num());
    Root->SetNumberField(TEXT("image_count"), ImageValues.Num());
    Root->SetNumberField(TEXT("missing_count"), MissingValues.Num());
    Root->SetNumberField(TEXT("error_count"), ErrorValues.Num());
    Root->SetNumberField(TEXT("width"), Width);
    Root->SetNumberField(TEXT("height"), Height);
    Root->SetNumberField(TEXT("jpeg_quality"), JpegQuality);
    Root->SetNumberField(TEXT("shared_pool_size"), SharedPoolSize);
    Root->SetNumberField(TEXT("depth_far_cm"), DepthFarCm);
    Root->SetBoolField(TEXT("force_capture"), bForceCapture);
    Root->SetBoolField(TEXT("initialize_only"), bInitializeOnly);
    Root->SetStringField(TEXT("capture_mode"), CaptureMode);
    Root->SetStringField(TEXT("segmentation_scope"), SegmentationScopeToString(SegmentationScope));
    Root->SetBoolField(TEXT("scene_object_ids_requested"), bRequestedSceneObjectIds);
    Root->SetBoolField(TEXT("camera_transform_override"), bAnyCameraTransformOverride);
    if (bHasCameraLocationOverride)
    {
        Root->SetArrayField(TEXT("camera_location_cm"), JsonVectorArray(CameraLocationOverride));
    }
    if (bHasCameraRotationOverride)
    {
        Root->SetArrayField(TEXT("camera_rotation_degrees"), JsonRotatorArray(CameraRotationOverride));
    }
    TArray<TSharedPtr<FJsonValue>> MaskActorTagValues;
    for (const FString& MaskActorTag : MaskActorTags)
    {
        MaskActorTagValues.Add(MakeShared<FJsonValueString>(MaskActorTag));
    }
    Root->SetArrayField(TEXT("mask_actor_tags"), MaskActorTagValues);
    TArray<TSharedPtr<FJsonValue>> CameraSourceValues;
    for (const FSpCameraSourceRequest& Source : CameraSources)
    {
        TSharedPtr<FJsonObject> SourceObject = MakeShared<FJsonObject>();
        SourceObject->SetStringField(TEXT("camera_id"), Source.CameraId);
        SourceObject->SetStringField(
            TEXT("source_type"),
            CameraSourceTypeToString(Source.SourceType));
        SourceObject->SetStringField(TEXT("camera_view"), Source.CameraView);
        if (!Source.AgentTag.IsEmpty())
        {
            SourceObject->SetStringField(TEXT("agent_tag"), Source.AgentTag);
        }
        if (Source.bHasLocation)
        {
            SourceObject->SetArrayField(TEXT("camera_location_cm"), JsonVectorArray(Source.LocationCm));
        }
        if (Source.bHasRotation)
        {
            SourceObject->SetArrayField(
                TEXT("camera_rotation_degrees"),
                JsonRotatorArray(Source.RotationDegrees));
        }
        CameraSourceValues.Add(MakeShared<FJsonValueObject>(SourceObject));
    }
    Root->SetArrayField(TEXT("camera_sources"), CameraSourceValues);
    TArray<TSharedPtr<FJsonValue>> ModalityValues;
    for (ESpAgentCaptureModality Modality : Modalities)
    {
        ModalityValues.Add(MakeShared<FJsonValueString>(ModalityToString(Modality)));
    }
    Root->SetArrayField(TEXT("modalities"), ModalityValues);
    Root->SetArrayField(TEXT("images"), ImageValues);
    Root->SetArrayField(TEXT("missing_agent_tags"), MissingValues);
    Root->SetArrayField(TEXT("missing_camera_sources"), MissingValues);
    Root->SetArrayField(TEXT("errors"), ErrorValues);

    TSharedPtr<FJsonObject> Timing = MakeShared<FJsonObject>();
    Timing->SetNumberField(TEXT("total_ms"), TotalMs);
    Timing->SetNumberField(TEXT("lookup_ms"), LookupMs);
    Timing->SetNumberField(TEXT("init_ms"), InitMs);
    Timing->SetNumberField(TEXT("capture_read_ms"), CaptureReadMs);
    Timing->SetNumberField(TEXT("encode_ms"), EncodeMs);
    Root->SetObjectField(TEXT("timing"), Timing);

    FString Output;
    const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
    FJsonSerializer::Serialize(Root, Writer);
    return Output;
}

FString USpCameraCapturePool::Agent_SpawnObservationFixtureJson(const FString& RequestJson)
{
    TSharedPtr<FJsonObject> Request;
    const TSharedRef<TJsonReader<>> Reader =
        TJsonReaderFactory<>::Create(RequestJson);
    const bool bValidJson =
        RequestJson.TrimStartAndEnd().IsEmpty() ||
        (FJsonSerializer::Deserialize(Reader, Request) && Request.IsValid());

    TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    TArray<TSharedPtr<FJsonValue>> RecordValues;
    TArray<TSharedPtr<FJsonValue>> ErrorValues;
    Root->SetStringField(TEXT("schema"), TEXT("simworld-observation-fixture-v1"));
    Root->SetBoolField(TEXT("success"), false);
    if (!bValidJson)
    {
        Root->SetStringField(TEXT("error"), TEXT("invalid_request_json"));
    }

    UWorld* World = GetWorld();
    if (!World)
    {
        Root->SetStringField(TEXT("error"), TEXT("no_world"));
    }
    else
    {
        const FString CameraTagText = JsonStringOrDefault(
            Request,
            TEXT("camera_agent_tag"),
            TEXT("SimWorldPublicCameraAgent_88"));
        const FString StaticTagText = JsonStringOrDefault(
            Request,
            TEXT("static_target_tag"),
            TEXT("RuntimeCube_001"));
        const FString StaticSphereTagText = JsonStringOrDefault(
            Request,
            TEXT("static_sphere_target_tag"),
            TEXT("RuntimeSphere_001"));
        const FString PedestrianTagText = JsonStringOrDefault(
            Request,
            TEXT("pedestrian_target_tag"),
            TEXT("RuntimePed_001"));
        const FString VehicleTagText = JsonStringOrDefault(
            Request,
            TEXT("vehicle_target_tag"),
            TEXT("RuntimeVehicle_001"));

        const FName CameraTag(*CameraTagText);
        const FName StaticTag(*StaticTagText);
        const FName StaticSphereTag(*StaticSphereTagText);
        const FName PedestrianTag(*PedestrianTagText);
        const FName VehicleTag(*VehicleTagText);
        UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(
            nullptr,
            TEXT("/Engine/BasicShapes/Cube.Cube"));
        UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(
            nullptr,
            TEXT("/Engine/BasicShapes/Sphere.Sphere"));

        FActorSpawnParameters Params;
        Params.SpawnCollisionHandlingOverride =
            ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

        auto AddRecord = [&RecordValues](
            const FString& Case,
            const FString& Tag,
            AActor* Actor,
            bool bSpawned,
            const FString& SpawnMethod)
        {
            TSharedPtr<FJsonObject> Record = MakeShared<FJsonObject>();
            Record->SetStringField(TEXT("case"), Case);
            Record->SetStringField(TEXT("tag"), Tag);
            Record->SetBoolField(TEXT("spawned"), bSpawned);
            Record->SetStringField(TEXT("spawn_method"), SpawnMethod);
            if (Actor)
            {
                Record->SetStringField(TEXT("actor_name"), Actor->GetName());
                Record->SetStringField(TEXT("class_name"), Actor->GetClass()->GetName());
                const FVector Loc = Actor->GetActorLocation();
                TArray<TSharedPtr<FJsonValue>> LocValues;
                LocValues.Add(MakeShared<FJsonValueNumber>(Loc.X));
                LocValues.Add(MakeShared<FJsonValueNumber>(Loc.Y));
                LocValues.Add(MakeShared<FJsonValueNumber>(Loc.Z));
                Record->SetArrayField(TEXT("loc_cm"), LocValues);
            }
            RecordValues.Add(MakeShared<FJsonValueObject>(Record));
        };

        auto AddError = [&ErrorValues](const FString& Case, const FString& Error)
        {
            TSharedPtr<FJsonObject> ErrorObject = MakeShared<FJsonObject>();
            ErrorObject->SetStringField(TEXT("case"), Case);
            ErrorObject->SetStringField(TEXT("error"), Error);
            ErrorValues.Add(MakeShared<FJsonValueObject>(ErrorObject));
        };

        bool bAllOk = true;

        APointLight* FixtureLight = Cast<APointLight>(
            FindFirstActorByTagTextInWorld(World, TEXT("SimWorldObservationFixtureLight")));
        if (!FixtureLight)
        {
            FixtureLight = World->SpawnActor<APointLight>(
                APointLight::StaticClass(),
                FVector(260.0, 0.0, 520.0),
                FRotator::ZeroRotator,
                Params);
        }
        if (FixtureLight)
        {
            FixtureLight->SetActorLocation(
                FVector(260.0, 0.0, 520.0),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            AddTagToActor(FixtureLight, FName(TEXT("SimWorldObservationFixtureLight")));
            UPointLightComponent* LightComponent = FixtureLight->PointLightComponent.Get();
            if (LightComponent)
            {
                LightComponent->SetMobility(EComponentMobility::Movable);
                LightComponent->SetIntensity(35000.0f);
                LightComponent->SetAttenuationRadius(2500.0f);
                LightComponent->SetLightColor(FLinearColor::White);
            }
        }

        bool bSpawnedCamera = false;
        ASpHumanoidAgent* CameraAgent = FindHumanoidByTagInWorld(World, CameraTag);
        if (!CameraAgent)
        {
            CameraAgent = World->SpawnActor<ASpHumanoidAgent>(
                ASpHumanoidAgent::StaticClass(),
                FVector(0.0, 0.0, 120.0),
                FRotator(0.0, 0.0, 0.0),
                Params);
            bSpawnedCamera = CameraAgent != nullptr;
        }
        if (CameraAgent)
        {
            MakeActorComponentsMovable(CameraAgent);
            CameraAgent->SetActorLocationAndRotation(
                FVector(0.0, 0.0, 120.0),
                FRotator(0.0, 0.0, 0.0),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            CameraAgent->Agent_SetAgentTag(CameraTag);
            CameraAgent->ConfigureCamera(320, 180, 90.0f);
            CameraAgent->InitializeCamera();
            AddRecord(
                TEXT("camera_agent"),
                CameraTagText,
                CameraAgent,
                bSpawnedCamera,
                TEXT("ASpHumanoidAgent runtime spawn with authored SceneCapture"));
        }
        else
        {
            bAllOk = false;
            AddError(TEXT("camera_agent"), TEXT("spawn_camera_agent_failed"));
        }

        bool bSpawnedStatic = false;
        AActor* StaticTarget = FindFirstActorByTagTextInWorld(World, StaticTagText);
        if (StaticTarget)
        {
            StaticTarget->Destroy();
            StaticTarget = nullptr;
        }
        if (!StaticTarget)
        {
            StaticTarget = World->SpawnActor<AStaticMeshActor>(
                AStaticMeshActor::StaticClass(),
                FVector(650.0, -140.0, 100.0),
                FRotator::ZeroRotator,
                Params);
            bSpawnedStatic = StaticTarget != nullptr;
        }
        if (StaticTarget)
        {
            MakeActorComponentsMovable(StaticTarget);
            StaticTarget->SetActorLocation(FVector(650.0, -140.0, 100.0), false, nullptr, ETeleportType::TeleportPhysics);
            AddTagToActor(StaticTarget, StaticTag);
            FString MeshError;
            if (!EnsureVisibleMeshComponent(
                    StaticTarget,
                    CubeMesh,
                    nullptr,
                    FLinearColor::White,
                    TEXT("SimWorldObservationStaticMesh"),
                    FVector::ZeroVector,
                    FVector(1.5, 1.5, 1.5),
                    MeshError))
            {
                bAllOk = false;
                AddError(TEXT("static_runtime_actor"), MeshError);
            }
            AddRecord(
                TEXT("static_runtime_actor"),
                StaticTagText,
                StaticTarget,
                bSpawnedStatic,
                TEXT("AStaticMeshActor runtime spawn with engine cube mesh"));
        }
        else
        {
            bAllOk = false;
            AddError(TEXT("static_runtime_actor"), TEXT("spawn_static_target_failed"));
        }

        bool bSpawnedStaticSphere = false;
        AActor* StaticSphereTarget = FindFirstActorByTagTextInWorld(World, StaticSphereTagText);
        if (StaticSphereTarget)
        {
            StaticSphereTarget->Destroy();
            StaticSphereTarget = nullptr;
        }
        if (!StaticSphereTarget)
        {
            StaticSphereTarget = World->SpawnActor<AStaticMeshActor>(
                AStaticMeshActor::StaticClass(),
                FVector(650.0, 140.0, 95.0),
                FRotator::ZeroRotator,
                Params);
            bSpawnedStaticSphere = StaticSphereTarget != nullptr;
        }
        if (StaticSphereTarget)
        {
            MakeActorComponentsMovable(StaticSphereTarget);
            StaticSphereTarget->SetActorLocation(
                FVector(650.0, 140.0, 95.0),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            AddTagToActor(StaticSphereTarget, StaticSphereTag);
            FString MeshError;
            if (!EnsureVisibleMeshComponent(
                    StaticSphereTarget,
                    SphereMesh,
                    nullptr,
                    FLinearColor::White,
                    TEXT("SimWorldObservationStaticSphereMesh"),
                    FVector::ZeroVector,
                    FVector(1.4, 1.4, 1.4),
                    MeshError))
            {
                bAllOk = false;
                AddError(TEXT("static_runtime_actor"), MeshError);
            }
            AddRecord(
                TEXT("static_runtime_actor_sphere"),
                StaticSphereTagText,
                StaticSphereTarget,
                bSpawnedStaticSphere,
                TEXT("AStaticMeshActor runtime spawn with engine sphere mesh"));
        }
        else
        {
            bAllOk = false;
            AddError(TEXT("static_runtime_actor"), TEXT("spawn_static_sphere_target_failed"));
        }

        bool bSpawnedPedestrian = false;
        ASpPedestrianAgentBase* PedestrianTarget = nullptr;
        if (AActor* ExistingPedestrian = FindFirstActorByTagTextInWorld(World, PedestrianTagText))
        {
            ExistingPedestrian->Destroy();
        }
        if (!PedestrianTarget)
        {
            PedestrianTarget = World->SpawnActor<ASpPedestrianAgentBase>(
                ASpPedestrianAgentBase::StaticClass(),
                FVector(850.0, 180.0, 120.0),
                FRotator(0.0, -15.0, 0.0),
                Params);
            bSpawnedPedestrian = PedestrianTarget != nullptr;
        }
        if (PedestrianTarget)
        {
            MakeActorComponentsMovable(PedestrianTarget);
            PedestrianTarget->SetActorLocationAndRotation(
                FVector(850.0, 180.0, 120.0),
                FRotator(0.0, -15.0, 0.0),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            PedestrianTarget->AgentTag = PedestrianTag;
            AddTagToActor(PedestrianTarget, PedestrianTag);
            FString MeshError;
            if (!EnsureVisibleMeshComponent(
                    PedestrianTarget,
                    CubeMesh,
                    nullptr,
                    FLinearColor::White,
                    TEXT("SimWorldObservationPedestrianMesh"),
                    FVector(0.0, 0.0, 80.0),
                    FVector(0.7, 0.7, 1.6),
                    MeshError))
            {
                bAllOk = false;
                AddError(TEXT("pedestrian_runtime_actor"), MeshError);
            }
            AddRecord(
                TEXT("pedestrian_runtime_actor"),
                PedestrianTagText,
                PedestrianTarget,
                bSpawnedPedestrian,
                TEXT("ASpPedestrianAgentBase runtime spawn with visible fixture mesh"));
        }
        else
        {
            bAllOk = false;
            AddError(TEXT("pedestrian_runtime_actor"), TEXT("spawn_pedestrian_target_failed"));
        }

        bool bSpawnedVehicle = false;
        AActor* VehicleTarget = FindFirstActorByTagTextInWorld(World, VehicleTagText);
        if (VehicleTarget)
        {
            VehicleTarget->Destroy();
            VehicleTarget = nullptr;
        }
        if (!VehicleTarget)
        {
            VehicleTarget = World->SpawnActor<AStaticMeshActor>(
                AStaticMeshActor::StaticClass(),
                FVector(520.0, 0.0, 95.0),
                FRotator(0.0, 0.0, 0.0),
                Params);
            bSpawnedVehicle = VehicleTarget != nullptr;
        }
        if (VehicleTarget)
        {
            MakeActorComponentsMovable(VehicleTarget);
            VehicleTarget->SetActorLocationAndRotation(
                FVector(520.0, 0.0, 95.0),
                FRotator(0.0, 0.0, 0.0),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            AddTagToActor(VehicleTarget, VehicleTag);
            FString MeshError;
            if (!EnsureVisibleMeshComponent(
                    VehicleTarget,
                    CubeMesh,
                    nullptr,
                    FLinearColor::White,
                    TEXT("SimWorldObservationVehicleMesh"),
                    FVector(0.0, 0.0, 70.0),
                    FVector(1.8, 0.9, 0.7),
                    MeshError))
            {
                bAllOk = false;
                AddError(TEXT("vehicle_runtime_actor"), MeshError);
            }
            AddRecord(
                TEXT("vehicle_runtime_actor"),
                VehicleTagText,
                VehicleTarget,
                bSpawnedVehicle,
                TEXT("vehicle-shaped AStaticMeshActor runtime spawn with visible fixture mesh"));
        }
        else
        {
            bAllOk = false;
            AddError(TEXT("vehicle_runtime_actor"), TEXT("spawn_vehicle_target_failed"));
        }

        TArray<TSharedPtr<FJsonValue>> AgentTagValues;
        AgentTagValues.Add(MakeShared<FJsonValueString>(CameraTagText));
        Root->SetArrayField(TEXT("agent_tags"), AgentTagValues);

        TArray<TSharedPtr<FJsonValue>> TargetTagValues;
        TargetTagValues.Add(MakeShared<FJsonValueString>(StaticTagText));
        TargetTagValues.Add(MakeShared<FJsonValueString>(StaticSphereTagText));
        TargetTagValues.Add(MakeShared<FJsonValueString>(PedestrianTagText));
        TargetTagValues.Add(MakeShared<FJsonValueString>(VehicleTagText));
        Root->SetArrayField(TEXT("target_tags"), TargetTagValues);

        TSharedPtr<FJsonObject> CaseTags = MakeShared<FJsonObject>();
        TArray<TSharedPtr<FJsonValue>> StaticCaseTagValues;
        StaticCaseTagValues.Add(MakeShared<FJsonValueString>(StaticTagText));
        StaticCaseTagValues.Add(MakeShared<FJsonValueString>(StaticSphereTagText));
        CaseTags->SetArrayField(TEXT("static_runtime_actor"), StaticCaseTagValues);
        CaseTags->SetStringField(TEXT("pedestrian_runtime_actor"), PedestrianTagText);
        CaseTags->SetStringField(TEXT("vehicle_runtime_actor"), VehicleTagText);
        Root->SetObjectField(TEXT("case_target_tags"), CaseTags);

        Root->SetBoolField(TEXT("success"), bAllOk && ErrorValues.IsEmpty());
    }

    Root->SetArrayField(TEXT("records"), RecordValues);
    Root->SetArrayField(TEXT("errors"), ErrorValues);

    FString Output;
    const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
    FJsonSerializer::Serialize(Root, Writer);
    return Output;
}

void USpCameraCapturePool::ApplyTierPreset(USceneCaptureComponent2D* Capture,
                                           ESpCameraQualityTier Tier)
{
    if (!Capture) return;

    const FTierPreset& Preset = GetPreset(Tier);

    // Render target — create-or-resize as needed.
    UTextureRenderTarget2D* RT = Capture->TextureTarget;
    if (!RT || RT->SizeX != Preset.Side || RT->SizeY != Preset.Side)
    {
        if (!RT)
        {
            RT = NewObject<UTextureRenderTarget2D>(Capture);
            RT->RenderTargetFormat = RTF_RGBA8;
            RT->ClearColor = FLinearColor::Black;
            RT->bAutoGenerateMips = false;
            RT->bGPUSharedFlag = false;
        }
        RT->InitAutoFormat(Preset.Side, Preset.Side);
        RT->UpdateResourceImmediate(true);
        Capture->TextureTarget = RT;
    }

    // Show flags: turn off the expensive features that LLM/Inspect
    // tiers don't need.  Hero leaves defaults intact.
    FEngineShowFlags& ShowFlags = Capture->ShowFlags;
    if (Preset.bDisableBloom)       ShowFlags.SetBloom(false);
    if (Preset.bDisableMotionBlur)  ShowFlags.SetMotionBlur(false);
    if (Preset.bDisableToneMapper)  ShowFlags.SetTonemapper(false);
    if (Preset.bDisableSSAO)        ShowFlags.SetAmbientOcclusion(false);

    // LLM + Inspect: also kill shadows + reflections for speed.
    if (Tier != ESpCameraQualityTier::Hero)
    {
        ShowFlags.SetDynamicShadows(false);
        ShowFlags.SetReflectionEnvironment(false);
        ShowFlags.SetLensFlares(false);
    }
}

void USpCameraCapturePool::ProcessRequest(const FSpCaptureRequest& Request,
                                          USceneCaptureComponent2D* Capture)
{
    FSpCaptureResult Result;
    Result.SequenceNumber = Request.SequenceNumber;
    Result.RequestId = Request.RequestId;

    if (!Capture || !Request.AttachTo)
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("null capture component or AttachTo actor");
        ResultsBySequence.Add(Request.SequenceNumber, Result);
        return;
    }

    ApplyTierPreset(Capture, Request.Tier);

    // Attach to the target actor's root; this puts the capture at the
    // actor's transform.  Future: per-tier offset + per-agent pose
    // (head bone for pedestrians, hood for vehicles).
    Capture->AttachToComponent(
        Request.AttachTo->GetRootComponent(),
        FAttachmentTransformRules::SnapToTargetIncludingScale);

    Capture->CaptureScene();

    // Synchronous readback.  This stalls the render thread until the
    // capture completes — fine for Phase 3.b; async FRHIGPUTextureReadback
    // is a future optimization.
    UTextureRenderTarget2D* RT = Capture->TextureTarget;
    if (!RT || !RT->GetResource())
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("render target has no resource");
        ResultsBySequence.Add(Request.SequenceNumber, Result);
        return;
    }

    FTextureRenderTargetResource* RTResource = RT->GameThread_GetRenderTargetResource();
    TArray<FColor> Bitmap;
    FReadSurfaceDataFlags Flags(RCM_UNorm, CubeFace_MAX);
    Flags.SetLinearToGamma(false);
    if (!RTResource->ReadPixels(Bitmap, Flags))
    {
        Result.bSuccess = false;
        Result.ErrorMessage = TEXT("RTResource->ReadPixels failed");
        ResultsBySequence.Add(Request.SequenceNumber, Result);
        return;
    }

    Result.Width = RT->SizeX;
    Result.Height = RT->SizeY;
    Result.PixelData.SetNumUninitialized(Bitmap.Num() * 4);
    // FColor is BGRA already; just memcpy.
    FMemory::Memcpy(Result.PixelData.GetData(), Bitmap.GetData(),
                    Bitmap.Num() * sizeof(FColor));
    Result.bSuccess = true;
    ResultsBySequence.Add(Request.SequenceNumber, Result);
}

void USpCameraCapturePool::Tick(float DeltaTime)
{
    // Drain up to K (=Pool.Num()) pending requests per tick — one per
    // pool slot.  FIFO order.  Per-tick budget = pool size.
    const int32 K = FMath::Min(Pool.Num(), PendingRequests.Num());
    for (int32 i = 0; i < K; ++i)
    {
        const FSpCaptureRequest Request = PendingRequests[0];  // copy
        PendingRequests.RemoveAt(0);
        USceneCaptureComponent2D* Capture = Pool[i].Get();
        ProcessRequest(Request, Capture);
    }

    // Queue overflow protection (carried from skeleton).
    constexpr int32 kMaxQueueDepth = 4096;
    if (PendingRequests.Num() > kMaxQueueDepth)
    {
        const int32 OverflowCount = PendingRequests.Num() - kMaxQueueDepth;
        PendingRequests.RemoveAt(0, OverflowCount);
        UE_LOG(LogTemp, Warning,
               TEXT("USpCameraCapturePool: queue overflow — dropped %d oldest request(s) "
                    "(depth was %d, max=%d).  Increase pool size or LLM batch rate."),
               OverflowCount, OverflowCount + kMaxQueueDepth, kMaxQueueDepth);
    }

    // Auto-evict stale results that nobody polled (>10s old).  Phase 3.b
    // gives 1 result slot per request; without eviction a forgotten poll
    // would leak forever.  Future: track per-result timestamps + age out.
    // For now we cap the result map size as a crude defense.
    constexpr int32 kMaxResults = 1024;
    if (ResultsBySequence.Num() > kMaxResults)
    {
        // Drop the lowest sequence numbers — oldest by construction.
        TArray<int64> Keys;
        ResultsBySequence.GetKeys(Keys);
        Keys.Sort();
        const int32 ToDrop = ResultsBySequence.Num() - kMaxResults;
        for (int32 i = 0; i < ToDrop; ++i)
        {
            ResultsBySequence.Remove(Keys[i]);
        }
        UE_LOG(LogTemp, Warning,
               TEXT("USpCameraCapturePool: result map evicted %d oldest entries "
                    "(cap=%d).  Increase poll rate."), ToDrop, kMaxResults);
    }
}

TStatId USpCameraCapturePool::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(USpCameraCapturePool, STATGROUP_Tickables);
}
