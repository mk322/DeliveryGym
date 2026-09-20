// Copyright (c) 2026 The SimWorld Development Team. Licensed under the MIT License.

#include "SpPixelGoalSubsystem.h"

#include "AIController.h"
#include "Camera/CameraTypes.h"
#include "Components/CapsuleComponent.h"
#include "Components/DirectionalLightComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Components/MeshComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/SkyAtmosphereComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/DirectionalLight.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/SpringArmComponent.h"
#include "Kismet/GameplayStatics.h"
#include "Materials/MaterialInterface.h"
#include "NavigationPath.h"
#include "NavigationSystem.h"
#include "NavMesh/NavMeshBoundsVolume.h"
#include "NavMesh/RecastNavMesh.h"
#include "SceneView.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "SpCameraCapturePool.h"
#include "SpHumanoidAgent.h"
#include "SpNavMeshHelper.h"
#include "SpUnrealTypes/SpSceneCaptureComponent2D.h"

#include <initializer_list>
#include <limits>

namespace
{
    constexpr int32 MaxRetainedSnapshots = 256;
    constexpr float CalibrationFloorTopZCm = 50010.f;
    constexpr double ParisPocCaptureWarmupSeconds = 6.0;
    constexpr TCHAR ParisPocCaptureContractVersion[] =
        TEXT("paris-agent-native-rgb-v2");
    constexpr float ParisPocExposureBias = 0.5f;
    constexpr float ParisPocExposureMinEv100 = 6.0f;
    constexpr float ParisPocExposureMaxEv100 = 12.0f;
    constexpr float ParisPocExposureSpeedUp = 10.0f;
    constexpr float ParisPocExposureSpeedDown = 10.0f;
    constexpr float ParisPocMieScatteringScale = 0.003996f;
    const FName CalibrationAgentTag(TEXT("PixelGoalCalibrationAgent"));
    const FName CalibrationFloorTag(TEXT("PixelGoalCalibrationFloor"));
    const FName CalibrationHelperTag(TEXT("PixelGoalCalibrationNavHelper"));
    const FName CalibrationLightTag(TEXT("PixelGoalCalibrationLight"));
    const FName CalibrationWallTag(TEXT("PixelGoalCalibrationWall"));
    const FName ParisPocAgentTag(TEXT("PixelGoalParisPocAgent"));
    const FName ParisPocHelperTag(TEXT("PixelGoalParisPocNavHelper"));
    const FString ParisPocScene(
        TEXT("/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints"));
    const FString ParisPocMapLeaf(TEXT("ParisCity_FinalBlueprints"));

    struct FViewPairCameraRigState
    {
        USpSceneCaptureComponent2D* Component = nullptr;
        USceneComponent* AttachParent = nullptr;
        UTextureRenderTarget2D* TextureTarget = nullptr;
        FTransform RelativeTransform = FTransform::Identity;
        FTransform WorldTransform = FTransform::Identity;
        FMatrix CustomProjectionMatrix = FMatrix::Identity;
        int32 Width = 0;
        int32 Height = 0;
        int32 TargetWidth = 0;
        int32 TargetHeight = 0;
        float FovDegrees = 0.f;
        int32 ProjectionType = 0;
        int32 CaptureSource = 0;
        int32 RenderTargetFormat = 0;
        int32 ChannelDataType = 0;
        int32 NumChannels = 0;
        float Overscan = 0.f;
        float PostProcessBlendWeight = 0.f;
        int32 AutoExposureMethod = 0;
        float AutoExposureMinBrightness = 0.f;
        float AutoExposureMaxBrightness = 0.f;
        float AutoExposureSpeedUp = 0.f;
        float AutoExposureSpeedDown = 0.f;
        float AutoExposureBias = 0.f;
        bool bUseCustomProjection = false;
        bool bOverrideCustomNearClip = false;
        bool bEnableClipPlane = false;
        bool bEnableFirstPersonFov = false;
        bool bEnableFirstPersonScale = false;
        bool bMainViewCamera = false;
        bool bMainViewResolution = false;
        bool bRenderInMainRenderer = false;
        bool bAlwaysPersist = false;
        bool bCaptureEveryFrame = false;
        bool bCaptureOnMovement = false;
        bool bOverrideRenderTargetFormat = false;
        bool bOverrideAutoExposureMethod = false;
        bool bOverrideAutoExposureMinBrightness = false;
        bool bOverrideAutoExposureMaxBrightness = false;
        bool bOverrideAutoExposureSpeedUp = false;
        bool bOverrideAutoExposureSpeedDown = false;
        bool bOverrideAutoExposureBias = false;

        explicit FViewPairCameraRigState(USpSceneCaptureComponent2D* InComponent)
            : Component(InComponent)
        {
            if (!IsValid(InComponent))
            {
                return;
            }
            AttachParent = InComponent->GetAttachParent();
            TextureTarget = InComponent->TextureTarget;
            RelativeTransform = InComponent->GetRelativeTransform();
            WorldTransform = InComponent->GetComponentTransform();
            CustomProjectionMatrix = InComponent->CustomProjectionMatrix;
            Width = InComponent->Width;
            Height = InComponent->Height;
            TargetWidth = IsValid(TextureTarget) ? TextureTarget->SizeX : 0;
            TargetHeight = IsValid(TextureTarget) ? TextureTarget->SizeY : 0;
            FovDegrees = InComponent->FOVAngle;
            ProjectionType = static_cast<int32>(InComponent->ProjectionType);
            CaptureSource = static_cast<int32>(InComponent->CaptureSource);
            RenderTargetFormat =
                static_cast<int32>(InComponent->TextureRenderTargetFormat);
            ChannelDataType = static_cast<int32>(InComponent->ChannelDataType);
            NumChannels = InComponent->NumChannelsPerPixel;
            Overscan = InComponent->Overscan;
            PostProcessBlendWeight = InComponent->PostProcessBlendWeight;
            AutoExposureMethod = static_cast<int32>(
                InComponent->PostProcessSettings.AutoExposureMethod.GetValue());
            AutoExposureMinBrightness =
                InComponent->PostProcessSettings.AutoExposureMinBrightness;
            AutoExposureMaxBrightness =
                InComponent->PostProcessSettings.AutoExposureMaxBrightness;
            AutoExposureSpeedUp =
                InComponent->PostProcessSettings.AutoExposureSpeedUp;
            AutoExposureSpeedDown =
                InComponent->PostProcessSettings.AutoExposureSpeedDown;
            AutoExposureBias =
                InComponent->PostProcessSettings.AutoExposureBias;
            bUseCustomProjection = InComponent->bUseCustomProjectionMatrix;
            bOverrideCustomNearClip =
                InComponent->bOverride_CustomNearClippingPlane;
            bEnableClipPlane = InComponent->bEnableClipPlane;
            bEnableFirstPersonFov =
                InComponent->bEnableFirstPersonFieldOfView;
            bEnableFirstPersonScale = InComponent->bEnableFirstPersonScale;
            bMainViewCamera = InComponent->bMainViewCamera;
            bMainViewResolution = InComponent->bMainViewResolution;
            bRenderInMainRenderer = InComponent->bRenderInMainRenderer;
            bAlwaysPersist = InComponent->bAlwaysPersistRenderingState;
            bCaptureEveryFrame = InComponent->bCaptureEveryFrame;
            bCaptureOnMovement = InComponent->bCaptureOnMovement;
            bOverrideRenderTargetFormat =
                InComponent->bOverrideTextureRenderTargetFormat;
            bOverrideAutoExposureMethod =
                InComponent->PostProcessSettings.bOverride_AutoExposureMethod;
            bOverrideAutoExposureMinBrightness =
                InComponent->PostProcessSettings.bOverride_AutoExposureMinBrightness;
            bOverrideAutoExposureMaxBrightness =
                InComponent->PostProcessSettings.bOverride_AutoExposureMaxBrightness;
            bOverrideAutoExposureSpeedUp =
                InComponent->PostProcessSettings.bOverride_AutoExposureSpeedUp;
            bOverrideAutoExposureSpeedDown =
                InComponent->PostProcessSettings.bOverride_AutoExposureSpeedDown;
            bOverrideAutoExposureBias =
                InComponent->PostProcessSettings.bOverride_AutoExposureBias;
        }

        bool Matches(USpSceneCaptureComponent2D* InComponent) const
        {
            return IsValid(InComponent) && InComponent == Component &&
                IsValid(AttachParent) &&
                InComponent->GetAttachParent() == AttachParent &&
                IsValid(TextureTarget) &&
                InComponent->TextureTarget == TextureTarget &&
                InComponent->IsInitialized() &&
                InComponent->GetRelativeTransform().Equals(RelativeTransform, 0.0) &&
                InComponent->GetComponentTransform().Equals(WorldTransform, 0.0) &&
                InComponent->CustomProjectionMatrix.Equals(
                    CustomProjectionMatrix, 0.f) &&
                InComponent->Width == Width && InComponent->Height == Height &&
                TextureTarget->SizeX == TargetWidth &&
                TextureTarget->SizeY == TargetHeight &&
                InComponent->FOVAngle == FovDegrees &&
                static_cast<int32>(InComponent->ProjectionType) == ProjectionType &&
                static_cast<int32>(InComponent->CaptureSource) == CaptureSource &&
                static_cast<int32>(InComponent->TextureRenderTargetFormat) ==
                    RenderTargetFormat &&
                static_cast<int32>(InComponent->ChannelDataType) ==
                    ChannelDataType &&
                InComponent->NumChannelsPerPixel == NumChannels &&
                InComponent->Overscan == Overscan &&
                InComponent->PostProcessBlendWeight == PostProcessBlendWeight &&
                static_cast<int32>(InComponent->PostProcessSettings.
                    AutoExposureMethod.GetValue()) == AutoExposureMethod &&
                InComponent->PostProcessSettings.AutoExposureMinBrightness ==
                    AutoExposureMinBrightness &&
                InComponent->PostProcessSettings.AutoExposureMaxBrightness ==
                    AutoExposureMaxBrightness &&
                InComponent->PostProcessSettings.AutoExposureSpeedUp ==
                    AutoExposureSpeedUp &&
                InComponent->PostProcessSettings.AutoExposureSpeedDown ==
                    AutoExposureSpeedDown &&
                InComponent->PostProcessSettings.AutoExposureBias ==
                    AutoExposureBias &&
                InComponent->bUseCustomProjectionMatrix == bUseCustomProjection &&
                InComponent->bOverride_CustomNearClippingPlane ==
                    bOverrideCustomNearClip &&
                InComponent->bEnableClipPlane == bEnableClipPlane &&
                InComponent->bEnableFirstPersonFieldOfView ==
                    bEnableFirstPersonFov &&
                InComponent->bEnableFirstPersonScale ==
                    bEnableFirstPersonScale &&
                InComponent->bMainViewCamera == bMainViewCamera &&
                InComponent->bMainViewResolution == bMainViewResolution &&
                InComponent->bRenderInMainRenderer == bRenderInMainRenderer &&
                InComponent->bAlwaysPersistRenderingState == bAlwaysPersist &&
                InComponent->bCaptureEveryFrame == bCaptureEveryFrame &&
                InComponent->bCaptureOnMovement == bCaptureOnMovement &&
                InComponent->bOverrideTextureRenderTargetFormat ==
                    bOverrideRenderTargetFormat &&
                InComponent->PostProcessSettings.bOverride_AutoExposureMethod ==
                    bOverrideAutoExposureMethod &&
                InComponent->PostProcessSettings.
                    bOverride_AutoExposureMinBrightness ==
                    bOverrideAutoExposureMinBrightness &&
                InComponent->PostProcessSettings.
                    bOverride_AutoExposureMaxBrightness ==
                    bOverrideAutoExposureMaxBrightness &&
                InComponent->PostProcessSettings.bOverride_AutoExposureSpeedUp ==
                    bOverrideAutoExposureSpeedUp &&
                InComponent->PostProcessSettings.bOverride_AutoExposureSpeedDown ==
                    bOverrideAutoExposureSpeedDown &&
                InComponent->PostProcessSettings.bOverride_AutoExposureBias ==
                    bOverrideAutoExposureBias;
        }
    };

#if WITH_DEV_AUTOMATION_TESTS
    int32 GeometryTraceEntryCount = 0;
#endif

    bool IsParisPocMapName(const FString& MapName)
    {
        if (MapName == ParisPocMapLeaf)
        {
            return true;
        }
        const FString PiePrefix(TEXT("UEDPIE_"));
        if (!MapName.StartsWith(PiePrefix))
        {
            return false;
        }
        const int32 Separator = MapName.Find(
            TEXT("_"),
            ESearchCase::CaseSensitive,
            ESearchDir::FromStart,
            PiePrefix.Len());
        const FString PieInstance = Separator == INDEX_NONE
            ? FString()
            : MapName.Mid(PiePrefix.Len(), Separator - PiePrefix.Len());
        return !PieInstance.IsEmpty() && PieInstance.IsNumeric() &&
            MapName.Mid(Separator + 1) == ParisPocMapLeaf;
    }

    bool PrepareParisPocCapture(
        ASpHumanoidAgent* Agent,
        bool bEnableRearCamera)
    {
        if (!Agent || !Agent->SceneCapture)
        {
            return false;
        }
        if (!bEnableRearCamera)
        {
            if (USpSceneCaptureComponent2D* Rear =
                    Agent->GetObservationCamera(TEXT("rear"));
                Rear && Rear->IsInitialized())
            {
                Agent->TerminateObservationCamera(TEXT("rear"));
            }
        }
        const FString Views[] = {TEXT("front"), TEXT("rear")};
        const int32 ViewCount = bEnableRearCamera ? 2 : 1;
        for (int32 Index = 0; Index < ViewCount; ++Index)
        {
            USpSceneCaptureComponent2D* Capture =
                Agent->GetObservationCamera(Views[Index]);
            if (!Capture)
            {
                return false;
            }
            const bool bSizeChanged = Capture->Width != 640 ||
                Capture->Height != 360 ||
                !FMath::IsNearlyEqual(Capture->FOVAngle, 90.f, 0.01f);
            if (Capture->IsInitialized() && bSizeChanged)
            {
                Agent->TerminateObservationCamera(Views[Index]);
            }
            if (!SpPixelGoal::ConfigurePersistentParisCapture(Capture) ||
                !Agent->ConfigureObservationCamera(Views[Index], 640, 360, 90.f))
            {
                return false;
            }
            if (!Capture->IsInitialized())
            {
                Agent->InitializeObservationCamera(Views[Index]);
            }
            if (!Capture->IsInitialized() || !Capture->TextureTarget)
            {
                return false;
            }
        }
        return true;
    }

    bool PrepareParisPocAtmosphere(
        UWorld* World,
        float& OutMieBefore,
        float& OutMieAfter)
    {
        if (!World)
        {
            return false;
        }
        for (TActorIterator<ASkyAtmosphere> It(World); It; ++It)
        {
            USkyAtmosphereComponent* Atmosphere = It->GetComponent();
            if (!Atmosphere)
            {
                continue;
            }
            OutMieBefore = Atmosphere->MieScatteringScale;
            if (!SpPixelGoal::ConfigureParisSkyAtmosphere(Atmosphere))
            {
                return false;
            }
            OutMieAfter = Atmosphere->MieScatteringScale;
            return true;
        }
        return false;
    }

    TArray<TSharedPtr<FJsonValue>> JsonVector(const FVector& Value)
    {
        TArray<TSharedPtr<FJsonValue>> Result;
        Result.Add(MakeShared<FJsonValueNumber>(Value.X));
        Result.Add(MakeShared<FJsonValueNumber>(Value.Y));
        Result.Add(MakeShared<FJsonValueNumber>(Value.Z));
        return Result;
    }

    TArray<TSharedPtr<FJsonValue>> JsonVectorPath(
        const TArray<FVector>& Points)
    {
        TArray<TSharedPtr<FJsonValue>> Result;
        Result.Reserve(Points.Num());
        for (const FVector& Point : Points)
        {
            Result.Add(MakeShared<FJsonValueArray>(JsonVector(Point)));
        }
        return Result;
    }

    TArray<TSharedPtr<FJsonValue>> JsonRotator(const FRotator& Value)
    {
        TArray<TSharedPtr<FJsonValue>> Result;
        Result.Add(MakeShared<FJsonValueNumber>(Value.Pitch));
        Result.Add(MakeShared<FJsonValueNumber>(Value.Yaw));
        Result.Add(MakeShared<FJsonValueNumber>(Value.Roll));
        return Result;
    }

    FString SerializeJson(const TSharedRef<FJsonObject>& Object)
    {
        FString Output;
        const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Output);
        FJsonSerializer::Serialize(Object, Writer);
        return Output;
    }

    TSharedPtr<FJsonObject> ParseJson(const FString& RequestJson)
    {
        TSharedPtr<FJsonObject> Object;
        const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(RequestJson);
        if (!FJsonSerializer::Deserialize(Reader, Object) || !Object.IsValid())
        {
            return nullptr;
        }
        return Object;
    }

    bool TryGetExactJsonString(
        const TSharedPtr<FJsonObject>& Object,
        const TCHAR* Field,
        FString& OutValue)
    {
        OutValue.Reset();
        const TSharedPtr<FJsonValue> Value = Object.IsValid()
            ? Object->TryGetField(Field)
            : nullptr;
        return Value.IsValid() && Value->Type == EJson::String &&
            Value->TryGetString(OutValue);
    }

    bool TryGetExactJsonNumber(
        const TSharedPtr<FJsonObject>& Object,
        const TCHAR* Field,
        double& OutValue)
    {
        const TSharedPtr<FJsonValue> Value = Object.IsValid()
            ? Object->TryGetField(Field)
            : nullptr;
        return Value.IsValid() && Value->Type == EJson::Number &&
            Value->TryGetNumber(OutValue) && FMath::IsFinite(OutValue);
    }

    bool TryGetExactJsonBool(
        const TSharedPtr<FJsonObject>& Object,
        const TCHAR* Field,
        bool& OutValue)
    {
        const TSharedPtr<FJsonValue> Value = Object.IsValid()
            ? Object->TryGetField(Field)
            : nullptr;
        return Value.IsValid() && Value->Type == EJson::Boolean &&
            Value->TryGetBool(OutValue);
    }

    FString JsonString(
        const TSharedPtr<FJsonObject>& Object,
        const TCHAR* Field,
        const FString& Default = FString())
    {
        FString Value;
        return Object.IsValid() && Object->TryGetStringField(Field, Value)
            ? Value
            : Default;
    }

    double JsonNumber(
        const TSharedPtr<FJsonObject>& Object,
        const TCHAR* Field,
        double Default)
    {
        double Value = Default;
        if (Object.IsValid())
        {
            Object->TryGetNumberField(Field, Value);
        }
        return Value;
    }

    bool IsFinite(const FVector2D& Value)
    {
        return FMath::IsFinite(Value.X) && FMath::IsFinite(Value.Y);
    }

    bool JsonVectorField(
        const TSharedPtr<FJsonObject>& Object,
        const TCHAR* Field,
        FVector& OutValue)
    {
        const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
        if (!Object.IsValid() ||
            !Object->TryGetArrayField(Field, Values) ||
            !Values ||
            Values->Num() != 3)
        {
            return false;
        }
        double Components[3] = {};
        for (int32 Index = 0; Index < 3; ++Index)
        {
            if (!(*Values)[Index].IsValid() ||
                !(*Values)[Index]->TryGetNumber(Components[Index]) ||
                !FMath::IsFinite(Components[Index]))
            {
                return false;
            }
        }
        OutValue = FVector(Components[0], Components[1], Components[2]);
        return true;
    }

    bool JsonVectorValue(
        const TSharedPtr<FJsonValue>& Value,
        FVector& OutValue)
    {
        const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
        if (!Value.IsValid() || !Value->TryGetArray(Values) ||
            !Values || Values->Num() != 3)
        {
            return false;
        }
        double Components[3] = {};
        for (int32 Index = 0; Index < 3; ++Index)
        {
            if (!(*Values)[Index].IsValid() ||
                (*Values)[Index]->Type != EJson::Number ||
                !(*Values)[Index]->TryGetNumber(Components[Index]) ||
                !FMath::IsFinite(Components[Index]))
            {
                return false;
            }
        }
        OutValue = FVector(Components[0], Components[1], Components[2]);
        return true;
    }

    bool HasExactJsonFields(
        const TSharedPtr<FJsonObject>& Object,
        std::initializer_list<const TCHAR*> Fields)
    {
        if (!Object.IsValid() || Object->Values.Num() != static_cast<int32>(Fields.size()))
        {
            return false;
        }
        for (const TCHAR* Field : Fields)
        {
            if (!Object->HasField(Field))
            {
                return false;
            }
        }
        return true;
    }

    FString NavNodeRefString(NavNodeRef Ref)
    {
        return FString::Printf(
            TEXT("%llu"), static_cast<unsigned long long>(Ref));
    }

    bool IsRealParisCrosswalkLabel(const FString& Label)
    {
        // Deliberately excludes unprefixed Crossswalk_* decoration and the
        // PR_CrossswalkBike_* meshes.  A trusted pedestrian crossing must be
        // one of the scene-authored PR_Crossswalk_* assets requested by the
        // benchmark contract.
        return Label.StartsWith(
            TEXT("PR_Crossswalk_"), ESearchCase::CaseSensitive);
    }

    struct FParisCrosswalkComponent
    {
        FString Label;
        FString ActorName;
        TWeakObjectPtr<UStaticMeshComponent> Component;
        FVector LocalMin = FVector::ZeroVector;
        FVector LocalMax = FVector::ZeroVector;
    };

    TArray<FParisCrosswalkComponent> CollectParisCrosswalkComponents(UWorld* World)
    {
        TArray<FParisCrosswalkComponent> Result;
        if (!World)
        {
            return Result;
        }
        for (TActorIterator<AActor> It(World); It; ++It)
        {
            AActor* Actor = *It;
            if (!IsValid(Actor))
            {
                continue;
            }
            const FString Label = Actor->GetActorNameOrLabel();
            if (!IsRealParisCrosswalkLabel(Label))
            {
                continue;
            }
            TArray<UStaticMeshComponent*> Components;
            Actor->GetComponents<UStaticMeshComponent>(Components);
            for (UStaticMeshComponent* Component : Components)
            {
                if (!IsValid(Component) || !IsValid(Component->GetStaticMesh()))
                {
                    continue;
                }
                FParisCrosswalkComponent Row;
                Row.Label = Label;
                Row.ActorName = Actor->GetName();
                Row.Component = Component;
                Component->GetLocalBounds(Row.LocalMin, Row.LocalMax);
                Result.Add(MoveTemp(Row));
            }
        }
        Result.Sort([](
            const FParisCrosswalkComponent& A,
            const FParisCrosswalkComponent& B)
        {
            if (A.Label != B.Label)
            {
                return A.Label < B.Label;
            }
            const UStaticMeshComponent* AComponent = A.Component.Get();
            const UStaticMeshComponent* BComponent = B.Component.Get();
            const FString AName = AComponent ? AComponent->GetName() : FString();
            const FString BName = BComponent ? BComponent->GetName() : FString();
            return AName < BName;
        });
        return Result;
    }

    TArray<FString> CrosswalkLabelsAtPoint(
        const FVector& WorldPoint,
        const TArray<FParisCrosswalkComponent>& Crosswalks,
        float HorizontalToleranceCm = 1.f)
    {
        TArray<FString> Labels;
        for (const FParisCrosswalkComponent& Row : Crosswalks)
        {
            const UStaticMeshComponent* Component = Row.Component.Get();
            if (!Component)
            {
                continue;
            }
            const FVector Local =
                Component->GetComponentTransform().InverseTransformPosition(WorldPoint);
            if (Local.X >= Row.LocalMin.X - HorizontalToleranceCm &&
                Local.X <= Row.LocalMax.X + HorizontalToleranceCm &&
                Local.Y >= Row.LocalMin.Y - HorizontalToleranceCm &&
                Local.Y <= Row.LocalMax.Y + HorizontalToleranceCm &&
                Local.Z >= Row.LocalMin.Z - 100.f &&
                Local.Z <= Row.LocalMax.Z + 200.f)
            {
                Labels.AddUnique(Row.Label);
            }
        }
        Labels.Sort();
        return Labels;
    }

    TArray<TSharedPtr<FJsonValue>> JsonStrings(const TArray<FString>& Values)
    {
        TArray<TSharedPtr<FJsonValue>> Result;
        Result.Reserve(Values.Num());
        for (const FString& Value : Values)
        {
            Result.Add(MakeShared<FJsonValueString>(Value));
        }
        return Result;
    }

    TSharedRef<FJsonObject> GroundHitJson(const FHitResult& Hit)
    {
        const TSharedRef<FJsonObject> Row = MakeShared<FJsonObject>();
        const AActor* Actor = Hit.GetActor();
        const UPrimitiveComponent* Primitive = Hit.GetComponent();
        Row->SetStringField(
            TEXT("actor_label"), Actor ? Actor->GetActorNameOrLabel() : FString());
        Row->SetStringField(
            TEXT("actor_name"), Actor ? Actor->GetName() : FString());
        Row->SetStringField(
            TEXT("actor_class"),
            Actor && Actor->GetClass() ? Actor->GetClass()->GetPathName() : FString());
        Row->SetStringField(
            TEXT("component_name"), Primitive ? Primitive->GetName() : FString());
        Row->SetStringField(
            TEXT("component_class"),
            Primitive && Primitive->GetClass()
                ? Primitive->GetClass()->GetPathName()
                : FString());
        Row->SetArrayField(TEXT("impact_point_cm"), JsonVector(Hit.ImpactPoint));
        Row->SetArrayField(TEXT("impact_normal"), JsonVector(Hit.ImpactNormal));
        Row->SetNumberField(TEXT("face_index"), Hit.FaceIndex);
        Row->SetNumberField(TEXT("element_index"), Hit.ElementIndex);

        int32 SurfaceSectionIndex = INDEX_NONE;
        const UMaterialInterface* SurfaceMaterial = Primitive
            ? Primitive->GetMaterialFromCollisionFaceIndex(
                Hit.FaceIndex, SurfaceSectionIndex)
            : nullptr;
        Row->SetNumberField(
            TEXT("surface_section_index"), SurfaceSectionIndex);
        Row->SetStringField(
            TEXT("surface_material_path"),
            SurfaceMaterial ? SurfaceMaterial->GetPathName() : FString());

        FString StaticMeshPath;
        if (const UStaticMeshComponent* StaticMesh =
                Cast<UStaticMeshComponent>(Primitive))
        {
            if (const UStaticMesh* Asset = StaticMesh->GetStaticMesh())
            {
                StaticMeshPath = Asset->GetPathName();
            }
        }
        Row->SetStringField(TEXT("static_mesh_path"), StaticMeshPath);

        TArray<FString> Materials;
        if (const UMeshComponent* Mesh = Cast<UMeshComponent>(Primitive))
        {
            const int32 Count = Mesh->GetNumMaterials();
            for (int32 Index = 0; Index < Count; ++Index)
            {
                if (const UMaterialInterface* Material = Mesh->GetMaterial(Index))
                {
                    Materials.AddUnique(Material->GetPathName());
                }
            }
        }
        Materials.Sort();
        Row->SetArrayField(TEXT("material_paths"), JsonStrings(Materials));
        return Row;
    }

    TArray<FHitResult> CollectGroundHits(
        UWorld* World,
        const FVector& NavPoint,
        const AActor* IgnoredActor)
    {
        TArray<FHitResult> Result;
        if (!World)
        {
            return Result;
        }
        FCollisionQueryParams QueryParams(
            SCENE_QUERY_STAT(PixelGoalPedestrianAudit), true, IgnoredActor);
        QueryParams.bReturnFaceIndex = true;
        if (IgnoredActor)
        {
            QueryParams.AddIgnoredActor(IgnoredActor);
        }
        const FVector TraceStart = NavPoint + FVector(0.f, 0.f, 250.f);
        const FVector TraceEnd = NavPoint - FVector(0.f, 0.f, 250.f);
        // A multi trace stops at the first blocking decal/prop and therefore
        // hides the pavement or road below it.  Peel one component at a time:
        // this keeps the exact top hit while also exposing its structural
        // underlay, which lets the authoring classifier reject an asphalt
        // manhole but retain the same prop when it sits on pavement.
        for (int32 Index = 0; Index < 8; ++Index)
        {
            FHitResult Hit;
            if (!World->LineTraceSingleByChannel(
                    Hit,
                    TraceStart,
                    TraceEnd,
                    ECC_Visibility,
                    QueryParams))
            {
                break;
            }
            Result.Add(Hit);
            UPrimitiveComponent* HitComponent = Hit.GetComponent();
            if (!HitComponent)
            {
                break;
            }
            QueryParams.AddIgnoredComponent(HitComponent);
        }
        return Result;
    }

    TArray<TSharedPtr<FJsonValue>> JsonGroundHits(
        const TArray<FHitResult>& Hits)
    {
        TArray<TSharedPtr<FJsonValue>> Result;
        Result.Reserve(Hits.Num());
        for (const FHitResult& Hit : Hits)
        {
            Result.Add(MakeShared<FJsonValueObject>(GroundHitJson(Hit)));
        }
        return Result;
    }

    FString HitSurfaceMaterialPath(const FHitResult& Hit)
    {
        const UPrimitiveComponent* Primitive = Hit.GetComponent();
        int32 SectionIndex = INDEX_NONE;
        const UMaterialInterface* Material = Primitive
            ? Primitive->GetMaterialFromCollisionFaceIndex(
                Hit.FaceIndex, SectionIndex)
            : nullptr;
        return Material ? Material->GetPathName() : FString();
    }

    FString HitStaticMeshPath(const FHitResult& Hit)
    {
        const UStaticMeshComponent* Component =
            Cast<UStaticMeshComponent>(Hit.GetComponent());
        const UStaticMesh* Mesh = Component ? Component->GetStaticMesh() : nullptr;
        return Mesh ? Mesh->GetPathName() : FString();
    }

    FString ClassifyPedestrianSurface(
        const TArray<FString>& CrosswalkLabels,
        const TArray<FHitResult>& Hits)
    {
        if (!CrosswalkLabels.IsEmpty())
        {
            return TEXT("marked_crossing");
        }
        for (const FHitResult& Hit : Hits)
        {
            const AActor* Actor = Hit.GetActor();
            const FString ActorLabel =
                Actor ? Actor->GetActorNameOrLabel() : FString();
            const FString MaterialPath = HitSurfaceMaterialPath(Hit);
            const FString MeshPath = HitStaticMeshPath(Hit);

            // Asphalt and the scene's road spline actors are always vehicle
            // surfaces here.  Dirt/manhole/marking overlays do not decide the
            // class: component peeling exposes their structural underlay on
            // the following hit.
            if (ActorLabel.StartsWith(
                    TEXT("BP_SplineRoad"), ESearchCase::CaseSensitive) ||
                MaterialPath.Contains(
                    TEXT("/Asphalt/"), ESearchCase::CaseSensitive))
            {
                return TEXT("road");
            }
            if (ActorLabel.StartsWith(
                    TEXT("BP_SplineSidewalk"), ESearchCase::CaseSensitive) ||
                ActorLabel.StartsWith(
                    TEXT("PR_Sidewalk"), ESearchCase::CaseSensitive) ||
                MeshPath.Contains(
                    TEXT("/Sidewalk/"), ESearchCase::CaseSensitive) ||
                MaterialPath.Contains(
                    TEXT("Pavement"), ESearchCase::CaseSensitive) ||
                MaterialPath.Contains(
                    TEXT("Sidewalk"), ESearchCase::CaseSensitive) ||
                MaterialPath.Contains(
                    TEXT("TruncatedDomes"), ESearchCase::CaseSensitive))
            {
                return TEXT("pavement");
            }
        }
        return TEXT("unverified");
    }

    struct FPedestrianPathSurfaceSample
    {
        FVector Point = FVector::ZeroVector;
        FString Surface;
    };

    FString PedestrianPathRejection(
        UWorld* World,
        const TArray<FVector>& PathPoints,
        const AActor* IgnoredActor,
        const TArray<FParisCrosswalkComponent>& Crosswalks,
        float SampleSpacingCm = 10.f,
        float MaxCrosswalkKerbTransitionCm = 50.f)
    {
        if (!World || PathPoints.IsEmpty())
        {
            return TEXT("controller_path_surface_unverified");
        }
        TArray<FPedestrianPathSurfaceSample> Samples;
        for (int32 SegmentIndex = 0;
             SegmentIndex < FMath::Max(PathPoints.Num() - 1, 1);
             ++SegmentIndex)
        {
            const FVector Start = PathPoints[SegmentIndex];
            const FVector End = PathPoints.Num() > 1
                ? PathPoints[SegmentIndex + 1]
                : Start;
            const int32 Steps = FMath::Max(
                1,
                FMath::CeilToInt(FVector::Dist2D(Start, End) / SampleSpacingCm));
            for (int32 Step = 0; Step <= Steps; ++Step)
            {
                // The first point of later segments was already checked.
                if (SegmentIndex > 0 && Step == 0)
                {
                    continue;
                }
                const FVector Point = FMath::Lerp(
                    Start, End, static_cast<float>(Step) / Steps);
                const TArray<FString> Labels =
                    CrosswalkLabelsAtPoint(Point, Crosswalks);
                FPedestrianPathSurfaceSample Sample;
                Sample.Point = Point;
                Sample.Surface = ClassifyPedestrianSurface(
                    Labels, CollectGroundHits(World, Point, IgnoredActor));
                Samples.Add(MoveTemp(Sample));
            }
        }

        if (Samples.IsEmpty() || Samples[0].Surface == TEXT("road") ||
            Samples.Last().Surface == TEXT("road"))
        {
            // A move may traverse a short kerb seam as part of a crossing,
            // but may never begin or finish standing on asphalt.
            return TEXT("controller_path_enters_unmarked_road");
        }
        for (const FPedestrianPathSurfaceSample& Sample : Samples)
        {
            if (Sample.Surface != TEXT("pavement") &&
                Sample.Surface != TEXT("marked_crossing") &&
                Sample.Surface != TEXT("road"))
            {
                return TEXT("controller_path_surface_unverified");
            }
        }

        int32 Index = 0;
        while (Index < Samples.Num())
        {
            if (Samples[Index].Surface != TEXT("road"))
            {
                ++Index;
                continue;
            }
            const int32 FirstRoad = Index;
            while (Index + 1 < Samples.Num() &&
                   Samples[Index + 1].Surface == TEXT("road"))
            {
                ++Index;
            }
            const int32 LastRoad = Index;
            const bool bTouchesMarkedCrossing =
                (FirstRoad > 0 &&
                 Samples[FirstRoad - 1].Surface == TEXT("marked_crossing")) ||
                (LastRoad + 1 < Samples.Num() &&
                 Samples[LastRoad + 1].Surface == TEXT("marked_crossing"));
            float RoadRunCm = 0.f;
            if (FirstRoad > 0)
            {
                RoadRunCm += 0.5f * FVector::Dist2D(
                    Samples[FirstRoad - 1].Point, Samples[FirstRoad].Point);
            }
            for (int32 RoadIndex = FirstRoad;
                 RoadIndex < LastRoad;
                 ++RoadIndex)
            {
                RoadRunCm += FVector::Dist2D(
                    Samples[RoadIndex].Point, Samples[RoadIndex + 1].Point);
            }
            if (LastRoad + 1 < Samples.Num())
            {
                RoadRunCm += 0.5f * FVector::Dist2D(
                    Samples[LastRoad].Point, Samples[LastRoad + 1].Point);
            }
            if (!bTouchesMarkedCrossing ||
                RoadRunCm > MaxCrosswalkKerbTransitionCm + KINDA_SMALL_NUMBER)
            {
                return TEXT("controller_path_enters_unmarked_road");
            }
            ++Index;
        }
        return FString();
    }

    FString PedestrianDiagnosticError(const FString& Code)
    {
        const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetBoolField(TEXT("success"), false);
        Root->SetStringField(TEXT("error"), Code);
        return SerializeJson(Root);
    }

    template <typename TActor>
    TActor* FindActorWithTag(UWorld* World, const FName Tag)
    {
        if (!World)
        {
            return nullptr;
        }
        for (TActorIterator<TActor> It(World); It; ++It)
        {
            if (It->ActorHasTag(Tag))
            {
                return *It;
            }
        }
        return nullptr;
    }

    AStaticMeshActor* SpawnCalibrationCube(
        UWorld* World,
        UStaticMesh* CubeMesh,
        const FName Tag,
        const FVector& Location,
        const FVector& Scale,
        bool bAffectsNavigation)
    {
        if (!World || !CubeMesh)
        {
            return nullptr;
        }
        const FTransform Transform(FRotator::ZeroRotator, Location, Scale);
        AStaticMeshActor* Actor = World->SpawnActorDeferred<AStaticMeshActor>(
            AStaticMeshActor::StaticClass(),
            Transform,
            nullptr,
            nullptr,
            ESpawnActorCollisionHandlingMethod::AlwaysSpawn);
        UStaticMeshComponent* Mesh = Actor ? Actor->GetStaticMeshComponent() : nullptr;
        if (!Actor || !Mesh)
        {
            if (Actor)
            {
                Actor->Destroy();
            }
            return nullptr;
        }
        Actor->Tags.AddUnique(Tag);
        Mesh->SetMobility(EComponentMobility::Movable);
        Mesh->SetStaticMesh(CubeMesh);
        Mesh->SetMobility(EComponentMobility::Static);
        Mesh->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        Mesh->SetCollisionResponseToAllChannels(ECR_Block);
        Mesh->SetCanEverAffectNavigation(bAffectsNavigation);
        UGameplayStatics::FinishSpawningActor(Actor, Transform);
        return Actor;
    }

    void AddRequestIdentity(
        const TSharedRef<FJsonObject>& Root,
        const FString& SnapshotId,
        const FString& IntrinsicsId,
        const FVector2D& UV,
        bool bHasBinding = false,
        const FString& ViewId = FString(),
        const FString& CaptureGroupId = FString())
    {
        Root->SetStringField(TEXT("camera_snapshot_id"), SnapshotId);
        Root->SetStringField(TEXT("camera_intrinsics_id"), IntrinsicsId);
        TArray<TSharedPtr<FJsonValue>> UVValues;
        UVValues.Add(MakeShared<FJsonValueNumber>(UV.X));
        UVValues.Add(MakeShared<FJsonValueNumber>(UV.Y));
        Root->SetArrayField(TEXT("requested_uv"), UVValues);
        if (bHasBinding)
        {
            Root->SetStringField(TEXT("view_id"), ViewId);
            Root->SetStringField(TEXT("capture_group_id"), CaptureGroupId);
        }
    }

    FString Reject(
        const FString& Reason,
        const FString& SnapshotId,
        const FString& IntrinsicsId,
        const FVector2D& UV,
        const FHitResult* Hit = nullptr,
        const FVector* ValidatedTarget = nullptr,
        float NavAdjustmentCm = -1.f,
        bool bHasBinding = false,
        const FString& ViewId = FString(),
        const FString& CaptureGroupId = FString(),
        const TArray<FVector>* ControllerPathPoints = nullptr,
        float ControllerPathLengthCm = -1.f,
        float ControllerPathDirectCm = -1.f,
        float ControllerPathStretchRatio = -1.f)
    {
        const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetBoolField(TEXT("accepted"), false);
        Root->SetStringField(TEXT("rejection_reason"), Reason);
        AddRequestIdentity(
            Root,
            SnapshotId,
            IntrinsicsId,
            UV,
            bHasBinding,
            ViewId,
            CaptureGroupId);
        if (Hit && Hit->bBlockingHit)
        {
            Root->SetArrayField(TEXT("raw_world_hit_cm"), JsonVector(Hit->ImpactPoint));
            Root->SetArrayField(TEXT("raw_world_normal"), JsonVector(Hit->ImpactNormal));
            Root->SetStringField(
                TEXT("raw_hit_actor"),
                Hit->GetActor() ? Hit->GetActor()->GetActorNameOrLabel() : FString());
        }
        if (ValidatedTarget)
        {
            Root->SetArrayField(
                TEXT("validated_navigation_target_cm"),
                JsonVector(*ValidatedTarget));
        }
        if (NavAdjustmentCm >= 0.f)
        {
            Root->SetNumberField(TEXT("navmesh_adjustment_cm"), NavAdjustmentCm);
        }
        if (ControllerPathPoints)
        {
            Root->SetArrayField(
                TEXT("controller_path_points_cm"),
                JsonVectorPath(*ControllerPathPoints));
        }
        if (ControllerPathLengthCm >= 0.f)
        {
            Root->SetNumberField(
                TEXT("controller_path_length_cm"), ControllerPathLengthCm);
        }
        if (ControllerPathDirectCm >= 0.f)
        {
            Root->SetNumberField(
                TEXT("controller_path_direct_cm"), ControllerPathDirectCm);
        }
        if (ControllerPathStretchRatio >= 0.f)
        {
            Root->SetNumberField(
                TEXT("controller_path_stretch_ratio"),
                ControllerPathStretchRatio);
        }
        return SerializeJson(Root);
    }

    FString MoveRequestResultString(EPathFollowingRequestResult::Type Result)
    {
        switch (Result)
        {
            case EPathFollowingRequestResult::RequestSuccessful:
                return TEXT("request_successful");
            case EPathFollowingRequestResult::AlreadyAtGoal:
                return TEXT("already_at_goal");
            case EPathFollowingRequestResult::Failed:
                return TEXT("failed");
        }
        return TEXT("unknown");
    }

    FString MoveCompletionResultString(EPathFollowingResult::Type Result)
    {
        switch (Result)
        {
            case EPathFollowingResult::Success: return TEXT("success");
            case EPathFollowingResult::Blocked: return TEXT("blocked");
            case EPathFollowingResult::OffPath: return TEXT("off_path");
            case EPathFollowingResult::Aborted: return TEXT("aborted");
            case EPathFollowingResult::Skipped_DEPRECATED: return TEXT("aborted");
            case EPathFollowingResult::Invalid: return TEXT("invalid");
        }
        return TEXT("invalid");
    }

    FSpPixelGoalCameraSnapshot MakeSnapshot(
        const FMinimalViewInfo& ViewInfo,
        const FIntPoint& RenderSize,
        const TOptional<FMatrix>& CustomProjectionMatrix)
    {
        FSpPixelGoalCameraSnapshot Snapshot;
        Snapshot.RenderSize = RenderSize;
        Snapshot.CameraLocation = ViewInfo.Location;
        Snapshot.CameraRotation = ViewInfo.Rotation;
        Snapshot.HorizontalFovDegrees = ViewInfo.FOV;

        const FMatrix ProjectionMatrix = AdjustProjectionMatrixForRHI(
            CustomProjectionMatrix.IsSet()
                ? CustomProjectionMatrix.GetValue()
                : ViewInfo.CalculateProjectionMatrix());
        Snapshot.InvProjectionMatrix = ProjectionMatrix.Inverse();
        Snapshot.InvViewMatrix = FMatrix(
            FPlane(0, 1, 0, 0),
            FPlane(0, 0, 1, 0),
            FPlane(1, 0, 0, 0),
            FPlane(0, 0, 0, 1)) *
            FRotationTranslationMatrix(ViewInfo.Rotation, ViewInfo.Location);
        return Snapshot;
    }

    bool MatricesExactlyEqual(const FMatrix& A, const FMatrix& B)
    {
        for (int32 Row = 0; Row < 4; ++Row)
        {
            for (int32 Column = 0; Column < 4; ++Column)
            {
                if (A.M[Row][Column] != B.M[Row][Column])
                {
                    return false;
                }
            }
        }
        return true;
    }
}

bool FSpPixelGoalCameraSnapshot::Deproject(
    const FVector2D& NormalizedUV,
    FVector& OutOrigin,
    FVector& OutDirection) const
{
    if (!IsFinite(NormalizedUV) ||
        NormalizedUV.X < 0.f || NormalizedUV.X > 1.f ||
        NormalizedUV.Y < 0.f || NormalizedUV.Y > 1.f ||
        RenderSize.X <= 0 || RenderSize.Y <= 0)
    {
        OutOrigin = FVector::ZeroVector;
        OutDirection = FVector::ZeroVector;
        return false;
    }
    // FSceneView::DeprojectScreenToWorld truncates its screen position to an
    // integer pixel. Pixel Goal coordinates are continuous image coordinates,
    // so unproject normalized device coordinates directly to keep the ray
    // independent of the render resolution.
    const float ScreenSpaceX = (NormalizedUV.X - 0.5f) * 2.0f;
    const float ScreenSpaceY = ((1.0f - NormalizedUV.Y) - 0.5f) * 2.0f;
    const FVector4 RayStartProjectionSpace(ScreenSpaceX, ScreenSpaceY, 1.0f, 1.0f);
    const FVector4 RayEndProjectionSpace(ScreenSpaceX, ScreenSpaceY, 0.01f, 1.0f);
    const FVector4 HomogeneousStart =
        InvProjectionMatrix.TransformFVector4(RayStartProjectionSpace);
    const FVector4 HomogeneousEnd =
        InvProjectionMatrix.TransformFVector4(RayEndProjectionSpace);

    FVector RayStartView(HomogeneousStart.X, HomogeneousStart.Y, HomogeneousStart.Z);
    FVector RayEndView(HomogeneousEnd.X, HomogeneousEnd.Y, HomogeneousEnd.Z);
    if (HomogeneousStart.W != 0.0f)
    {
        RayStartView /= HomogeneousStart.W;
    }
    if (HomogeneousEnd.W != 0.0f)
    {
        RayEndView /= HomogeneousEnd.W;
    }

    const FVector RayDirectionView = (RayEndView - RayStartView).GetSafeNormal();
    OutOrigin = CameraLocation;
    OutDirection = InvViewMatrix.TransformVector(RayDirectionView).GetSafeNormal();
    return !OutDirection.IsNearlyZero();
}

#if WITH_DEV_AUTOMATION_TESTS
FSpPixelGoalCameraSnapshot FSpPixelGoalCameraSnapshot::PerspectiveForTest(
    const FVector& Location,
    const FRotator& Rotation,
    int32 Width,
    int32 Height,
    float HorizontalFovDegrees)
{
    FMinimalViewInfo ViewInfo;
    ViewInfo.Location = Location;
    ViewInfo.Rotation = Rotation;
    ViewInfo.FOV = HorizontalFovDegrees;
    ViewInfo.AspectRatio = static_cast<float>(Width) / static_cast<float>(Height);
    ViewInfo.bConstrainAspectRatio = false;
    ViewInfo.ProjectionMode = ECameraProjectionMode::Perspective;
    return MakeSnapshot(ViewInfo, FIntPoint(Width, Height), TOptional<FMatrix>());
}
#endif

ESpPixelGoalHitClass SpPixelGoal::ClassifyFirstHit(
    const FVector& ImpactNormal,
    float MaxGroundSlopeDegrees)
{
    const FVector Normal = ImpactNormal.GetSafeNormal();
    const float MinimumUpDot = FMath::Cos(FMath::DegreesToRadians(MaxGroundSlopeDegrees));
    return FVector::DotProduct(Normal, FVector::UpVector) + KINDA_SMALL_NUMBER >= MinimumUpDot
        ? ESpPixelGoalHitClass::WalkableGround
        : ESpPixelGoalHitClass::NotWalkableGround;
}

bool SpPixelGoal::WithinNavAdjustment(
    const FVector& RawHit,
    const FVector& ProjectedTarget,
    float MaxAdjustmentCm)
{
    return FVector::Distance(RawHit, ProjectedTarget) <= MaxAdjustmentCm + KINDA_SMALL_NUMBER;
}

float SpPixelGoal::PlanarErrorCm(const FVector& A, const FVector& B)
{
    return FVector::Dist2D(A, B);
}

float SpPixelGoal::ControllerPathLengthCm(const TArray<FVector>& Points)
{
    float LengthCm = 0.f;
    for (int32 Index = 1; Index < Points.Num(); ++Index)
    {
        LengthCm += FVector::Dist2D(Points[Index - 1], Points[Index]);
    }
    return LengthCm;
}

bool SpPixelGoal::ControllerPathDetourExceeded(
    float PathLengthCm,
    float DirectDistanceCm,
    float MaxStretchRatio,
    float DetourAllowanceCm)
{
    if (!FMath::IsFinite(PathLengthCm) ||
        !FMath::IsFinite(DirectDistanceCm) ||
        !FMath::IsFinite(MaxStretchRatio) ||
        !FMath::IsFinite(DetourAllowanceCm) ||
        PathLengthCm < 0.f || DirectDistanceCm < 0.f ||
        MaxStretchRatio < 1.f || DetourAllowanceCm < 0.f)
    {
        return true;
    }
    const float AllowedLengthCm = FMath::Max(
        DirectDistanceCm * MaxStretchRatio,
        DirectDistanceCm + DetourAllowanceCm);
    return PathLengthCm > AllowedLengthCm + KINDA_SMALL_NUMBER;
}

FString SpPixelGoal::SnapshotBindingError(
    const FString& SnapshotViewId,
    const FString& SnapshotCaptureGroupId,
    const FString& RequestedViewId,
    const FString& RequestedCaptureGroupId)
{
    if (SnapshotViewId != RequestedViewId)
    {
        return TEXT("camera_snapshot_view_mismatch");
    }
    if (SnapshotCaptureGroupId != RequestedCaptureGroupId)
    {
        return TEXT("camera_snapshot_group_mismatch");
    }
    return FString();
}

FString SpPixelGoal::ParseViewPairCaptureRequest(
    const TSharedPtr<FJsonObject>& Request,
    FSpPixelGoalViewPairCaptureRequest& OutRequest)
{
    OutRequest = FSpPixelGoalViewPairCaptureRequest();
    if (!Request.IsValid())
    {
        return TEXT("invalid_request_json");
    }
    if (!TryGetExactJsonString(
            Request, TEXT("agent_tag"), OutRequest.AgentTag) ||
        OutRequest.AgentTag.IsEmpty())
    {
        return TEXT("missing_agent_tag");
    }

    const auto ReadOptionalNumber = [&Request](
        const TCHAR* Field,
        double Default,
        double Minimum,
        double Maximum,
        double& OutValue)
    {
        OutValue = Default;
        if (Request->HasField(Field) &&
            !TryGetExactJsonNumber(Request, Field, OutValue))
        {
            return false;
        }
        return OutValue >= Minimum && OutValue <= Maximum;
    };
    const auto IsExactInteger = [](double Value)
    {
        return Value == static_cast<double>(static_cast<int32>(Value));
    };

    double Width = 640.0;
    double Height = 360.0;
    double Fov = 90.0;
    double JpegQuality = 90.0;
    if (!ReadOptionalNumber(TEXT("width"), 640.0, 16.0, 4096.0, Width) ||
        !ReadOptionalNumber(TEXT("height"), 360.0, 16.0, 4096.0, Height) ||
        !IsExactInteger(Width) || !IsExactInteger(Height))
    {
        return TEXT("invalid_capture_dimensions");
    }
    if (!ReadOptionalNumber(TEXT("fov_degrees"), 90.0, 5.0, 170.0, Fov))
    {
        return TEXT("invalid_camera_fov");
    }
    if (!ReadOptionalNumber(
            TEXT("jpeg_quality"), 90.0, 1.0, 95.0, JpegQuality) ||
        !IsExactInteger(JpegQuality))
    {
        return TEXT("invalid_jpeg_quality");
    }
    if (Request->HasField(TEXT("validate")) &&
        !TryGetExactJsonBool(
            Request, TEXT("validate"), OutRequest.bValidate))
    {
        return TEXT("invalid_validate_flag");
    }
    if (Request->HasField(TEXT("commit_snapshot")) &&
        !TryGetExactJsonBool(
            Request, TEXT("commit_snapshot"), OutRequest.bCommitSnapshots))
    {
        return TEXT("invalid_commit_snapshot_flag");
    }

    OutRequest.Width = static_cast<int32>(Width);
    OutRequest.Height = static_cast<int32>(Height);
    OutRequest.HorizontalFovDegrees = static_cast<float>(Fov);
    OutRequest.JpegQuality = static_cast<int32>(JpegQuality);
    return FString();
}

FString SpPixelGoal::WarmedViewPairCameraError(
    const ASpHumanoidAgent* Agent,
    int32 Width,
    int32 Height,
    float HorizontalFovDegrees)
{
    if (!IsValid(Agent) || !IsValid(Agent->SpringArm))
    {
        return TEXT("view_pair_cameras_not_ready");
    }
    USpSceneCaptureComponent2D* Front =
        Agent->GetObservationCamera(TEXT("front"));
    USpSceneCaptureComponent2D* Rear =
        Agent->GetObservationCamera(TEXT("rear"));
    if (!IsValid(Front) || !IsValid(Rear) || Front == Rear ||
        Front != Agent->SceneCapture || Rear != Agent->RearSceneCapture.Get() ||
        !Front->IsInitialized() || !Rear->IsInitialized() ||
        !IsValid(Front->TextureTarget) || !IsValid(Rear->TextureTarget))
    {
        return TEXT("view_pair_cameras_not_ready");
    }

    const auto MatchesRequestedOptics = [Width, Height, HorizontalFovDegrees](
        const USpSceneCaptureComponent2D* Capture)
    {
        return Capture->Width == Width && Capture->Height == Height &&
            Capture->TextureTarget->SizeX == Width &&
            Capture->TextureTarget->SizeY == Height &&
            Capture->FOVAngle == HorizontalFovDegrees &&
            Capture->ProjectionType == ECameraProjectionMode::Perspective &&
            Capture->Overscan == 0.f &&
            !Capture->bUseCustomProjectionMatrix &&
            !Capture->bOverride_CustomNearClippingPlane &&
            !Capture->bEnableClipPlane &&
            !Capture->bEnableFirstPersonFieldOfView &&
            !Capture->bEnableFirstPersonScale &&
            !Capture->bMainViewCamera &&
            !Capture->bMainViewResolution &&
            !Capture->bRenderInMainRenderer;
    };
    const auto MatchesPersistentRgbContract = [](
        const USpSceneCaptureComponent2D* Capture)
    {
        return Capture->bAlwaysPersistRenderingState &&
            Capture->bCaptureEveryFrame && !Capture->bCaptureOnMovement &&
            Capture->CaptureSource == ESceneCaptureSource::SCS_FinalColorLDR &&
            Capture->bOverrideTextureRenderTargetFormat &&
            Capture->TextureRenderTargetFormat ==
                ETextureRenderTargetFormat::RTF_RGBA8_SRGB &&
            Capture->NumChannelsPerPixel == 4 &&
            Capture->ChannelDataType == ESpArrayDataType::UInt8 &&
            Capture->PostProcessBlendWeight == 1.f &&
            Capture->PostProcessSettings.bOverride_AutoExposureMethod &&
            Capture->PostProcessSettings.AutoExposureMethod == AEM_Histogram &&
            Capture->PostProcessSettings.
                bOverride_AutoExposureMinBrightness &&
            Capture->PostProcessSettings.AutoExposureMinBrightness ==
                ParisPocExposureMinEv100 &&
            Capture->PostProcessSettings.
                bOverride_AutoExposureMaxBrightness &&
            Capture->PostProcessSettings.AutoExposureMaxBrightness ==
                ParisPocExposureMaxEv100 &&
            Capture->PostProcessSettings.bOverride_AutoExposureSpeedUp &&
            Capture->PostProcessSettings.AutoExposureSpeedUp ==
                ParisPocExposureSpeedUp &&
            Capture->PostProcessSettings.bOverride_AutoExposureSpeedDown &&
            Capture->PostProcessSettings.AutoExposureSpeedDown ==
                ParisPocExposureSpeedDown &&
            Capture->PostProcessSettings.bOverride_AutoExposureBias &&
            Capture->PostProcessSettings.AutoExposureBias ==
                ParisPocExposureBias;
    };
    const bool bExpectedIdentity =
        Front->GetOwner() == Agent && Rear->GetOwner() == Agent &&
        Front->GetAttachParent() == Agent->SpringArm &&
        Rear->GetAttachParent() == Agent->SpringArm;
    const bool bExpectedRelativePose =
        Front->GetRelativeLocation().Equals(Rear->GetRelativeLocation(), 0.f) &&
        Front->GetRelativeRotation().Equals(FRotator::ZeroRotator, 0.0) &&
        Rear->GetRelativeRotation().Equals(FRotator(0.f, 180.f, 0.f), 0.0) &&
        Front->GetComponentLocation().Equals(
            Rear->GetComponentLocation(), 0.f);
    const bool bMatchingRgbProperties =
        Front->CaptureSource == Rear->CaptureSource &&
        Front->bOverrideTextureRenderTargetFormat ==
            Rear->bOverrideTextureRenderTargetFormat &&
        Front->TextureRenderTargetFormat == Rear->TextureRenderTargetFormat &&
        Front->NumChannelsPerPixel == Rear->NumChannelsPerPixel &&
        Front->ChannelDataType == Rear->ChannelDataType;
    if (!bExpectedIdentity || !bExpectedRelativePose ||
        !MatchesRequestedOptics(Front) || !MatchesRequestedOptics(Rear) ||
        !MatchesPersistentRgbContract(Front) ||
        !MatchesPersistentRgbContract(Rear) || !bMatchingRgbProperties)
    {
        return TEXT("view_pair_camera_configuration_mismatch");
    }
    return FString();
}

FString SpPixelGoal::ViewPairSnapshotError(
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
    const USceneCaptureComponent2D* RearComponent)
{
    if (FrontSnapshot.SnapshotId == RearSnapshot.SnapshotId ||
        FrontImageSnapshotId == RearImageSnapshotId)
    {
        return TEXT("duplicate_view_pair_snapshot");
    }
    if (FrontImageSnapshotId.IsEmpty() || RearImageSnapshotId.IsEmpty() ||
        FrontImageIntrinsicsId.IsEmpty() || RearImageIntrinsicsId.IsEmpty() ||
        FrontSnapshot.SnapshotId != FrontImageSnapshotId ||
        RearSnapshot.SnapshotId != RearImageSnapshotId ||
        FrontSnapshot.ViewId != TEXT("front") ||
        RearSnapshot.ViewId != TEXT("rear") ||
        FrontSnapshot.CaptureGroupId != CaptureGroupId ||
        RearSnapshot.CaptureGroupId != CaptureGroupId ||
        FrontSnapshot.CaptureComponent.Get() != FrontComponent ||
        RearSnapshot.CaptureComponent.Get() != RearComponent)
    {
        return TEXT("invalid_view_pair_snapshot");
    }
    if (FrontImageIntrinsicsId != FrontSnapshot.IntrinsicsId ||
        RearImageIntrinsicsId != RearSnapshot.IntrinsicsId)
    {
        return TEXT("view_pair_image_snapshot_intrinsics_mismatch");
    }
    if (FrontSnapshot.IntrinsicsId.IsEmpty() ||
        FrontSnapshot.IntrinsicsId != RearSnapshot.IntrinsicsId)
    {
        return TEXT("view_pair_intrinsics_mismatch");
    }
    if (!FrontSnapshot.CameraLocation.Equals(
            RearSnapshot.CameraLocation, 0.f))
    {
        return TEXT("view_pair_optical_centre_mismatch");
    }
    if (ExpectedWidth <= 0 || ExpectedHeight <= 0 ||
        !FMath::IsFinite(ExpectedHorizontalFovDegrees))
    {
        return TEXT("view_pair_effective_intrinsics_mismatch");
    }

    FMinimalViewInfo ExpectedViewInfo;
    ExpectedViewInfo.FOV = ExpectedHorizontalFovDegrees;
    ExpectedViewInfo.AspectRatio =
        static_cast<float>(ExpectedWidth) / static_cast<float>(ExpectedHeight);
    ExpectedViewInfo.bConstrainAspectRatio = false;
    ExpectedViewInfo.ProjectionMode = ECameraProjectionMode::Perspective;
    const FSpPixelGoalCameraSnapshot ExpectedSnapshot = MakeSnapshot(
        ExpectedViewInfo,
        FIntPoint(ExpectedWidth, ExpectedHeight),
        TOptional<FMatrix>());
    const auto MatchesExpectedEffectiveIntrinsics = [
        &ExpectedSnapshot,
        ExpectedHorizontalFovDegrees](const FSpPixelGoalCameraSnapshot& Snapshot)
    {
        // MakeSnapshot records a deterministic matrix derived synchronously
        // from these same request inputs; no renderer estimate is involved.
        // Exact element comparison therefore rejects every distinct ray model.
        return Snapshot.RenderSize == ExpectedSnapshot.RenderSize &&
            Snapshot.HorizontalFovDegrees == ExpectedHorizontalFovDegrees &&
            MatricesExactlyEqual(
                Snapshot.InvProjectionMatrix,
                ExpectedSnapshot.InvProjectionMatrix);
    };
    if (!MatchesExpectedEffectiveIntrinsics(FrontSnapshot) ||
        !MatchesExpectedEffectiveIntrinsics(RearSnapshot))
    {
        return TEXT("view_pair_effective_intrinsics_mismatch");
    }
    return FString();
}

FString SpPixelGoal::ViewPairImageError(
    const TSharedPtr<FJsonObject>& Image,
    const FString& ExpectedViewId,
    const FString& AgentTag,
    const FString& CaptureGroupId,
    int32 ExpectedWidth,
    int32 ExpectedHeight,
    const FVector& ExpectedAgentLocation,
    double ExpectedAgentYawDegrees,
    FString& OutSnapshotId,
    FString& OutIntrinsicsId)
{
    OutSnapshotId.Reset();
    OutIntrinsicsId.Reset();
    const auto ExactVectorField = [&Image](
        const TCHAR* Field,
        FVector& OutValue)
    {
        const TSharedPtr<FJsonValue> Value = Image.IsValid()
            ? Image->TryGetField(Field)
            : nullptr;
        const TArray<TSharedPtr<FJsonValue>>* Components = nullptr;
        if (!Value.IsValid() || Value->Type != EJson::Array ||
            !Value->TryGetArray(Components) || !Components ||
            Components->Num() != 3)
        {
            return false;
        }
        double Numbers[3] = {};
        for (int32 Index = 0; Index < 3; ++Index)
        {
            if (!(*Components)[Index].IsValid() ||
                (*Components)[Index]->Type != EJson::Number ||
                !(*Components)[Index]->TryGetNumber(Numbers[Index]) ||
                !FMath::IsFinite(Numbers[Index]))
            {
                return false;
            }
        }
        OutValue = FVector(Numbers[0], Numbers[1], Numbers[2]);
        return true;
    };

    bool bImageSuccess = false;
    double ImageWidth = 0.0;
    double ImageHeight = 0.0;
    double ImageAgentYaw = 0.0;
    FVector ImageAgentLocation;
    FString DataUrl;
    FString FieldValue;
    FString SnapshotId;
    FString IntrinsicsId;
    if (!TryGetExactJsonBool(Image, TEXT("success"), bImageSuccess) ||
        !bImageSuccess ||
        !TryGetExactJsonString(Image, TEXT("modality"), FieldValue) ||
        FieldValue != TEXT("rgb") ||
        !TryGetExactJsonString(Image, TEXT("camera_id"), FieldValue) ||
        FieldValue != ExpectedViewId ||
        !TryGetExactJsonString(Image, TEXT("camera_view"), FieldValue) ||
        FieldValue != ExpectedViewId ||
        !TryGetExactJsonString(Image, TEXT("source_type"), FieldValue) ||
        FieldValue != TEXT("agent") ||
        !TryGetExactJsonString(
            Image, TEXT("source_agent_tag"), FieldValue) ||
        FieldValue != AgentTag ||
        !TryGetExactJsonString(Image, TEXT("capture_mode"), FieldValue) ||
        FieldValue != TEXT("agent_native") ||
        !TryGetExactJsonString(Image, TEXT("view_id"), FieldValue) ||
        FieldValue != ExpectedViewId ||
        !TryGetExactJsonString(
            Image, TEXT("capture_group_id"), FieldValue) ||
        FieldValue != CaptureGroupId ||
        !TryGetExactJsonString(Image, TEXT("data_url"), DataUrl) ||
        DataUrl.IsEmpty() ||
        !TryGetExactJsonNumber(Image, TEXT("width"), ImageWidth) ||
        ImageWidth != static_cast<double>(ExpectedWidth) ||
        !TryGetExactJsonNumber(Image, TEXT("height"), ImageHeight) ||
        ImageHeight != static_cast<double>(ExpectedHeight) ||
        !ExactVectorField(TEXT("loc_cm"), ImageAgentLocation) ||
        !ImageAgentLocation.Equals(ExpectedAgentLocation, 0.f) ||
        !TryGetExactJsonNumber(Image, TEXT("yaw_deg"), ImageAgentYaw) ||
        ImageAgentYaw != ExpectedAgentYawDegrees ||
        !TryGetExactJsonString(
            Image, TEXT("camera_snapshot_id"), SnapshotId) ||
        SnapshotId.IsEmpty() ||
        !TryGetExactJsonString(
            Image, TEXT("camera_intrinsics_id"), IntrinsicsId) ||
        IntrinsicsId.IsEmpty())
    {
        return TEXT("invalid_view_pair_image");
    }
    OutSnapshotId = SnapshotId;
    OutIntrinsicsId = IntrinsicsId;
    return FString();
}

#if WITH_DEV_AUTOMATION_TESTS
void SpPixelGoal::ResetGeometryTraceEntryCountForTest()
{
    GeometryTraceEntryCount = 0;
}

int32 SpPixelGoal::GeometryTraceEntryCountForTest()
{
    return GeometryTraceEntryCount;
}

bool SpPixelGoal::PrepareParisPocCaptureForTest(
    ASpHumanoidAgent* Agent,
    bool bEnableRearCamera)
{
    return PrepareParisPocCapture(Agent, bEnableRearCamera);
}
#endif

FString SpPixelGoal::CaptureModeForAgentTag(const FString& AgentTag)
{
    return AgentTag == ParisPocAgentTag.ToString()
        ? FString(TEXT("agent_native"))
        : FString(TEXT("shared_pool"));
}

bool SpPixelGoal::ConfigurePersistentParisCapture(
    USpSceneCaptureComponent2D* Capture)
{
    if (!Capture)
    {
        return false;
    }
    Capture->bAlwaysPersistRenderingState = true;
    Capture->bCaptureEveryFrame = true;
    Capture->bCaptureOnMovement = false;
    Capture->PostProcessBlendWeight = 1.f;
    Capture->PostProcessSettings.bOverride_AutoExposureMethod = true;
    Capture->PostProcessSettings.AutoExposureMethod = AEM_Histogram;
    Capture->PostProcessSettings.bOverride_AutoExposureMinBrightness = true;
    Capture->PostProcessSettings.AutoExposureMinBrightness =
        ParisPocExposureMinEv100;
    Capture->PostProcessSettings.bOverride_AutoExposureMaxBrightness = true;
    Capture->PostProcessSettings.AutoExposureMaxBrightness =
        ParisPocExposureMaxEv100;
    Capture->PostProcessSettings.bOverride_AutoExposureSpeedUp = true;
    Capture->PostProcessSettings.AutoExposureSpeedUp =
        ParisPocExposureSpeedUp;
    Capture->PostProcessSettings.bOverride_AutoExposureSpeedDown = true;
    Capture->PostProcessSettings.AutoExposureSpeedDown =
        ParisPocExposureSpeedDown;
    Capture->PostProcessSettings.bOverride_AutoExposureBias = true;
    Capture->PostProcessSettings.AutoExposureBias = ParisPocExposureBias;
    return true;
}

bool SpPixelGoal::ConfigureParisSkyAtmosphere(
    USkyAtmosphereComponent* Atmosphere)
{
    if (!Atmosphere)
    {
        return false;
    }
    Atmosphere->SetMieScatteringScale(ParisPocMieScatteringScale);
    return FMath::IsNearlyEqual(
        Atmosphere->MieScatteringScale,
        ParisPocMieScatteringScale,
        1.e-6f);
}

bool SpPixelGoal::ShouldSampleMoveAudit(const FString& State)
{
    return State == TEXT("moving");
}

bool SpPixelGoal::ParseParisPocSetupRequest(
    const FString& RequestJson,
    const FString& CurrentMapName,
    FSpPixelGoalParisPocSetup& OutSetup,
    FString& OutError)
{
    OutSetup = FSpPixelGoalParisPocSetup();
    OutError.Reset();
    const TSharedPtr<FJsonObject> Request = ParseJson(RequestJson);
    if (!Request.IsValid())
    {
        OutError = TEXT("invalid_request_json");
        return false;
    }
    if (!IsParisPocMapName(CurrentMapName))
    {
        OutError = TEXT("paris_map_not_loaded");
        return false;
    }
    if (JsonString(Request, TEXT("scene")) != ParisPocScene)
    {
        OutError = TEXT("scene_mismatch");
        return false;
    }
    if (Request->HasField(TEXT("spawn_floor")))
    {
        OutError = TEXT("synthetic_floor_forbidden");
        return false;
    }
    if (Request->HasField(TEXT("enable_rear_camera")) &&
        !TryGetExactJsonBool(
            Request,
            TEXT("enable_rear_camera"), OutSetup.bEnableRearCamera))
    {
        OutError = TEXT("invalid_enable_rear_camera");
        return false;
    }
    OutSetup.RegionName = JsonString(Request, TEXT("region_name"));
    if (OutSetup.RegionName.IsEmpty() || OutSetup.RegionName.Len() > 96)
    {
        OutError = TEXT("invalid_region_name");
        return false;
    }
    if (!JsonVectorField(Request, TEXT("agent_spawn_cm"), OutSetup.AgentSpawn) ||
        !JsonVectorField(
            Request, TEXT("nav_bounds_center_cm"), OutSetup.NavBoundsCenter) ||
        !JsonVectorField(
            Request, TEXT("nav_bounds_extent_cm"), OutSetup.NavBoundsExtent))
    {
        OutError = TEXT("invalid_setup_vector");
        return false;
    }
    const double Yaw = JsonNumber(
        Request,
        TEXT("agent_yaw_deg"),
        std::numeric_limits<double>::quiet_NaN());
    if (!FMath::IsFinite(Yaw))
    {
        OutError = TEXT("invalid_agent_yaw");
        return false;
    }
    OutSetup.AgentYawDegrees = static_cast<float>(Yaw);
    if (OutSetup.NavBoundsExtent.X <= 0.f ||
        OutSetup.NavBoundsExtent.Y <= 0.f ||
        OutSetup.NavBoundsExtent.Z <= 0.f ||
        OutSetup.NavBoundsExtent.X > 5000.f ||
        OutSetup.NavBoundsExtent.Y > 5000.f ||
        OutSetup.NavBoundsExtent.Z > 1000.f)
    {
        OutError = TEXT("nav_bounds_not_local");
        return false;
    }
    const FVector Delta = (OutSetup.AgentSpawn - OutSetup.NavBoundsCenter).GetAbs();
    if (Delta.X > OutSetup.NavBoundsExtent.X ||
        Delta.Y > OutSetup.NavBoundsExtent.Y ||
        Delta.Z > OutSetup.NavBoundsExtent.Z)
    {
        OutError = TEXT("agent_spawn_outside_nav_bounds");
        return false;
    }
    return true;
}

void USpPixelGoalSubsystem::Deinitialize()
{
    for (TPair<FString, FMoveAudit>& Pair : MoveAudits)
    {
        if (AAIController* Controller = Pair.Value.Controller.Get())
        {
            Controller->ReceiveMoveCompleted.RemoveDynamic(
                this, &USpPixelGoalSubsystem::HandleMoveCompleted);
        }
    }
    CameraSnapshots.Empty();
    SnapshotOrder.Empty();
    ActiveCaptureGroupId.Reset();
    MoveAudits.Empty();
    CalibrationHelper = nullptr;
    CalibrationFloor = nullptr;
    CalibrationAgent = nullptr;
    CalibrationLight = nullptr;
    CalibrationWall = nullptr;
    ParisPocHelper = nullptr;
    ParisPocAgent = nullptr;
    ParisPocSetup = FSpPixelGoalParisPocSetup();
    Super::Deinitialize();
}

void USpPixelGoalSubsystem::RecordCameraSnapshot(
    USceneCaptureComponent2D* CaptureComponent,
    ASpHumanoidAgent* Agent,
    int32 Width,
    int32 Height,
    TSharedPtr<FJsonObject>& ImageJson,
    const FString& ViewId,
    const FString& CaptureGroupId)
{
    if (!CaptureComponent || !Agent || !ImageJson.IsValid() || Width <= 0 || Height <= 0)
    {
        return;
    }

    FMinimalViewInfo ViewInfo;
    CaptureComponent->GetCameraView(0.f, ViewInfo);
    const TOptional<FMatrix> CustomProjection = CaptureComponent->bUseCustomProjectionMatrix
        ? TOptional<FMatrix>(CaptureComponent->CustomProjectionMatrix)
        : TOptional<FMatrix>();
    FSpPixelGoalCameraSnapshot Snapshot = MakeSnapshot(
        ViewInfo, FIntPoint(Width, Height), CustomProjection);
    Snapshot.SnapshotId = FString::Printf(
        TEXT("pixel-snapshot-%llu"), NextSnapshotNumber++);
    Snapshot.IntrinsicsId = FString::Printf(
        TEXT("perspective-%dx%d-hfov%.3f"), Width, Height, ViewInfo.FOV);
    Snapshot.ViewId = ViewId.IsEmpty()
        ? (CaptureComponent == Agent->RearSceneCapture.Get()
            ? FString(TEXT("rear"))
            : FString(TEXT("front")))
        : ViewId;
    Snapshot.CaptureGroupId = CaptureGroupId.IsEmpty()
        ? ActiveCaptureGroupId
        : CaptureGroupId;
    Snapshot.CaptureComponent = CaptureComponent;
    Snapshot.Agent = Agent;
    Snapshot.AgentLocation = Agent->GetActorLocation();
    Snapshot.AgentRotation = Agent->GetActorRotation();
    Snapshot.CapturedWorldTimeSeconds = GetWorld() ? GetWorld()->GetTimeSeconds() : 0.0;

    CameraSnapshots.Add(Snapshot.SnapshotId, Snapshot);
    SnapshotOrder.Add(Snapshot.SnapshotId);
    EvictOldSnapshots();

    ImageJson->SetStringField(TEXT("camera_snapshot_id"), Snapshot.SnapshotId);
    ImageJson->SetStringField(TEXT("camera_intrinsics_id"), Snapshot.IntrinsicsId);
    ImageJson->SetStringField(TEXT("view_id"), Snapshot.ViewId);
    ImageJson->SetStringField(TEXT("capture_group_id"), Snapshot.CaptureGroupId);
    ImageJson->SetArrayField(
        TEXT("camera_location_cm"), JsonVector(Snapshot.CameraLocation));
    ImageJson->SetArrayField(
        TEXT("camera_rotation_degrees"), JsonRotator(Snapshot.CameraRotation));
    ImageJson->SetNumberField(TEXT("camera_horizontal_fov_degrees"), ViewInfo.FOV);
    ImageJson->SetNumberField(TEXT("camera_snapshot_world_time_s"),
                              Snapshot.CapturedWorldTimeSeconds);
}

void USpPixelGoalSubsystem::EvictOldSnapshots()
{
    while (SnapshotOrder.Num() > MaxRetainedSnapshots)
    {
        const FString Oldest = SnapshotOrder[0];
        SnapshotOrder.RemoveAt(0, 1, EAllowShrinking::No);
        CameraSnapshots.Remove(Oldest);
    }
}

FString USpPixelGoalSubsystem::PixelGoal_CaptureFrameJson(const FString& RequestJson)
{
    const TSharedPtr<FJsonObject> Request = ParseJson(RequestJson);
    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetBoolField(TEXT("success"), false);
    if (!Request.IsValid())
    {
        Root->SetStringField(TEXT("error"), TEXT("invalid_request_json"));
        return SerializeJson(Root);
    }

    const FString AgentTag = JsonString(Request, TEXT("agent_tag"));
    const int32 Width = FMath::Clamp(
        FMath::RoundToInt(JsonNumber(Request, TEXT("width"), 640.0)), 16, 4096);
    const int32 Height = FMath::Clamp(
        FMath::RoundToInt(JsonNumber(Request, TEXT("height"), 360.0)), 16, 4096);
    const double Fov = FMath::Clamp(
        JsonNumber(Request, TEXT("fov_degrees"), 90.0), 5.0, 170.0);
    const int32 JpegQuality = FMath::Clamp(
        FMath::RoundToInt(JsonNumber(Request, TEXT("jpeg_quality"), 90.0)), 1, 95);
    if (AgentTag.IsEmpty())
    {
        Root->SetStringField(TEXT("error"), TEXT("missing_agent_tag"));
        return SerializeJson(Root);
    }

    UWorld* World = GetWorld();
    USpCameraCapturePool* Pool = World ? World->GetSubsystem<USpCameraCapturePool>() : nullptr;
    if (!Pool)
    {
        Root->SetStringField(TEXT("error"), TEXT("camera_capture_pool_unavailable"));
        return SerializeJson(Root);
    }

    const TSharedRef<FJsonObject> CaptureRequest = MakeShared<FJsonObject>();
    TArray<TSharedPtr<FJsonValue>> AgentTags;
    AgentTags.Add(MakeShared<FJsonValueString>(AgentTag));
    CaptureRequest->SetArrayField(TEXT("agent_tags"), AgentTags);
    CaptureRequest->SetNumberField(TEXT("width"), Width);
    CaptureRequest->SetNumberField(TEXT("height"), Height);
    CaptureRequest->SetNumberField(TEXT("fov_degrees"), Fov);
    CaptureRequest->SetNumberField(TEXT("jpeg_quality"), JpegQuality);
    const FString CaptureMode = SpPixelGoal::CaptureModeForAgentTag(AgentTag);
    CaptureRequest->SetStringField(TEXT("capture_mode"), CaptureMode);
    CaptureRequest->SetNumberField(TEXT("shared_pool_size"), 1);
    CaptureRequest->SetBoolField(TEXT("force_capture"), true);
    CaptureRequest->SetBoolField(TEXT("validate"), true);
    TArray<TSharedPtr<FJsonValue>> Modalities;
    Modalities.Add(MakeShared<FJsonValueString>(TEXT("rgb")));
    CaptureRequest->SetArrayField(TEXT("modalities"), Modalities);

    const TSharedPtr<FJsonObject> CaptureResponse = ParseJson(
        Pool->Camera_CaptureCamerasJson(SerializeJson(CaptureRequest)));
    if (!CaptureResponse.IsValid())
    {
        Root->SetStringField(TEXT("error"), TEXT("invalid_camera_capture_response"));
        return SerializeJson(Root);
    }
    const TArray<TSharedPtr<FJsonValue>>* Images = nullptr;
    if (!CaptureResponse->TryGetArrayField(TEXT("images"), Images) ||
        !Images || Images->Num() != 1)
    {
        Root->SetStringField(TEXT("error"), TEXT("pixel_goal_rgb_capture_failed"));
        Root->SetObjectField(TEXT("camera_capture_response"), CaptureResponse);
        return SerializeJson(Root);
    }
    const TSharedPtr<FJsonObject> Image = (*Images)[0]->AsObject();
    if (!Image.IsValid())
    {
        Root->SetStringField(TEXT("error"), TEXT("invalid_pixel_goal_rgb_image"));
        return SerializeJson(Root);
    }

    const FString DataUrl = JsonString(Image, TEXT("data_url"));
    const FString SnapshotId = JsonString(Image, TEXT("camera_snapshot_id"));
    const FString IntrinsicsId = JsonString(Image, TEXT("camera_intrinsics_id"));
    if (DataUrl.IsEmpty() || SnapshotId.IsEmpty() || IntrinsicsId.IsEmpty())
    {
        Root->SetStringField(TEXT("error"), TEXT("rgb_missing_pixel_goal_snapshot"));
        return SerializeJson(Root);
    }
    Root->SetBoolField(TEXT("success"), true);
    Root->SetStringField(TEXT("rgb_data_url"), DataUrl);
    Root->SetStringField(TEXT("camera_snapshot_id"), SnapshotId);
    Root->SetStringField(TEXT("camera_intrinsics_id"), IntrinsicsId);
    Root->SetNumberField(TEXT("width"), Width);
    Root->SetNumberField(TEXT("height"), Height);
    Root->SetStringField(TEXT("agent_tag"), AgentTag);
    Root->SetStringField(TEXT("capture_mode"), CaptureMode);
    Root->SetBoolField(
        TEXT("capture_render_state_persistent"),
        CaptureMode == TEXT("agent_native"));
    if (CaptureResponse->HasField(TEXT("timing")))
    {
        Root->SetField(
            TEXT("capture_timing"),
            CaptureResponse->TryGetField(TEXT("timing")));
    }
    if (Image->HasField(TEXT("camera_location_cm")))
    {
        Root->SetField(TEXT("camera_location_cm"), Image->TryGetField(TEXT("camera_location_cm")));
        Root->SetField(
            TEXT("camera_rotation_degrees"),
            Image->TryGetField(TEXT("camera_rotation_degrees")));
    }
    return SerializeJson(Root);
}

FString USpPixelGoalSubsystem::PixelGoal_CaptureViewPairJson(
    const FString& RequestJson)
{
    const double PairStartSeconds = FPlatformTime::Seconds();
    const TSharedPtr<FJsonObject> Request = ParseJson(RequestJson);
    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetBoolField(TEXT("success"), false);
    if (!Request.IsValid())
    {
        Root->SetStringField(TEXT("error"), TEXT("invalid_request_json"));
        return SerializeJson(Root);
    }

    FSpPixelGoalViewPairCaptureRequest PairRequest;
    const FString PairRequestError = SpPixelGoal::ParseViewPairCaptureRequest(
        Request, PairRequest);
    if (!PairRequestError.IsEmpty())
    {
        Root->SetStringField(TEXT("error"), PairRequestError);
        return SerializeJson(Root);
    }
    const FString& AgentTag = PairRequest.AgentTag;
    const int32 Width = PairRequest.Width;
    const int32 Height = PairRequest.Height;
    const float FovDegrees = PairRequest.HorizontalFovDegrees;
    const int32 JpegQuality = PairRequest.JpegQuality;
    const bool bValidate = PairRequest.bValidate;
    const bool bCommitSnapshots = PairRequest.bCommitSnapshots;

    for (const TPair<FString, FMoveAudit>& Pair : MoveAudits)
    {
        if (SpPixelGoal::ShouldSampleMoveAudit(Pair.Value.State))
        {
            Root->SetStringField(TEXT("error"), TEXT("pixel_goal_move_in_progress"));
            return SerializeJson(Root);
        }
    }
    if (!ParisPocSetup.bEnableRearCamera)
    {
        Root->SetStringField(TEXT("error"), TEXT("rear_camera_not_enabled"));
        return SerializeJson(Root);
    }

    UWorld* World = GetWorld();
    USpCameraCapturePool* Pool = World
        ? World->GetSubsystem<USpCameraCapturePool>()
        : nullptr;
    ASpHumanoidAgent* Agent = World
        ? FindActorWithTag<ASpHumanoidAgent>(World, FName(*AgentTag))
        : nullptr;
    if (!World || !Pool)
    {
        Root->SetStringField(TEXT("error"), TEXT("camera_capture_pool_unavailable"));
        return SerializeJson(Root);
    }
    if (!Agent)
    {
        Root->SetStringField(TEXT("error"), TEXT("agent_not_found"));
        return SerializeJson(Root);
    }
    AController* Controller = Agent->GetController();
    USpSceneCaptureComponent2D* FrontCapture =
        Agent->GetObservationCamera(TEXT("front"));
    USpSceneCaptureComponent2D* RearCapture =
        Agent->GetObservationCamera(TEXT("rear"));
    if (!Controller)
    {
        Root->SetStringField(TEXT("error"), TEXT("controller_unavailable"));
        return SerializeJson(Root);
    }
    if (!IsValid(FrontCapture) || !IsValid(RearCapture) ||
        !FrontCapture->IsInitialized() || !RearCapture->IsInitialized() ||
        !IsValid(FrontCapture->TextureTarget) ||
        !IsValid(RearCapture->TextureTarget))
    {
        Root->SetStringField(TEXT("error"), TEXT("view_pair_cameras_not_ready"));
        return SerializeJson(Root);
    }
    const FString WarmedCameraError = SpPixelGoal::WarmedViewPairCameraError(
        Agent, Width, Height, FovDegrees);
    if (!WarmedCameraError.IsEmpty())
    {
        Root->SetStringField(TEXT("error"), WarmedCameraError);
        return SerializeJson(Root);
    }

    const FString CaptureGroupId = FString::Printf(
        TEXT("view-pair-%llu"), NextCaptureGroupNumber++);
    const FVector ActorLocationBefore = Agent->GetActorLocation();
    const FRotator ActorRotationBefore = Agent->GetActorRotation();
    const FRotator ControlRotationBefore = Controller->GetControlRotation();
    USpringArmComponent* const SpringArmBefore = Agent->SpringArm;
    const FVector SpringLocationBefore = SpringArmBefore->GetRelativeLocation();
    const FRotator SpringRotationBefore = SpringArmBefore->GetRelativeRotation();
    const FViewPairCameraRigState FrontRigBefore(FrontCapture);
    const FViewPairCameraRigState RearRigBefore(RearCapture);
    const TMap<FString, FSpPixelGoalCameraSnapshot> CameraSnapshotsBefore =
        CameraSnapshots;
    const TArray<FString> SnapshotOrderBefore = SnapshotOrder;

    const auto RestoreSnapshotState = [
        this,
        &CameraSnapshotsBefore,
        &SnapshotOrderBefore]()
    {
        CameraSnapshots = CameraSnapshotsBefore;
        SnapshotOrder = SnapshotOrderBefore;
    };
    const auto FailPair = [&Root, &RestoreSnapshotState](const FString& Error)
    {
        RestoreSnapshotState();
        Root->SetStringField(TEXT("error"), Error);
        return SerializeJson(Root);
    };
    const auto CameraSourceJson = [&AgentTag](
        const FString& CameraId,
        const FString& ViewId)
    {
        const TSharedRef<FJsonObject> Source = MakeShared<FJsonObject>();
        Source->SetStringField(TEXT("camera_id"), CameraId);
        Source->SetStringField(TEXT("agent_tag"), AgentTag);
        Source->SetStringField(TEXT("camera_view"), ViewId);
        return MakeShared<FJsonValueObject>(Source);
    };

    const TSharedRef<FJsonObject> CaptureRequest = MakeShared<FJsonObject>();
    CaptureRequest->SetStringField(TEXT("capture_group_id"), CaptureGroupId);
    CaptureRequest->SetStringField(TEXT("capture_mode"), TEXT("agent_native"));
    CaptureRequest->SetArrayField(TEXT("camera_sources"), {
        CameraSourceJson(TEXT("front"), TEXT("front")),
        CameraSourceJson(TEXT("rear"), TEXT("rear")),
    });
    CaptureRequest->SetNumberField(TEXT("width"), Width);
    CaptureRequest->SetNumberField(TEXT("height"), Height);
    CaptureRequest->SetNumberField(TEXT("fov_degrees"), FovDegrees);
    CaptureRequest->SetNumberField(TEXT("jpeg_quality"), JpegQuality);
    // Preview pairs read the continuously rendered persistent targets.  An
    // explicit CaptureScene on bCaptureEveryFrame cameras renders twice and
    // materially increases the very GPU pressure this readiness path audits.
    // The final actionable pair still forces one pose-fresh capture, and the
    // caller verifies that it matches the converged previews.
    CaptureRequest->SetBoolField(TEXT("force_capture"), bCommitSnapshots);
    CaptureRequest->SetBoolField(TEXT("validate"), bValidate);
    CaptureRequest->SetArrayField(TEXT("modalities"), {
        MakeShared<FJsonValueString>(TEXT("rgb")),
    });

    ActiveCaptureGroupId = CaptureGroupId;
    const TSharedPtr<FJsonObject> CaptureResponse = ParseJson(
        Pool->Camera_CaptureCamerasJson(SerializeJson(CaptureRequest)));
    ActiveCaptureGroupId.Reset();

#if WITH_DEV_AUTOMATION_TESTS
    if (!ViewPairFailureAfterBatchForTest.IsEmpty())
    {
        const FString InjectedError = ViewPairFailureAfterBatchForTest;
        ViewPairFailureAfterBatchForTest.Reset();
        return FailPair(InjectedError);
    }
#endif

    const bool bActorPoseUnchanged =
        IsValid(Agent) &&
        Agent->GetActorLocation().Equals(ActorLocationBefore, 0.f) &&
        Agent->GetActorRotation().Equals(ActorRotationBefore, 0.0);
    const bool bControllerUnchanged =
        IsValid(Agent) && IsValid(Controller) &&
        Controller == Agent->GetController() &&
        Controller->GetControlRotation().Equals(ControlRotationBefore, 0.0);
    const bool bSpringUnchanged =
        IsValid(Agent) && IsValid(SpringArmBefore) &&
        Agent->SpringArm == SpringArmBefore &&
        SpringArmBefore->GetRelativeLocation().Equals(SpringLocationBefore, 0.f) &&
        SpringArmBefore->GetRelativeRotation().Equals(
            SpringRotationBefore, 0.0);
    const bool bCameraRigUnchanged =
        IsValid(Agent) &&
        FrontRigBefore.Matches(Agent->GetObservationCamera(TEXT("front"))) &&
        RearRigBefore.Matches(Agent->GetObservationCamera(TEXT("rear"))) &&
        SpPixelGoal::WarmedViewPairCameraError(
            Agent, Width, Height, FovDegrees).IsEmpty();
    if (!bActorPoseUnchanged || !bSpringUnchanged || !bCameraRigUnchanged)
    {
        return FailPair(TEXT("view_pair_pose_changed"));
    }
    if (!bControllerUnchanged)
    {
        return FailPair(TEXT("view_pair_controller_changed"));
    }
    if (!CaptureResponse.IsValid())
    {
        return FailPair(TEXT("invalid_camera_capture_response"));
    }

    const TArray<TSharedPtr<FJsonValue>>* Images = nullptr;
    if (!CaptureResponse->TryGetArrayField(TEXT("images"), Images) ||
        !Images || Images->Num() != 2)
    {
        return FailPair(TEXT("incomplete_view_pair"));
    }

    TArray<TSharedPtr<FJsonValue>> OutputViews;
    FString SnapshotIds[2];
    FString IntrinsicsIds[2];
    const FString ExpectedViews[] = {TEXT("front"), TEXT("rear")};
    USceneCaptureComponent2D* ExpectedComponents[] = {FrontCapture, RearCapture};
    for (int32 Index = 0; Index < 2; ++Index)
    {
        const TSharedPtr<FJsonObject> Image = (*Images)[Index].IsValid()
            ? (*Images)[Index]->AsObject()
            : nullptr;
        const FString ImageError = SpPixelGoal::ViewPairImageError(
            Image,
            ExpectedViews[Index],
            AgentTag,
            CaptureGroupId,
            Width,
            Height,
            ActorLocationBefore,
            ActorRotationBefore.Yaw,
            SnapshotIds[Index],
            IntrinsicsIds[Index]);
        if (!ImageError.IsEmpty())
        {
            return FailPair(ImageError);
        }
        const FSpPixelGoalCameraSnapshot* Snapshot =
            CameraSnapshots.Find(SnapshotIds[Index]);
        if (!Snapshot ||
            Snapshot->Agent.Get() != Agent ||
            !Snapshot->AgentLocation.Equals(ActorLocationBefore, 0.f) ||
            !Snapshot->AgentRotation.Equals(ActorRotationBefore, 0.0))
        {
            return FailPair(TEXT("invalid_view_pair_snapshot"));
        }
        Image->SetStringField(TEXT("rgb_data_url"),
            JsonString(Image, TEXT("data_url")));
        OutputViews.Add(MakeShared<FJsonValueObject>(Image));
    }
    const FSpPixelGoalCameraSnapshot* FrontSnapshot =
        CameraSnapshots.Find(SnapshotIds[0]);
    const FSpPixelGoalCameraSnapshot* RearSnapshot =
        CameraSnapshots.Find(SnapshotIds[1]);
    if (!FrontSnapshot || !RearSnapshot)
    {
        return FailPair(TEXT("invalid_view_pair_snapshot"));
    }
    const FString SnapshotPairError = SpPixelGoal::ViewPairSnapshotError(
        *FrontSnapshot,
        *RearSnapshot,
        SnapshotIds[0],
        SnapshotIds[1],
        IntrinsicsIds[0],
        IntrinsicsIds[1],
        Width,
        Height,
        FovDegrees,
        CaptureGroupId,
        ExpectedComponents[0],
        ExpectedComponents[1]);
    if (!SnapshotPairError.IsEmpty())
    {
        return FailPair(SnapshotPairError);
    }

    const TSharedRef<FJsonObject> Pose = MakeShared<FJsonObject>();
    Pose->SetNumberField(TEXT("x_cm"), ActorLocationBefore.X);
    Pose->SetNumberField(TEXT("y_cm"), ActorLocationBefore.Y);
    Pose->SetNumberField(TEXT("z_cm"), ActorLocationBefore.Z);
    Pose->SetNumberField(TEXT("yaw_deg"), ActorRotationBefore.Yaw);
    Root->SetBoolField(TEXT("success"), true);
    Root->SetBoolField(TEXT("snapshots_committed"), bCommitSnapshots);
    Root->SetStringField(TEXT("capture_group_id"), CaptureGroupId);
    Root->SetStringField(TEXT("agent_tag"), AgentTag);
    Root->SetObjectField(TEXT("pose"), Pose);
    Root->SetArrayField(TEXT("views"), OutputViews);
    Root->SetNumberField(
        TEXT("pair_capture_wall_ms"),
        FMath::Max(0.0, (FPlatformTime::Seconds() - PairStartSeconds) * 1000.0));
    if (CaptureResponse->HasField(TEXT("timing")))
    {
        Root->SetField(TEXT("capture_timing"), CaptureResponse->TryGetField(TEXT("timing")));
    }
    if (!bCommitSnapshots)
    {
        RestoreSnapshotState();
    }
    return SerializeJson(Root);
}

FString USpPixelGoalSubsystem::PixelGoal_SpawnCalibrationFixtureJson(
    const FString& RequestJson)
{
    const TSharedPtr<FJsonObject> Request = RequestJson.TrimStartAndEnd().IsEmpty()
        ? MakeShared<FJsonObject>()
        : ParseJson(RequestJson);
    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetBoolField(TEXT("success"), false);
    if (!Request.IsValid())
    {
        Root->SetStringField(TEXT("error"), TEXT("invalid_request_json"));
        return SerializeJson(Root);
    }
    UWorld* World = GetWorld();
    if (!World)
    {
        Root->SetStringField(TEXT("error"), TEXT("no_world"));
        return SerializeJson(Root);
    }

    CalibrationHelper = FindActorWithTag<ASpNavMeshHelper>(World, CalibrationHelperTag);
    CalibrationFloor = FindActorWithTag<AActor>(World, CalibrationFloorTag);
    CalibrationAgent = FindActorWithTag<ASpHumanoidAgent>(World, CalibrationAgentTag);
    CalibrationLight = FindActorWithTag<AActor>(World, CalibrationLightTag);
    CalibrationWall = FindActorWithTag<AActor>(World, CalibrationWallTag);
    if (CalibrationHelper && CalibrationFloor && CalibrationAgent &&
        CalibrationLight && CalibrationWall)
    {
        Root->SetBoolField(TEXT("success"), true);
        Root->SetBoolField(TEXT("reused"), true);
        Root->SetStringField(TEXT("agent_tag"), CalibrationAgentTag.ToString());
        return SerializeJson(Root);
    }

    FActorSpawnParameters Params;
    Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    CalibrationHelper = World->SpawnActor<ASpNavMeshHelper>(
        ASpNavMeshHelper::StaticClass(),
        FVector(0.f, 0.f, 50000.f),
        FRotator::ZeroRotator,
        Params);
    if (!CalibrationHelper)
    {
        Root->SetStringField(TEXT("error"), TEXT("calibration_nav_helper_spawn_failed"));
        return SerializeJson(Root);
    }
    CalibrationHelper->Tags.AddUnique(CalibrationHelperTag);
    UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(
        nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
    CalibrationFloor = SpawnCalibrationCube(
        World,
        CubeMesh,
        CalibrationFloorTag,
        FVector(0.f, 0.f, 50000.f),
        FVector(60.f, 60.f, 0.2f),
        true);
    if (!CalibrationFloor || !CalibrationHelper->SpawnNavMeshBounds(
            FVector(0.f, 0.f, 50100.f), FVector(2500.f, 2500.f, 500.f)))
    {
        Root->SetStringField(TEXT("error"), TEXT("calibration_floor_or_nav_bounds_failed"));
        return SerializeJson(Root);
    }

    ADirectionalLight* DirectionalLight = World->SpawnActor<ADirectionalLight>(
        ADirectionalLight::StaticClass(),
        FVector::ZeroVector,
        FRotator(-45.f, 0.f, 0.f),
        Params);
    UDirectionalLightComponent* LightComponent = DirectionalLight
        ? Cast<UDirectionalLightComponent>(DirectionalLight->GetLightComponent())
        : nullptr;
    if (!DirectionalLight || !LightComponent)
    {
        Root->SetStringField(TEXT("error"), TEXT("calibration_light_spawn_failed"));
        return SerializeJson(Root);
    }
    DirectionalLight->Tags.AddUnique(CalibrationLightTag);
    LightComponent->SetMobility(EComponentMobility::Movable);
    LightComponent->SetIntensity(20.f);
    LightComponent->SetLightColor(FLinearColor(1.f, 0.95f, 0.85f));
    CalibrationLight = DirectionalLight;

    CalibrationAgent = World->SpawnActor<ASpHumanoidAgent>(
        ASpHumanoidAgent::StaticClass(),
        FVector(0.f, 0.f, CalibrationFloorTopZCm + 90.f),
        FRotator::ZeroRotator,
        Params);
    if (!CalibrationAgent)
    {
        Root->SetStringField(TEXT("error"), TEXT("calibration_agent_spawn_failed"));
        return SerializeJson(Root);
    }
    CalibrationAgent->AgentTag = CalibrationAgentTag;
    CalibrationAgent->Tags.AddUnique(CalibrationAgentTag);
    const float CapsuleHalfHeight = CalibrationAgent->GetCapsuleComponent()
        ? CalibrationAgent->GetCapsuleComponent()->GetScaledCapsuleHalfHeight()
        : 90.f;
    CalibrationAgent->SetActorLocation(
        FVector(0.f, 0.f, CalibrationFloorTopZCm + CapsuleHalfHeight));
    CalibrationAgent->SetActorRotation(FRotator::ZeroRotator);
    if (CalibrationAgent->SpringArm)
    {
        CalibrationAgent->SpringArm->TargetArmLength = 0.f;
        CalibrationAgent->SpringArm->SetRelativeLocation(FVector(20.f, 0.f, 60.f));
        CalibrationAgent->SpringArm->SetRelativeRotation(FRotator::ZeroRotator);
    }
    if (CalibrationAgent->SceneCapture)
    {
        CalibrationAgent->SceneCapture->SetRelativeLocation(FVector::ZeroVector);
        CalibrationAgent->SceneCapture->SetRelativeRotation(FRotator::ZeroRotator);
    }
    CalibrationAgent->ConfigureCamera(640, 360, 90.f);

    AStaticMeshActor* Wall = SpawnCalibrationCube(
        World,
        CubeMesh,
        CalibrationWallTag,
        FVector(1500.f, 0.f, CalibrationFloorTopZCm + 100.f),
        FVector(0.2f, 4.f, 2.f),
        false);
    if (!Wall || !Wall->GetStaticMeshComponent())
    {
        Root->SetStringField(TEXT("error"), TEXT("calibration_wall_spawn_failed"));
        return SerializeJson(Root);
    }
    CalibrationWall = Wall;

    CalibrationHelper->BuildNavMesh();
    Root->SetBoolField(TEXT("success"), true);
    Root->SetBoolField(TEXT("reused"), false);
    Root->SetStringField(TEXT("agent_tag"), CalibrationAgentTag.ToString());
    Root->SetNumberField(TEXT("floor_top_z_cm"), CalibrationFloorTopZCm);
    Root->SetArrayField(TEXT("floor_size_cm"), JsonVector(FVector(6000.f, 6000.f, 20.f)));
    Root->SetArrayField(TEXT("nav_bounds_extent_cm"), JsonVector(FVector(2500.f, 2500.f, 500.f)));
    return SerializeJson(Root);
}

FString USpPixelGoalSubsystem::PixelGoal_GetCalibrationStatusJson(
    const FString& RequestJson)
{
    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    UWorld* World = GetWorld();
    if (!CalibrationHelper)
    {
        CalibrationHelper = FindActorWithTag<ASpNavMeshHelper>(
            World, CalibrationHelperTag);
    }
    if (!CalibrationAgent)
    {
        CalibrationAgent = FindActorWithTag<ASpHumanoidAgent>(
            World, CalibrationAgentTag);
    }
    if (!CalibrationFloor)
    {
        CalibrationFloor = FindActorWithTag<AActor>(World, CalibrationFloorTag);
    }
    if (!CalibrationWall)
    {
        CalibrationWall = FindActorWithTag<AActor>(World, CalibrationWallTag);
    }
    if (!CalibrationLight)
    {
        CalibrationLight = FindActorWithTag<AActor>(World, CalibrationLightTag);
    }
    const bool bFixturePresent = CalibrationHelper && CalibrationFloor &&
        CalibrationAgent && CalibrationLight && CalibrationWall;
    const UCharacterMovementComponent* Movement = CalibrationAgent
        ? CalibrationAgent->GetCharacterMovement()
        : nullptr;
    const bool bAgentGrounded = Movement && Movement->IsMovingOnGround();
    bool bNavMeshReady = false;
    float NavMeshAdjustmentCm = -1.f;
    if (bFixturePresent && CalibrationHelper->IsNavMeshReady())
    {
        UNavigationSystemV1* NavSys = UNavigationSystemV1::GetNavigationSystem(World);
        FNavLocation NavLocation;
        const FVector Feet = GetFeetLocation(CalibrationAgent);
        bNavMeshReady = NavSys && NavSys->ProjectPointToNavigation(
            Feet,
            NavLocation,
            FVector(50.f, 50.f, 100.f),
            &CalibrationAgent->GetNavAgentPropertiesRef());
        if (bNavMeshReady)
        {
            NavMeshAdjustmentCm = FVector::Distance(Feet, NavLocation.Location);
        }
    }
    Root->SetBoolField(TEXT("fixture_present"),
                       bFixturePresent);
    Root->SetBoolField(TEXT("agent_grounded"), bAgentGrounded);
    Root->SetBoolField(TEXT("navmesh_ready"), bNavMeshReady);
    Root->SetBoolField(TEXT("fixture_ready"), bFixturePresent && bAgentGrounded && bNavMeshReady);
    Root->SetNumberField(TEXT("registered_nav_data"),
                         CalibrationHelper ? CalibrationHelper->GetNumRegisteredNavData() : 0);
    Root->SetNumberField(TEXT("navmesh_adjustment_at_agent_cm"), NavMeshAdjustmentCm);
    if (CalibrationAgent)
    {
        Root->SetArrayField(TEXT("agent_feet_position_cm"),
                            JsonVector(GetFeetLocation(CalibrationAgent)));
        Root->SetNumberField(
            TEXT("agent_vertical_speed_cm_s"),
            Movement ? Movement->Velocity.Z : 0.f);
    }
    return SerializeJson(Root);
}

FString USpPixelGoalSubsystem::PixelGoal_SetupParisPocJson(
    const FString& RequestJson)
{
    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetBoolField(TEXT("success"), false);
    UWorld* World = GetWorld();
    if (!World)
    {
        Root->SetStringField(TEXT("error"), TEXT("no_world"));
        return SerializeJson(Root);
    }

    FSpPixelGoalParisPocSetup RequestedSetup;
    FString Error;
    if (!SpPixelGoal::ParseParisPocSetupRequest(
            RequestJson, World->GetMapName(), RequestedSetup, Error))
    {
        Root->SetStringField(TEXT("error"), Error);
        Root->SetStringField(TEXT("active_map"), World->GetMapName());
        return SerializeJson(Root);
    }

    float ParisMieBefore = -1.f;
    float ParisMieAfter = -1.f;
    const bool bParisSkyAtmosphereConfigured = PrepareParisPocAtmosphere(
        World, ParisMieBefore, ParisMieAfter);
    const auto AddParisRenderingFields = [&]()
    {
        Root->SetBoolField(
            TEXT("paris_sky_atmosphere_configured"),
            bParisSkyAtmosphereConfigured);
        Root->SetNumberField(
            TEXT("paris_sky_mie_before"),
            ParisMieBefore);
        Root->SetNumberField(
            TEXT("paris_sky_mie_scattering_scale"),
            ParisMieAfter);
        Root->SetStringField(
            TEXT("rgb_capture_contract_version"),
            ParisPocCaptureContractVersion);
        Root->SetStringField(
            TEXT("rgb_capture_auto_exposure_method"), TEXT("histogram"));
        Root->SetNumberField(
            TEXT("rgb_capture_auto_exposure_min_ev100"),
            ParisPocExposureMinEv100);
        Root->SetNumberField(
            TEXT("rgb_capture_auto_exposure_max_ev100"),
            ParisPocExposureMaxEv100);
        Root->SetNumberField(
            TEXT("rgb_capture_auto_exposure_speed_up"),
            ParisPocExposureSpeedUp);
        Root->SetNumberField(
            TEXT("rgb_capture_auto_exposure_speed_down"),
            ParisPocExposureSpeedDown);
    };

    ParisPocHelper = FindActorWithTag<ASpNavMeshHelper>(World, ParisPocHelperTag);
    ParisPocAgent = FindActorWithTag<ASpHumanoidAgent>(World, ParisPocAgentTag);
    if (ParisPocHelper && ParisPocAgent)
    {
        if (!PrepareParisPocCapture(
                ParisPocAgent, RequestedSetup.bEnableRearCamera))
        {
            Root->SetStringField(
                TEXT("error"), TEXT("paris_persistent_capture_init_failed"));
            return SerializeJson(Root);
        }
        ParisPocSetup = RequestedSetup;
        Root->SetBoolField(TEXT("success"), true);
        Root->SetBoolField(TEXT("reused"), true);
        Root->SetBoolField(TEXT("synthetic_floor_spawned"), false);
        Root->SetStringField(TEXT("active_map"), World->GetMapName());
        Root->SetStringField(TEXT("agent_tag"), ParisPocAgentTag.ToString());
        Root->SetStringField(TEXT("region_name"), RequestedSetup.RegionName);
        Root->SetStringField(TEXT("rgb_capture_mode"), TEXT("agent_native"));
        Root->SetBoolField(TEXT("rgb_capture_render_state_persistent"), true);
        Root->SetBoolField(
            TEXT("rear_camera_enabled"), RequestedSetup.bEnableRearCamera);
        Root->SetNumberField(
            TEXT("rgb_capture_warmup_seconds"),
            ParisPocCaptureWarmupSeconds);
        Root->SetNumberField(
            TEXT("rgb_capture_auto_exposure_bias"),
            ParisPocExposureBias);
        AddParisRenderingFields();
        return SerializeJson(Root);
    }

    FActorSpawnParameters Params;
    Params.SpawnCollisionHandlingOverride =
        ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    ParisPocHelper = World->SpawnActor<ASpNavMeshHelper>(
        ASpNavMeshHelper::StaticClass(),
        RequestedSetup.NavBoundsCenter,
        FRotator::ZeroRotator,
        Params);
    if (!ParisPocHelper)
    {
        Root->SetStringField(TEXT("error"), TEXT("paris_nav_helper_spawn_failed"));
        return SerializeJson(Root);
    }
    ParisPocHelper->Tags.AddUnique(ParisPocHelperTag);
    if (!ParisPocHelper->SpawnNavMeshBounds(
            RequestedSetup.NavBoundsCenter, RequestedSetup.NavBoundsExtent))
    {
        Root->SetStringField(TEXT("error"), TEXT("paris_nav_bounds_spawn_failed"));
        return SerializeJson(Root);
    }

    ParisPocAgent = World->SpawnActor<ASpHumanoidAgent>(
        ASpHumanoidAgent::StaticClass(),
        RequestedSetup.AgentSpawn,
        FRotator(0.f, RequestedSetup.AgentYawDegrees, 0.f),
        Params);
    if (!ParisPocAgent)
    {
        Root->SetStringField(TEXT("error"), TEXT("paris_agent_spawn_failed"));
        return SerializeJson(Root);
    }
    ParisPocAgent->Agent_SetAgentTag(ParisPocAgentTag);
    ParisPocAgent->SetActorLocation(RequestedSetup.AgentSpawn);
    ParisPocAgent->SetActorRotation(
        FRotator(0.f, RequestedSetup.AgentYawDegrees, 0.f));
    if (ParisPocAgent->SpringArm)
    {
        ParisPocAgent->SpringArm->TargetArmLength = 0.f;
        ParisPocAgent->SpringArm->SetRelativeLocation(FVector(20.f, 0.f, 60.f));
        ParisPocAgent->SpringArm->SetRelativeRotation(FRotator::ZeroRotator);
    }
    if (ParisPocAgent->SceneCapture)
    {
        ParisPocAgent->SceneCapture->SetRelativeLocation(FVector::ZeroVector);
        ParisPocAgent->SceneCapture->SetRelativeRotation(FRotator::ZeroRotator);
    }
    if (!PrepareParisPocCapture(
            ParisPocAgent, RequestedSetup.bEnableRearCamera))
    {
        Root->SetStringField(
            TEXT("error"), TEXT("paris_persistent_capture_init_failed"));
        return SerializeJson(Root);
    }

    ParisPocSetup = RequestedSetup;
    ParisPocHelper->BuildNavMesh();
    const bool bDynamicRuntimeGeneration =
        ParisPocHelper->IsDynamicRuntimeGeneration();
    ParisPocHelper->DumpNavState();

    Root->SetBoolField(TEXT("success"), true);
    Root->SetBoolField(TEXT("reused"), false);
    Root->SetBoolField(TEXT("synthetic_floor_spawned"), false);
    Root->SetBoolField(
        TEXT("dynamic_runtime_generation"), bDynamicRuntimeGeneration);
    Root->SetBoolField(
        TEXT("supports_runtime_generation"),
        ParisPocHelper->SupportsRuntimeNavGeneration());
    Root->SetBoolField(
        TEXT("navmesh_has_valid_data"), ParisPocHelper->HasValidNavMeshData());
    Root->SetNumberField(
        TEXT("active_navmesh_tiles"),
        ParisPocHelper->GetNumActiveNavMeshTiles());
    Root->SetStringField(TEXT("active_map"), World->GetMapName());
    Root->SetStringField(TEXT("scene"), ParisPocScene);
    Root->SetStringField(TEXT("region_name"), RequestedSetup.RegionName);
    Root->SetStringField(TEXT("agent_tag"), ParisPocAgentTag.ToString());
    Root->SetStringField(TEXT("rgb_capture_mode"), TEXT("agent_native"));
    Root->SetBoolField(TEXT("rgb_capture_render_state_persistent"), true);
    Root->SetBoolField(
        TEXT("rear_camera_enabled"), RequestedSetup.bEnableRearCamera);
    Root->SetNumberField(
        TEXT("rgb_capture_warmup_seconds"),
        ParisPocCaptureWarmupSeconds);
    Root->SetNumberField(
        TEXT("rgb_capture_auto_exposure_bias"),
        ParisPocExposureBias);
    Root->SetArrayField(
        TEXT("agent_spawn_cm"), JsonVector(RequestedSetup.AgentSpawn));
    Root->SetNumberField(
        TEXT("agent_yaw_deg"), RequestedSetup.AgentYawDegrees);
    Root->SetArrayField(
        TEXT("nav_bounds_center_cm"), JsonVector(RequestedSetup.NavBoundsCenter));
    Root->SetArrayField(
        TEXT("nav_bounds_extent_cm"), JsonVector(RequestedSetup.NavBoundsExtent));
    AddParisRenderingFields();
    return SerializeJson(Root);
}

FString USpPixelGoalSubsystem::PixelGoal_GetParisPocStatusJson(
    const FString& RequestJson)
{
    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    UWorld* World = GetWorld();
    if (!ParisPocHelper)
    {
        ParisPocHelper = FindActorWithTag<ASpNavMeshHelper>(
            World, ParisPocHelperTag);
    }
    if (!ParisPocAgent)
    {
        ParisPocAgent = FindActorWithTag<ASpHumanoidAgent>(World, ParisPocAgentTag);
    }

    const bool bMapMatches = World && IsParisPocMapName(World->GetMapName());
    const bool bSetupPresent = ParisPocHelper && ParisPocAgent;
    const bool bControllerPresent = ParisPocAgent && ParisPocAgent->GetController();
    const UCharacterMovementComponent* Movement = ParisPocAgent
        ? ParisPocAgent->GetCharacterMovement()
        : nullptr;
    const bool bAgentGrounded = Movement && Movement->IsMovingOnGround();
    bool bAgentOnNavMesh = false;
    float NavMeshAdjustmentCm = -1.f;
    FVector ProjectedFeet = FVector::ZeroVector;
    const bool bNavDataReady = ParisPocHelper && ParisPocHelper->IsNavMeshReady();
    const bool bDynamicRuntimeGeneration = ParisPocHelper &&
        ParisPocHelper->IsDynamicRuntimeGeneration();
    const bool bSupportsRuntimeGeneration = ParisPocHelper &&
        ParisPocHelper->SupportsRuntimeNavGeneration();
    const bool bNavMeshHasValidData = ParisPocHelper &&
        ParisPocHelper->HasValidNavMeshData();
    const int32 ActiveNavMeshTiles = ParisPocHelper
        ? ParisPocHelper->GetNumActiveNavMeshTiles()
        : 0;
    if (bSetupPresent && bNavDataReady)
    {
        UNavigationSystemV1* NavSys =
            UNavigationSystemV1::GetNavigationSystem(World);
        FNavLocation NavLocation;
        const FVector Feet = GetFeetLocation(ParisPocAgent);
        bAgentOnNavMesh = NavSys && NavSys->ProjectPointToNavigation(
            Feet,
            NavLocation,
            FVector(50.f, 50.f, 100.f),
            &ParisPocAgent->GetNavAgentPropertiesRef());
        if (bAgentOnNavMesh)
        {
            ProjectedFeet = NavLocation.Location;
            NavMeshAdjustmentCm = FVector::Distance(Feet, ProjectedFeet);
        }
    }
    const bool bReady = bMapMatches && bSetupPresent && bControllerPresent &&
        bAgentGrounded && bNavDataReady && bAgentOnNavMesh &&
        bDynamicRuntimeGeneration && bSupportsRuntimeGeneration &&
        bNavMeshHasValidData && ActiveNavMeshTiles > 0;
    Root->SetBoolField(TEXT("poc_ready"), bReady);
    Root->SetBoolField(TEXT("map_matches"), bMapMatches);
    Root->SetBoolField(TEXT("setup_present"), bSetupPresent);
    Root->SetBoolField(TEXT("controller_present"), bControllerPresent);
    Root->SetBoolField(TEXT("agent_grounded"), bAgentGrounded);
    Root->SetBoolField(TEXT("nav_data_ready"), bNavDataReady);
    Root->SetBoolField(TEXT("agent_on_navmesh"), bAgentOnNavMesh);
    Root->SetBoolField(
        TEXT("dynamic_runtime_generation"), bDynamicRuntimeGeneration);
    Root->SetBoolField(
        TEXT("supports_runtime_generation"), bSupportsRuntimeGeneration);
    Root->SetBoolField(TEXT("navmesh_has_valid_data"), bNavMeshHasValidData);
    Root->SetNumberField(TEXT("active_navmesh_tiles"), ActiveNavMeshTiles);
    Root->SetBoolField(TEXT("synthetic_floor_spawned"), false);
    Root->SetStringField(
        TEXT("active_map"), World ? World->GetMapName() : FString());
    Root->SetStringField(TEXT("region_name"), ParisPocSetup.RegionName);
    Root->SetBoolField(
        TEXT("rear_camera_enabled"), ParisPocSetup.bEnableRearCamera);
    Root->SetNumberField(
        TEXT("registered_nav_data"),
        ParisPocHelper ? ParisPocHelper->GetNumRegisteredNavData() : 0);
    Root->SetNumberField(
        TEXT("registered_nav_bounds"),
        ParisPocHelper ? ParisPocHelper->GetNumNavBounds() : 0);
    Root->SetNumberField(
        TEXT("navmesh_adjustment_at_agent_cm"), NavMeshAdjustmentCm);
    if (ParisPocAgent)
    {
        Root->SetArrayField(
            TEXT("agent_feet_position_cm"), JsonVector(GetFeetLocation(ParisPocAgent)));
        Root->SetArrayField(
            TEXT("agent_position_cm"), JsonVector(ParisPocAgent->GetActorLocation()));
    }
    if (bAgentOnNavMesh)
    {
        Root->SetArrayField(
            TEXT("projected_agent_feet_cm"), JsonVector(ProjectedFeet));
    }
    return SerializeJson(Root);
}

FString USpPixelGoalSubsystem::PixelGoal_ResetParisPocTrialJson(
    const FString& RequestJson)
{
    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetBoolField(TEXT("success"), false);
    UWorld* World = GetWorld();
    if (!World || !IsParisPocMapName(World->GetMapName()))
    {
        Root->SetStringField(TEXT("error"), TEXT("paris_map_not_loaded"));
        return SerializeJson(Root);
    }
    if (!ParisPocAgent)
    {
        ParisPocAgent = FindActorWithTag<ASpHumanoidAgent>(
            World, ParisPocAgentTag);
    }
    if (!ParisPocHelper)
    {
        ParisPocHelper = FindActorWithTag<ASpNavMeshHelper>(
            World, ParisPocHelperTag);
    }
    if (!ParisPocAgent || !ParisPocHelper || ParisPocSetup.RegionName.IsEmpty())
    {
        Root->SetStringField(TEXT("error"), TEXT("paris_poc_not_setup"));
        return SerializeJson(Root);
    }
    for (const TPair<FString, FMoveAudit>& Pair : MoveAudits)
    {
        if (SpPixelGoal::ShouldSampleMoveAudit(Pair.Value.State))
        {
            Root->SetStringField(TEXT("error"), TEXT("pixel_goal_move_in_progress"));
            return SerializeJson(Root);
        }
    }

    AAIController* Controller = Cast<AAIController>(ParisPocAgent->GetController());
    if (!Controller)
    {
        Root->SetStringField(TEXT("error"), TEXT("paris_controller_unavailable"));
        return SerializeJson(Root);
    }
    Controller->StopMovement();
    Controller->SetControlRotation(
        FRotator(0.f, ParisPocSetup.AgentYawDegrees, 0.f));
    if (UCharacterMovementComponent* Movement =
            ParisPocAgent->GetCharacterMovement())
    {
        Movement->StopMovementImmediately();
    }
    const bool bReset = ParisPocAgent->SetActorLocationAndRotation(
        ParisPocSetup.AgentSpawn,
        FRotator(0.f, ParisPocSetup.AgentYawDegrees, 0.f),
        false,
        nullptr,
        ETeleportType::TeleportPhysics);
    if (!bReset)
    {
        Root->SetStringField(TEXT("error"), TEXT("paris_agent_reset_failed"));
        return SerializeJson(Root);
    }

    // A trial reset is an internal fixture operation, never a policy action.
    // Invalidate every pre-reset view so the next Pixel Goal can only use the
    // exact RGB/camera snapshot captured after the deterministic reset.
    CameraSnapshots.Empty();
    SnapshotOrder.Empty();

    Root->SetBoolField(TEXT("success"), true);
    Root->SetBoolField(TEXT("controller_stopped"), true);
    Root->SetBoolField(TEXT("camera_snapshots_invalidated"), true);
    Root->SetStringField(TEXT("agent_tag"), ParisPocAgentTag.ToString());
    Root->SetStringField(TEXT("region_name"), ParisPocSetup.RegionName);
    Root->SetArrayField(
        TEXT("reset_agent_position_cm"),
        JsonVector(ParisPocAgent->GetActorLocation()));
    Root->SetArrayField(
        TEXT("reset_feet_position_cm"), JsonVector(GetFeetLocation(ParisPocAgent)));
    Root->SetNumberField(
        TEXT("reset_yaw_deg"), ParisPocAgent->GetActorRotation().Yaw);
    return SerializeJson(Root);
}

FString USpPixelGoalSubsystem::PixelGoal_GetCrosswalkCatalogJson(
    const FString& RequestJson)
{
    const TSharedPtr<FJsonObject> Request = ParseJson(RequestJson);
    FString AgentTag;
    if (!HasExactJsonFields(Request, {TEXT("agent_tag")}) ||
        !TryGetExactJsonString(Request, TEXT("agent_tag"), AgentTag) ||
        AgentTag != ParisPocAgentTag.ToString())
    {
        return PedestrianDiagnosticError(TEXT("invalid_request"));
    }
    UWorld* World = GetWorld();
    if (!World || !IsValid(ParisPocAgent) ||
        !ParisPocAgent->ActorHasTag(ParisPocAgentTag) ||
        !ParisPocHelper || !ParisPocHelper->IsNavMeshReady() ||
        !ParisPocHelper->HasValidNavMeshData())
    {
        return PedestrianDiagnosticError(TEXT("paris_recast_not_ready"));
    }

    const TArray<FParisCrosswalkComponent> Crosswalks =
        CollectParisCrosswalkComponents(World);
    TArray<TSharedPtr<FJsonValue>> Rows;
    Rows.Reserve(Crosswalks.Num());
    for (const FParisCrosswalkComponent& Crosswalk : Crosswalks)
    {
        const UStaticMeshComponent* Component = Crosswalk.Component.Get();
        if (!Component || !Component->GetStaticMesh())
        {
            continue;
        }
        const FTransform Transform = Component->GetComponentTransform();
        const TSharedRef<FJsonObject> Row = MakeShared<FJsonObject>();
        Row->SetStringField(TEXT("label"), Crosswalk.Label);
        Row->SetStringField(TEXT("actor_name"), Crosswalk.ActorName);
        Row->SetStringField(TEXT("component_name"), Component->GetName());
        Row->SetStringField(
            TEXT("static_mesh_path"), Component->GetStaticMesh()->GetPathName());
        Row->SetArrayField(TEXT("location_cm"), JsonVector(Transform.GetLocation()));
        Row->SetArrayField(
            TEXT("rotation_deg"), JsonRotator(Transform.Rotator()));
        Row->SetArrayField(TEXT("scale"), JsonVector(Transform.GetScale3D()));
        Row->SetArrayField(TEXT("local_min_cm"), JsonVector(Crosswalk.LocalMin));
        Row->SetArrayField(TEXT("local_max_cm"), JsonVector(Crosswalk.LocalMax));
        Rows.Add(MakeShared<FJsonValueObject>(Row));
    }

    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetBoolField(TEXT("success"), true);
    Root->SetStringField(TEXT("active_map"), World->GetMapName());
    Root->SetStringField(TEXT("region_name"), ParisPocSetup.RegionName);
    Root->SetStringField(
        TEXT("required_label_prefix"), TEXT("PR_Crossswalk_"));
    Root->SetNumberField(TEXT("component_count"), Rows.Num());
    Root->SetArrayField(TEXT("crosswalk_components"), Rows);
    return SerializeJson(Root);
}

FString USpPixelGoalSubsystem::PixelGoal_GetBuildingEntranceCatalogJson(
    const FString& RequestJson)
{
    const TSharedPtr<FJsonObject> Request = ParseJson(RequestJson);
    FString AgentTag;
    if (!HasExactJsonFields(Request, {TEXT("agent_tag")}) ||
        !TryGetExactJsonString(Request, TEXT("agent_tag"), AgentTag) ||
        AgentTag != ParisPocAgentTag.ToString())
    {
        return PedestrianDiagnosticError(TEXT("invalid_request"));
    }
    UWorld* World = GetWorld();
    if (!World || !IsValid(ParisPocAgent) ||
        !ParisPocAgent->ActorHasTag(ParisPocAgentTag) ||
        !ParisPocHelper || !ParisPocHelper->IsNavMeshReady() ||
        !ParisPocHelper->HasValidNavMeshData())
    {
        return PedestrianDiagnosticError(TEXT("paris_recast_not_ready"));
    }

    TArray<TSharedPtr<FJsonValue>> Rows;
    for (TActorIterator<AActor> It(World); It; ++It)
    {
        AActor* Actor = *It;
        if (!IsValid(Actor))
        {
            continue;
        }
        TArray<UStaticMeshComponent*> Components;
        Actor->GetComponents<UStaticMeshComponent>(Components);
        for (UStaticMeshComponent* Component : Components)
        {
            UStaticMesh* Mesh = IsValid(Component)
                ? Component->GetStaticMesh()
                : nullptr;
            if (!IsValid(Mesh) || !Mesh->GetPathName().Contains(
                    TEXT("Entrance"), ESearchCase::IgnoreCase))
            {
                continue;
            }
            const FBox LocalBounds = Mesh->GetBoundingBox();
            const UInstancedStaticMeshComponent* Instances =
                Cast<UInstancedStaticMeshComponent>(Component);
            const int32 InstanceCount = Instances
                ? Instances->GetInstanceCount()
                : 1;
            for (int32 InstanceIndex = 0;
                 InstanceIndex < InstanceCount;
                 ++InstanceIndex)
            {
                FTransform Transform = Component->GetComponentTransform();
                if (Instances && !Instances->GetInstanceTransform(
                        InstanceIndex, Transform, true))
                {
                    continue;
                }
                const FBox WorldBounds = LocalBounds.TransformBy(Transform);
                const TSharedRef<FJsonObject> Row = MakeShared<FJsonObject>();
                Row->SetStringField(
                    TEXT("entrance_asset_id"),
                    FString::Printf(
                        TEXT("%s|%s|%d"),
                        *Actor->GetName(), *Component->GetName(), InstanceIndex));
                Row->SetStringField(
                    TEXT("actor_name"), Actor->GetName());
                Row->SetStringField(
                    TEXT("actor_label"), Actor->GetActorNameOrLabel());
                Row->SetStringField(
                    TEXT("component_name"), Component->GetName());
                Row->SetNumberField(TEXT("instance_index"), InstanceIndex);
                Row->SetStringField(
                    TEXT("static_mesh_path"), Mesh->GetPathName());
                Row->SetArrayField(
                    TEXT("location_cm"), JsonVector(Transform.GetLocation()));
                Row->SetArrayField(
                    TEXT("rotation_deg"), JsonRotator(Transform.Rotator()));
                Row->SetArrayField(
                    TEXT("scale"), JsonVector(Transform.GetScale3D()));
                Row->SetArrayField(
                    TEXT("local_min_cm"), JsonVector(LocalBounds.Min));
                Row->SetArrayField(
                    TEXT("local_max_cm"), JsonVector(LocalBounds.Max));
                Row->SetArrayField(
                    TEXT("world_bounds_center_cm"), JsonVector(WorldBounds.GetCenter()));
                Row->SetArrayField(
                    TEXT("world_bounds_extent_cm"), JsonVector(WorldBounds.GetExtent()));
                Rows.Add(MakeShared<FJsonValueObject>(Row));
            }
        }
    }
    Rows.Sort([](
        const TSharedPtr<FJsonValue>& A,
        const TSharedPtr<FJsonValue>& B)
    {
        FString AId;
        FString BId;
        A->AsObject()->TryGetStringField(TEXT("entrance_asset_id"), AId);
        B->AsObject()->TryGetStringField(TEXT("entrance_asset_id"), BId);
        return AId < BId;
    });

    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetBoolField(TEXT("success"), true);
    Root->SetStringField(TEXT("active_map"), World->GetMapName());
    Root->SetStringField(TEXT("region_name"), ParisPocSetup.RegionName);
    Root->SetStringField(TEXT("required_mesh_token"), TEXT("Entrance"));
    Root->SetNumberField(TEXT("component_count"), Rows.Num());
    Root->SetArrayField(TEXT("entrance_components"), Rows);
    return SerializeJson(Root);
}

FString USpPixelGoalSubsystem::PixelGoal_AuditPedestrianPointsJson(
    const FString& RequestJson)
{
    const TSharedPtr<FJsonObject> Request = ParseJson(RequestJson);
    FString AgentTag;
    double ProjectionExtentCm = 0.0;
    bool bCompact = false;
    const TArray<TSharedPtr<FJsonValue>>* PointValues = nullptr;
    const bool bLegacyFields = HasExactJsonFields(
        Request,
        {TEXT("agent_tag"), TEXT("points_cm"),
         TEXT("projection_extent_cm")});
    const bool bCompactFields = HasExactJsonFields(
        Request,
        {TEXT("agent_tag"), TEXT("points_cm"),
         TEXT("projection_extent_cm"), TEXT("compact")});
    if ((!bLegacyFields &&
         (!bCompactFields ||
          !TryGetExactJsonBool(Request, TEXT("compact"), bCompact))) ||
        !TryGetExactJsonString(Request, TEXT("agent_tag"), AgentTag) ||
        AgentTag != ParisPocAgentTag.ToString() ||
        !TryGetExactJsonNumber(
            Request, TEXT("projection_extent_cm"), ProjectionExtentCm) ||
        ProjectionExtentCm < 0.1 || ProjectionExtentCm > 100.0 ||
        !Request->TryGetArrayField(TEXT("points_cm"), PointValues) ||
        !PointValues || PointValues->Num() < 1 || PointValues->Num() > 512)
    {
        return PedestrianDiagnosticError(TEXT("invalid_request"));
    }

    UWorld* World = GetWorld();
    UNavigationSystemV1* NavSys = World
        ? UNavigationSystemV1::GetNavigationSystem(World)
        : nullptr;
    const ANavigationData* AgentNavData = NavSys && IsValid(ParisPocAgent)
        ? NavSys->GetNavDataForProps(
            ParisPocAgent->GetNavAgentPropertiesRef(),
            ParisPocAgent->GetActorLocation())
        : nullptr;
    ARecastNavMesh* Recast =
        Cast<ARecastNavMesh>(const_cast<ANavigationData*>(AgentNavData));
    if (!World || !IsValid(ParisPocAgent) || !ParisPocHelper ||
        !ParisPocHelper->IsNavMeshReady() ||
        !ParisPocHelper->HasValidNavMeshData() || !NavSys || !Recast)
    {
        return PedestrianDiagnosticError(TEXT("paris_recast_not_ready"));
    }

    TArray<FVector> Points;
    Points.Reserve(PointValues->Num());
    for (const TSharedPtr<FJsonValue>& Value : *PointValues)
    {
        FVector Point;
        if (!JsonVectorValue(Value, Point))
        {
            return PedestrianDiagnosticError(TEXT("invalid_request"));
        }
        const FVector Delta = (Point - ParisPocSetup.NavBoundsCenter).GetAbs();
        if (Delta.X > ParisPocSetup.NavBoundsExtent.X ||
            Delta.Y > ParisPocSetup.NavBoundsExtent.Y ||
            Delta.Z > ParisPocSetup.NavBoundsExtent.Z)
        {
            return PedestrianDiagnosticError(TEXT("point_outside_active_bounds"));
        }
        Points.Add(Point);
    }

    const TArray<FParisCrosswalkComponent> Crosswalks =
        CollectParisCrosswalkComponents(World);
    TArray<TSharedPtr<FJsonValue>> Rows;
    Rows.Reserve(Points.Num());
    for (const FVector& Point : Points)
    {
        const TSharedRef<FJsonObject> Row = MakeShared<FJsonObject>();
        Row->SetArrayField(TEXT("input_cm"), JsonVector(Point));
        FNavLocation Projected;
        const bool bProjected = NavSys->ProjectPointToNavigation(
            Point,
            Projected,
            FVector(static_cast<float>(ProjectionExtentCm)),
            &ParisPocAgent->GetNavAgentPropertiesRef());
        Row->SetBoolField(TEXT("projected"), bProjected);
        if (bProjected)
        {
            FVector PolyCenter = FVector::ZeroVector;
            TArray<FVector> PolyVertices;
            Recast->GetPolyCenter(Projected.NodeRef, PolyCenter);
            Recast->GetPolyVerts(Projected.NodeRef, PolyVertices);
            Row->SetStringField(
                TEXT("poly_ref"), NavNodeRefString(Projected.NodeRef));
            Row->SetArrayField(
                TEXT("projected_cm"), JsonVector(Projected.Location));
            Row->SetNumberField(
                TEXT("projection_error_cm"),
                FVector::Distance(Point, Projected.Location));
            if (!bCompact)
            {
                Row->SetArrayField(
                    TEXT("poly_center_cm"), JsonVector(PolyCenter));
                Row->SetArrayField(
                    TEXT("poly_vertices_cm"), JsonVectorPath(PolyVertices));
            }
            const TArray<FString> CrosswalkLabels =
                CrosswalkLabelsAtPoint(Projected.Location, Crosswalks);
            const TArray<FHitResult> GroundHits =
                CollectGroundHits(World, Projected.Location, ParisPocAgent);
            Row->SetArrayField(
                TEXT("crosswalk_labels"), JsonStrings(CrosswalkLabels));
            Row->SetStringField(
                TEXT("surface_class"),
                ClassifyPedestrianSurface(CrosswalkLabels, GroundHits));
            if (!bCompact)
            {
                Row->SetArrayField(
                    TEXT("ground_hits"), JsonGroundHits(GroundHits));
            }
        }
        Rows.Add(MakeShared<FJsonValueObject>(Row));
    }

    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetBoolField(TEXT("success"), true);
    Root->SetStringField(TEXT("active_map"), World->GetMapName());
    Root->SetStringField(TEXT("region_name"), ParisPocSetup.RegionName);
    Root->SetBoolField(TEXT("compact"), bCompact);
    Root->SetNumberField(TEXT("projection_extent_cm"), ProjectionExtentCm);
    Root->SetNumberField(TEXT("point_count"), Rows.Num());
    Root->SetArrayField(TEXT("points"), Rows);
    return SerializeJson(Root);
}

FString USpPixelGoalSubsystem::PixelGoal_GetRecastPolygonsJson(
    const FString& RequestJson)
{
    const TSharedPtr<FJsonObject> Request = ParseJson(RequestJson);
    FString AgentTag;
    FVector BoundsMin;
    FVector BoundsMax;
    double OffsetValue = 0.0;
    double LimitValue = 0.0;
    if (!HasExactJsonFields(
            Request,
            {TEXT("agent_tag"), TEXT("bounds_min_cm"),
             TEXT("bounds_max_cm"), TEXT("offset"), TEXT("limit")}) ||
        !TryGetExactJsonString(Request, TEXT("agent_tag"), AgentTag) ||
        AgentTag != ParisPocAgentTag.ToString() ||
        !JsonVectorField(Request, TEXT("bounds_min_cm"), BoundsMin) ||
        !JsonVectorField(Request, TEXT("bounds_max_cm"), BoundsMax) ||
        !TryGetExactJsonNumber(Request, TEXT("offset"), OffsetValue) ||
        !TryGetExactJsonNumber(Request, TEXT("limit"), LimitValue) ||
        OffsetValue < 0.0 || OffsetValue != FMath::FloorToDouble(OffsetValue) ||
        LimitValue < 1.0 || LimitValue > 250.0 ||
        LimitValue != FMath::FloorToDouble(LimitValue) ||
        BoundsMin.X >= BoundsMax.X || BoundsMin.Y >= BoundsMax.Y ||
        BoundsMin.Z >= BoundsMax.Z)
    {
        return PedestrianDiagnosticError(TEXT("invalid_request"));
    }
    const FVector ActiveMin =
        ParisPocSetup.NavBoundsCenter - ParisPocSetup.NavBoundsExtent;
    const FVector ActiveMax =
        ParisPocSetup.NavBoundsCenter + ParisPocSetup.NavBoundsExtent;
    if (BoundsMin.X < ActiveMin.X || BoundsMin.Y < ActiveMin.Y ||
        BoundsMin.Z < ActiveMin.Z || BoundsMax.X > ActiveMax.X ||
        BoundsMax.Y > ActiveMax.Y || BoundsMax.Z > ActiveMax.Z)
    {
        return PedestrianDiagnosticError(TEXT("bounds_outside_active_region"));
    }

    UWorld* World = GetWorld();
    UNavigationSystemV1* NavSys = World
        ? UNavigationSystemV1::GetNavigationSystem(World)
        : nullptr;
    const ANavigationData* AgentNavData = NavSys && IsValid(ParisPocAgent)
        ? NavSys->GetNavDataForProps(
            ParisPocAgent->GetNavAgentPropertiesRef(),
            ParisPocAgent->GetActorLocation())
        : nullptr;
    ARecastNavMesh* Recast =
        Cast<ARecastNavMesh>(const_cast<ANavigationData*>(AgentNavData));
    if (!World || !IsValid(ParisPocAgent) || !ParisPocHelper ||
        !ParisPocHelper->IsNavMeshReady() ||
        !ParisPocHelper->HasValidNavMeshData() || !Recast)
    {
        return PedestrianDiagnosticError(TEXT("paris_recast_not_ready"));
    }

    // ARecastNavMesh::GetPolysInBox is hard-capped at 256 hits inside UE.  A
    // city region can exceed that without an error flag, which previously
    // produced a plausible-looking but truncated graph.  Enumerate every
    // active tile instead, then filter complete tile contents by the requested
    // polygon bounds before our own deterministic paging.
    const FBox QueryBounds(BoundsMin, BoundsMax);
    TArray<FNavTileRef> TileRefs;
    Recast->GetAllNavMeshTiles(TileRefs);
    int32 QueriedTileCount = 0;
    TArray<FNavPoly> RawPolys;
    for (FNavTileRef TileRef : TileRefs)
    {
        const FBox TileBounds = Recast->GetNavMeshTileBounds(TileRef);
        if (!TileBounds.IsValid || !TileBounds.Intersect(QueryBounds))
        {
            continue;
        }
        ++QueriedTileCount;
        TArray<FNavPoly> TilePolys;
        if (!Recast->GetPolysInTile(TileRef, TilePolys))
        {
            return PedestrianDiagnosticError(TEXT("recast_tile_read_failed"));
        }
        for (const FNavPoly& Poly : TilePolys)
        {
            FVector Center = FVector::ZeroVector;
            TArray<FVector> Vertices;
            if (Poly.Ref == INVALID_NAVNODEREF ||
                !Recast->GetPolyCenter(Poly.Ref, Center) ||
                !Recast->GetPolyVerts(Poly.Ref, Vertices) ||
                Vertices.IsEmpty())
            {
                continue;
            }
            FBox PolyBounds(ForceInit);
            for (const FVector& Vertex : Vertices)
            {
                PolyBounds += Vertex;
            }
            if (PolyBounds.Intersect(QueryBounds) &&
                QueryBounds.IsInsideOrOn(Center))
            {
                RawPolys.Add(Poly);
            }
        }
    }
    // De-duplicate by native polygon ref before sorting and paging.  Tile
    // ownership should already be unique; retaining this guard makes any
    // engine-side overlap explicit in raw_polygon_count.
    TArray<FNavPoly> Polys;
    Polys.Reserve(RawPolys.Num());
    TSet<NavNodeRef> SeenPolyRefs;
    for (const FNavPoly& Poly : RawPolys)
    {
        if (Poly.Ref != INVALID_NAVNODEREF &&
            !SeenPolyRefs.Contains(Poly.Ref))
        {
            SeenPolyRefs.Add(Poly.Ref);
            Polys.Add(Poly);
        }
    }
    Polys.Sort([](const FNavPoly& A, const FNavPoly& B)
    {
        return A.Ref < B.Ref;
    });
    const int32 Offset = static_cast<int32>(OffsetValue);
    const int32 Limit = static_cast<int32>(LimitValue);
    const int32 End = FMath::Min(Offset + Limit, Polys.Num());
    if (Offset > Polys.Num())
    {
        return PedestrianDiagnosticError(TEXT("offset_out_of_range"));
    }

    const TArray<FParisCrosswalkComponent> Crosswalks =
        CollectParisCrosswalkComponents(World);
    TArray<TSharedPtr<FJsonValue>> Rows;
    Rows.Reserve(FMath::Max(End - Offset, 0));
    for (int32 Index = Offset; Index < End; ++Index)
    {
        const FNavPoly& Poly = Polys[Index];
        FVector Center = FVector::ZeroVector;
        TArray<FVector> Vertices;
        TArray<NavNodeRef> Neighbours;
        if (!Recast->GetPolyCenter(Poly.Ref, Center) ||
            !Recast->GetPolyVerts(Poly.Ref, Vertices))
        {
            continue;
        }
        Recast->GetPolyNeighbors(Poly.Ref, Neighbours);
        Neighbours.Sort();

        const TSharedRef<FJsonObject> Row = MakeShared<FJsonObject>();
        Row->SetStringField(TEXT("poly_ref"), NavNodeRefString(Poly.Ref));
        Row->SetArrayField(TEXT("center_cm"), JsonVector(Center));
        Row->SetArrayField(TEXT("vertices_cm"), JsonVectorPath(Vertices));
        Row->SetNumberField(TEXT("area_id"), Recast->GetPolyAreaID(Poly.Ref));
        TArray<TSharedPtr<FJsonValue>> NeighbourValues;
        NeighbourValues.Reserve(Neighbours.Num());
        for (NavNodeRef Neighbour : Neighbours)
        {
            NeighbourValues.Add(MakeShared<FJsonValueString>(
                NavNodeRefString(Neighbour)));
        }
        Row->SetArrayField(TEXT("neighbor_refs"), NeighbourValues);

        TArray<FVector> Samples;
        Samples.Add(Center);
        for (const FVector& Vertex : Vertices)
        {
            Samples.Add((Center + Vertex) * 0.5f);
        }
        TArray<TSharedPtr<FJsonValue>> SampleValues;
        SampleValues.Reserve(Samples.Num());
        for (const FVector& Sample : Samples)
        {
            const TSharedRef<FJsonObject> SampleRow = MakeShared<FJsonObject>();
            SampleRow->SetArrayField(TEXT("point_cm"), JsonVector(Sample));
            const TArray<FString> CrosswalkLabels =
                CrosswalkLabelsAtPoint(Sample, Crosswalks);
            const TArray<FHitResult> GroundHits =
                CollectGroundHits(World, Sample, ParisPocAgent);
            SampleRow->SetArrayField(
                TEXT("crosswalk_labels"), JsonStrings(CrosswalkLabels));
            SampleRow->SetStringField(
                TEXT("surface_class"),
                ClassifyPedestrianSurface(CrosswalkLabels, GroundHits));
            SampleRow->SetArrayField(
                TEXT("ground_hits"), JsonGroundHits(GroundHits));
            SampleValues.Add(MakeShared<FJsonValueObject>(SampleRow));
        }
        Row->SetArrayField(TEXT("surface_samples"), SampleValues);
        Rows.Add(MakeShared<FJsonValueObject>(Row));
    }

    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetBoolField(TEXT("success"), true);
    Root->SetStringField(TEXT("active_map"), World->GetMapName());
    Root->SetStringField(TEXT("region_name"), ParisPocSetup.RegionName);
    Root->SetArrayField(TEXT("bounds_min_cm"), JsonVector(BoundsMin));
    Root->SetArrayField(TEXT("bounds_max_cm"), JsonVector(BoundsMax));
    Root->SetNumberField(TEXT("active_tile_count"), TileRefs.Num());
    Root->SetNumberField(TEXT("queried_tile_count"), QueriedTileCount);
    Root->SetNumberField(TEXT("raw_polygon_count"), RawPolys.Num());
    Root->SetNumberField(TEXT("total_polygons"), Polys.Num());
    Root->SetNumberField(TEXT("offset"), Offset);
    Root->SetNumberField(TEXT("returned"), Rows.Num());
    Root->SetBoolField(TEXT("has_more"), End < Polys.Num());
    Root->SetArrayField(TEXT("polygons"), Rows);
    return SerializeJson(Root);
}

FString USpPixelGoalSubsystem::PixelGoal_ResolveAndMoveJson(const FString& RequestJson)
{
    const TSharedPtr<FJsonObject> Request = ParseJson(RequestJson);
    if (!Request.IsValid())
    {
        return Reject(
            TEXT("invalid_request_json"), FString(), FString(), FVector2D::ZeroVector);
    }
    const FString SnapshotId = JsonString(Request, TEXT("camera_snapshot_id"));
    const bool bHasViewField = Request->HasField(TEXT("view_id"));
    const bool bHasCaptureGroupField = Request->HasField(TEXT("capture_group_id"));
    const bool bHasBinding = bHasViewField || bHasCaptureGroupField;
    FString RequestedViewId;
    FString RequestedCaptureGroupId;
    const bool bValidViewField = !bHasViewField ||
        TryGetExactJsonString(Request, TEXT("view_id"), RequestedViewId);
    const bool bValidCaptureGroupField = !bHasCaptureGroupField ||
        TryGetExactJsonString(
            Request,
            TEXT("capture_group_id"), RequestedCaptureGroupId);
    const FVector2D UV(
        JsonNumber(Request, TEXT("u_norm"), std::numeric_limits<double>::quiet_NaN()),
        JsonNumber(Request, TEXT("v_norm"), std::numeric_limits<double>::quiet_NaN()));
    const auto RejectRequest = [
        &SnapshotId,
        &UV,
        bHasBinding,
        &RequestedViewId,
        &RequestedCaptureGroupId](
            const FString& Reason,
            const FString& IntrinsicsId,
            const FHitResult* Hit = nullptr,
            const FVector* ValidatedTarget = nullptr,
            float NavAdjustmentCm = -1.f,
            const TArray<FVector>* ControllerPathPoints = nullptr,
            float ControllerPathLengthCm = -1.f,
            float ControllerPathDirectCm = -1.f,
            float ControllerPathStretchRatio = -1.f)
    {
        return Reject(
            Reason,
            SnapshotId,
            IntrinsicsId,
            UV,
            Hit,
            ValidatedTarget,
            NavAdjustmentCm,
            bHasBinding,
            RequestedViewId,
            RequestedCaptureGroupId,
            ControllerPathPoints,
            ControllerPathLengthCm,
            ControllerPathDirectCm,
            ControllerPathStretchRatio);
    };
    if (!IsFinite(UV) || UV.X < 0.f || UV.X > 1.f || UV.Y < 0.f || UV.Y > 1.f)
    {
        return RejectRequest(TEXT("invalid_normalized_coordinate"), FString());
    }
    FSpPixelGoalCameraSnapshot* Snapshot = CameraSnapshots.Find(SnapshotId);
    if (!Snapshot)
    {
        return RejectRequest(TEXT("camera_snapshot_not_found"), FString());
    }
    if (bHasBinding)
    {
        if (!bHasViewField || !bHasCaptureGroupField)
        {
            return RejectRequest(
                TEXT("camera_snapshot_binding_incomplete"), Snapshot->IntrinsicsId);
        }
        if (!bValidViewField || !bValidCaptureGroupField ||
            RequestedViewId.IsEmpty() || RequestedCaptureGroupId.IsEmpty())
        {
            return RejectRequest(
                TEXT("camera_snapshot_binding_invalid"), Snapshot->IntrinsicsId);
        }
        const FString BindingError = SpPixelGoal::SnapshotBindingError(
            Snapshot->ViewId,
            Snapshot->CaptureGroupId,
            RequestedViewId,
            RequestedCaptureGroupId);
        if (!BindingError.IsEmpty())
        {
            return RejectRequest(BindingError, Snapshot->IntrinsicsId);
        }
    }
    ASpHumanoidAgent* Agent = Snapshot->Agent.Get();
    if (!Agent)
    {
        return RejectRequest(
            TEXT("camera_snapshot_agent_unavailable"),
            Snapshot->IntrinsicsId);
    }
    const FString RequestedAgentTag = JsonString(Request, TEXT("agent_tag"));
    if (!RequestedAgentTag.IsEmpty() && !Agent->ActorHasTag(FName(*RequestedAgentTag)))
    {
        return RejectRequest(
            TEXT("camera_snapshot_agent_mismatch"),
            Snapshot->IntrinsicsId);
    }
    USceneCaptureComponent2D* SnapshotCapture = Snapshot->CaptureComponent.Get();
    if (!SnapshotCapture ||
        FVector::Distance(
            SnapshotCapture->GetComponentLocation(), Snapshot->CameraLocation) > 1.f ||
        !SnapshotCapture->GetComponentRotation().Equals(
            Snapshot->CameraRotation, 0.1f) ||
        FVector::Distance(Agent->GetActorLocation(), Snapshot->AgentLocation) > 1.f ||
        !Agent->GetActorRotation().Equals(Snapshot->AgentRotation, 0.1f))
    {
        return RejectRequest(TEXT("camera_snapshot_stale"), Snapshot->IntrinsicsId);
    }

    const float TraceDistanceCm = FMath::Clamp(
        static_cast<float>(JsonNumber(Request, TEXT("trace_distance_cm"), 100000.0)),
        1.f,
        10000000.f);
    const float MaxGroundSlopeDegrees = FMath::Clamp(
        static_cast<float>(JsonNumber(Request, TEXT("max_ground_slope_deg"), 30.0)),
        0.f,
        89.9f);
    const float MaxNavAdjustmentCm = FMath::Clamp(
        static_cast<float>(JsonNumber(Request, TEXT("max_navmesh_adjustment_cm"), 10.0)),
        0.1f,
        10000.f);
    const float AcceptanceRadiusCm = FMath::Clamp(
        static_cast<float>(JsonNumber(Request, TEXT("acceptance_radius_cm"), 15.0)),
        0.1f,
        10000.f);
    const float MaxControllerPathStretchRatio = FMath::Clamp(
        static_cast<float>(JsonNumber(
            Request, TEXT("max_controller_path_stretch_ratio"), 1.35)),
        1.f,
        10.f);
    const float ControllerPathDetourAllowanceCm = FMath::Clamp(
        static_cast<float>(JsonNumber(
            Request, TEXT("controller_path_detour_allowance_cm"), 100.0)),
        0.f,
        10000.f);

    FVector RayOrigin;
    FVector RayDirection;
    if (!Snapshot->Deproject(UV, RayOrigin, RayDirection))
    {
        return RejectRequest(TEXT("deprojection_failed"), Snapshot->IntrinsicsId);
    }

    UWorld* World = GetWorld();
    FHitResult Hit;
    FCollisionQueryParams QueryParams(SCENE_QUERY_STAT(PixelGoal), true, Agent);
    QueryParams.AddIgnoredActor(Agent);
#if WITH_DEV_AUTOMATION_TESTS
    ++GeometryTraceEntryCount;
#endif
    const bool bHit = World && World->LineTraceSingleByChannel(
        Hit,
        RayOrigin,
        RayOrigin + RayDirection * TraceDistanceCm,
        ECC_Visibility,
        QueryParams);
    if (!bHit || !Hit.bBlockingHit)
    {
        return RejectRequest(TEXT("no_geometry_hit"), Snapshot->IntrinsicsId);
    }
    if (SpPixelGoal::ClassifyFirstHit(Hit.ImpactNormal, MaxGroundSlopeDegrees) !=
        ESpPixelGoalHitClass::WalkableGround)
    {
        return RejectRequest(
            TEXT("hit_not_walkable_ground"),
            Snapshot->IntrinsicsId,
            &Hit);
    }

    AAIController* Controller = Cast<AAIController>(Agent->GetController());
    UNavigationSystemV1* NavSys = World
        ? UNavigationSystemV1::GetNavigationSystem(World)
        : nullptr;
    if (!Controller || !NavSys || !NavSys->GetDefaultNavDataInstance())
    {
        return RejectRequest(
            TEXT("navmesh_unavailable"),
            Snapshot->IntrinsicsId,
            &Hit);
    }

    FNavLocation NavLocation;
    const FVector QueryExtent(MaxNavAdjustmentCm);
    if (!NavSys->ProjectPointToNavigation(
            Hit.ImpactPoint,
            NavLocation,
            QueryExtent,
            &Agent->GetNavAgentPropertiesRef()))
    {
        return RejectRequest(
            TEXT("navmesh_projection_failed"),
            Snapshot->IntrinsicsId,
            &Hit);
    }
    const FVector ValidatedTarget = NavLocation.Location;
    const float NavAdjustmentCm = FVector::Distance(Hit.ImpactPoint, ValidatedTarget);
    if (!SpPixelGoal::WithinNavAdjustment(
            Hit.ImpactPoint, ValidatedTarget, MaxNavAdjustmentCm))
    {
        return RejectRequest(
            TEXT("navmesh_adjustment_exceeded"),
            Snapshot->IntrinsicsId,
            &Hit,
            &ValidatedTarget,
            NavAdjustmentCm);
    }

    const ANavigationData* NavData = NavSys->GetNavDataForProps(
        Agent->GetNavAgentPropertiesRef(), Agent->GetActorLocation());
    if (!NavData)
    {
        return RejectRequest(
            TEXT("navmesh_unavailable"),
            Snapshot->IntrinsicsId,
            &Hit,
            &ValidatedTarget,
            NavAdjustmentCm);
    }
    const FPathFindingQuery PathQuery(
        Controller,
        *NavData,
        Agent->GetActorLocation(),
        ValidatedTarget);
    const FPathFindingResult PathResult = NavSys->FindPathSync(
        Agent->GetNavAgentPropertiesRef(), PathQuery, EPathFindingMode::Regular);
    if (!PathResult.IsSuccessful() || !PathResult.Path.IsValid() ||
        PathResult.Path->IsPartial())
    {
        return RejectRequest(
            TEXT("no_controller_path"),
            Snapshot->IntrinsicsId,
            &Hit,
            &ValidatedTarget,
            NavAdjustmentCm);
    }

    TArray<FVector> PlannedPathPoints;
    for (const FNavPathPoint& Point : PathResult.Path->GetPathPoints())
    {
        PlannedPathPoints.Add(Point.Location);
    }
    const float PlannedPathLengthCm =
        SpPixelGoal::ControllerPathLengthCm(PlannedPathPoints);
    const float PlannedPathDirectCm = FVector::Dist2D(
        Agent->GetActorLocation(), ValidatedTarget);
    const float PlannedPathStretchRatio = PlannedPathDirectCm > KINDA_SMALL_NUMBER
        ? PlannedPathLengthCm / PlannedPathDirectCm
        : 1.f;
    if (SpPixelGoal::ControllerPathDetourExceeded(
            PlannedPathLengthCm,
            PlannedPathDirectCm,
            MaxControllerPathStretchRatio,
            ControllerPathDetourAllowanceCm))
    {
        return RejectRequest(
            TEXT("controller_path_detour_exceeded"),
            Snapshot->IntrinsicsId,
            &Hit,
            &ValidatedTarget,
            NavAdjustmentCm,
            &PlannedPathPoints,
            PlannedPathLengthCm,
            PlannedPathDirectCm,
            PlannedPathStretchRatio);
    }

    // Recast is the connectivity authority, but this CityCore scene also
    // bakes vehicle lanes into that NavMesh.  In the Paris pedestrian mode a
    // path is executable only when dense samples stay on physical pavement or
    // inside an exact PR_Crossswalk_* component.  This uses the same classifier
    // exported to the graph authoring tool, so the phone graph cannot promise a
    // route the controller is allowed to execute.
    if (World && IsParisPocMapName(World->GetMapName()) &&
        Agent->ActorHasTag(ParisPocAgentTag))
    {
        const TArray<FParisCrosswalkComponent> Crosswalks =
            CollectParisCrosswalkComponents(World);
        const FString PedestrianRejection = PedestrianPathRejection(
            World, PlannedPathPoints, Agent, Crosswalks);
        if (!PedestrianRejection.IsEmpty())
        {
            return RejectRequest(
                PedestrianRejection,
                Snapshot->IntrinsicsId,
                &Hit,
                &ValidatedTarget,
                NavAdjustmentCm,
                &PlannedPathPoints,
                PlannedPathLengthCm,
                PlannedPathDirectCm,
                PlannedPathStretchRatio);
        }
    }

    FAIMoveRequest MoveRequest;
    MoveRequest.SetGoalLocation(ValidatedTarget);
    MoveRequest.SetAcceptanceRadius(AcceptanceRadiusCm);
    MoveRequest.SetUsePathfinding(true);
    MoveRequest.SetProjectGoalLocation(false);
    MoveRequest.SetAllowPartialPath(false);
    MoveRequest.SetReachTestIncludesAgentRadius(false);
    MoveRequest.SetReachTestIncludesGoalRadius(false);
    MoveRequest.SetCanStrafe(false);

    FNavPathSharedPtr ControllerPath;
    const FPathFollowingRequestResult ControllerRequest = Controller->MoveTo(
        MoveRequest, &ControllerPath);
    if (ControllerRequest.Code == EPathFollowingRequestResult::Failed)
    {
        return RejectRequest(
            TEXT("controller_request_failed"),
            Snapshot->IntrinsicsId,
            &Hit,
            &ValidatedTarget,
            NavAdjustmentCm);
    }
    if (ControllerPath.IsValid())
    {
        PlannedPathPoints.Reset();
        for (const FNavPathPoint& Point : ControllerPath->GetPathPoints())
        {
            PlannedPathPoints.Add(Point.Location);
        }
    }
    const float ControllerPathLengthCm =
        SpPixelGoal::ControllerPathLengthCm(PlannedPathPoints);
    const float ControllerPathStretchRatio = PlannedPathDirectCm > KINDA_SMALL_NUMBER
        ? ControllerPathLengthCm / PlannedPathDirectCm
        : 1.f;
    if (World && IsParisPocMapName(World->GetMapName()) &&
        Agent->ActorHasTag(ParisPocAgentTag))
    {
        const TArray<FParisCrosswalkComponent> Crosswalks =
            CollectParisCrosswalkComponents(World);
        const FString PedestrianRejection = PedestrianPathRejection(
            World, PlannedPathPoints, Agent, Crosswalks);
        if (!PedestrianRejection.IsEmpty())
        {
            Controller->StopMovement();
            return RejectRequest(
                PedestrianRejection,
                Snapshot->IntrinsicsId,
                &Hit,
                &ValidatedTarget,
                NavAdjustmentCm,
                &PlannedPathPoints,
                ControllerPathLengthCm,
                PlannedPathDirectCm,
                ControllerPathStretchRatio);
        }
    }

    const FString MoveId = FString::Printf(TEXT("pixel-move-%llu"), NextMoveNumber++);
    FMoveAudit Audit;
    Audit.RequestId = MoveId;
    Audit.CameraSnapshotId = SnapshotId;
    Audit.CameraIntrinsicsId = Snapshot->IntrinsicsId;
    Audit.ViewId = RequestedViewId;
    Audit.CaptureGroupId = RequestedCaptureGroupId;
    Audit.bHasSnapshotBinding = bHasBinding;
    Audit.RequestedUV = UV;
    Audit.RawWorldHit = Hit.ImpactPoint;
    Audit.AcceptedTarget = ValidatedTarget;
    Audit.InitialFeet = GetFeetLocation(Agent);
    Audit.FinalFeet = Audit.InitialFeet;
    Audit.FinalAgentPosition = Agent->GetActorLocation();
    Audit.LastSampledFeet = Audit.InitialFeet;
    Audit.ControllerPathPoints = PlannedPathPoints;
    Audit.ControllerPathLengthCm = ControllerPathLengthCm;
    Audit.ControllerPathDirectCm = PlannedPathDirectCm;
    Audit.ControllerPathStretchRatio = ControllerPathStretchRatio;
    Audit.ControllerRequestId = ControllerRequest.MoveId;
    Audit.Agent = Agent;
    Audit.Controller = Controller;
    Audit.StartWorldTimeSeconds = World->GetTimeSeconds();
    if (ControllerRequest.Code == EPathFollowingRequestResult::AlreadyAtGoal)
    {
        Audit.State = TEXT("completed");
        Audit.ControllerResult = TEXT("success");
    }
    else
    {
        Controller->ReceiveMoveCompleted.AddUniqueDynamic(
            this, &USpPixelGoalSubsystem::HandleMoveCompleted);
    }
    MoveAudits.Add(MoveId, Audit);

    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetBoolField(TEXT("accepted"), true);
    Root->SetStringField(TEXT("request_id"), MoveId);
    AddRequestIdentity(
        Root,
        SnapshotId,
        Snapshot->IntrinsicsId,
        UV,
        bHasBinding,
        RequestedViewId,
        RequestedCaptureGroupId);
    Root->SetArrayField(TEXT("raw_world_hit_cm"), JsonVector(Hit.ImpactPoint));
    Root->SetArrayField(TEXT("raw_world_normal"), JsonVector(Hit.ImpactNormal));
    Root->SetStringField(
        TEXT("raw_hit_actor"),
        Hit.GetActor() ? Hit.GetActor()->GetActorNameOrLabel() : FString());
    Root->SetArrayField(
        TEXT("validated_navigation_target_cm"), JsonVector(ValidatedTarget));
    Root->SetNumberField(TEXT("navmesh_adjustment_cm"), NavAdjustmentCm);
    Root->SetArrayField(
        TEXT("controller_path_points_cm"),
        JsonVectorPath(Audit.ControllerPathPoints));
    Root->SetNumberField(
        TEXT("controller_path_length_cm"), Audit.ControllerPathLengthCm);
    Root->SetNumberField(
        TEXT("controller_path_direct_cm"), Audit.ControllerPathDirectCm);
    Root->SetNumberField(
        TEXT("controller_path_stretch_ratio"),
        Audit.ControllerPathStretchRatio);
    Root->SetArrayField(TEXT("initial_feet_position_cm"), JsonVector(Audit.InitialFeet));
    Root->SetStringField(
        TEXT("controller_request_result"),
        MoveRequestResultString(ControllerRequest.Code));
    Root->SetStringField(
        TEXT("controller_request_id"), ControllerRequest.MoveId.ToString());
    return SerializeJson(Root);
}

FString USpPixelGoalSubsystem::PixelGoal_GetMoveStatusJson(const FString& RequestJson)
{
    const TSharedPtr<FJsonObject> Request = ParseJson(RequestJson);
    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    if (!Request.IsValid())
    {
        Root->SetStringField(TEXT("error"), TEXT("invalid_request_json"));
        return SerializeJson(Root);
    }
    const FString RequestId = JsonString(Request, TEXT("request_id"));
    FMoveAudit* Audit = MoveAudits.Find(RequestId);
    if (!Audit)
    {
        Root->SetStringField(TEXT("error"), TEXT("move_request_not_found"));
        Root->SetStringField(TEXT("request_id"), RequestId);
        return SerializeJson(Root);
    }
    return SerializeMoveAudit(*Audit);
}

FString USpPixelGoalSubsystem::PixelGoal_CancelMoveJson(const FString& RequestJson)
{
    const TSharedPtr<FJsonObject> Request = ParseJson(RequestJson);
    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    if (!Request.IsValid())
    {
        Root->SetBoolField(TEXT("cancelled"), false);
        Root->SetStringField(TEXT("error"), TEXT("invalid_request_json"));
        return SerializeJson(Root);
    }
    const FString RequestId = JsonString(Request, TEXT("request_id"));
    const FString Reason = JsonString(Request, TEXT("reason"));
    FMoveAudit* Audit = MoveAudits.Find(RequestId);
    if (!Audit)
    {
        Root->SetBoolField(TEXT("cancelled"), false);
        Root->SetStringField(TEXT("error"), TEXT("move_request_not_found"));
        Root->SetStringField(TEXT("request_id"), RequestId);
        return SerializeJson(Root);
    }
    if (Reason.IsEmpty())
    {
        Root->SetBoolField(TEXT("cancelled"), false);
        Root->SetStringField(TEXT("error"), TEXT("cancel_reason_required"));
        Root->SetStringField(TEXT("request_id"), RequestId);
        return SerializeJson(Root);
    }
    if (Audit->State != TEXT("moving"))
    {
        return SerializeMoveAudit(*Audit, false);
    }

    AAIController* Controller = Audit->Controller.Get();
    if (!Controller)
    {
        Root->SetBoolField(TEXT("cancelled"), false);
        Root->SetStringField(TEXT("error"), TEXT("controller_unavailable"));
        Root->SetStringField(TEXT("request_id"), RequestId);
        return SerializeJson(Root);
    }
    if (!Audit->ControllerRequestId.IsEquivalent(Controller->GetCurrentMoveRequestID()))
    {
        Root->SetBoolField(TEXT("cancelled"), false);
        Root->SetStringField(TEXT("error"), TEXT("controller_request_not_active"));
        Root->SetStringField(TEXT("request_id"), RequestId);
        return SerializeJson(Root);
    }

    // StopMovement may synchronously deliver HandleMoveCompleted.  Overwrite
    // that generic "aborted" audit afterwards with the caller's explicit
    // terminal reason so a timeout cannot continue mutating the agent.
    Controller->StopMovement();
    Audit = MoveAudits.Find(RequestId);
    check(Audit);
    Audit->State = TEXT("failed");
    Audit->ControllerResult = TEXT("aborted");
    Audit->FailureReason = Reason;
    UpdateMoveSample(*Audit);
    return SerializeMoveAudit(*Audit, true);
}

FString USpPixelGoalSubsystem::SerializeMoveAudit(
    FMoveAudit& Audit,
    bool bCancelled)
{
    if (SpPixelGoal::ShouldSampleMoveAudit(Audit.State))
    {
        UpdateMoveSample(Audit);
    }
    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetBoolField(TEXT("cancelled"), bCancelled);
    Root->SetStringField(TEXT("request_id"), Audit.RequestId);
    Root->SetStringField(TEXT("state"), Audit.State);
    Root->SetStringField(TEXT("controller_result"), Audit.ControllerResult);
    Root->SetStringField(TEXT("failure_reason"), Audit.FailureReason);
    Root->SetStringField(TEXT("camera_snapshot_id"), Audit.CameraSnapshotId);
    Root->SetStringField(TEXT("camera_intrinsics_id"), Audit.CameraIntrinsicsId);
    if (Audit.bHasSnapshotBinding)
    {
        Root->SetStringField(TEXT("view_id"), Audit.ViewId);
        Root->SetStringField(TEXT("capture_group_id"), Audit.CaptureGroupId);
    }
    Root->SetArrayField(TEXT("raw_world_hit_cm"), JsonVector(Audit.RawWorldHit));
    Root->SetArrayField(TEXT("accepted_target_cm"), JsonVector(Audit.AcceptedTarget));
    Root->SetArrayField(TEXT("initial_feet_position_cm"), JsonVector(Audit.InitialFeet));
    Root->SetArrayField(TEXT("final_feet_position_cm"), JsonVector(Audit.FinalFeet));
    Root->SetArrayField(
        TEXT("final_agent_position_cm"), JsonVector(Audit.FinalAgentPosition));
    Root->SetNumberField(TEXT("final_yaw_degrees"), Audit.FinalYawDegrees);
    Root->SetNumberField(TEXT("elapsed_sim_s"), Audit.ElapsedSimSeconds);
    Root->SetNumberField(TEXT("distance_travelled_cm"), Audit.DistanceTravelledCm);
    Root->SetArrayField(
        TEXT("controller_path_points_cm"),
        JsonVectorPath(Audit.ControllerPathPoints));
    Root->SetNumberField(
        TEXT("controller_path_length_cm"), Audit.ControllerPathLengthCm);
    Root->SetNumberField(
        TEXT("controller_path_direct_cm"), Audit.ControllerPathDirectCm);
    Root->SetNumberField(
        TEXT("controller_path_stretch_ratio"),
        Audit.ControllerPathStretchRatio);
    Root->SetNumberField(
        TEXT("execution_error_planar_m"),
        SpPixelGoal::PlanarErrorCm(Audit.FinalFeet, Audit.AcceptedTarget) / 100.f);
    Root->SetNumberField(
        TEXT("execution_error_3d_m"),
        FVector::Distance(Audit.FinalFeet, Audit.AcceptedTarget) / 100.f);
    return SerializeJson(Root);
}

FVector USpPixelGoalSubsystem::GetFeetLocation(const ASpHumanoidAgent* Agent)
{
    if (!Agent)
    {
        return FVector::ZeroVector;
    }
    const UCapsuleComponent* Capsule = Agent->GetCapsuleComponent();
    const float HalfHeight = Capsule ? Capsule->GetScaledCapsuleHalfHeight() : 0.f;
    return Agent->GetActorLocation() - FVector(0.f, 0.f, HalfHeight);
}

void USpPixelGoalSubsystem::UpdateMoveSample(FMoveAudit& Audit)
{
    ASpHumanoidAgent* Agent = Audit.Agent.Get();
    UWorld* World = GetWorld();
    if (!Agent || !World)
    {
        if (Audit.State == TEXT("moving"))
        {
            Audit.State = TEXT("failed");
            Audit.ControllerResult = TEXT("invalid");
            Audit.FailureReason = TEXT("agent_unavailable_during_execution");
        }
        return;
    }
    const FVector Feet = GetFeetLocation(Agent);
    Audit.DistanceTravelledCm += FVector::Dist2D(Audit.LastSampledFeet, Feet);
    Audit.LastSampledFeet = Feet;
    Audit.FinalFeet = Feet;
    Audit.FinalAgentPosition = Agent->GetActorLocation();
    Audit.FinalYawDegrees = Agent->GetActorRotation().Yaw;
    Audit.ElapsedSimSeconds = World->GetTimeSeconds() - Audit.StartWorldTimeSeconds;
}

void USpPixelGoalSubsystem::HandleMoveCompleted(
    FAIRequestID RequestId,
    EPathFollowingResult::Type Result)
{
    for (TPair<FString, FMoveAudit>& Pair : MoveAudits)
    {
        FMoveAudit& Audit = Pair.Value;
        if (!Audit.ControllerRequestId.IsEquivalent(RequestId))
        {
            continue;
        }
        Audit.ControllerResult = MoveCompletionResultString(Result);
        Audit.State = Result == EPathFollowingResult::Success
            ? TEXT("completed")
            : TEXT("failed");
        Audit.FailureReason = Result == EPathFollowingResult::Success
            ? FString()
            : Audit.ControllerResult;
        UpdateMoveSample(Audit);
        return;
    }
}
