// Copyright (c) 2026 The SimWorld Development Team. Licensed under the MIT License.

#if WITH_DEV_AUTOMATION_TESTS

#include "SpCameraCapturePool.h"
#include "SpHumanoidAgent.h"
#include "SpPixelGoalSubsystem.h"

#include "AIController.h"
#include "Components/SkyAtmosphereComponent.h"
#include "Engine/Engine.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "GameFramework/SpringArmComponent.h"
#include "Misc/AutomationTest.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "SpUnrealTypes/SpSceneCaptureComponent2D.h"

#include <limits>

namespace
{
    TSharedPtr<FJsonObject> ParseTestJson(const FString& Json)
    {
        TSharedPtr<FJsonObject> Object;
        const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Json);
        return FJsonSerializer::Deserialize(Reader, Object) && Object.IsValid()
            ? Object
            : nullptr;
    }

    FString TestJsonString(
        const TSharedPtr<FJsonObject>& Object,
        const TCHAR* Field)
    {
        FString Value;
        return Object.IsValid() && Object->TryGetStringField(Field, Value)
            ? Value
            : FString();
    }

    double TestJsonNumber(
        const TSharedPtr<FJsonObject>& Object,
        const TCHAR* Field,
        double Default = -1.0)
    {
        double Value = Default;
        if (Object.IsValid())
        {
            Object->TryGetNumberField(Field, Value);
        }
        return Value;
    }

    TSharedPtr<FJsonObject> MakeValidViewPairImageForTest()
    {
        const TSharedRef<FJsonObject> Image = MakeShared<FJsonObject>();
        Image->SetBoolField(TEXT("success"), true);
        Image->SetStringField(TEXT("modality"), TEXT("rgb"));
        Image->SetStringField(TEXT("camera_id"), TEXT("front"));
        Image->SetStringField(TEXT("camera_view"), TEXT("front"));
        Image->SetStringField(TEXT("source_type"), TEXT("agent"));
        Image->SetStringField(
            TEXT("source_agent_tag"), TEXT("PixelGoalParisPocAgent"));
        Image->SetStringField(TEXT("capture_mode"), TEXT("agent_native"));
        Image->SetStringField(TEXT("view_id"), TEXT("front"));
        Image->SetStringField(TEXT("capture_group_id"), TEXT("view-pair-4"));
        Image->SetStringField(TEXT("data_url"), TEXT("data:image/jpeg;base64,AA=="));
        Image->SetNumberField(TEXT("width"), 640.0);
        Image->SetNumberField(TEXT("height"), 360.0);
        Image->SetArrayField(TEXT("loc_cm"), {
            MakeShared<FJsonValueNumber>(120.0),
            MakeShared<FJsonValueNumber>(-340.0),
            MakeShared<FJsonValueNumber>(160.0),
        });
        Image->SetNumberField(TEXT("yaw_deg"), 37.0);
        Image->SetStringField(
            TEXT("camera_snapshot_id"), TEXT("camera-snapshot-10"));
        Image->SetStringField(
            TEXT("camera_intrinsics_id"), TEXT("intrinsics-640x360-fov90"));
        return Image;
    }

    TSharedPtr<FJsonObject> MakeValidViewPairRequestForTest()
    {
        const TSharedRef<FJsonObject> Request = MakeShared<FJsonObject>();
        Request->SetStringField(
            TEXT("agent_tag"), TEXT("PixelGoalParisPocAgent"));
        return Request;
    }

    struct FTestViewPairCameraState
    {
        USpSceneCaptureComponent2D* Component = nullptr;
        USceneComponent* AttachParent = nullptr;
        UTextureRenderTarget2D* TextureTarget = nullptr;
        FTransform RelativeTransform = FTransform::Identity;
        FMatrix CustomProjectionMatrix = FMatrix::Identity;
        int32 Width = 0;
        int32 Height = 0;
        int32 TargetWidth = 0;
        int32 TargetHeight = 0;
        float FovDegrees = 0.f;
        float Overscan = 0.f;
        int32 ProjectionType = 0;
        int32 CaptureSource = 0;
        int32 RenderTargetFormat = 0;
        int32 ChannelDataType = 0;
        int32 NumChannels = 0;
        float PostProcessBlendWeight = 0.f;
        int32 AutoExposureMethod = 0;
        float AutoExposureMinBrightness = 0.f;
        float AutoExposureMaxBrightness = 0.f;
        float AutoExposureSpeedUp = 0.f;
        float AutoExposureSpeedDown = 0.f;
        float AutoExposureBias = 0.f;
        bool bInitialized = false;
        bool bUseCustomProjection = false;
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

        static FTestViewPairCameraState Capture(
            USpSceneCaptureComponent2D* InComponent)
        {
            FTestViewPairCameraState State;
            State.Component = InComponent;
            if (!InComponent)
            {
                return State;
            }
            State.AttachParent = InComponent->GetAttachParent();
            State.TextureTarget = InComponent->TextureTarget;
            State.RelativeTransform = InComponent->GetRelativeTransform();
            State.CustomProjectionMatrix = InComponent->CustomProjectionMatrix;
            State.Width = InComponent->Width;
            State.Height = InComponent->Height;
            State.TargetWidth = InComponent->TextureTarget
                ? InComponent->TextureTarget->SizeX
                : 0;
            State.TargetHeight = InComponent->TextureTarget
                ? InComponent->TextureTarget->SizeY
                : 0;
            State.FovDegrees = InComponent->FOVAngle;
            State.Overscan = InComponent->Overscan;
            State.ProjectionType = static_cast<int32>(InComponent->ProjectionType);
            State.CaptureSource = static_cast<int32>(InComponent->CaptureSource);
            State.RenderTargetFormat =
                static_cast<int32>(InComponent->TextureRenderTargetFormat);
            State.ChannelDataType = static_cast<int32>(InComponent->ChannelDataType);
            State.NumChannels = InComponent->NumChannelsPerPixel;
            State.PostProcessBlendWeight = InComponent->PostProcessBlendWeight;
            State.AutoExposureMethod = static_cast<int32>(
                InComponent->PostProcessSettings.AutoExposureMethod.GetValue());
            State.AutoExposureMinBrightness =
                InComponent->PostProcessSettings.AutoExposureMinBrightness;
            State.AutoExposureMaxBrightness =
                InComponent->PostProcessSettings.AutoExposureMaxBrightness;
            State.AutoExposureSpeedUp =
                InComponent->PostProcessSettings.AutoExposureSpeedUp;
            State.AutoExposureSpeedDown =
                InComponent->PostProcessSettings.AutoExposureSpeedDown;
            State.AutoExposureBias =
                InComponent->PostProcessSettings.AutoExposureBias;
            State.bInitialized = InComponent->IsInitialized();
            State.bUseCustomProjection = InComponent->bUseCustomProjectionMatrix;
            State.bAlwaysPersist = InComponent->bAlwaysPersistRenderingState;
            State.bCaptureEveryFrame = InComponent->bCaptureEveryFrame;
            State.bCaptureOnMovement = InComponent->bCaptureOnMovement;
            State.bOverrideRenderTargetFormat =
                InComponent->bOverrideTextureRenderTargetFormat;
            State.bOverrideAutoExposureMethod =
                InComponent->PostProcessSettings.bOverride_AutoExposureMethod;
            State.bOverrideAutoExposureMinBrightness =
                InComponent->PostProcessSettings.bOverride_AutoExposureMinBrightness;
            State.bOverrideAutoExposureMaxBrightness =
                InComponent->PostProcessSettings.bOverride_AutoExposureMaxBrightness;
            State.bOverrideAutoExposureSpeedUp =
                InComponent->PostProcessSettings.bOverride_AutoExposureSpeedUp;
            State.bOverrideAutoExposureSpeedDown =
                InComponent->PostProcessSettings.bOverride_AutoExposureSpeedDown;
            State.bOverrideAutoExposureBias =
                InComponent->PostProcessSettings.bOverride_AutoExposureBias;
            return State;
        }

        bool Equals(const FTestViewPairCameraState& Other) const
        {
            return Component == Other.Component &&
                AttachParent == Other.AttachParent &&
                TextureTarget == Other.TextureTarget &&
                RelativeTransform.Equals(Other.RelativeTransform, 0.0) &&
                CustomProjectionMatrix.Equals(Other.CustomProjectionMatrix, 0.f) &&
                Width == Other.Width && Height == Other.Height &&
                TargetWidth == Other.TargetWidth &&
                TargetHeight == Other.TargetHeight &&
                FovDegrees == Other.FovDegrees &&
                Overscan == Other.Overscan &&
                ProjectionType == Other.ProjectionType &&
                CaptureSource == Other.CaptureSource &&
                RenderTargetFormat == Other.RenderTargetFormat &&
                ChannelDataType == Other.ChannelDataType &&
                NumChannels == Other.NumChannels &&
                PostProcessBlendWeight == Other.PostProcessBlendWeight &&
                AutoExposureMethod == Other.AutoExposureMethod &&
                AutoExposureMinBrightness == Other.AutoExposureMinBrightness &&
                AutoExposureMaxBrightness == Other.AutoExposureMaxBrightness &&
                AutoExposureSpeedUp == Other.AutoExposureSpeedUp &&
                AutoExposureSpeedDown == Other.AutoExposureSpeedDown &&
                AutoExposureBias == Other.AutoExposureBias &&
                bInitialized == Other.bInitialized &&
                bUseCustomProjection == Other.bUseCustomProjection &&
                bAlwaysPersist == Other.bAlwaysPersist &&
                bCaptureEveryFrame == Other.bCaptureEveryFrame &&
                bCaptureOnMovement == Other.bCaptureOnMovement &&
                bOverrideRenderTargetFormat == Other.bOverrideRenderTargetFormat &&
                bOverrideAutoExposureMethod ==
                    Other.bOverrideAutoExposureMethod &&
                bOverrideAutoExposureMinBrightness ==
                    Other.bOverrideAutoExposureMinBrightness &&
                bOverrideAutoExposureMaxBrightness ==
                    Other.bOverrideAutoExposureMaxBrightness &&
                bOverrideAutoExposureSpeedUp ==
                    Other.bOverrideAutoExposureSpeedUp &&
                bOverrideAutoExposureSpeedDown ==
                    Other.bOverrideAutoExposureSpeedDown &&
                bOverrideAutoExposureBias == Other.bOverrideAutoExposureBias;
        }
    };

    bool TestSnapshotsEqual(
        const FSpPixelGoalCameraSnapshot& A,
        const FSpPixelGoalCameraSnapshot& B)
    {
        return A.SnapshotId == B.SnapshotId &&
            A.IntrinsicsId == B.IntrinsicsId &&
            A.RenderSize == B.RenderSize &&
            A.InvViewMatrix.Equals(B.InvViewMatrix, 0.f) &&
            A.InvProjectionMatrix.Equals(B.InvProjectionMatrix, 0.f) &&
            A.CameraLocation.Equals(B.CameraLocation, 0.f) &&
            A.CameraRotation.Equals(B.CameraRotation, 0.f) &&
            A.AgentLocation.Equals(B.AgentLocation, 0.f) &&
            A.AgentRotation.Equals(B.AgentRotation, 0.f) &&
            A.HorizontalFovDegrees == B.HorizontalFovDegrees &&
            A.CapturedWorldTimeSeconds == B.CapturedWorldTimeSeconds &&
            A.ViewId == B.ViewId &&
            A.CaptureGroupId == B.CaptureGroupId &&
            A.CaptureComponent == B.CaptureComponent &&
            A.Agent == B.Agent;
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpCameraViewNormalizationTest,
    "SimWorld.PixelGoal.CameraViews.CaptureSourceNormalization",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpCameraViewNormalizationTest::RunTest(const FString& Parameters)
{
    FString View;
    FString Error;
    TestTrue(TEXT("omitted view defaults to front"),
        SpCameraCapture::NormalizeCameraView(FString(), View, Error));
    TestEqual(TEXT("omitted view is canonical front"), View, FString(TEXT("front")));
    TestTrue(TEXT("omitted view has no error"), Error.IsEmpty());

    View.Reset();
    Error = TEXT("stale");
    TestTrue(TEXT("mixed-case front is accepted"),
        SpCameraCapture::NormalizeCameraView(TEXT("FrOnT"), View, Error));
    TestEqual(TEXT("front is canonicalized"), View, FString(TEXT("front")));
    TestTrue(TEXT("front clears stale error"), Error.IsEmpty());

    View.Reset();
    Error.Reset();
    TestTrue(TEXT("mixed-case rear is accepted"),
        SpCameraCapture::NormalizeCameraView(TEXT("rEaR"), View, Error));
    TestEqual(TEXT("rear is canonicalized"), View, FString(TEXT("rear")));

    View = TEXT("stale");
    Error.Reset();
    TestFalse(TEXT("left is not a hidden camera view"),
        SpCameraCapture::NormalizeCameraView(TEXT("left"), View, Error));
    TestTrue(TEXT("unsupported view clears the output view"), View.IsEmpty());
    TestEqual(TEXT("unsupported view has the stable error"),
        Error, FString(TEXT("unsupported_camera_view")));

    View.Reset();
    Error.Reset();
    TestFalse(TEXT("surrounding whitespace is not another accepted spelling"),
        SpCameraCapture::NormalizeCameraView(TEXT(" front "), View, Error));
    TestEqual(TEXT("whitespace spelling has the stable error"),
        Error, FString(TEXT("unsupported_camera_view")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpCameraViewSourceParserTest,
    "SimWorld.PixelGoal.CameraViews.CaptureSourceParsing",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpCameraViewSourceParserTest::RunTest(const FString& Parameters)
{
    USpCameraCapturePool* Pool = NewObject<USpCameraCapturePool>();
    TestNotNull(TEXT("capture pool exists"), Pool);
    if (!Pool)
    {
        return false;
    }

    const FString PairRequest = TEXT(
        "{\"camera_sources\":["
        "{\"camera_id\":\"front\",\"agent_tag\":\"PixelGoalParisPocAgent\","
        "\"camera_view\":\"front\"},"
        "{\"camera_id\":\"rear\",\"agent_tag\":\"PixelGoalParisPocAgent\","
        "\"camera_view\":\"rear\"}],"
        "\"capture_mode\":\"agent_native\",\"initialize_only\":true}");
    const TSharedPtr<FJsonObject> PairResponse = ParseTestJson(
        Pool->Camera_CaptureCamerasJson(PairRequest));
    TestNotNull(TEXT("pair response parses"), PairResponse.Get());
    if (!PairResponse.IsValid())
    {
        return false;
    }

    const TArray<TSharedPtr<FJsonValue>>* Sources = nullptr;
    TestTrue(TEXT("pair response contains parsed sources"),
        PairResponse->TryGetArrayField(TEXT("camera_sources"), Sources));
    if (!Sources || Sources->Num() != 2)
    {
        AddError(TEXT("pair response must preserve exactly two camera sources"));
        return false;
    }
    const TSharedPtr<FJsonObject> FrontSource = (*Sources)[0]->AsObject();
    const TSharedPtr<FJsonObject> RearSource = (*Sources)[1]->AsObject();
    TestEqual(TEXT("first source remains front"),
        TestJsonString(FrontSource, TEXT("camera_id")), FString(TEXT("front")));
    TestEqual(TEXT("second source remains rear"),
        TestJsonString(RearSource, TEXT("camera_id")), FString(TEXT("rear")));
    TestEqual(TEXT("front source is agent-native"),
        TestJsonString(FrontSource, TEXT("source_type")), FString(TEXT("agent")));
    TestEqual(TEXT("rear source is agent-native"),
        TestJsonString(RearSource, TEXT("source_type")), FString(TEXT("agent")));
    TestEqual(TEXT("front source keeps agent tag"),
        TestJsonString(FrontSource, TEXT("agent_tag")),
        FString(TEXT("PixelGoalParisPocAgent")));
    TestEqual(TEXT("rear source keeps same agent tag"),
        TestJsonString(RearSource, TEXT("agent_tag")),
        FString(TEXT("PixelGoalParisPocAgent")));
    TestEqual(TEXT("first view is canonical front"),
        TestJsonString(FrontSource, TEXT("camera_view")), FString(TEXT("front")));
    TestEqual(TEXT("second view is canonical rear"),
        TestJsonString(RearSource, TEXT("camera_view")), FString(TEXT("rear")));

    const TSharedPtr<FJsonObject> InvalidResponse = ParseTestJson(
        Pool->Camera_CaptureCamerasJson(TEXT(
            "{\"camera_sources\":[{\"camera_id\":\"bad\","
            "\"agent_tag\":\"PixelGoalParisPocAgent\",\"camera_view\":\"left\"}],"
            "\"capture_mode\":\"agent_native\",\"initialize_only\":true}")));
    TestNotNull(TEXT("invalid-view response parses"), InvalidResponse.Get());
    const TArray<TSharedPtr<FJsonValue>>* Errors = nullptr;
    if (!InvalidResponse.IsValid() ||
        !InvalidResponse->TryGetArrayField(TEXT("errors"), Errors) ||
        !Errors || Errors->Num() != 1)
    {
        AddError(TEXT("invalid camera view must produce one source-level error"));
        return false;
    }
    TestEqual(TEXT("invalid camera view is rejected explicitly"),
        TestJsonString((*Errors)[0]->AsObject(), TEXT("error")),
        FString(TEXT("unsupported_camera_view")));

    const TSharedPtr<FJsonObject> LegacyResponse = ParseTestJson(
        Pool->Camera_CaptureCamerasJson(TEXT(
            "{\"agent_tags\":[\"PixelGoalParisPocAgent\"],"
            "\"capture_mode\":\"agent_native\",\"initialize_only\":true}")));
    const TArray<TSharedPtr<FJsonValue>>* LegacySources = nullptr;
    if (!LegacyResponse.IsValid() ||
        !LegacyResponse->TryGetArrayField(TEXT("camera_sources"), LegacySources) ||
        !LegacySources || LegacySources->Num() != 1)
    {
        AddError(TEXT("legacy agent_tags request must still produce one source"));
        return false;
    }
    TestEqual(TEXT("legacy agent_tags source defaults to front"),
        TestJsonString((*LegacySources)[0]->AsObject(), TEXT("camera_view")),
        FString(TEXT("front")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpCameraViewNativePairCaptureTest,
    "SimWorld.PixelGoal.CameraViews.NativePairCapture",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpCameraViewNativePairCaptureTest::RunTest(const FString& Parameters)
{
    UWorld* World = UWorld::CreateWorld(EWorldType::Game, false);
    TestNotNull(TEXT("native capture world exists"), World);
    if (!World || !GEngine)
    {
        if (World)
        {
            World->DestroyWorld(false);
        }
        return false;
    }
    FWorldContext& WorldContext = GEngine->CreateNewWorldContext(EWorldType::Game);
    WorldContext.SetCurrentWorld(World);

    ASpHumanoidAgent* Agent = World->SpawnActor<ASpHumanoidAgent>();
    TestNotNull(TEXT("native capture agent spawns"), Agent);
    if (!Agent)
    {
        World->DestroyWorld(false);
        GEngine->DestroyWorldContext(World);
        return false;
    }
    Agent->Agent_SetAgentTag(FName(TEXT("PixelGoalParisPocAgent")));
    Agent->SetActorLocationAndRotation(
        FVector(120.f, -340.f, 160.f), FRotator(0.f, 37.f, 0.f));
    Agent->SpawnDefaultController();
    AController* Controller = Agent->GetController();
    TestNotNull(TEXT("native capture agent has a controller"), Controller);
    if (!Controller)
    {
        World->DestroyWorld(false);
        GEngine->DestroyWorldContext(World);
        return false;
    }
    Controller->SetControlRotation(FRotator(0.f, -23.f, 0.f));

    TestTrue(TEXT("front slot configures for pair"),
        Agent->ConfigureObservationCamera(TEXT("front"), 20, 18, 75.f));
    TestTrue(TEXT("rear slot configures for pair"),
        Agent->ConfigureObservationCamera(TEXT("rear"), 24, 20, 80.f));
    Agent->InitializeObservationCamera(TEXT("front"));
    Agent->InitializeObservationCamera(TEXT("rear"));

    const FVector LocationBefore = Agent->GetActorLocation();
    const FRotator RotationBefore = Agent->GetActorRotation();
    const FRotator ControlRotationBefore = Controller->GetControlRotation();
    const FRotator SpringArmRotationBefore = Agent->SpringArm->GetRelativeRotation();

    USpCameraCapturePool* Pool = NewObject<USpCameraCapturePool>(World);
    TestNotNull(TEXT("world capture pool exists"), Pool);

    const TSharedPtr<FJsonObject> MixedErrorResponse = Pool
        ? ParseTestJson(Pool->Camera_CaptureCamerasJson(TEXT(
            "{\"camera_sources\":["
            "{\"camera_id\":\"missing-first\",\"agent_tag\":\"MissingPixelGoalAgent\","
            "\"camera_view\":\"front\"},"
            "{\"camera_id\":\"invalid-second\",\"agent_tag\":\"PixelGoalParisPocAgent\","
            "\"camera_view\":\"left\"}],"
            "\"capture_mode\":\"agent_native\",\"initialize_only\":true}")))
        : nullptr;
    const TArray<TSharedPtr<FJsonValue>>* MixedErrors = nullptr;
    TestTrue(TEXT("mixed-error response contains errors"),
        MixedErrorResponse.IsValid() && MixedErrorResponse->TryGetArrayField(
            TEXT("errors"), MixedErrors));
    TestEqual(TEXT("mixed-error response has one error per source"),
        MixedErrors ? MixedErrors->Num() : -1, 2);
    if (MixedErrors && MixedErrors->Num() == 2)
    {
        TestEqual(TEXT("first source error remains first"),
            TestJsonString((*MixedErrors)[0]->AsObject(), TEXT("error")),
            FString(TEXT("agent_not_visible_on_this_client")));
        TestEqual(TEXT("second source error remains second"),
            TestJsonString((*MixedErrors)[1]->AsObject(), TEXT("error")),
            FString(TEXT("unsupported_camera_view")));
    }

    const FString ResponseText = Pool
        ? Pool->Camera_CaptureCamerasJson(TEXT(
            "{\"camera_sources\":["
            "{\"camera_id\":\"front\",\"agent_tag\":\"PixelGoalParisPocAgent\","
            "\"camera_view\":\"front\"},"
            "{\"camera_id\":\"rear\",\"agent_tag\":\"PixelGoalParisPocAgent\","
            "\"camera_view\":\"rear\"}],"
            "\"capture_mode\":\"agent_native\",\"width\":32,\"height\":24,"
            "\"fov_degrees\":90,\"jpeg_quality\":50,\"force_capture\":true,"
            "\"validate\":false,\"modalities\":[\"rgb\"]}"))
        : FString();
    const TSharedPtr<FJsonObject> Response = ParseTestJson(ResponseText);
    TestNotNull(TEXT("native pair response parses"), Response.Get());

    const TArray<TSharedPtr<FJsonValue>>* Images = nullptr;
    if (!Response.IsValid() ||
        !Response->TryGetArrayField(TEXT("images"), Images) ||
        !Images || Images->Num() != 2)
    {
        AddError(FString::Printf(
            TEXT("native pair must return two images; response=%s"), *ResponseText));
        Agent->TerminateObservationCamera(TEXT("front"));
        Agent->TerminateObservationCamera(TEXT("rear"));
        World->DestroyWorld(false);
        GEngine->DestroyWorldContext(World);
        return false;
    }

    TestEqual(TEXT("two images"), Images->Num(), 2);
    const TSharedPtr<FJsonObject> FrontImage = (*Images)[0]->AsObject();
    const TSharedPtr<FJsonObject> RearImage = (*Images)[1]->AsObject();
    TestEqual(TEXT("first is front"),
        TestJsonString(FrontImage, TEXT("camera_view")), FString(TEXT("front")));
    TestEqual(TEXT("second is rear"),
        TestJsonString(RearImage, TEXT("camera_view")), FString(TEXT("rear")));
    TestEqual(TEXT("front component was configured by its source"),
        Agent->SceneCapture->Width, 32);
    TestEqual(TEXT("rear component was configured by its source"),
        Agent->RearSceneCapture->Width, 32);
    TestEqual(TEXT("front component height matches request"),
        Agent->SceneCapture->Height, 24);
    TestEqual(TEXT("rear component height matches request"),
        Agent->RearSceneCapture->Height, 24);

    const TSharedPtr<FJsonObject>* RootTiming = nullptr;
    TestTrue(TEXT("root aggregate timing exists"),
        Response->TryGetObjectField(TEXT("timing"), RootTiming));
    double PerCaptureReadMs[2] = {-1.0, -1.0};
    double PerEncodeMs[2] = {-1.0, -1.0};
    double PerWallMs[2] = {-1.0, -1.0};
    const TSharedPtr<FJsonObject> ImageObjects[2] = {FrontImage, RearImage};
    for (int32 Index = 0; Index < 2; ++Index)
    {
        const TSharedPtr<FJsonObject>* ImageTiming = nullptr;
        TestTrue(FString::Printf(TEXT("image %d timing exists"), Index),
            ImageObjects[Index]->TryGetObjectField(TEXT("timing"), ImageTiming));
        if (!ImageTiming || !ImageTiming->IsValid())
        {
            continue;
        }
        PerCaptureReadMs[Index] =
            TestJsonNumber(*ImageTiming, TEXT("capture_read_ms"));
        PerEncodeMs[Index] = TestJsonNumber(*ImageTiming, TEXT("encode_ms"));
        PerWallMs[Index] = TestJsonNumber(*ImageTiming, TEXT("wall_ms"));
        TestTrue(FString::Printf(TEXT("image %d capture/read is non-negative"), Index),
            PerCaptureReadMs[Index] >= 0.0);
        TestTrue(FString::Printf(TEXT("image %d encode is non-negative"), Index),
            PerEncodeMs[Index] >= 0.0);
        TestTrue(FString::Printf(TEXT("image %d wall is non-negative"), Index),
            PerWallMs[Index] >= 0.0);
    }
    if (RootTiming && RootTiming->IsValid())
    {
        const double RootCaptureReadMs =
            TestJsonNumber(*RootTiming, TEXT("capture_read_ms"));
        const double RootEncodeMs = TestJsonNumber(*RootTiming, TEXT("encode_ms"));
        const double RootTotalMs = TestJsonNumber(*RootTiming, TEXT("total_ms"));
        for (int32 Index = 0; Index < 2; ++Index)
        {
            TestTrue(FString::Printf(TEXT("root capture/read covers image %d"), Index),
                RootCaptureReadMs + 1.e-6 >= PerCaptureReadMs[Index]);
            TestTrue(FString::Printf(TEXT("root encode covers image %d"), Index),
                RootEncodeMs + 1.e-6 >= PerEncodeMs[Index]);
            TestTrue(FString::Printf(TEXT("root total covers image %d wall"), Index),
                RootTotalMs + 1.e-6 >= PerWallMs[Index]);
        }
    }

    const TSharedPtr<FJsonObject> InvalidLiveResponse = ParseTestJson(
        Pool->Camera_CaptureCamerasJson(TEXT(
            "{\"camera_sources\":[{\"camera_id\":\"bad\","
            "\"agent_tag\":\"PixelGoalParisPocAgent\",\"camera_view\":\"left\"}],"
            "\"capture_mode\":\"agent_native\",\"width\":64,\"height\":48,"
            "\"validate\":false,\"modalities\":[\"rgb\"]}")));
    const TArray<TSharedPtr<FJsonValue>>* InvalidLiveImages = nullptr;
    const TArray<TSharedPtr<FJsonValue>>* InvalidLiveErrors = nullptr;
    TestTrue(TEXT("live invalid-view response has images array"),
        InvalidLiveResponse.IsValid() && InvalidLiveResponse->TryGetArrayField(
            TEXT("images"), InvalidLiveImages));
    TestTrue(TEXT("live invalid-view response has errors array"),
        InvalidLiveResponse.IsValid() && InvalidLiveResponse->TryGetArrayField(
            TEXT("errors"), InvalidLiveErrors));
    TestEqual(TEXT("invalid view captures no fallback image"),
        InvalidLiveImages ? InvalidLiveImages->Num() : -1, 0);
    TestEqual(TEXT("invalid view emits one live source error"),
        InvalidLiveErrors ? InvalidLiveErrors->Num() : -1, 1);
    if (InvalidLiveErrors && InvalidLiveErrors->Num() == 1)
    {
        TestEqual(TEXT("live invalid view uses stable error"),
            TestJsonString((*InvalidLiveErrors)[0]->AsObject(), TEXT("error")),
            FString(TEXT("unsupported_camera_view")));
    }
    TestEqual(TEXT("invalid view does not reconfigure front"),
        Agent->SceneCapture->Width, 32);
    TestEqual(TEXT("invalid view does not reconfigure rear"),
        Agent->RearSceneCapture->Width, 32);

    struct FMalformedCameraViewCase
    {
        const TCHAR* Name;
        const TCHAR* JsonValue;
    };
    const FMalformedCameraViewCase MalformedCases[] = {
        {TEXT("number"), TEXT("17")},
        {TEXT("boolean"), TEXT("true")},
        {TEXT("null"), TEXT("null")},
        {TEXT("array"), TEXT("[]")},
        {TEXT("object"), TEXT("{}")},
    };
    for (const FMalformedCameraViewCase& MalformedCase : MalformedCases)
    {
        const FString MalformedRequest = FString::Printf(
            TEXT("{\"camera_sources\":[{\"camera_id\":\"malformed-%s\","
                 "\"agent_tag\":\"PixelGoalParisPocAgent\",\"camera_view\":%s}],"
                 "\"capture_mode\":\"agent_native\",\"width\":96,\"height\":72,"
                 "\"validate\":false,\"modalities\":[\"rgb\"]}"),
            MalformedCase.Name,
            MalformedCase.JsonValue);
        const TSharedPtr<FJsonObject> MalformedResponse = ParseTestJson(
            Pool->Camera_CaptureCamerasJson(MalformedRequest));
        const TArray<TSharedPtr<FJsonValue>>* MalformedImages = nullptr;
        const TArray<TSharedPtr<FJsonValue>>* MalformedErrors = nullptr;
        TestTrue(FString::Printf(TEXT("%s view response has images"), MalformedCase.Name),
            MalformedResponse.IsValid() && MalformedResponse->TryGetArrayField(
                TEXT("images"), MalformedImages));
        TestTrue(FString::Printf(TEXT("%s view response has errors"), MalformedCase.Name),
            MalformedResponse.IsValid() && MalformedResponse->TryGetArrayField(
                TEXT("errors"), MalformedErrors));
        TestEqual(FString::Printf(TEXT("%s view captures no fallback"), MalformedCase.Name),
            MalformedImages ? MalformedImages->Num() : -1, 0);
        TestEqual(FString::Printf(TEXT("%s view has one source error"), MalformedCase.Name),
            MalformedErrors ? MalformedErrors->Num() : -1, 1);
        if (MalformedErrors && MalformedErrors->Num() == 1)
        {
            TestEqual(FString::Printf(TEXT("%s view uses stable error"), MalformedCase.Name),
                TestJsonString((*MalformedErrors)[0]->AsObject(), TEXT("error")),
                FString(TEXT("unsupported_camera_view")));
        }
        TestEqual(FString::Printf(TEXT("%s view leaves front unchanged"), MalformedCase.Name),
            Agent->SceneCapture->Width, 32);
        TestEqual(FString::Printf(TEXT("%s view leaves rear unchanged"), MalformedCase.Name),
            Agent->RearSceneCapture->Width, 32);
    }

    const TSharedPtr<FJsonObject> LegacyLiveResponse = ParseTestJson(
        Pool->Camera_CaptureCamerasJson(TEXT(
            "{\"agent_tags\":[\"PixelGoalParisPocAgent\"],"
            "\"capture_mode\":\"agent_native\",\"width\":40,\"height\":24,"
            "\"fov_degrees\":90,\"jpeg_quality\":50,\"force_capture\":true,"
            "\"validate\":false,\"modalities\":[\"rgb\"]}")));
    const TArray<TSharedPtr<FJsonValue>>* LegacyLiveImages = nullptr;
    TestTrue(TEXT("live legacy response contains images"),
        LegacyLiveResponse.IsValid() && LegacyLiveResponse->TryGetArrayField(
            TEXT("images"), LegacyLiveImages));
    TestEqual(TEXT("live legacy request returns one image"),
        LegacyLiveImages ? LegacyLiveImages->Num() : -1, 1);
    if (LegacyLiveImages && LegacyLiveImages->Num() == 1)
    {
        TestEqual(TEXT("live legacy image is front"),
            TestJsonString((*LegacyLiveImages)[0]->AsObject(), TEXT("camera_view")),
            FString(TEXT("front")));
    }
    TestEqual(TEXT("live legacy request configures front"),
        Agent->SceneCapture->Width, 40);
    TestEqual(TEXT("live legacy request leaves rear unchanged"),
        Agent->RearSceneCapture->Width, 32);

    Agent->TerminateObservationCamera(TEXT("front"));
    TestFalse(TEXT("front render target is unavailable for rear-only capture"),
        Agent->SceneCapture->IsInitialized());
    TestNull(TEXT("front texture target is unavailable for rear-only capture"),
        Agent->SceneCapture->TextureTarget);
    TestTrue(TEXT("rear remains initialized for rear-only capture"),
        Agent->RearSceneCapture->IsInitialized());
    TestNotNull(TEXT("rear texture target remains available"),
        Agent->RearSceneCapture->TextureTarget.Get());

    const TSharedPtr<FJsonObject> RearOnlyResponse = ParseTestJson(
        Pool->Camera_CaptureCamerasJson(TEXT(
            "{\"camera_sources\":[{\"camera_id\":\"rear-only\","
            "\"agent_tag\":\"PixelGoalParisPocAgent\",\"camera_view\":\"rear\"}],"
            "\"capture_mode\":\"agent_native\",\"width\":32,\"height\":24,"
            "\"fov_degrees\":90,\"jpeg_quality\":50,\"force_capture\":true,"
            "\"validate\":false,\"modalities\":[\"rgb\"]}")));
    const TArray<TSharedPtr<FJsonValue>>* RearOnlyImages = nullptr;
    const TArray<TSharedPtr<FJsonValue>>* RearOnlyErrors = nullptr;
    TestTrue(TEXT("rear-only response contains images"),
        RearOnlyResponse.IsValid() && RearOnlyResponse->TryGetArrayField(
            TEXT("images"), RearOnlyImages));
    TestTrue(TEXT("rear-only response contains errors"),
        RearOnlyResponse.IsValid() && RearOnlyResponse->TryGetArrayField(
            TEXT("errors"), RearOnlyErrors));
    TestEqual(TEXT("rear-only capture reads one selected render target"),
        RearOnlyImages ? RearOnlyImages->Num() : -1, 1);
    TestEqual(TEXT("rear-only capture has no front-target error"),
        RearOnlyErrors ? RearOnlyErrors->Num() : -1, 0);
    if (RearOnlyImages && RearOnlyImages->Num() == 1)
    {
        TestEqual(TEXT("rear-only image metadata remains rear"),
            TestJsonString((*RearOnlyImages)[0]->AsObject(), TEXT("camera_view")),
            FString(TEXT("rear")));
    }
    TestFalse(TEXT("rear-only capture does not initialize front"),
        Agent->SceneCapture->IsInitialized());

    TestTrue(TEXT("pair capture preserves actor location"),
        Agent->GetActorLocation().Equals(LocationBefore, 0.0));
    TestTrue(TEXT("pair capture preserves actor rotation"),
        Agent->GetActorRotation().Equals(RotationBefore, 0.0));
    TestTrue(TEXT("pair capture preserves controller rotation"),
        Controller->GetControlRotation().Equals(ControlRotationBefore, 0.0));
    TestTrue(TEXT("pair capture preserves spring-arm rotation"),
        Agent->SpringArm->GetRelativeRotation().Equals(
            SpringArmRotationBefore, 0.0));

    Agent->TerminateObservationCamera(TEXT("front"));
    Agent->TerminateObservationCamera(TEXT("rear"));
    World->DestroyWorld(false);
    GEngine->DestroyWorldContext(World);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalCameraViewIdentityTest,
    "SimWorld.PixelGoal.CameraViews.IdentityAndRotation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalCameraViewIdentityTest::RunTest(const FString& Parameters)
{
    TestTrue(TEXT("front is supported"), SpPixelGoal::IsSupportedCameraView(TEXT("front")));
    TestTrue(TEXT("rear is supported"), SpPixelGoal::IsSupportedCameraView(TEXT("rear")));
    TestFalse(TEXT("left is not a hidden third view"), SpPixelGoal::IsSupportedCameraView(TEXT("left")));
    TestEqual(TEXT("front yaw is zero"),
        SpPixelGoal::CameraViewRelativeRotation(TEXT("front")).Yaw, 0.0);
    TestEqual(TEXT("rear yaw is opposite"),
        SpPixelGoal::CameraViewRelativeRotation(TEXT("rear")).Yaw, 180.0);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalHumanoidObservationCamerasTest,
    "SimWorld.PixelGoal.CameraViews.HumanoidConfiguration",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalHumanoidObservationCamerasTest::RunTest(const FString& Parameters)
{
    UWorld* World = UWorld::CreateWorld(EWorldType::Game, false);
    TestNotNull(TEXT("transient world exists"), World);
    if (!World)
    {
        return false;
    }

    FWorldContext& WorldContext = GEngine->CreateNewWorldContext(EWorldType::Game);
    WorldContext.SetCurrentWorld(World);

    ASpHumanoidAgent* Agent = World->SpawnActor<ASpHumanoidAgent>();
    TestNotNull(TEXT("humanoid agent spawns"), Agent);
    if (!Agent)
    {
        World->DestroyWorld(false);
        GEngine->DestroyWorldContext(World);
        return false;
    }

    TestNotNull(TEXT("front capture exists"), Agent->SceneCapture);
    TestNotNull(TEXT("rear capture exists"), Agent->RearSceneCapture.Get());
    if (!Agent->SceneCapture || !Agent->RearSceneCapture)
    {
        World->DestroyWorld(false);
        GEngine->DestroyWorldContext(World);
        return false;
    }

    TestEqual(TEXT("front slot resolves front capture"),
        Agent->GetObservationCamera(TEXT("front")), Agent->SceneCapture);
    TestEqual(TEXT("rear slot resolves rear capture"),
        Agent->GetObservationCamera(TEXT("rear")), Agent->RearSceneCapture.Get());
    TestNull(TEXT("unsupported slot has no capture"),
        Agent->GetObservationCamera(TEXT("left")));
    TestEqual(TEXT("same attachment"),
        Agent->SceneCapture->GetAttachParent(), Agent->RearSceneCapture->GetAttachParent());
    TestTrue(TEXT("same relative location"),
        Agent->SceneCapture->GetRelativeLocation().Equals(
            Agent->RearSceneCapture->GetRelativeLocation(), 0.01f));
    TestTrue(TEXT("rear is 180 degrees from front"), FMath::IsNearlyEqual(
        FMath::Abs(FMath::FindDeltaAngleDegrees(
            Agent->SceneCapture->GetRelativeRotation().Yaw,
            Agent->RearSceneCapture->GetRelativeRotation().Yaw)), 180.0, 0.01));

    Agent->SceneCapture->Width = 1;
    Agent->SceneCapture->Height = 2;
    Agent->SceneCapture->FOVAngle = 3.f;
    Agent->RearSceneCapture->Width = 4;
    Agent->RearSceneCapture->Height = 5;
    Agent->RearSceneCapture->FOVAngle = 6.f;
    TestTrue(TEXT("front configures"),
        Agent->ConfigureObservationCamera(TEXT("front"), 640, 360, 90.f));
    TestTrue(TEXT("rear configures"),
        Agent->ConfigureObservationCamera(TEXT("rear"), 640, 360, 90.f));
    TestFalse(TEXT("unsupported slot is rejected"),
        Agent->ConfigureObservationCamera(TEXT("left"), 640, 360, 90.f));

    TestEqual(TEXT("front width is configured"), Agent->SceneCapture->Width, 640);
    TestEqual(TEXT("front height is configured"), Agent->SceneCapture->Height, 360);
    TestEqual(TEXT("front FOV is configured"), Agent->SceneCapture->FOVAngle, 90.f);
    TestEqual(TEXT("rear width matches front"),
        Agent->RearSceneCapture->Width, Agent->SceneCapture->Width);
    TestEqual(TEXT("rear height matches front"),
        Agent->RearSceneCapture->Height, Agent->SceneCapture->Height);
    TestEqual(TEXT("rear FOV matches front"),
        Agent->RearSceneCapture->FOVAngle, Agent->SceneCapture->FOVAngle);
    TestEqual(TEXT("capture source matches"),
        Agent->RearSceneCapture->CaptureSource, Agent->SceneCapture->CaptureSource);
    TestEqual(TEXT("projection matches"),
        Agent->RearSceneCapture->ProjectionType, Agent->SceneCapture->ProjectionType);
    TestEqual(TEXT("render-target format matches"),
        Agent->RearSceneCapture->TextureRenderTargetFormat,
        Agent->SceneCapture->TextureRenderTargetFormat);

    Agent->RearSceneCapture->Width = 320;
    TestTrue(TEXT("legacy configuration wrapper succeeds"),
        Agent->ConfigureCamera(800, 450, 100.f));
    TestEqual(TEXT("legacy wrapper configures front width"),
        Agent->SceneCapture->Width, 800);
    TestEqual(TEXT("legacy wrapper leaves rear width unchanged"),
        Agent->RearSceneCapture->Width, 320);

    Agent->InitializeCamera();
    TestTrue(TEXT("legacy initialization wrapper initializes front"),
        Agent->SceneCapture->IsInitialized());
    TestFalse(TEXT("legacy initialization wrapper leaves rear uninitialized"),
        Agent->RearSceneCapture->IsInitialized());
    Agent->TerminateCamera();
    TestFalse(TEXT("legacy termination wrapper terminates front"),
        Agent->SceneCapture->IsInitialized());

    Agent->InitializeObservationCamera(TEXT("rear"));
    TestTrue(TEXT("rear slot initializes rear"),
        Agent->RearSceneCapture->IsInitialized());
    Agent->TerminateObservationCamera(TEXT("rear"));
    TestFalse(TEXT("rear slot terminates rear"),
        Agent->RearSceneCapture->IsInitialized());

    World->DestroyWorld(false);
    GEngine->DestroyWorldContext(World);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalCenterRayTest,
    "SimWorld.PixelGoal.Deprojection.CenterRay",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalCenterRayTest::RunTest(const FString& Parameters)
{
    const FVector CameraLocation(100.f, 200.f, 160.f);
    const FRotator Rotation(0.f, 30.f, 0.f);
    const FSpPixelGoalCameraSnapshot Snapshot =
        FSpPixelGoalCameraSnapshot::PerspectiveForTest(
            CameraLocation, Rotation, 640, 360, 90.f);

    FVector Origin;
    FVector Direction;
    TestTrue(
        TEXT("centre deprojects"),
        Snapshot.Deproject(FVector2D(0.5, 0.5), Origin, Direction));
    TestTrue(
        TEXT("centre follows camera forward"),
        Direction.Equals(Rotation.Vector(), 1.e-4f));
    const FVector2D SampleUVs[] = {
        FVector2D(0.5, 0.5),
        FVector2D(0.17, 0.29),
        FVector2D(0.83, 0.71),
    };
    for (int32 Index = 0; Index < UE_ARRAY_COUNT(SampleUVs); ++Index)
    {
        TestTrue(
            FString::Printf(TEXT("sample %d deprojects"), Index),
            Snapshot.Deproject(SampleUVs[Index], Origin, Direction));
        TestTrue(
            FString::Printf(TEXT("sample %d starts at the exact camera location"), Index),
            Origin.Equals(CameraLocation, 0.0));
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalFrontRearRayTest,
    "SimWorld.PixelGoal.Deprojection.FrontRearRays",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalFrontRearRayTest::RunTest(const FString& Parameters)
{
    const FSpPixelGoalCameraSnapshot Front =
        FSpPixelGoalCameraSnapshot::PerspectiveForTest(
            FVector(0.f, 0.f, 160.f), FRotator(0.f, 0.f, 0.f), 640, 360, 90.f);
    const FSpPixelGoalCameraSnapshot Rear =
        FSpPixelGoalCameraSnapshot::PerspectiveForTest(
            FVector(0.f, 0.f, 160.f), FRotator(0.f, 180.f, 0.f), 640, 360, 90.f);

    FVector FrontOrigin;
    FVector FrontCentre;
    FVector RearOrigin;
    FVector RearCentre;
    TestTrue(TEXT("front lower centre deprojects"),
        Front.Deproject(FVector2D(0.5, 0.75), FrontOrigin, FrontCentre));
    TestTrue(TEXT("rear lower centre deprojects"),
        Rear.Deproject(FVector2D(0.5, 0.75), RearOrigin, RearCentre));
    TestTrue(TEXT("front and rear share one optical centre"),
        FrontOrigin.Equals(RearOrigin, 1.e-3f));
    const FVector2D FrontPlanar(FrontCentre.X, FrontCentre.Y);
    const FVector2D RearPlanar(RearCentre.X, RearCentre.Y);
    TestTrue(TEXT("front and rear point into opposite horizontal hemispheres"),
        FrontPlanar.GetSafeNormal().Dot(RearPlanar.GetSafeNormal()) < -0.99f);

    const FSpPixelGoalCameraSnapshot* Views[] = {&Front, &Rear};
    const TCHAR* ViewNames[] = {TEXT("front"), TEXT("rear")};
    for (int32 Index = 0; Index < 2; ++Index)
    {
        FVector Origin;
        FVector Centre;
        FVector Left;
        FVector Right;
        TestTrue(FString::Printf(TEXT("%s centre deprojects"), ViewNames[Index]),
            Views[Index]->Deproject(FVector2D(0.5, 0.5), Origin, Centre));
        TestTrue(FString::Printf(TEXT("%s left deprojects"), ViewNames[Index]),
            Views[Index]->Deproject(FVector2D(0.25, 0.5), Origin, Left));
        TestTrue(FString::Printf(TEXT("%s right deprojects"), ViewNames[Index]),
            Views[Index]->Deproject(FVector2D(0.75, 0.5), Origin, Right));
        TestTrue(FString::Printf(TEXT("%s left/right are camera-local symmetric"),
                ViewNames[Index]),
            FMath::IsNearlyEqual(
                FVector::DotProduct(Left, Centre),
                FVector::DotProduct(Right, Centre),
                1.e-4f));
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalSnapshotBindingTest,
    "SimWorld.PixelGoal.Snapshots.Binding",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalSnapshotBindingTest::RunTest(const FString& Parameters)
{
    TestTrue(TEXT("exact snapshot binding is accepted"),
        SpPixelGoal::SnapshotBindingError(
            TEXT("rear"), TEXT("view-pair-7"), TEXT("rear"), TEXT("view-pair-7"))
            .IsEmpty());
    TestEqual(TEXT("view mismatch fails closed"),
        SpPixelGoal::SnapshotBindingError(
            TEXT("front"), TEXT("view-pair-7"), TEXT("rear"), TEXT("view-pair-7")),
        FString(TEXT("camera_snapshot_view_mismatch")));
    TestEqual(TEXT("group mismatch fails closed"),
        SpPixelGoal::SnapshotBindingError(
            TEXT("rear"), TEXT("view-pair-6"), TEXT("rear"), TEXT("view-pair-7")),
        FString(TEXT("camera_snapshot_group_mismatch")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalViewPairSnapshotValidationTest,
    "SimWorld.PixelGoal.Snapshots.ViewPairValidation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalViewPairSnapshotValidationTest::RunTest(const FString& Parameters)
{
    USpSceneCaptureComponent2D* FrontComponent =
        NewObject<USpSceneCaptureComponent2D>();
    USpSceneCaptureComponent2D* RearComponent =
        NewObject<USpSceneCaptureComponent2D>();
    FSpPixelGoalCameraSnapshot Front =
        FSpPixelGoalCameraSnapshot::PerspectiveForTest(
            FVector::ZeroVector, FRotator::ZeroRotator, 640, 360, 90.f);
    FSpPixelGoalCameraSnapshot Rear =
        FSpPixelGoalCameraSnapshot::PerspectiveForTest(
            FVector::ZeroVector, FRotator(0.f, 180.f, 0.f), 640, 360, 90.f);
    Front.SnapshotId = TEXT("camera-snapshot-10");
    Rear.SnapshotId = TEXT("camera-snapshot-11");
    Front.IntrinsicsId = TEXT("intrinsics-640x360-fov90");
    Rear.IntrinsicsId = Front.IntrinsicsId;
    Front.ViewId = TEXT("front");
    Rear.ViewId = TEXT("rear");
    Front.CaptureGroupId = TEXT("view-pair-4");
    Rear.CaptureGroupId = Front.CaptureGroupId;
    Front.CaptureComponent = FrontComponent;
    Rear.CaptureComponent = RearComponent;

    const auto Validate = [&](const FSpPixelGoalCameraSnapshot& TestFront,
                              const FSpPixelGoalCameraSnapshot& TestRear,
                              const FString& FrontImageSnapshotId,
                              const FString& RearImageSnapshotId,
                              const FString& FrontImageIntrinsicsId,
                              const FString& RearImageIntrinsicsId)
    {
        return SpPixelGoal::ViewPairSnapshotError(
            TestFront,
            TestRear,
            FrontImageSnapshotId,
            RearImageSnapshotId,
            FrontImageIntrinsicsId,
            RearImageIntrinsicsId,
            640,
            360,
            90.f,
            TEXT("view-pair-4"),
            FrontComponent,
            RearComponent);
    };

    TestTrue(TEXT("exact pair snapshot metadata and intrinsics are accepted"),
        Validate(
            Front, Rear, Front.SnapshotId, Rear.SnapshotId,
            Front.IntrinsicsId, Rear.IntrinsicsId).IsEmpty());
    TestEqual(TEXT("swapped view order is rejected"),
        Validate(
            Rear, Front, Rear.SnapshotId, Front.SnapshotId,
            Rear.IntrinsicsId, Front.IntrinsicsId),
        FString(TEXT("invalid_view_pair_snapshot")));

    FSpPixelGoalCameraSnapshot WrongGroup = Rear;
    WrongGroup.CaptureGroupId = TEXT("view-pair-elsewhere");
    TestEqual(TEXT("wrong retained capture group is rejected"),
        Validate(
            Front, WrongGroup, Front.SnapshotId, WrongGroup.SnapshotId,
            Front.IntrinsicsId, WrongGroup.IntrinsicsId),
        FString(TEXT("invalid_view_pair_snapshot")));

    FSpPixelGoalCameraSnapshot WrongComponent = Rear;
    WrongComponent.CaptureComponent = FrontComponent;
    TestEqual(TEXT("wrong retained component identity is rejected"),
        Validate(
            Front, WrongComponent, Front.SnapshotId, WrongComponent.SnapshotId,
            Front.IntrinsicsId, WrongComponent.IntrinsicsId),
        FString(TEXT("invalid_view_pair_snapshot")));

    FSpPixelGoalCameraSnapshot Duplicate = Rear;
    Duplicate.SnapshotId = Front.SnapshotId;
    TestEqual(TEXT("duplicate retained snapshot ids are rejected"),
        Validate(
            Front, Duplicate, Front.SnapshotId, Duplicate.SnapshotId,
            Front.IntrinsicsId, Duplicate.IntrinsicsId),
        FString(TEXT("duplicate_view_pair_snapshot")));

    TestEqual(TEXT("image id must identify the exact retained snapshot"),
        Validate(
            Front, Rear, TEXT("camera-snapshot-other"), Rear.SnapshotId,
            Front.IntrinsicsId, Rear.IntrinsicsId),
        FString(TEXT("invalid_view_pair_snapshot")));
    TestEqual(TEXT("image intrinsics must match its retained snapshot"),
        Validate(
            Front, Rear, Front.SnapshotId, Rear.SnapshotId,
            TEXT("intrinsics-image-only"), Rear.IntrinsicsId),
        FString(TEXT("view_pair_image_snapshot_intrinsics_mismatch")));

    FSpPixelGoalCameraSnapshot DifferentIntrinsicsId = Rear;
    DifferentIntrinsicsId.IntrinsicsId = TEXT("intrinsics-retained-other");
    TestEqual(TEXT("retained intrinsics ids must match"),
        Validate(
            Front, DifferentIntrinsicsId,
            Front.SnapshotId, DifferentIntrinsicsId.SnapshotId,
            Front.IntrinsicsId, DifferentIntrinsicsId.IntrinsicsId),
        FString(TEXT("view_pair_intrinsics_mismatch")));

    FSpPixelGoalCameraSnapshot DifferentSize = Rear;
    DifferentSize.RenderSize = FIntPoint(800, 450);
    TestEqual(TEXT("retained render dimensions must match"),
        Validate(
            Front, DifferentSize, Front.SnapshotId, DifferentSize.SnapshotId,
            Front.IntrinsicsId, DifferentSize.IntrinsicsId),
        FString(TEXT("view_pair_effective_intrinsics_mismatch")));

    FSpPixelGoalCameraSnapshot DifferentFov = Rear;
    DifferentFov.HorizontalFovDegrees = 89.f;
    TestEqual(TEXT("retained horizontal FOV must match"),
        Validate(
            Front, DifferentFov, Front.SnapshotId, DifferentFov.SnapshotId,
            Front.IntrinsicsId, DifferentFov.IntrinsicsId),
        FString(TEXT("view_pair_effective_intrinsics_mismatch")));

    FSpPixelGoalCameraSnapshot DifferentProjection = Rear;
    DifferentProjection.InvProjectionMatrix.M[0][0] += 0.01f;
    TestEqual(TEXT("each retained projection must match the request"),
        Validate(
            Front, DifferentProjection,
            Front.SnapshotId, DifferentProjection.SnapshotId,
            Front.IntrinsicsId, DifferentProjection.IntrinsicsId),
        FString(TEXT("view_pair_effective_intrinsics_mismatch")));

    FSpPixelGoalCameraSnapshot SubToleranceFovFront = Front;
    FSpPixelGoalCameraSnapshot SubToleranceFovRear = Rear;
    SubToleranceFovFront.HorizontalFovDegrees += 0.00005f;
    SubToleranceFovRear.HorizontalFovDegrees += 0.00005f;
    TestEqual(TEXT("common sub-tolerance FOV drift from request is rejected"),
        Validate(
            SubToleranceFovFront,
            SubToleranceFovRear,
            SubToleranceFovFront.SnapshotId,
            SubToleranceFovRear.SnapshotId,
            SubToleranceFovFront.IntrinsicsId,
            SubToleranceFovRear.IntrinsicsId),
        FString(TEXT("view_pair_effective_intrinsics_mismatch")));

    FSpPixelGoalCameraSnapshot SubToleranceProjectionFront = Front;
    FSpPixelGoalCameraSnapshot SubToleranceProjectionRear = Rear;
    SubToleranceProjectionFront.InvProjectionMatrix.M[0][0] += 0.000001f;
    SubToleranceProjectionRear.InvProjectionMatrix.M[0][0] += 0.000001f;
    TestEqual(TEXT("common sub-tolerance projection drift is rejected"),
        Validate(
            SubToleranceProjectionFront,
            SubToleranceProjectionRear,
            SubToleranceProjectionFront.SnapshotId,
            SubToleranceProjectionRear.SnapshotId,
            SubToleranceProjectionFront.IntrinsicsId,
            SubToleranceProjectionRear.IntrinsicsId),
        FString(TEXT("view_pair_effective_intrinsics_mismatch")));

    FSpPixelGoalCameraSnapshot NearOpticalCentre = Rear;
    NearOpticalCentre.CameraLocation.X += 0.05f;
    TestEqual(TEXT("sub-centimetre optical-centre mismatch is rejected"),
        Validate(
            Front,
            NearOpticalCentre,
            Front.SnapshotId,
            NearOpticalCentre.SnapshotId,
            Front.IntrinsicsId,
            NearOpticalCentre.IntrinsicsId),
        FString(TEXT("view_pair_optical_centre_mismatch")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalViewPairImageValidationTest,
    "SimWorld.PixelGoal.Snapshots.ViewPairImageValidation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalViewPairImageValidationTest::RunTest(const FString& Parameters)
{
    const auto Validate = [](const TSharedPtr<FJsonObject>& Image,
                             FString& OutSnapshotId,
                             FString& OutIntrinsicsId)
    {
        return SpPixelGoal::ViewPairImageError(
            Image,
            TEXT("front"),
            TEXT("PixelGoalParisPocAgent"),
            TEXT("view-pair-4"),
            640,
            360,
            FVector(120.f, -340.f, 160.f),
            37.f,
            OutSnapshotId,
            OutIntrinsicsId);
    };

    FString SnapshotId;
    FString IntrinsicsId;
    TestTrue(TEXT("complete exact front image metadata is accepted"),
        Validate(
            MakeValidViewPairImageForTest(), SnapshotId, IntrinsicsId).IsEmpty());
    TestEqual(TEXT("validator extracts exact snapshot id"),
        SnapshotId, FString(TEXT("camera-snapshot-10")));
    TestEqual(TEXT("validator extracts exact intrinsics id"),
        IntrinsicsId, FString(TEXT("intrinsics-640x360-fov90")));

    struct FInvalidDimensionCase
    {
        const TCHAR* Name;
        const TCHAR* Field;
        TSharedPtr<FJsonValue> Value;
        bool bRemoveField = false;
    };
    const FInvalidDimensionCase Cases[] = {
        {TEXT("fractional width"), TEXT("width"),
            MakeShared<FJsonValueNumber>(639.6)},
        {TEXT("fractional height"), TEXT("height"),
            MakeShared<FJsonValueNumber>(359.6)},
        {TEXT("missing width"), TEXT("width"), nullptr, true},
        {TEXT("string width"), TEXT("width"),
            MakeShared<FJsonValueString>(TEXT("640"))},
        {TEXT("boolean width"), TEXT("width"),
            MakeShared<FJsonValueBoolean>(true)},
        {TEXT("null width"), TEXT("width"), MakeShared<FJsonValueNull>()},
        {TEXT("array width"), TEXT("width"),
            MakeShared<FJsonValueArray>(TArray<TSharedPtr<FJsonValue>>())},
        {TEXT("object width"), TEXT("width"),
            MakeShared<FJsonValueObject>(MakeShared<FJsonObject>())},
        {TEXT("NaN width"), TEXT("width"),
            MakeShared<FJsonValueNumber>(
                std::numeric_limits<double>::quiet_NaN())},
        {TEXT("infinite width"), TEXT("width"),
            MakeShared<FJsonValueNumber>(
                std::numeric_limits<double>::infinity())},
    };
    for (const FInvalidDimensionCase& Case : Cases)
    {
        TSharedPtr<FJsonObject> Image = MakeValidViewPairImageForTest();
        if (Case.bRemoveField)
        {
            Image->RemoveField(Case.Field);
        }
        else
        {
            Image->SetField(Case.Field, Case.Value);
        }
        SnapshotId = TEXT("stale");
        IntrinsicsId = TEXT("stale");
        TestEqual(FString::Printf(TEXT("%s is rejected"), Case.Name),
            Validate(Image, SnapshotId, IntrinsicsId),
            FString(TEXT("invalid_view_pair_image")));
        TestTrue(FString::Printf(TEXT("%s clears snapshot output"), Case.Name),
            SnapshotId.IsEmpty());
        TestTrue(FString::Printf(TEXT("%s clears intrinsics output"), Case.Name),
            IntrinsicsId.IsEmpty());
    }

    TSharedPtr<FJsonObject> WrongOrder = MakeValidViewPairImageForTest();
    WrongOrder->SetStringField(TEXT("camera_view"), TEXT("rear"));
    TestEqual(TEXT("wrong image view/order metadata is rejected"),
        Validate(WrongOrder, SnapshotId, IntrinsicsId),
        FString(TEXT("invalid_view_pair_image")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalViewPairRequestValidationTest,
    "SimWorld.PixelGoal.Snapshots.ViewPairRequestValidation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalViewPairRequestValidationTest::RunTest(const FString& Parameters)
{
    FSpPixelGoalViewPairCaptureRequest Parsed;
    TestTrue(TEXT("minimal exact pair request uses documented defaults"),
        SpPixelGoal::ParseViewPairCaptureRequest(
            MakeValidViewPairRequestForTest(), Parsed).IsEmpty());
    TestEqual(TEXT("exact agent tag is retained"),
        Parsed.AgentTag, FString(TEXT("PixelGoalParisPocAgent")));
    TestEqual(TEXT("default width is 640"), Parsed.Width, 640);
    TestEqual(TEXT("default height is 360"), Parsed.Height, 360);
    TestEqual(TEXT("default FOV is 90"), Parsed.HorizontalFovDegrees, 90.f);
    TestEqual(TEXT("default JPEG quality is 90"), Parsed.JpegQuality, 90);
    TestTrue(TEXT("validation defaults enabled"), Parsed.bValidate);
    TestTrue(TEXT("snapshot commit defaults enabled"), Parsed.bCommitSnapshots);

    TSharedPtr<FJsonObject> PreviewRequest = MakeValidViewPairRequestForTest();
    PreviewRequest->SetBoolField(TEXT("commit_snapshot"), false);
    TestTrue(TEXT("exact preview flag parses"),
        SpPixelGoal::ParseViewPairCaptureRequest(
            PreviewRequest, Parsed).IsEmpty());
    TestFalse(TEXT("preview flag disables snapshot commit"),
        Parsed.bCommitSnapshots);

    TSharedPtr<FJsonObject> FractionalFov = MakeValidViewPairRequestForTest();
    FractionalFov->SetNumberField(TEXT("fov_degrees"), 90.5);
    TestTrue(TEXT("finite in-range fractional FOV remains valid"),
        SpPixelGoal::ParseViewPairCaptureRequest(
            FractionalFov, Parsed).IsEmpty());
    TestEqual(TEXT("fractional FOV is retained as camera float"),
        Parsed.HorizontalFovDegrees, 90.5f);

    struct FInvalidRequestCase
    {
        const TCHAR* Name;
        const TCHAR* Field;
        TSharedPtr<FJsonValue> Value;
        const TCHAR* ExpectedError;
    };
    const TSharedPtr<FJsonValue> Null = MakeShared<FJsonValueNull>();
    const TSharedPtr<FJsonValue> EmptyArray =
        MakeShared<FJsonValueArray>(TArray<TSharedPtr<FJsonValue>>());
    const TSharedPtr<FJsonValue> EmptyObject =
        MakeShared<FJsonValueObject>(MakeShared<FJsonObject>());
    const FInvalidRequestCase Cases[] = {
        {TEXT("numeric agent tag"), TEXT("agent_tag"),
            MakeShared<FJsonValueNumber>(17.0), TEXT("missing_agent_tag")},
        {TEXT("boolean agent tag"), TEXT("agent_tag"),
            MakeShared<FJsonValueBoolean>(true), TEXT("missing_agent_tag")},
        {TEXT("null agent tag"), TEXT("agent_tag"), Null,
            TEXT("missing_agent_tag")},
        {TEXT("array agent tag"), TEXT("agent_tag"), EmptyArray,
            TEXT("missing_agent_tag")},
        {TEXT("object agent tag"), TEXT("agent_tag"), EmptyObject,
            TEXT("missing_agent_tag")},
        {TEXT("empty agent tag"), TEXT("agent_tag"),
            MakeShared<FJsonValueString>(TEXT("")), TEXT("missing_agent_tag")},

        {TEXT("string width"), TEXT("width"),
            MakeShared<FJsonValueString>(TEXT("640")),
            TEXT("invalid_capture_dimensions")},
        {TEXT("boolean width"), TEXT("width"),
            MakeShared<FJsonValueBoolean>(true),
            TEXT("invalid_capture_dimensions")},
        {TEXT("null width"), TEXT("width"), Null,
            TEXT("invalid_capture_dimensions")},
        {TEXT("array width"), TEXT("width"), EmptyArray,
            TEXT("invalid_capture_dimensions")},
        {TEXT("object width"), TEXT("width"), EmptyObject,
            TEXT("invalid_capture_dimensions")},
        {TEXT("NaN width"), TEXT("width"),
            MakeShared<FJsonValueNumber>(
                std::numeric_limits<double>::quiet_NaN()),
            TEXT("invalid_capture_dimensions")},
        {TEXT("infinite width"), TEXT("width"),
            MakeShared<FJsonValueNumber>(
                std::numeric_limits<double>::infinity()),
            TEXT("invalid_capture_dimensions")},
        {TEXT("fractional width"), TEXT("width"),
            MakeShared<FJsonValueNumber>(639.99999),
            TEXT("invalid_capture_dimensions")},

        {TEXT("string height"), TEXT("height"),
            MakeShared<FJsonValueString>(TEXT("360")),
            TEXT("invalid_capture_dimensions")},
        {TEXT("boolean height"), TEXT("height"),
            MakeShared<FJsonValueBoolean>(true),
            TEXT("invalid_capture_dimensions")},
        {TEXT("null height"), TEXT("height"), Null,
            TEXT("invalid_capture_dimensions")},
        {TEXT("array height"), TEXT("height"), EmptyArray,
            TEXT("invalid_capture_dimensions")},
        {TEXT("object height"), TEXT("height"), EmptyObject,
            TEXT("invalid_capture_dimensions")},
        {TEXT("NaN height"), TEXT("height"),
            MakeShared<FJsonValueNumber>(
                std::numeric_limits<double>::quiet_NaN()),
            TEXT("invalid_capture_dimensions")},
        {TEXT("infinite height"), TEXT("height"),
            MakeShared<FJsonValueNumber>(
                std::numeric_limits<double>::infinity()),
            TEXT("invalid_capture_dimensions")},
        {TEXT("fractional height"), TEXT("height"),
            MakeShared<FJsonValueNumber>(359.99999),
            TEXT("invalid_capture_dimensions")},

        {TEXT("string FOV"), TEXT("fov_degrees"),
            MakeShared<FJsonValueString>(TEXT("90")),
            TEXT("invalid_camera_fov")},
        {TEXT("boolean FOV"), TEXT("fov_degrees"),
            MakeShared<FJsonValueBoolean>(true), TEXT("invalid_camera_fov")},
        {TEXT("null FOV"), TEXT("fov_degrees"), Null,
            TEXT("invalid_camera_fov")},
        {TEXT("array FOV"), TEXT("fov_degrees"), EmptyArray,
            TEXT("invalid_camera_fov")},
        {TEXT("object FOV"), TEXT("fov_degrees"), EmptyObject,
            TEXT("invalid_camera_fov")},
        {TEXT("NaN FOV"), TEXT("fov_degrees"),
            MakeShared<FJsonValueNumber>(
                std::numeric_limits<double>::quiet_NaN()),
            TEXT("invalid_camera_fov")},
        {TEXT("infinite FOV"), TEXT("fov_degrees"),
            MakeShared<FJsonValueNumber>(
                std::numeric_limits<double>::infinity()),
            TEXT("invalid_camera_fov")},

        {TEXT("string JPEG quality"), TEXT("jpeg_quality"),
            MakeShared<FJsonValueString>(TEXT("90")),
            TEXT("invalid_jpeg_quality")},
        {TEXT("boolean JPEG quality"), TEXT("jpeg_quality"),
            MakeShared<FJsonValueBoolean>(true), TEXT("invalid_jpeg_quality")},
        {TEXT("null JPEG quality"), TEXT("jpeg_quality"), Null,
            TEXT("invalid_jpeg_quality")},
        {TEXT("array JPEG quality"), TEXT("jpeg_quality"), EmptyArray,
            TEXT("invalid_jpeg_quality")},
        {TEXT("object JPEG quality"), TEXT("jpeg_quality"), EmptyObject,
            TEXT("invalid_jpeg_quality")},
        {TEXT("NaN JPEG quality"), TEXT("jpeg_quality"),
            MakeShared<FJsonValueNumber>(
                std::numeric_limits<double>::quiet_NaN()),
            TEXT("invalid_jpeg_quality")},
        {TEXT("infinite JPEG quality"), TEXT("jpeg_quality"),
            MakeShared<FJsonValueNumber>(
                std::numeric_limits<double>::infinity()),
            TEXT("invalid_jpeg_quality")},
        {TEXT("fractional JPEG quality"), TEXT("jpeg_quality"),
            MakeShared<FJsonValueNumber>(89.99999),
            TEXT("invalid_jpeg_quality")},

        {TEXT("numeric validate"), TEXT("validate"),
            MakeShared<FJsonValueNumber>(1.0), TEXT("invalid_validate_flag")},
        {TEXT("string validate"), TEXT("validate"),
            MakeShared<FJsonValueString>(TEXT("true")),
            TEXT("invalid_validate_flag")},
        {TEXT("null validate"), TEXT("validate"), Null,
            TEXT("invalid_validate_flag")},
        {TEXT("array validate"), TEXT("validate"), EmptyArray,
            TEXT("invalid_validate_flag")},
        {TEXT("object validate"), TEXT("validate"), EmptyObject,
            TEXT("invalid_validate_flag")},

        {TEXT("numeric snapshot commit"), TEXT("commit_snapshot"),
            MakeShared<FJsonValueNumber>(1.0),
            TEXT("invalid_commit_snapshot_flag")},
        {TEXT("string snapshot commit"), TEXT("commit_snapshot"),
            MakeShared<FJsonValueString>(TEXT("false")),
            TEXT("invalid_commit_snapshot_flag")},
        {TEXT("null snapshot commit"), TEXT("commit_snapshot"), Null,
            TEXT("invalid_commit_snapshot_flag")},
        {TEXT("array snapshot commit"), TEXT("commit_snapshot"), EmptyArray,
            TEXT("invalid_commit_snapshot_flag")},
        {TEXT("object snapshot commit"), TEXT("commit_snapshot"), EmptyObject,
            TEXT("invalid_commit_snapshot_flag")},
    };
    for (const FInvalidRequestCase& Case : Cases)
    {
        TSharedPtr<FJsonObject> Request = MakeValidViewPairRequestForTest();
        Request->SetField(Case.Field, Case.Value);
        TestEqual(FString::Printf(TEXT("%s is rejected"), Case.Name),
            SpPixelGoal::ParseViewPairCaptureRequest(Request, Parsed),
            FString(Case.ExpectedError));
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalViewPairCaptureTest,
    "SimWorld.PixelGoal.Snapshots.AtomicViewPair",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalViewPairCaptureTest::RunTest(const FString& Parameters)
{
    UWorld* World = UWorld::CreateWorld(EWorldType::Game, false);
    TestNotNull(TEXT("view-pair world exists"), World);
    if (!World || !GEngine)
    {
        if (World)
        {
            World->DestroyWorld(false);
        }
        return false;
    }
    FWorldContext& WorldContext = GEngine->CreateNewWorldContext(EWorldType::Game);
    WorldContext.SetCurrentWorld(World);

    ASpHumanoidAgent* Agent = World->SpawnActor<ASpHumanoidAgent>();
    USpPixelGoalSubsystem* PixelGoal = World->GetSubsystem<USpPixelGoalSubsystem>();
    TestNotNull(TEXT("view-pair agent spawns"), Agent);
    TestNotNull(TEXT("pixel-goal subsystem exists"), PixelGoal);
    if (!Agent || !PixelGoal)
    {
        World->DestroyWorld(false);
        GEngine->DestroyWorldContext(World);
        return false;
    }
    Agent->Agent_SetAgentTag(FName(TEXT("PixelGoalParisPocAgent")));
    Agent->SetActorLocationAndRotation(
        FVector(120.f, -340.f, 160.f), FRotator(0.f, 37.f, 0.f));
    Agent->SpawnDefaultController();
    AController* Controller = Agent->GetController();
    TestNotNull(TEXT("view-pair controller exists"), Controller);
    if (!Controller)
    {
        World->DestroyWorld(false);
        GEngine->DestroyWorldContext(World);
        return false;
    }
    Controller->SetControlRotation(FRotator(0.f, -23.f, 0.f));
    TestTrue(TEXT("front configures for view pair"),
        Agent->ConfigureObservationCamera(TEXT("front"), 640, 360, 90.f));
    TestTrue(TEXT("rear configures for view pair"),
        Agent->ConfigureObservationCamera(TEXT("rear"), 640, 360, 90.f));
    Agent->InitializeObservationCamera(TEXT("front"));
    Agent->InitializeObservationCamera(TEXT("rear"));
    TestTrue(TEXT("front receives the Paris rendering contract"),
        SpPixelGoal::ConfigurePersistentParisCapture(Agent->SceneCapture));
    TestTrue(TEXT("rear receives the Paris rendering contract"),
        SpPixelGoal::ConfigurePersistentParisCapture(
            Agent->RearSceneCapture.Get()));
    PixelGoal->ParisPocSetup.bEnableRearCamera = true;

    const FVector ActorLocationBefore = Agent->GetActorLocation();
    const FRotator ActorRotationBefore = Agent->GetActorRotation();
    const FRotator ControlRotationBefore = Controller->GetControlRotation();
    const FRotator SpringArmRotationBefore = Agent->SpringArm->GetRelativeRotation();
    const FTestViewPairCameraState FrontBeforeMismatch =
        FTestViewPairCameraState::Capture(Agent->SceneCapture);
    const FTestViewPairCameraState RearBeforeMismatch =
        FTestViewPairCameraState::Capture(Agent->RearSceneCapture);
    const uint64 SnapshotNumberBeforeMismatch = PixelGoal->NextSnapshotNumber;
    const uint64 GroupNumberBeforeMismatch = PixelGoal->NextCaptureGroupNumber;
    const int32 SnapshotCountBeforeMismatch = PixelGoal->CameraSnapshots.Num();
    const int32 SnapshotOrderBeforeMismatch = PixelGoal->SnapshotOrder.Num();
    struct FMalformedPairRpcCase
    {
        const TCHAR* Name;
        const TCHAR* Request;
        const TCHAR* ExpectedError;
    };
    const FMalformedPairRpcCase MalformedPairRequests[] = {
        {TEXT("numeric agent tag"), TEXT("{\"agent_tag\":17}"),
            TEXT("missing_agent_tag")},
        {TEXT("fractional width"),
            TEXT("{\"agent_tag\":\"PixelGoalParisPocAgent\",\"width\":639.6}"),
            TEXT("invalid_capture_dimensions")},
        {TEXT("string height"),
            TEXT("{\"agent_tag\":\"PixelGoalParisPocAgent\",\"height\":\"360\"}"),
            TEXT("invalid_capture_dimensions")},
        {TEXT("boolean FOV"),
            TEXT("{\"agent_tag\":\"PixelGoalParisPocAgent\",\"fov_degrees\":false}"),
            TEXT("invalid_camera_fov")},
        {TEXT("null JPEG quality"),
            TEXT("{\"agent_tag\":\"PixelGoalParisPocAgent\",\"jpeg_quality\":null}"),
            TEXT("invalid_jpeg_quality")},
        {TEXT("numeric validate"),
            TEXT("{\"agent_tag\":\"PixelGoalParisPocAgent\",\"validate\":1}"),
            TEXT("invalid_validate_flag")},
        {TEXT("numeric snapshot commit"),
            TEXT("{\"agent_tag\":\"PixelGoalParisPocAgent\","
                 "\"commit_snapshot\":1}"),
            TEXT("invalid_commit_snapshot_flag")},
    };
    for (const FMalformedPairRpcCase& Case : MalformedPairRequests)
    {
        const TSharedPtr<FJsonObject> MalformedPair = ParseTestJson(
            PixelGoal->PixelGoal_CaptureViewPairJson(Case.Request));
        TestEqual(FString::Printf(TEXT("%s pair request is rejected"), Case.Name),
            TestJsonString(MalformedPair, TEXT("error")),
            FString(Case.ExpectedError));
        TestEqual(FString::Printf(TEXT("%s mints no snapshot"), Case.Name),
            PixelGoal->NextSnapshotNumber, SnapshotNumberBeforeMismatch);
        TestEqual(FString::Printf(TEXT("%s mints no group"), Case.Name),
            PixelGoal->NextCaptureGroupNumber, GroupNumberBeforeMismatch);
        TestEqual(FString::Printf(TEXT("%s preserves snapshot map"), Case.Name),
            PixelGoal->CameraSnapshots.Num(), SnapshotCountBeforeMismatch);
        TestEqual(FString::Printf(TEXT("%s preserves snapshot order"), Case.Name),
            PixelGoal->SnapshotOrder.Num(), SnapshotOrderBeforeMismatch);
        TestTrue(FString::Printf(TEXT("%s preserves front rig"), Case.Name),
            FrontBeforeMismatch.Equals(
                FTestViewPairCameraState::Capture(Agent->SceneCapture)));
        TestTrue(FString::Printf(TEXT("%s preserves rear rig"), Case.Name),
            RearBeforeMismatch.Equals(
                FTestViewPairCameraState::Capture(Agent->RearSceneCapture)));
    }

    const TSharedPtr<FJsonObject> MismatchedPair = ParseTestJson(
        PixelGoal->PixelGoal_CaptureViewPairJson(TEXT(
            "{\"agent_tag\":\"PixelGoalParisPocAgent\",\"width\":320,"
            "\"height\":360,\"fov_degrees\":90,\"jpeg_quality\":50,"
            "\"validate\":false}")));
    TestEqual(TEXT("mismatched warmed dimensions fail closed"),
        TestJsonString(MismatchedPair, TEXT("error")),
        FString(TEXT("view_pair_camera_configuration_mismatch")));
    TestEqual(TEXT("mismatch mints no snapshot id"),
        PixelGoal->NextSnapshotNumber, SnapshotNumberBeforeMismatch);
    TestEqual(TEXT("mismatch mints no capture group"),
        PixelGoal->NextCaptureGroupNumber, GroupNumberBeforeMismatch);
    TestEqual(TEXT("mismatch retains snapshot map size"),
        PixelGoal->CameraSnapshots.Num(), SnapshotCountBeforeMismatch);
    TestEqual(TEXT("mismatch retains snapshot order size"),
        PixelGoal->SnapshotOrder.Num(), SnapshotOrderBeforeMismatch);
    TestTrue(TEXT("mismatch leaves front camera state byte-equivalent"),
        FrontBeforeMismatch.Equals(
            FTestViewPairCameraState::Capture(Agent->SceneCapture)));
    TestTrue(TEXT("mismatch leaves rear camera state byte-equivalent"),
        RearBeforeMismatch.Equals(
            FTestViewPairCameraState::Capture(Agent->RearSceneCapture)));

    Agent->RearSceneCapture->FOVAngle = 90.00005f;
    const FTestViewPairCameraState RearBeforeNearFovMismatch =
        FTestViewPairCameraState::Capture(Agent->RearSceneCapture);
    const uint64 GroupNumberBeforeNearFovMismatch =
        PixelGoal->NextCaptureGroupNumber;
    const TSharedPtr<FJsonObject> NearFovMismatchedPair = ParseTestJson(
        PixelGoal->PixelGoal_CaptureViewPairJson(TEXT(
            "{\"agent_tag\":\"PixelGoalParisPocAgent\",\"width\":640,"
            "\"height\":360,\"fov_degrees\":90,\"jpeg_quality\":50,"
            "\"validate\":false}")));
    TestEqual(TEXT("near but non-exact warmed FOV fails closed"),
        TestJsonString(NearFovMismatchedPair, TEXT("error")),
        FString(TEXT("view_pair_camera_configuration_mismatch")));
    TestEqual(TEXT("near FOV mismatch mints no capture group"),
        PixelGoal->NextCaptureGroupNumber,
        GroupNumberBeforeNearFovMismatch);
    TestTrue(TEXT("near FOV mismatch preserves exact rear camera state"),
        RearBeforeNearFovMismatch.Equals(
            FTestViewPairCameraState::Capture(Agent->RearSceneCapture)));
    Agent->RearSceneCapture->FOVAngle = 90.f;

    Agent->RearSceneCapture->Overscan = 0.1f;
    const FTestViewPairCameraState RearBeforeOverscanMismatch =
        FTestViewPairCameraState::Capture(Agent->RearSceneCapture);
    const uint64 GroupNumberBeforeOverscanMismatch =
        PixelGoal->NextCaptureGroupNumber;
    const TSharedPtr<FJsonObject> OverscanMismatchedPair = ParseTestJson(
        PixelGoal->PixelGoal_CaptureViewPairJson(TEXT(
            "{\"agent_tag\":\"PixelGoalParisPocAgent\",\"width\":640,"
            "\"height\":360,\"fov_degrees\":90,\"jpeg_quality\":50,"
            "\"validate\":false}")));
    TestEqual(TEXT("unmodeled overscan fails closed before capture"),
        TestJsonString(OverscanMismatchedPair, TEXT("error")),
        FString(TEXT("view_pair_camera_configuration_mismatch")));
    TestEqual(TEXT("overscan mismatch mints no capture group"),
        PixelGoal->NextCaptureGroupNumber,
        GroupNumberBeforeOverscanMismatch);
    TestTrue(TEXT("overscan mismatch preserves exact rear camera state"),
        RearBeforeOverscanMismatch.Equals(
            FTestViewPairCameraState::Capture(Agent->RearSceneCapture)));
    Agent->RearSceneCapture->Overscan = 0.f;

    const TMap<FString, FSpPixelGoalCameraSnapshot>
        CameraSnapshotsBeforePreview = PixelGoal->CameraSnapshots;
    const TArray<FString> SnapshotOrderBeforePreview = PixelGoal->SnapshotOrder;
    const FString PreviewText = PixelGoal->PixelGoal_CaptureViewPairJson(TEXT(
        "{\"agent_tag\":\"PixelGoalParisPocAgent\",\"width\":640,"
        "\"height\":360,\"fov_degrees\":90,\"jpeg_quality\":50,"
        "\"validate\":false,\"commit_snapshot\":false}"));
    const TSharedPtr<FJsonObject> Preview = ParseTestJson(PreviewText);
    bool bPreviewSuccess = false;
    bool bPreviewCommitted = true;
    TestTrue(TEXT("noncommitting preview capture succeeds"),
        Preview.IsValid() &&
            Preview->TryGetBoolField(TEXT("success"), bPreviewSuccess) &&
            bPreviewSuccess);
    TestTrue(TEXT("preview explicitly reports noncommitting snapshots"),
        Preview.IsValid() && Preview->TryGetBoolField(
            TEXT("snapshots_committed"), bPreviewCommitted) &&
            !bPreviewCommitted);
    const TArray<TSharedPtr<FJsonValue>>* PreviewViews = nullptr;
    TestTrue(TEXT("preview contains both RGB views"),
        Preview.IsValid() &&
            Preview->TryGetArrayField(TEXT("views"), PreviewViews) &&
            PreviewViews && PreviewViews->Num() == 2);
    if (PreviewViews && PreviewViews->Num() == 2)
    {
        for (int32 Index = 0; Index < 2; ++Index)
        {
            const FString PreviewSnapshotId = TestJsonString(
                (*PreviewViews)[Index]->AsObject(), TEXT("camera_snapshot_id"));
            TestTrue(FString::Printf(
                    TEXT("preview %d still names its captured RGB"), Index),
                !PreviewSnapshotId.IsEmpty());
            TestFalse(FString::Printf(
                    TEXT("preview %d cannot bind a later action"), Index),
                PixelGoal->CameraSnapshots.Contains(PreviewSnapshotId));
        }
    }
    TestEqual(TEXT("preview restores retained snapshot count"),
        PixelGoal->CameraSnapshots.Num(), CameraSnapshotsBeforePreview.Num());
    TestEqual(TEXT("preview restores retained snapshot order"),
        PixelGoal->SnapshotOrder.Num(), SnapshotOrderBeforePreview.Num());

    const FString PairText = PixelGoal->PixelGoal_CaptureViewPairJson(TEXT(
        "{\"agent_tag\":\"PixelGoalParisPocAgent\",\"width\":640,\"height\":360,"
        "\"fov_degrees\":90,\"jpeg_quality\":50,\"validate\":false}"));
    const TSharedPtr<FJsonObject> Pair = ParseTestJson(PairText);
    bool bPairSuccess = false;
    TestTrue(TEXT("pair response parses"), Pair.IsValid());
    TestTrue(TEXT("pair capture succeeds"),
        Pair.IsValid() && Pair->TryGetBoolField(TEXT("success"), bPairSuccess) &&
            bPairSuccess);
    bool bPairCommitted = false;
    TestTrue(TEXT("action pair explicitly reports committed snapshots"),
        Pair.IsValid() && Pair->TryGetBoolField(
            TEXT("snapshots_committed"), bPairCommitted) && bPairCommitted);
    const FString GroupId = TestJsonString(Pair, TEXT("capture_group_id"));
    TestTrue(TEXT("pair has minted group id"), GroupId.StartsWith(TEXT("view-pair-")));

    const TArray<TSharedPtr<FJsonValue>>* Views = nullptr;
    TestTrue(TEXT("pair contains views"),
        Pair.IsValid() && Pair->TryGetArrayField(TEXT("views"), Views));
    if (!Views || Views->Num() != 2)
    {
        AddError(FString::Printf(TEXT("view pair must contain exactly two views: %s"),
            *PairText));
        Agent->TerminateObservationCamera(TEXT("front"));
        Agent->TerminateObservationCamera(TEXT("rear"));
        World->DestroyWorld(false);
        GEngine->DestroyWorldContext(World);
        return false;
    }

    const TSharedPtr<FJsonObject> FrontImage = (*Views)[0]->AsObject();
    const TSharedPtr<FJsonObject> RearImage = (*Views)[1]->AsObject();
    TestEqual(TEXT("first pair view is front"),
        TestJsonString(FrontImage, TEXT("view_id")), FString(TEXT("front")));
    TestEqual(TEXT("second pair view is rear"),
        TestJsonString(RearImage, TEXT("view_id")), FString(TEXT("rear")));
    TestEqual(TEXT("front belongs to minted group"),
        TestJsonString(FrontImage, TEXT("capture_group_id")), GroupId);
    TestEqual(TEXT("rear belongs to minted group"),
        TestJsonString(RearImage, TEXT("capture_group_id")), GroupId);
    const FString FrontSnapshotId =
        TestJsonString(FrontImage, TEXT("camera_snapshot_id"));
    const FString RearSnapshotId =
        TestJsonString(RearImage, TEXT("camera_snapshot_id"));
    TestTrue(TEXT("front snapshot exists"), !FrontSnapshotId.IsEmpty());
    TestTrue(TEXT("rear snapshot exists"), !RearSnapshotId.IsEmpty());
    TestNotEqual(TEXT("pair snapshots are distinct"), FrontSnapshotId, RearSnapshotId);
    TestEqual(TEXT("pair intrinsics match"),
        TestJsonString(FrontImage, TEXT("camera_intrinsics_id")),
        TestJsonString(RearImage, TEXT("camera_intrinsics_id")));

    const FSpPixelGoalCameraSnapshot* FrontSnapshot =
        PixelGoal->CameraSnapshots.Find(FrontSnapshotId);
    const FSpPixelGoalCameraSnapshot* RearSnapshot =
        PixelGoal->CameraSnapshots.Find(RearSnapshotId);
    TestNotNull(TEXT("front immutable snapshot is retained"), FrontSnapshot);
    TestNotNull(TEXT("rear immutable snapshot is retained"), RearSnapshot);
    if (FrontSnapshot && RearSnapshot)
    {
        TestTrue(TEXT("front snapshot stores front component"),
            FrontSnapshot->CaptureComponent.Get() == Agent->SceneCapture);
        TestTrue(TEXT("rear snapshot stores rear component"),
            RearSnapshot->CaptureComponent.Get() == Agent->RearSceneCapture.Get());
        TestEqual(TEXT("front snapshot view identity"),
            FrontSnapshot->ViewId, FString(TEXT("front")));
        TestEqual(TEXT("rear snapshot view identity"),
            RearSnapshot->ViewId, FString(TEXT("rear")));
        TestEqual(TEXT("front snapshot group identity"),
            FrontSnapshot->CaptureGroupId, GroupId);
        TestEqual(TEXT("rear snapshot group identity"),
            RearSnapshot->CaptureGroupId, GroupId);
        TestTrue(TEXT("front snapshot stores actor location"),
            FrontSnapshot->AgentLocation.Equals(ActorLocationBefore, 0.f));
        TestTrue(TEXT("rear snapshot stores actor rotation"),
            RearSnapshot->AgentRotation.Equals(ActorRotationBefore, 0.0));
    }

    TestTrue(TEXT("pair capture preserves actor location"),
        Agent->GetActorLocation().Equals(ActorLocationBefore, 0.f));
    TestTrue(TEXT("pair capture preserves actor rotation"),
        Agent->GetActorRotation().Equals(ActorRotationBefore, 0.0));
    TestTrue(TEXT("pair capture preserves controller rotation"),
        Controller->GetControlRotation().Equals(ControlRotationBefore, 0.0));
    TestTrue(TEXT("pair capture preserves spring-arm rotation"),
        Agent->SpringArm->GetRelativeRotation().Equals(SpringArmRotationBefore, 0.0));

    SpPixelGoal::ResetGeometryTraceEntryCountForTest();
    const TSharedPtr<FJsonObject> ViewMismatch = ParseTestJson(
        PixelGoal->PixelGoal_ResolveAndMoveJson(FString::Printf(
            TEXT("{\"camera_snapshot_id\":\"%s\",\"view_id\":\"front\","
                 "\"capture_group_id\":\"%s\",\"u_norm\":0.5,\"v_norm\":0.75}"),
            *RearSnapshotId, *GroupId)));
    TestEqual(TEXT("rear snapshot refuses front binding"),
        TestJsonString(ViewMismatch, TEXT("rejection_reason")),
        FString(TEXT("camera_snapshot_view_mismatch")));
    TestEqual(TEXT("rejected audit echoes requested view"),
        TestJsonString(ViewMismatch, TEXT("view_id")), FString(TEXT("front")));
    TestEqual(TEXT("rejected audit echoes requested group"),
        TestJsonString(ViewMismatch, TEXT("capture_group_id")), GroupId);
    TestEqual(TEXT("view mismatch enters no geometry trace"),
        SpPixelGoal::GeometryTraceEntryCountForTest(), 0);

    const TSharedPtr<FJsonObject> GroupMismatch = ParseTestJson(
        PixelGoal->PixelGoal_ResolveAndMoveJson(FString::Printf(
            TEXT("{\"camera_snapshot_id\":\"%s\",\"view_id\":\"rear\","
                 "\"capture_group_id\":\"wrong-group\",\"u_norm\":0.5,"
                 "\"v_norm\":0.75}"), *RearSnapshotId)));
    TestEqual(TEXT("rear snapshot refuses wrong group"),
        TestJsonString(GroupMismatch, TEXT("rejection_reason")),
        FString(TEXT("camera_snapshot_group_mismatch")));
    TestEqual(TEXT("group mismatch enters no geometry trace"),
        SpPixelGoal::GeometryTraceEntryCountForTest(), 0);

    const TSharedPtr<FJsonObject> PartialBinding = ParseTestJson(
        PixelGoal->PixelGoal_ResolveAndMoveJson(FString::Printf(
            TEXT("{\"camera_snapshot_id\":\"%s\",\"view_id\":\"rear\","
                 "\"u_norm\":0.5,\"v_norm\":0.75}"), *RearSnapshotId)));
    TestEqual(TEXT("partial binding fails closed"),
        TestJsonString(PartialBinding, TEXT("rejection_reason")),
        FString(TEXT("camera_snapshot_binding_incomplete")));
    TestEqual(TEXT("partial binding enters no geometry trace"),
        SpPixelGoal::GeometryTraceEntryCountForTest(), 0);

    struct FMalformedBindingCase
    {
        const TCHAR* Name;
        const TCHAR* Fields;
    };
    const FMalformedBindingCase MalformedBindings[] = {
        {TEXT("numeric view"),
            TEXT("\"view_id\":17,\"capture_group_id\":\"view-pair-1\"")},
        {TEXT("boolean view"),
            TEXT("\"view_id\":false,\"capture_group_id\":\"view-pair-1\"")},
        {TEXT("null view"),
            TEXT("\"view_id\":null,\"capture_group_id\":\"view-pair-1\"")},
        {TEXT("array view"),
            TEXT("\"view_id\":[],\"capture_group_id\":\"view-pair-1\"")},
        {TEXT("object view"),
            TEXT("\"view_id\":{},\"capture_group_id\":\"view-pair-1\"")},
        {TEXT("numeric group"),
            TEXT("\"view_id\":\"rear\",\"capture_group_id\":17")},
        {TEXT("boolean group"),
            TEXT("\"view_id\":\"rear\",\"capture_group_id\":false")},
        {TEXT("null group"),
            TEXT("\"view_id\":\"rear\",\"capture_group_id\":null")},
        {TEXT("array group"),
            TEXT("\"view_id\":\"rear\",\"capture_group_id\":[]")},
        {TEXT("object group"),
            TEXT("\"view_id\":\"rear\",\"capture_group_id\":{}")},
        {TEXT("numeric fields"), TEXT("\"view_id\":17,\"capture_group_id\":18")},
        {TEXT("boolean fields"),
            TEXT("\"view_id\":true,\"capture_group_id\":false")},
        {TEXT("null fields"), TEXT("\"view_id\":null,\"capture_group_id\":null")},
        {TEXT("array fields"), TEXT("\"view_id\":[],\"capture_group_id\":[]")},
        {TEXT("object fields"), TEXT("\"view_id\":{},\"capture_group_id\":{}")},
        {TEXT("empty fields"), TEXT("\"view_id\":\"\",\"capture_group_id\":\"\"")},
    };
    for (const FMalformedBindingCase& Case : MalformedBindings)
    {
        const TSharedPtr<FJsonObject> MalformedBinding = ParseTestJson(
            PixelGoal->PixelGoal_ResolveAndMoveJson(FString::Printf(
                TEXT("{\"camera_snapshot_id\":\"%s\",%s,\"u_norm\":0.5,"
                     "\"v_norm\":0.75}"),
                *RearSnapshotId,
                Case.Fields)));
        TestEqual(FString::Printf(TEXT("%s binding fails closed"), Case.Name),
            TestJsonString(MalformedBinding, TEXT("rejection_reason")),
            FString(TEXT("camera_snapshot_binding_invalid")));
        TestTrue(FString::Printf(TEXT("%s binding echoes view"), Case.Name),
            MalformedBinding->HasField(TEXT("view_id")));
        TestTrue(FString::Printf(TEXT("%s binding echoes group"), Case.Name),
            MalformedBinding->HasField(TEXT("capture_group_id")));
        TestEqual(FString::Printf(TEXT("%s binding skips trace"), Case.Name),
            SpPixelGoal::GeometryTraceEntryCountForTest(), 0);
    }

    const TSharedPtr<FJsonObject> ExactRear = ParseTestJson(
        PixelGoal->PixelGoal_ResolveAndMoveJson(FString::Printf(
            TEXT("{\"camera_snapshot_id\":\"%s\",\"view_id\":\"rear\","
                 "\"capture_group_id\":\"%s\",\"u_norm\":0.5,\"v_norm\":0.75}"),
            *RearSnapshotId, *GroupId)));
    TestEqual(TEXT("exact rear binding reaches geometry"),
        TestJsonString(ExactRear, TEXT("rejection_reason")),
        FString(TEXT("no_geometry_hit")));
    TestEqual(TEXT("exact rear binding enters one geometry trace"),
        SpPixelGoal::GeometryTraceEntryCountForTest(), 1);

    const FRotator FrontRelativeRotation = Agent->SceneCapture->GetRelativeRotation();
    Agent->SceneCapture->SetRelativeRotation(
        FrontRelativeRotation + FRotator(0.f, 5.f, 0.f));
    const TSharedPtr<FJsonObject> RearAfterFrontChange = ParseTestJson(
        PixelGoal->PixelGoal_ResolveAndMoveJson(FString::Printf(
            TEXT("{\"camera_snapshot_id\":\"%s\",\"view_id\":\"rear\","
                 "\"capture_group_id\":\"%s\",\"u_norm\":0.5,\"v_norm\":0.75}"),
            *RearSnapshotId, *GroupId)));
    TestEqual(TEXT("rear staleness does not consult front component"),
        TestJsonString(RearAfterFrontChange, TEXT("rejection_reason")),
        FString(TEXT("no_geometry_hit")));
    TestEqual(TEXT("rear remains traceable after front-only change"),
        SpPixelGoal::GeometryTraceEntryCountForTest(), 2);
    Agent->SceneCapture->SetRelativeRotation(FrontRelativeRotation);

    const FRotator RearRelativeRotation =
        Agent->RearSceneCapture->GetRelativeRotation();
    Agent->RearSceneCapture->SetRelativeRotation(
        RearRelativeRotation + FRotator(0.f, 5.f, 0.f));
    const TSharedPtr<FJsonObject> RearAfterRearChange = ParseTestJson(
        PixelGoal->PixelGoal_ResolveAndMoveJson(FString::Printf(
            TEXT("{\"camera_snapshot_id\":\"%s\",\"view_id\":\"rear\","
                 "\"capture_group_id\":\"%s\",\"u_norm\":0.5,\"v_norm\":0.75}"),
            *RearSnapshotId, *GroupId)));
    TestEqual(TEXT("rear snapshot is stale after rear-camera yaw changes"),
        TestJsonString(RearAfterRearChange, TEXT("rejection_reason")),
        FString(TEXT("camera_snapshot_stale")));
    TestEqual(TEXT("rear-camera staleness rejects before geometry trace"),
        SpPixelGoal::GeometryTraceEntryCountForTest(), 2);
    Agent->RearSceneCapture->SetRelativeRotation(RearRelativeRotation);

    TSharedPtr<FJsonObject> LegacyImage = MakeShared<FJsonObject>();
    PixelGoal->RecordCameraSnapshot(
        Agent->SceneCapture, Agent, 32, 24, LegacyImage);
    const FString LegacySnapshotId =
        TestJsonString(LegacyImage, TEXT("camera_snapshot_id"));
    TestEqual(TEXT("legacy snapshot image is identified as front"),
        TestJsonString(LegacyImage, TEXT("view_id")), FString(TEXT("front")));
    TestEqual(TEXT("legacy snapshot image has an empty group"),
        TestJsonString(LegacyImage, TEXT("capture_group_id")), FString());
    const TSharedPtr<FJsonObject> LegacyResolve = ParseTestJson(
        PixelGoal->PixelGoal_ResolveAndMoveJson(FString::Printf(
            TEXT("{\"camera_snapshot_id\":\"%s\",\"u_norm\":0.5,"
                 "\"v_norm\":0.75}"), *LegacySnapshotId)));
    TestEqual(TEXT("legacy unbound snapshot follows existing geometry path"),
        TestJsonString(LegacyResolve, TEXT("rejection_reason")),
        FString(TEXT("no_geometry_hit")));
    TestFalse(TEXT("legacy rejected audit does not add view id"),
        LegacyResolve->HasField(TEXT("view_id")));
    TestFalse(TEXT("legacy rejected audit does not add capture group id"),
        LegacyResolve->HasField(TEXT("capture_group_id")));
    TestEqual(TEXT("legacy resolution enters geometry trace"),
        SpPixelGoal::GeometryTraceEntryCountForTest(), 3);

    Agent->SetActorLocation(ActorLocationBefore + FVector(5.f, 0.f, 0.f));
    const FString SnapshotIds[] = {FrontSnapshotId, RearSnapshotId};
    const TCHAR* ViewIds[] = {TEXT("front"), TEXT("rear")};
    for (int32 Index = 0; Index < 2; ++Index)
    {
        const TSharedPtr<FJsonObject> Stale = ParseTestJson(
            PixelGoal->PixelGoal_ResolveAndMoveJson(FString::Printf(
                TEXT("{\"camera_snapshot_id\":\"%s\",\"view_id\":\"%s\","
                     "\"capture_group_id\":\"%s\",\"u_norm\":0.5,"
                     "\"v_norm\":0.75}"),
                *SnapshotIds[Index], ViewIds[Index], *GroupId)));
        TestEqual(FString::Printf(TEXT("%s snapshot is stale after actor movement"),
                ViewIds[Index]),
            TestJsonString(Stale, TEXT("rejection_reason")),
            FString(TEXT("camera_snapshot_stale")));
    }
    TestEqual(TEXT("stale snapshots enter no geometry trace"),
        SpPixelGoal::GeometryTraceEntryCountForTest(), 3);

    Agent->SetActorLocation(ActorLocationBefore);
    Agent->SetActorRotation(ActorRotationBefore + FRotator(0.f, 5.f, 0.f));
    for (int32 Index = 0; Index < 2; ++Index)
    {
        const TSharedPtr<FJsonObject> Stale = ParseTestJson(
            PixelGoal->PixelGoal_ResolveAndMoveJson(FString::Printf(
                TEXT("{\"camera_snapshot_id\":\"%s\",\"view_id\":\"%s\","
                     "\"capture_group_id\":\"%s\",\"u_norm\":0.5,"
                     "\"v_norm\":0.75}"),
                *SnapshotIds[Index], ViewIds[Index], *GroupId)));
        TestEqual(FString::Printf(TEXT("%s snapshot is stale after actor yaw"),
                ViewIds[Index]),
            TestJsonString(Stale, TEXT("rejection_reason")),
            FString(TEXT("camera_snapshot_stale")));
    }
    TestEqual(TEXT("yaw-stale snapshots enter no geometry trace"),
        SpPixelGoal::GeometryTraceEntryCountForTest(), 3);
    Agent->SetActorRotation(ActorRotationBefore);
    // The actor yaw round-trip can preserve a signed-zero quaternion-derived
    // pitch/roll. Restore the test rig's exact canonical relative rotations
    // before exercising the strict warmed-pair preflight and failure hook.
    Agent->SceneCapture->SetRelativeRotationExact(FRotator::ZeroRotator);
    Agent->RearSceneCapture->SetRelativeRotationExact(
        FRotator(0.f, 180.f, 0.f));

    const USpSceneCaptureComponent2D* FrontBeforeRollback =
        Agent->SceneCapture;
    const USpSceneCaptureComponent2D* RearBeforeRollback =
        Agent->RearSceneCapture;
    TestTrue(TEXT("rollback fixture restores the front camera state"),
        FrontBeforeMismatch.Equals(FTestViewPairCameraState::Capture(
            Agent->SceneCapture)));
    TestTrue(TEXT("rollback fixture restores the rear camera state"),
        RearBeforeMismatch.Equals(FTestViewPairCameraState::Capture(
            Agent->RearSceneCapture)));
    TestTrue(TEXT("rollback fixture retains front/rear common optical centre"),
        FrontBeforeRollback->GetComponentLocation().Equals(
            RearBeforeRollback->GetComponentLocation(), 0.f));
    TestTrue(TEXT("rollback fixture retains the warmed pair predicate"),
        SpPixelGoal::WarmedViewPairCameraError(Agent, 640, 360, 90.f).IsEmpty());
    while (PixelGoal->SnapshotOrder.Num() < 256)
    {
        TSharedPtr<FJsonObject> CapacityImage = MakeShared<FJsonObject>();
        PixelGoal->RecordCameraSnapshot(
            Agent->SceneCapture, Agent, 32, 24, CapacityImage);
    }
    const TArray<FString> SnapshotOrderBeforeFailedPair = PixelGoal->SnapshotOrder;
    const TMap<FString, FSpPixelGoalCameraSnapshot>
        CameraSnapshotsBeforeFailedPair = PixelGoal->CameraSnapshots;
    const int32 SnapshotCountBeforeFailedPair = PixelGoal->CameraSnapshots.Num();
    PixelGoal->ViewPairFailureAfterBatchForTest =
        TEXT("forced_view_pair_failure");
    const TSharedPtr<FJsonObject> FailedPair = ParseTestJson(
        PixelGoal->PixelGoal_CaptureViewPairJson(TEXT(
            "{\"agent_tag\":\"PixelGoalParisPocAgent\",\"width\":640,"
            "\"height\":360,\"fov_degrees\":90,\"jpeg_quality\":50,"
            "\"validate\":false}")));
    bool bFailedPairSuccess = true;
    TestTrue(TEXT("invalid pair reports failure"),
        FailedPair.IsValid() &&
            FailedPair->TryGetBoolField(TEXT("success"), bFailedPairSuccess) &&
            !bFailedPairSuccess);
    TestEqual(TEXT("injected post-batch pair failure is explicit"),
        TestJsonString(FailedPair, TEXT("error")),
        FString(TEXT("forced_view_pair_failure")));
    TestEqual(TEXT("failed pair restores snapshot count after eviction"),
        PixelGoal->CameraSnapshots.Num(), SnapshotCountBeforeFailedPair);
    TestEqual(TEXT("failed pair restores snapshot order length"),
        PixelGoal->SnapshotOrder.Num(), SnapshotOrderBeforeFailedPair.Num());
    for (int32 Index = 0; Index < SnapshotOrderBeforeFailedPair.Num(); ++Index)
    {
        TestEqual(FString::Printf(TEXT("failed pair restores snapshot order %d"), Index),
            PixelGoal->SnapshotOrder[Index], SnapshotOrderBeforeFailedPair[Index]);
        TestTrue(FString::Printf(TEXT("failed pair restores snapshot %d"), Index),
            PixelGoal->CameraSnapshots.Contains(SnapshotOrderBeforeFailedPair[Index]));
        const FSpPixelGoalCameraSnapshot* BeforeSnapshot =
            CameraSnapshotsBeforeFailedPair.Find(SnapshotOrderBeforeFailedPair[Index]);
        const FSpPixelGoalCameraSnapshot* AfterSnapshot =
            PixelGoal->CameraSnapshots.Find(SnapshotOrderBeforeFailedPair[Index]);
        TestTrue(FString::Printf(
                TEXT("failed pair restores exact snapshot value %d"), Index),
            BeforeSnapshot && AfterSnapshot &&
                TestSnapshotsEqual(*BeforeSnapshot, *AfterSnapshot));
    }

    Agent->TerminateObservationCamera(TEXT("front"));
    Agent->TerminateObservationCamera(TEXT("rear"));
    World->DestroyWorld(false);
    GEngine->DestroyWorldContext(World);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalHorizontalDirectionTest,
    "SimWorld.PixelGoal.Deprojection.HorizontalDirection",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalHorizontalDirectionTest::RunTest(const FString& Parameters)
{
    const FSpPixelGoalCameraSnapshot Snapshot =
        FSpPixelGoalCameraSnapshot::PerspectiveForTest(
            FVector::ZeroVector, FRotator::ZeroRotator, 640, 360, 90.f);
    FVector Origin;
    FVector Left;
    FVector Right;
    TestTrue(TEXT("left deprojects"), Snapshot.Deproject(FVector2D(0.25, 0.5), Origin, Left));
    TestTrue(TEXT("right deprojects"), Snapshot.Deproject(FVector2D(0.75, 0.5), Origin, Right));
    TestTrue(TEXT("left pixel points to negative UE Y"), Left.Y < 0.f);
    TestTrue(TEXT("right pixel points to positive UE Y"), Right.Y > 0.f);
    TestTrue(TEXT("directions are symmetric"), FMath::IsNearlyEqual(Left.Y, -Right.Y, 1.e-4f));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalResolutionIndependenceTest,
    "SimWorld.PixelGoal.Deprojection.ResolutionIndependence",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalResolutionIndependenceTest::RunTest(const FString& Parameters)
{
    const FSpPixelGoalCameraSnapshot Small =
        FSpPixelGoalCameraSnapshot::PerspectiveForTest(
            FVector::ZeroVector, FRotator::ZeroRotator, 320, 180, 90.f);
    const FSpPixelGoalCameraSnapshot Large =
        FSpPixelGoalCameraSnapshot::PerspectiveForTest(
            FVector::ZeroVector, FRotator::ZeroRotator, 1280, 720, 90.f);
    FVector SmallOrigin;
    FVector SmallDirection;
    FVector LargeOrigin;
    FVector LargeDirection;
    TestTrue(TEXT("small deprojects"), Small.Deproject(FVector2D(0.31, 0.77), SmallOrigin, SmallDirection));
    TestTrue(TEXT("large deprojects"), Large.Deproject(FVector2D(0.31, 0.77), LargeOrigin, LargeDirection));
    TestTrue(
        TEXT("normalised point produces the same ray"),
        SmallDirection.Equals(LargeDirection, 1.e-4f));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalFlatGroundIntersectionTest,
    "SimWorld.PixelGoal.Deprojection.FlatGroundIntersection",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalFlatGroundIntersectionTest::RunTest(const FString& Parameters)
{
    const FSpPixelGoalCameraSnapshot Snapshot =
        FSpPixelGoalCameraSnapshot::PerspectiveForTest(
            FVector(0.f, 0.f, 160.f), FRotator::ZeroRotator, 640, 360, 90.f);
    FVector Origin;
    FVector Direction;
    TestTrue(
        TEXT("lower-image point deprojects"),
        Snapshot.Deproject(FVector2D(0.5, 0.75), Origin, Direction));
    TestTrue(TEXT("lower-image ray points down"), Direction.Z < 0.f);
    const float Scale = -Origin.Z / Direction.Z;
    const FVector GroundHit = Origin + Direction * Scale;
    TestTrue(TEXT("ground Z is zero"), FMath::IsNearlyZero(GroundHit.Z, 1.e-3f));
    TestTrue(
        TEXT("90-degree 16:9 analytic forward distance is 568.89 cm"),
        FMath::IsNearlyEqual(GroundHit.X, 568.8889f, 0.1f));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalFirstHitClassificationTest,
    "SimWorld.PixelGoal.Geometry.FirstHitClassification",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalFirstHitClassificationTest::RunTest(const FString& Parameters)
{
    TestEqual(
        TEXT("flat surface is walkable ground"),
        SpPixelGoal::ClassifyFirstHit(FVector::UpVector, 30.f),
        ESpPixelGoalHitClass::WalkableGround);
    TestEqual(
        TEXT("vertical first hit is not repaired through to ground"),
        SpPixelGoal::ClassifyFirstHit(FVector::BackwardVector, 30.f),
        ESpPixelGoalHitClass::NotWalkableGround);
    TestEqual(
        TEXT("slope beyond configured limit is rejected"),
        SpPixelGoal::ClassifyFirstHit(FVector(0.6f, 0.f, 0.8f), 30.f),
        ESpPixelGoalHitClass::NotWalkableGround);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalStrictNavAdjustmentTest,
    "SimWorld.PixelGoal.Navigation.StrictAdjustment",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalStrictNavAdjustmentTest::RunTest(const FString& Parameters)
{
    const FVector RawHit(100.f, 200.f, 10.f);
    TestTrue(
        TEXT("configured ten-centimetre boundary is accepted"),
        SpPixelGoal::WithinNavAdjustment(RawHit, RawHit + FVector(6.f, 8.f, 0.f), 10.f));
    TestFalse(
        TEXT("target just beyond the configured boundary is rejected"),
        SpPixelGoal::WithinNavAdjustment(RawHit, RawHit + FVector(10.01f, 0.f, 0.f), 10.f));
    TestTrue(
        TEXT("adjustment threshold remains configurable"),
        SpPixelGoal::WithinNavAdjustment(RawHit, RawHit + FVector(12.f, 0.f, 0.f), 15.f));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalExecutionErrorTest,
    "SimWorld.PixelGoal.Navigation.PlanarExecutionError",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalExecutionErrorTest::RunTest(const FString& Parameters)
{
    TestEqual(
        TEXT("planar error ignores feet height"),
        SpPixelGoal::PlanarErrorCm(FVector(0.f, 0.f, 100.f), FVector(3.f, 4.f, 0.f)),
        5.f);
    TestTrue(
        TEXT("fifteen-centimetre acceptance radius is a caller-level comparison"),
        SpPixelGoal::PlanarErrorCm(FVector::ZeroVector, FVector(9.f, 12.f, 500.f)) <= 15.f);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalControllerPathAuditTest,
    "SimWorld.PixelGoal.Navigation.ControllerPathAudit",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalControllerPathAuditTest::RunTest(const FString& Parameters)
{
    const TArray<FVector> BentPath = {
        FVector(0.f, 0.f, 0.f),
        FVector(30.f, 40.f, 20.f),
        FVector(30.f, 140.f, -10.f),
    };
    TestEqual(
        TEXT("controller path audit uses the planar walking metric"),
        SpPixelGoal::ControllerPathLengthCm(BentPath),
        150.f);
    TestFalse(
        TEXT("ordinary visible-ground curvature remains accepted"),
        SpPixelGoal::ControllerPathDetourExceeded(630.f, 500.f, 1.35f, 100.f));
    TestTrue(
        TEXT("the observed five-to-eight-metre hidden detour is rejected"),
        SpPixelGoal::ControllerPathDetourExceeded(780.f, 504.f, 1.35f, 100.f));
    TestFalse(
        TEXT("allowance protects short local targets from ratio noise"),
        SpPixelGoal::ControllerPathDetourExceeded(190.f, 100.f, 1.35f, 100.f));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalTerminalAuditFreezeTest,
    "SimWorld.PixelGoal.Navigation.TerminalAuditFreeze",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalTerminalAuditFreezeTest::RunTest(const FString& Parameters)
{
    TestTrue(
        TEXT("moving audit continues sampling"),
        SpPixelGoal::ShouldSampleMoveAudit(TEXT("moving")));
    TestFalse(
        TEXT("completed audit remains frozen"),
        SpPixelGoal::ShouldSampleMoveAudit(TEXT("completed")));
    TestFalse(
        TEXT("failed audit remains frozen"),
        SpPixelGoal::ShouldSampleMoveAudit(TEXT("failed")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalParisPocSetupValidationTest,
    "SimWorld.PixelGoal.ParisPoc.SetupValidation",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalParisPocSetupValidationTest::RunTest(const FString& Parameters)
{
    const FString ValidRequest = TEXT(
        "{\"scene\":\"/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints\","
        "\"region_name\":\"rue_de_rivoli_sidewalk\","
        "\"agent_spawn_cm\":[-26805.82,9437.53,90.0],"
        "\"agent_yaw_deg\":165.0056,"
        "\"nav_bounds_center_cm\":[-27385.38,9592.82,100.0],"
        "\"nav_bounds_extent_cm\":[1000.0,450.0,300.0]}" );
    FSpPixelGoalParisPocSetup Parsed;
    FString Error;
    TestTrue(
        TEXT("valid local Paris request parses"),
        SpPixelGoal::ParseParisPocSetupRequest(
            ValidRequest,
            TEXT("UEDPIE_0_ParisCity_FinalBlueprints"),
            Parsed,
            Error));
    TestEqual(TEXT("region is preserved"), Parsed.RegionName, FString(TEXT("rue_de_rivoli_sidewalk")));
    TestTrue(
        TEXT("bounds are local"),
        Parsed.NavBoundsExtent.Equals(FVector(1000.f, 450.f, 300.f), 0.01f));
    TestFalse(TEXT("omitted rear-camera opt-in remains legacy front-only"),
        Parsed.bEnableRearCamera);

    const FString DualViewRequest =
        ValidRequest.LeftChop(1) + TEXT(",\"enable_rear_camera\":true}");
    TestTrue(
        TEXT("internal rear-camera opt-in parses"),
        SpPixelGoal::ParseParisPocSetupRequest(
            DualViewRequest,
            TEXT("ParisCity_FinalBlueprints"),
            Parsed,
            Error));
    TestTrue(TEXT("rear-camera opt-in is retained"), Parsed.bEnableRearCamera);

    const TCHAR* MalformedRearValues[] = {
        TEXT("\"true\""),
        TEXT("1"),
        TEXT("null"),
        TEXT("[]"),
        TEXT("{}"),
    };
    for (int32 Index = 0; Index < UE_ARRAY_COUNT(MalformedRearValues); ++Index)
    {
        const FString MalformedRearRequest = ValidRequest.LeftChop(1) +
            FString::Printf(
                TEXT(",\"enable_rear_camera\":%s}"),
                MalformedRearValues[Index]);
        TestFalse(
            FString::Printf(
                TEXT("non-boolean rear-camera opt-in %d is rejected"), Index),
            SpPixelGoal::ParseParisPocSetupRequest(
                MalformedRearRequest,
                TEXT("ParisCity_FinalBlueprints"),
                Parsed,
                Error));
        TestEqual(FString::Printf(
                TEXT("malformed rear opt-in %d rejection is explicit"), Index),
            Error, FString(TEXT("invalid_enable_rear_camera")));
        TestFalse(FString::Printf(
                TEXT("malformed rear opt-in %d leaves setup front-only"), Index),
            Parsed.bEnableRearCamera);
    }

    TestFalse(
        TEXT("map-name substring is not accepted as Paris"),
        SpPixelGoal::ParseParisPocSetupRequest(
            ValidRequest,
            TEXT("ParisCity_FinalBlueprints_Copy"),
            Parsed,
            Error));
    TestEqual(TEXT("near-match rejection is explicit"), Error, FString(TEXT("paris_map_not_loaded")));

    TestFalse(
        TEXT("non-Paris live world is rejected"),
        SpPixelGoal::ParseParisPocSetupRequest(
            ValidRequest,
            TEXT("UEDPIE_0_EmptyLevel"),
            Parsed,
            Error));
    TestEqual(TEXT("map mismatch is explicit"), Error, FString(TEXT("paris_map_not_loaded")));

    const FString CityWideRequest = ValidRequest.Replace(
        TEXT("[1000.0,450.0,300.0]"),
        TEXT("[40000.0,40000.0,1000.0]"));
    TestFalse(
        TEXT("city-wide bounds are rejected"),
        SpPixelGoal::ParseParisPocSetupRequest(
            CityWideRequest,
            TEXT("ParisCity_FinalBlueprints"),
            Parsed,
            Error));
    TestEqual(TEXT("bounds rejection is explicit"), Error, FString(TEXT("nav_bounds_not_local")));

    const FString FloorRequest = ValidRequest.LeftChop(1) + TEXT(",\"spawn_floor\":true}");
    TestFalse(
        TEXT("synthetic replacement ground is rejected"),
        SpPixelGoal::ParseParisPocSetupRequest(
            FloorRequest,
            TEXT("ParisCity_FinalBlueprints"),
            Parsed,
            Error));
    TestEqual(TEXT("floor rejection is explicit"), Error, FString(TEXT("synthetic_floor_forbidden")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalParisCaptureModeTest,
    "SimWorld.PixelGoal.ParisPoc.PersistentCapture.ModeSelection",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalParisCaptureModeTest::RunTest(const FString& Parameters)
{
    TestEqual(
        TEXT("Paris PoC uses its dedicated native capture"),
        SpPixelGoal::CaptureModeForAgentTag(TEXT("PixelGoalParisPocAgent")),
        FString(TEXT("agent_native")));
    TestEqual(
        TEXT("calibration preserves the existing shared pool"),
        SpPixelGoal::CaptureModeForAgentTag(TEXT("PixelGoalCalibrationAgent")),
        FString(TEXT("shared_pool")));
    TestEqual(
        TEXT("near-match does not opt into the Paris-only path"),
        SpPixelGoal::CaptureModeForAgentTag(TEXT("PixelGoalParisPocAgent_Copy")),
        FString(TEXT("shared_pool")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalParisPersistentCaptureConfigurationTest,
    "SimWorld.PixelGoal.ParisPoc.PersistentCapture.Configuration",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalParisPersistentCaptureConfigurationTest::RunTest(
    const FString& Parameters)
{
    USpSceneCaptureComponent2D* Capture =
        NewObject<USpSceneCaptureComponent2D>();
    TestNotNull(TEXT("test capture exists"), Capture);
    if (!Capture)
    {
        return false;
    }
    Capture->bAlwaysPersistRenderingState = false;
    Capture->bCaptureEveryFrame = false;
    Capture->bCaptureOnMovement = true;
    Capture->PostProcessBlendWeight = 0.f;
    Capture->PostProcessSettings.bOverride_AutoExposureMethod = false;
    Capture->PostProcessSettings.bOverride_AutoExposureMinBrightness = false;
    Capture->PostProcessSettings.bOverride_AutoExposureMaxBrightness = false;
    Capture->PostProcessSettings.bOverride_AutoExposureSpeedUp = false;
    Capture->PostProcessSettings.bOverride_AutoExposureSpeedDown = false;
    Capture->PostProcessSettings.bOverride_AutoExposureBias = false;
    Capture->PostProcessSettings.AutoExposureBias = 0.f;

    TestTrue(
        TEXT("valid dedicated capture configures successfully"),
        SpPixelGoal::ConfigurePersistentParisCapture(Capture));
    TestTrue(
        TEXT("render state persists across snapshots"),
        Capture->bAlwaysPersistRenderingState);
    TestTrue(
        TEXT("capture renders continuously during warm-up"),
        Capture->bCaptureEveryFrame);
    TestFalse(
        TEXT("movement does not create a second capture policy"),
        Capture->bCaptureOnMovement);
    TestTrue(TEXT("camera settings fully override the scene exposure volume"),
        FMath::IsNearlyEqual(Capture->PostProcessBlendWeight, 1.f, 1.e-6f));
    TestTrue(TEXT("Paris capture uses adaptive histogram exposure"),
        Capture->PostProcessSettings.bOverride_AutoExposureMethod &&
            Capture->PostProcessSettings.AutoExposureMethod == AEM_Histogram);
    TestTrue(TEXT("Paris capture can adapt into storefront shadow"),
        Capture->PostProcessSettings.bOverride_AutoExposureMinBrightness &&
            FMath::IsNearlyEqual(
                Capture->PostProcessSettings.AutoExposureMinBrightness,
                6.f, 1.e-6f));
    TestTrue(TEXT("Paris capture retains daylight headroom"),
        Capture->PostProcessSettings.bOverride_AutoExposureMaxBrightness &&
            FMath::IsNearlyEqual(
                Capture->PostProcessSettings.AutoExposureMaxBrightness,
                12.f, 1.e-6f));
    TestTrue(TEXT("Paris capture converges quickly when walking into light"),
        Capture->PostProcessSettings.bOverride_AutoExposureSpeedUp &&
            FMath::IsNearlyEqual(
                Capture->PostProcessSettings.AutoExposureSpeedUp,
                10.f, 1.e-6f));
    TestTrue(TEXT("Paris capture converges quickly when walking into shadow"),
        Capture->PostProcessSettings.bOverride_AutoExposureSpeedDown &&
            FMath::IsNearlyEqual(
                Capture->PostProcessSettings.AutoExposureSpeedDown,
                10.f, 1.e-6f));
    TestTrue(
        TEXT("Paris capture owns its exposure compensation"),
        Capture->PostProcessSettings.bOverride_AutoExposureBias);
    TestTrue(
        TEXT("Paris capture preserves foreground midtones after the Mie fix"),
        FMath::IsNearlyEqual(
            Capture->PostProcessSettings.AutoExposureBias,
            0.5f,
            1.e-6f));
    TestFalse(
        TEXT("null capture is rejected"),
        SpPixelGoal::ConfigurePersistentParisCapture(nullptr));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalParisRearCaptureOptInTest,
    "SimWorld.PixelGoal.ParisPoc.PersistentCapture.RearOptInLifecycle",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalParisRearCaptureOptInTest::RunTest(const FString& Parameters)
{
    UWorld* World = UWorld::CreateWorld(EWorldType::Game, false);
    TestNotNull(TEXT("rear-opt-in world exists"), World);
    if (!World || !GEngine)
    {
        if (World)
        {
            World->DestroyWorld(false);
        }
        return false;
    }
    FWorldContext& WorldContext = GEngine->CreateNewWorldContext(EWorldType::Game);
    WorldContext.SetCurrentWorld(World);
    ASpHumanoidAgent* Agent = World->SpawnActor<ASpHumanoidAgent>();
    TestNotNull(TEXT("rear-opt-in agent spawns"), Agent);
    if (!Agent)
    {
        World->DestroyWorld(false);
        GEngine->DestroyWorldContext(World);
        return false;
    }

    TestTrue(TEXT("dual setup initializes both persistent captures"),
        SpPixelGoal::PrepareParisPocCaptureForTest(Agent, true));
    TestTrue(TEXT("dual setup leaves front initialized"),
        Agent->SceneCapture->IsInitialized());
    TestTrue(TEXT("dual setup initializes rear"),
        Agent->RearSceneCapture->IsInitialized());
    TestNotNull(TEXT("dual setup gives rear a render target"),
        Agent->RearSceneCapture->TextureTarget.Get());

    TestTrue(TEXT("omitted/false setup returns to front-only readiness"),
        SpPixelGoal::PrepareParisPocCaptureForTest(Agent, false));
    TestTrue(TEXT("front-only setup preserves initialized front"),
        Agent->SceneCapture->IsInitialized());
    TestFalse(TEXT("front-only setup terminates previously enabled rear"),
        Agent->RearSceneCapture->IsInitialized());
    TestNull(TEXT("front-only setup releases rear render target"),
        Agent->RearSceneCapture->TextureTarget.Get());

    Agent->TerminateObservationCamera(TEXT("front"));
    World->DestroyWorld(false);
    GEngine->DestroyWorldContext(World);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FSpPixelGoalParisSkyAtmosphereConfigurationTest,
    "SimWorld.PixelGoal.ParisPoc.Rendering.SkyAtmosphereMie",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FSpPixelGoalParisSkyAtmosphereConfigurationTest::RunTest(
    const FString& Parameters)
{
    USkyAtmosphereComponent* Atmosphere =
        NewObject<USkyAtmosphereComponent>();
    TestNotNull(TEXT("test atmosphere exists"), Atmosphere);
    if (!Atmosphere)
    {
        return false;
    }
    Atmosphere->SetMieScatteringScale(0.316f);

    TestTrue(
        TEXT("Paris capture normalizes the authored Mie scattering"),
        SpPixelGoal::ConfigureParisSkyAtmosphere(Atmosphere));
    TestTrue(
        TEXT("Paris capture uses the UE default Mie scale"),
        FMath::IsNearlyEqual(
            Atmosphere->MieScatteringScale,
            0.003996f,
            1.e-6f));
    TestFalse(
        TEXT("null atmosphere is rejected"),
        SpPixelGoal::ConfigureParisSkyAtmosphere(nullptr));
    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
