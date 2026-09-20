// SpNavMeshHelper.cpp

#include "SpNavMeshHelper.h"

#include "NavigationSystem.h"
#include "NavMesh/NavMeshBoundsVolume.h"
#include "NavMesh/RecastNavMesh.h"
#include "Components/BrushComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "UObject/ConstructorHelpers.h"

#include "SpCore/Log.h"
#include "SpCore/Unreal.h"

ASpNavMeshHelper::ASpNavMeshHelper()
{
    PrimaryActorTick.bCanEverTick = false;
}

bool ASpNavMeshHelper::SpawnNavMeshBounds(FVector Center, FVector Extent)
{
    UWorld* World = GetWorld();
    if (!World)
    {
        SP_LOG("SpNavMeshHelper::SpawnNavMeshBounds: no world");
        return false;
    }

    // Use ASpExplicitNavMeshBoundsVolume which overrides
    // GetComponentsBoundingBox to return an explicit FBox.  This
    // works around the issue that ANavMeshBoundsVolume's brush model
    // (UModel/BSP) isn't rebuilt at runtime when scale changes, so the
    // default GetComponentsBoundingBox returns a 0-volume box.
    FActorSpawnParameters Params;
    Params.Owner = this;
    Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    Params.bDeferConstruction = true;

    ASpExplicitNavMeshBoundsVolume* Volume = World->SpawnActor<ASpExplicitNavMeshBoundsVolume>(
        ASpExplicitNavMeshBoundsVolume::StaticClass(), Center, FRotator::ZeroRotator, Params);

    if (!Volume)
    {
        SP_LOG("SpNavMeshHelper::SpawnNavMeshBounds: SpawnActor returned nullptr");
        return false;
    }

    // Set explicit min/max BEFORE FinishSpawning so that when
    // PostRegisterAllComponents fires (which calls OnNavigationBoundsAdded),
    // our overridden GetComponentsBoundingBox returns the right box.
    Volume->ExplicitMin = Center - Extent;
    Volume->ExplicitMax = Center + Extent;
    Volume->FinishSpawning(FTransform(FRotator::ZeroRotator, Center, FVector(1, 1, 1)));

    // Belt+suspenders: explicitly re-notify with the (now correct)
    // bounds, in case PostRegisterAllComponents ran early.
    if (UNavigationSystemV1* NavSys = UNavigationSystemV1::GetNavigationSystem(World))
    {
        NavSys->OnNavigationBoundsUpdated(Volume);
    }

    BoundsVolume = Volume;
    SP_LOG("SpNavMeshHelper::SpawnNavMeshBounds (explicit-box subclass): center=(",
           Center.X, ",", Center.Y, ",", Center.Z,
           ") extent=(", Extent.X, ",", Extent.Y, ",", Extent.Z,
           ") -> box min=(", Volume->ExplicitMin.X, ",", Volume->ExplicitMin.Y, ",", Volume->ExplicitMin.Z,
           ") max=(", Volume->ExplicitMax.X, ",", Volume->ExplicitMax.Y, ",", Volume->ExplicitMax.Z, ")");
    return true;
}

void ASpNavMeshHelper::BuildNavMesh()
{
    UWorld* World = GetWorld();
    if (!World)
    {
        SP_LOG("SpNavMeshHelper::BuildNavMesh: no world");
        return;
    }
    UNavigationSystemV1* NavSys = UNavigationSystemV1::GetNavigationSystem(World);
    if (!NavSys)
    {
        SP_LOG("SpNavMeshHelper::BuildNavMesh: no NavigationSystem");
        return;
    }
    SP_LOG("SpNavMeshHelper::BuildNavMesh: triggering Build()");
    NavSys->Build();
    SP_LOG("  registered NavData count after Build: ", NavSys->NavDataSet.Num());
}

bool ASpNavMeshHelper::IsNavMeshReady() const
{
    UWorld* World = GetWorld();
    if (!World) return false;
    UNavigationSystemV1* NavSys = UNavigationSystemV1::GetNavigationSystem(World);
    if (!NavSys) return false;
    ANavigationData* NavData = NavSys->GetDefaultNavDataInstance();
    if (!NavData) return false;
    // "ready" means: nav data exists and is not flagged as needing rebuild
    // (NavigationSystem clears NeedsRebuild after async generation completes).
    return !NavData->NeedsRebuild();
}

int32 ASpNavMeshHelper::GetNumRegisteredNavData() const
{
    UWorld* World = GetWorld();
    if (!World) return 0;
    UNavigationSystemV1* NavSys = UNavigationSystemV1::GetNavigationSystem(World);
    if (!NavSys) return 0;
    return NavSys->NavDataSet.Num();
}

int32 ASpNavMeshHelper::GetNumNavBounds() const
{
    UWorld* World = GetWorld();
    if (!World) return 0;
    UNavigationSystemV1* NavSys = UNavigationSystemV1::GetNavigationSystem(World);
    if (!NavSys) return 0;
    const TSet<FNavigationBounds>& Bounds = NavSys->GetNavigationBounds();
    SP_LOG("GetNumNavBounds: ", Bounds.Num(), " bounds registered");
    for (const FNavigationBounds& B : Bounds)
    {
        SP_LOG("  bound: box min=(", B.AreaBox.Min.X, ",", B.AreaBox.Min.Y, ",", B.AreaBox.Min.Z,
               ") max=(", B.AreaBox.Max.X, ",", B.AreaBox.Max.Y, ",", B.AreaBox.Max.Z, ")");
    }
    return Bounds.Num();
}

FString ASpNavMeshHelper::DumpNavState()
{
    FString Out;
    UWorld* World = GetWorld();
    if (!World) { Out += TEXT("no world\n"); return Out; }
    UNavigationSystemV1* NavSys = UNavigationSystemV1::GetNavigationSystem(World);
    if (!NavSys) { Out += TEXT("no NavSys\n"); return Out; }

    Out += FString::Printf(TEXT("NavSys.NavDataSet.Num()=%d\n"), NavSys->NavDataSet.Num());
    Out += FString::Printf(TEXT("NavSys.bWholeWorldNavigable=%d\n"),
                            (int)NavSys->ShouldGenerateNavigationEverywhere());
    const TSet<FNavigationBounds>& Bounds = NavSys->GetNavigationBounds();
    Out += FString::Printf(TEXT("NavSys.RegisteredNavBounds.Num()=%d\n"), Bounds.Num());
    for (const FNavigationBounds& B : Bounds)
    {
        Out += FString::Printf(TEXT("  bound: min=(%.0f,%.0f,%.0f) max=(%.0f,%.0f,%.0f) area=%.0f\n"),
                                B.AreaBox.Min.X, B.AreaBox.Min.Y, B.AreaBox.Min.Z,
                                B.AreaBox.Max.X, B.AreaBox.Max.Y, B.AreaBox.Max.Z,
                                B.AreaBox.GetVolume());
    }
    ANavigationData* NavData = NavSys->GetDefaultNavDataInstance();
    if (NavData)
    {
        Out += FString::Printf(TEXT("DefaultNavData.Name=%s class=%s\n"),
                                *NavData->GetName(),
                                *NavData->GetClass()->GetName());
        Out += FString::Printf(TEXT("DefaultNavData.RuntimeGenerationMode=%d (0=Static,1=ModifiersOnly,2=Dynamic)\n"),
                                (int)NavData->GetRuntimeGenerationMode());
        Out += FString::Printf(TEXT("DefaultNavData.NeedsRebuild()=%d\n"),
                                (int)NavData->NeedsRebuild());
        Out += FString::Printf(TEXT("DefaultNavData.SupportsRuntimeGeneration()=%d\n"),
                                (int)NavData->SupportsRuntimeGeneration());
        if (ARecastNavMesh* Recast = Cast<ARecastNavMesh>(NavData))
        {
            Out += FString::Printf(TEXT("RecastNavMesh.AgentRadius=%.1f AgentHeight=%.1f\n"),
                                    Recast->AgentRadius, Recast->AgentHeight);
        }
    }
    else
    {
        Out += TEXT("DefaultNavData=null\n");
    }
    SP_LOG("DumpNavState:\n", TCHAR_TO_UTF8(*Out));
    return Out;
}

bool ASpNavMeshHelper::ForceDynamicRuntimeGeneration()
{
    // RuntimeGeneration is a protected UPROPERTY on ANavigationData;
    // there's no public setter exposed.  Best we can do is verify the
    // current mode in DumpNavState() and rely on the DefaultEngine.ini
    // `[/Script/NavigationSystem.RecastNavMesh] RuntimeGeneration=Dynamic`
    // to have applied at instance creation.  Always returns false to
    // signal "no-op".
    return false;
}

bool ASpNavMeshHelper::IsDynamicRuntimeGeneration() const
{
    UWorld* World = GetWorld();
    const UNavigationSystemV1* NavSys = World
        ? UNavigationSystemV1::GetNavigationSystem(World)
        : nullptr;
    const ANavigationData* NavData = NavSys
        ? NavSys->GetDefaultNavDataInstance()
        : nullptr;
    return NavData &&
        NavData->GetRuntimeGenerationMode() == ERuntimeGenerationType::Dynamic;
}

bool ASpNavMeshHelper::SupportsRuntimeNavGeneration() const
{
    UWorld* World = GetWorld();
    const UNavigationSystemV1* NavSys = World
        ? UNavigationSystemV1::GetNavigationSystem(World)
        : nullptr;
    const ANavigationData* NavData = NavSys
        ? NavSys->GetDefaultNavDataInstance()
        : nullptr;
    return NavData && NavData->SupportsRuntimeGeneration();
}

bool ASpNavMeshHelper::HasValidNavMeshData() const
{
    UWorld* World = GetWorld();
    const UNavigationSystemV1* NavSys = World
        ? UNavigationSystemV1::GetNavigationSystem(World)
        : nullptr;
    const ARecastNavMesh* Recast = NavSys
        ? Cast<ARecastNavMesh>(NavSys->GetDefaultNavDataInstance())
        : nullptr;
    return Recast && Recast->HasValidNavmesh();
}

int32 ASpNavMeshHelper::GetNumActiveNavMeshTiles() const
{
    UWorld* World = GetWorld();
    const UNavigationSystemV1* NavSys = World
        ? UNavigationSystemV1::GetNavigationSystem(World)
        : nullptr;
    const ARecastNavMesh* Recast = NavSys
        ? Cast<ARecastNavMesh>(NavSys->GetDefaultNavDataInstance())
        : nullptr;
    return Recast ? Recast->GetNumActiveTiles() : 0;
}

bool ASpNavMeshHelper::ForceNavDataRebuild()
{
    UWorld* World = GetWorld();
    if (!World) return false;
    UNavigationSystemV1* NavSys = UNavigationSystemV1::GetNavigationSystem(World);
    if (!NavSys) return false;
    ANavigationData* NavData = NavSys->GetDefaultNavDataInstance();
    if (!NavData)
    {
        SP_LOG("ForceNavDataRebuild: no NavData");
        return false;
    }
    SP_LOG("ForceNavDataRebuild: calling NavData->RebuildAll() on ",
           Unreal::toStdString(NavData->GetName()));
    NavData->RebuildAll();
    return true;
}

bool ASpNavMeshHelper::SpawnFloorPlate(FVector Center, FVector Extent)
{
    UWorld* World = GetWorld();
    if (!World)
    {
        SP_LOG("SpawnFloorPlate: no world");
        return false;
    }

    UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(
        nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
    if (!CubeMesh)
    {
        SP_LOG("SpawnFloorPlate: failed to load /Engine/BasicShapes/Cube.Cube");
        return false;
    }

    FActorSpawnParameters Params;
    Params.Owner = this;
    Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

    AStaticMeshActor* Floor = World->SpawnActor<AStaticMeshActor>(
        AStaticMeshActor::StaticClass(), Center, FRotator::ZeroRotator, Params);
    if (!Floor)
    {
        SP_LOG("SpawnFloorPlate: SpawnActor returned nullptr");
        return false;
    }

    UStaticMeshComponent* SMC = Floor->GetStaticMeshComponent();
    if (!SMC)
    {
        SP_LOG("SpawnFloorPlate: no StaticMeshComponent");
        Floor->Destroy();
        return false;
    }

    // CRITICAL: must be Static (not Movable) for navmesh recast pickup --
    // movable meshes are excluded by default from nav geometry queries.
    SMC->SetMobility(EComponentMobility::Static);
    SMC->SetStaticMesh(CubeMesh);
    // /Engine/BasicShapes/Cube is 100x100x100 unscaled (half-extent 50 per axis).
    // We want a plate Extent.X by Extent.Y wide, 20cm thick.
    // Scale X = Extent.X / 100  (full extent / 100 to get from 100cm cube to Extent cm)
    // For Z: 20cm thick / 100cm = 0.2
    Floor->SetActorScale3D(FVector(Extent.X / 100.f, Extent.Y / 100.f, 0.2f));
    SMC->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    SMC->SetCollisionResponseToAllChannels(ECR_Block);
    SMC->SetCanEverAffectNavigation(true);

    FloorPlate = Floor;
    SP_LOG("SpawnFloorPlate: spawned StaticMeshActor at (",
           Center.X, ",", Center.Y, ",", Center.Z,
           ") size=(", Extent.X, ",", Extent.Y, ",", 20.0f, ")");
    return true;
}
