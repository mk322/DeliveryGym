#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
simworld_root="$repo_root/.simworld-ue"
citycore_source="${CITYCORE_PARIS_CONTENT:?set CITYCORE_PARIS_CONTENT to the CityCore_Paris content directory}"
citycore_target="$simworld_root/Content/CityCore_Paris"
python_bin="${PIXEL_GOAL_PYTHON:-$(command -v python3 || true)}"
editor_bin="${PIXEL_GOAL_UNREAL_EDITOR:?set PIXEL_GOAL_UNREAL_EDITOR to the UnrealEditor binary}"
spear_config="$repo_root/tools/pixel_goal_1b_poc_spear.yaml"
runtime_config="$simworld_root/Saved/PixelGoal1BPoc/config.yaml"
scene="/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints"
rpc_port="${SIMWORLD_RPC_PORT:-30128}"
graphics_adapter="${PIXEL_GOAL_GPU:-0}"
editor_stdout="$simworld_root/Saved/Logs/PixelGoalParisPoc.stdout.log"
editor_log="$simworld_root/Saved/Logs/PixelGoalParisPoc.log"
load_ready_marker="Load map complete /Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints"
run_id="pixel-goal-1b-poc-$(date +%s%N)-$$"

if [[ ! -d "$citycore_source" ]]; then
  echo "CityCore_Paris content is missing: $citycore_source" >&2
  exit 2
fi
if [[ -w "$citycore_source" ]]; then
  echo "CityCore_Paris source must be read-only for this PoC: $citycore_source" >&2
  exit 2
fi
if [[ ! -f "$citycore_source/Scenes/ParisCity_FinalBlueprints.umap" ]]; then
  echo "Paris map asset is missing under: $citycore_source" >&2
  exit 2
fi
if [[ ! -f "$simworld_root/SimWorld.uproject" ]]; then
  echo "SimWorld.uproject is missing: $simworld_root/SimWorld.uproject" >&2
  exit 2
fi
if [[ ! -x "$editor_bin" ]]; then
  echo "Shared UnrealEditor is missing: $editor_bin" >&2
  exit 2
fi

mkdir -p \
  "$simworld_root/Content" \
  "$simworld_root/Saved/Logs" \
  "$(dirname "$runtime_config")"
if [[ -L "$citycore_target" ]]; then
  if [[ "$(readlink -f "$citycore_target")" != "$(readlink -f "$citycore_source")" ]]; then
    echo "CityCore_Paris link points somewhere unexpected: $citycore_target" >&2
    exit 2
  fi
elif [[ -e "$citycore_target" ]]; then
  echo "Refusing to replace existing CityCore_Paris content: $citycore_target" >&2
  exit 2
else
  ln -s "$citycore_source" "$citycore_target"
fi

export SIMWORLD_RPC_PORT="$rpc_port"
export SIMWORLD_SPEAR_PYTHON="${SIMWORLD_SPEAR_PYTHON:-$repo_root/.spear/python}"
spear_ext_python="${SIMWORLD_SPEAR_EXT_PYTHON:-$repo_root/.spear/python_ext/python}"
export UE_SKIP_UBT_SDK_SETUP=1
export PYTHONPATH="$repo_root:$SIMWORLD_SPEAR_PYTHON:$spear_ext_python${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" - "$spear_config" "$runtime_config" <<'PY'
import sys

import spear

config = spear.get_config(user_config_files=[sys.argv[1]])
with open(sys.argv[2], "w", encoding="utf-8") as output:
    config.dump(stream=output, default_flow_style=False)
PY

editor_pid=""
cleanup_editor() {
  if [[ -n "$editor_pid" ]] && kill -0 "$editor_pid" 2>/dev/null; then
    kill "$editor_pid" 2>/dev/null || true
    wait "$editor_pid" 2>/dev/null || true
  fi
}
trap cleanup_editor EXIT INT TERM

"$editor_bin" "$simworld_root/SimWorld.uproject" "$scene" \
  -game \
  -PixelGoalRunId="$run_id" \
  '-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorExperimentalSettings]:bEnableAsyncStaticMeshCompilation=False,[/Script/UnrealEd.EditorExperimentalSettings]:bEnableAsyncTextureCompilation=False,[/Script/UnrealEd.EditorExperimentalSettings]:bEnableAsyncSkinnedAssetCompilation=False' \
  -RenderOffScreen -resx=1280 -resy=720 -nosound -unattended -nopause \
  -NoSplash -NoUba -NoCompile -NoCompileEditor -notrace \
  -AssetRegistry.DisableDirectoryWatcher=1 \
  -ddc=NoZenLocalFallback \
  -LocalDataCachePath="$simworld_root/Saved/DerivedDataCache" \
  -graphicsadapter="$graphics_adapter" \
  -AbsLog="$editor_log" \
  -sp-config-file="$runtime_config" \
  > "$editor_stdout" 2>&1 &
editor_pid="$!"

"$python_bin" - "$rpc_port" "$editor_pid" "$editor_log" "$load_ready_marker" "$run_id" <<'PY'
import os
from pathlib import Path
import socket
import sys
import time

port = int(sys.argv[1])
pid = int(sys.argv[2])
log_path = Path(sys.argv[3])
ready_marker = sys.argv[4]
run_id = sys.argv[5]
deadline = time.monotonic() + 1800.0
map_ready = False
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise SystemExit(f"UnrealEditor exited before opening RPC port {port}: {exc}")
    if not map_ready and log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        run_index = log_text.find(run_id)
        map_ready = (
            run_index >= 0
            and log_text.find(ready_marker, run_index) > run_index
        )
    if not map_ready:
        time.sleep(1.0)
        continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(1.0)
        try:
            client.connect(("127.0.0.1", port))
        except OSError:
            time.sleep(1.0)
        else:
            raise SystemExit(0)
raise SystemExit(
    f"timed out waiting for Paris map readiness and UnrealEditor RPC port {port}"
)
PY

"$python_bin" "$repo_root/tools/run_pixel_goal_1b_poc.py" \
  --simworld-root "$simworld_root" \
  --citycore-content "$citycore_source" \
  --spear-config "$runtime_config" \
  --launch-mode attach \
  --shutdown-attached-editor \
  "$@"
