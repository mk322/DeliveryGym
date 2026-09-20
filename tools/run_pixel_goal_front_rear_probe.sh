#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
simworld_root="$repo_root/.simworld-ue"
citycore_source="${CITYCORE_PARIS_CONTENT:?set CITYCORE_PARIS_CONTENT to the CityCore_Paris content directory}"
citycore_target="$simworld_root/Content/CityCore_Paris"
python_bin="${PIXEL_GOAL_PYTHON:-$(command -v python3 || true)}"
project_editor="$simworld_root/Binaries/Linux/SimWorldEditor"
editor_bin="${PIXEL_GOAL_UNREAL_EDITOR:-$project_editor}"
identity_helper="$repo_root/tools/pixel_goal_launcher_identity.py"
graphics_adapter="${PIXEL_GOAL_GPU:-5}"
rpc_port="${SIMWORLD_RPC_PORT:-30154}"
spear_config="$repo_root/tools/pixel_goal_1b_poc_spear.yaml"
runtime_config="$simworld_root/Saved/PixelGoalFrontRearProbe/config.yaml"
scene="/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints"
editor_stdout="$simworld_root/Saved/Logs/PixelGoalFrontRearProbe.stdout.log"
editor_log="$simworld_root/Saved/Logs/PixelGoalFrontRearProbe.log"
load_ready_marker="Load map complete /Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints"
run_id="pixel-goal-front-rear-probe-$(date +%s%N)-$$"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${PIXEL_GOAL_OUTPUT:-$repo_root/artifacts/pixel_goal_front_rear_probe/$run_stamp}"

if [[ ! -d "$citycore_source" || -w "$citycore_source" ]]; then
  echo "CityCore_Paris must exist and be read-only: $citycore_source" >&2
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
if [[ ! -x "$editor_bin" || ! -x "$python_bin" || ! -f "$identity_helper" ]]; then
  echo "A required UE or Python executable is missing" >&2
  exit 2
fi
if ! editor_bin="$($python_bin "$identity_helper" validate-bundle \
    --expected-editor "$project_editor" \
    --selected-editor "$editor_bin")"; then
  exit 2
fi

plugin_library_path="$({
  find "$simworld_root/Plugins" -type d -path '*/Binaries/Linux' -print \
    | sort \
    | paste -sd:
} || true)"
project_library_path="$simworld_root/Binaries/Linux"
if [[ -n "$plugin_library_path" ]]; then
  project_library_path="$project_library_path:$plugin_library_path"
fi
for dependency_path in \
  "$simworld_root/Plugins/spear/third_party/boost/stage/lib" \
  "$simworld_root/Plugins/spear/third_party/rpclib/BUILD/Linux" \
  "$simworld_root/Plugins/spear/third_party/yaml-cpp/BUILD/Linux"
do
  if [[ -d "$dependency_path" ]]; then
    project_library_path="$project_library_path:$dependency_path"
  fi
done
export LD_LIBRARY_PATH="$project_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

port_is_free() {
  "$python_bin" - "$1" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
}

if ! port_is_free "$rpc_port"; then
  echo "SimWorld RPC port is already occupied: $rpc_port" >&2
  exit 2
fi

used_mib="$(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v wanted="$graphics_adapter" '$1 + 0 == wanted {gsub(/ /, "", $2); print $2}'
)"
if [[ -z "$used_mib" ]]; then
  echo "UE GPU index is unavailable: $graphics_adapter" >&2
  exit 2
fi
if (( used_mib > 4096 )); then
  echo "UE GPU $graphics_adapter is using ${used_mib} MiB; refusing to interfere" >&2
  exit 2
fi

mkdir -p \
  "$output_dir" \
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
import os
import sys

import spear

config = spear.get_config(user_config_files=[sys.argv[1]])
config.defrost()
config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = int(os.environ["SIMWORLD_RPC_PORT"])
config.freeze()
with open(sys.argv[2], "w", encoding="utf-8") as output:
    config.dump(stream=output, default_flow_style=False)
PY

editor_pid=""
cleanup_processes() {
  if [[ -n "$editor_pid" ]] && kill -0 "$editor_pid" 2>/dev/null; then
    kill "$editor_pid" 2>/dev/null || true
    wait "$editor_pid" 2>/dev/null || true
  fi
}
trap cleanup_processes EXIT INT TERM

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

if ! "$python_bin" "$identity_helper" verify-process \
    --pid "$editor_pid" \
    --expected-editor "$editor_bin" \
    --timeout-s 1.0 >/dev/null; then
  echo "Launched editor process identity failed before readiness" >&2
  exit 2
fi

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
    except OSError as error:
        raise SystemExit(f"UnrealEditor exited before readiness: {error}")
    if not map_ready and log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        run_index = log_text.find(run_id)
        map_ready = run_index >= 0 and log_text.find(ready_marker, run_index) > run_index
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
raise SystemExit("timed out waiting for Paris map and SimWorld RPC")
PY

# Narrow test-only synchronization for the controlled same-PID exec regression.
# It cannot select another executable and is inert for every normal launch.
case "${PIXEL_GOAL_LAUNCHER_TEST_SIGNAL_AFTER_READINESS:-}" in
  "")
    if [[ -n "${PIXEL_GOAL_LAUNCHER_TEST_SWITCH_ACK:-}" ]]; then
      echo "Test-only switch acknowledgement requires the test signal" >&2
      exit 2
    fi
    ;;
  1)
    test_switch_ack="${PIXEL_GOAL_LAUNCHER_TEST_SWITCH_ACK:-}"
    if [[ "$test_switch_ack" != /* || -e "$test_switch_ack" ]]; then
      echo "Test-only switch acknowledgement must be a new absolute path" >&2
      exit 2
    fi
    if ! kill -USR1 "$editor_pid"; then
      echo "Test-only post-readiness signal could not reach launched editor PID" >&2
      exit 2
    fi
    if ! "$python_bin" - "$test_switch_ack" "$editor_pid" <<'PY'
import os
from pathlib import Path
import sys
import time

acknowledgement = Path(sys.argv[1])
pid = int(sys.argv[2])
deadline = time.monotonic() + 2.0
while time.monotonic() < deadline:
    if acknowledgement.is_file() and not acknowledgement.is_symlink():
        raise SystemExit(0)
    try:
        os.kill(pid, 0)
    except OSError as error:
        raise SystemExit(
            f"launched editor exited before switch acknowledgement: {error}")
    time.sleep(0.005)
raise SystemExit("timed out waiting for test-only process-switch acknowledgement")
PY
    then
      exit 2
    fi
    ;;
  *)
    echo "PIXEL_GOAL_LAUNCHER_TEST_SIGNAL_AFTER_READINESS must be unset or 1" >&2
    exit 2
    ;;
esac

if ! "$python_bin" "$identity_helper" verify-process \
    --pid "$editor_pid" \
    --expected-editor "$editor_bin" \
    --timeout-s 0.25 >/dev/null; then
  echo "Launched editor process identity failed before runner dispatch" >&2
  exit 2
fi

"$python_bin" "$repo_root/tools/run_pixel_goal_front_rear_probe.py" \
  --simworld-root "$simworld_root" \
  --citycore-content "$citycore_source" \
  --spear-config "$runtime_config" \
  --launch-mode attach \
  --shutdown-attached-editor \
  --output "$output_dir" \
  "$@"

echo "Front/rear Pixel Goal probe artifacts: $output_dir"
