#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
simworld_root="$repo_root/.simworld-ue"
# The two things this launcher cannot find for you: the CityCore Paris
# content (an Epic marketplace asset) and the SPEAR python client. Both are
# named by environment variable; docs/PIXEL_GOAL.md says where they come from.
citycore_source="${CITYCORE_PARIS_CONTENT:?set CITYCORE_PARIS_CONTENT to the CityCore_Paris content directory}"
citycore_target="$simworld_root/Content/CityCore_Paris"
# The interpreter needs cairosvg, Pillow, pydantic, pyyaml and SPEAR's own
# dependencies (yacs, numpy, psutil, scipy, opencv, msgpack); see
# docs/PIXEL_GOAL.md. Set PIXEL_GOAL_PYTHON to that environment's python.
python_bin="${PIXEL_GOAL_PYTHON:-$(command -v python3 || true)}"
project_editor="$simworld_root/Binaries/Linux/SimWorldEditor"
editor_bin="${PIXEL_GOAL_UNREAL_EDITOR:-$project_editor}"
identity_helper="$repo_root/tools/pixel_goal_launcher_identity.py"
qwen_endpoint="${QWEN_ENDPOINT:-http://127.0.0.1:30001/v1/chat/completions}"
qwen_model_name="${QWEN_MODEL_NAME:-qwen3-vl-8b}"
qwen_gpu="${QWEN_GPU:-2}"
# Four photographs plus the phone's map per turn; the two-view runner needs 3.
model_image_capacity="${QWEN_MAX_IMAGES_PER_PROMPT:-5}"
graphics_adapter="${PIXEL_GOAL_GPU:-5}"
rpc_port="${SIMWORLD_RPC_PORT:-30155}"
spear_config="$repo_root/tools/pixel_goal_1b_poc_spear.yaml"
runtime_config="$simworld_root/Saved/PixelGoalFrontRearDelivery/config.yaml"
scene="/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints"
load_ready_marker="Load map complete /Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints"
run_id="pixel-goal-front-rear-delivery-$(date +%s%N)-$$"
editor_stdout="$simworld_root/Saved/Logs/PixelGoalFrontRearDelivery-${run_id}.stdout.log"
editor_log="$simworld_root/Saved/Logs/PixelGoalFrontRearDelivery-${run_id}.log"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${PIXEL_GOAL_OUTPUT:-$repo_root/artifacts/pixel_goal_front_rear_delivery/$run_stamp}"

if [[ ! "$model_image_capacity" =~ ^[0-9]+$ ]] || (( model_image_capacity < 3 )); then
  echo "Front/rear delivery requires model capacity for at least 3 images" >&2
  exit 2
fi
if [[ "$qwen_gpu" == "$graphics_adapter" ]]; then
  echo "Externally owned Qwen and UE must use different GPUs" >&2
  exit 2
fi
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
if [[ ! -x "${editor_bin}" || ! -x "${python_bin}" || ! -f "$identity_helper" ]]; then
  echo "A required UE or Python executable is missing" >&2
  exit 2
fi
if ! editor_bin="$($python_bin "$identity_helper" validate-bundle \
    --expected-editor "$project_editor" \
    --selected-editor "$editor_bin")"; then
  exit 2
fi

if ! "$python_bin" - <<'PY'
def verify_phone_map_rasterizer():
    from io import BytesIO

    import cairosvg
    from PIL import Image

    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2" '
        b'viewBox="0 0 2 2"><rect width="2" height="2" fill="#fff"/></svg>'
    )
    data = cairosvg.svg2png(bytestring=svg)
    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.format != "PNG" or image.size != (2, 2):
            raise RuntimeError("CairoSVG returned an unexpected raster")


try:
    verify_phone_map_rasterizer()
except BaseException:
    raise SystemExit(1)
PY
then
  echo "Front/rear delivery requires a functional cairosvg phone-map rasterizer" >&2
  exit 2
fi

# Read-only ownership boundary: verify the already-running model and its served
# name before any UE process is started.  Never launch, reconfigure, or stop it.
"$python_bin" - "$qwen_endpoint" "$qwen_model_name" <<'PY'
import json
import sys
import time
import urllib.parse
import urllib.request

endpoint, expected = sys.argv[1:]
parts = urllib.parse.urlsplit(endpoint)
if parts.path != "/v1/chat/completions":
    raise SystemExit("QWEN_ENDPOINT must end in /v1/chat/completions")
models_url = urllib.parse.urlunsplit(
    (parts.scheme, parts.netloc, "/v1/models", "", ""))
deadline = time.monotonic() + 60.0
last_error = None
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(models_url, timeout=2.0) as response:
            models = json.loads(response.read().decode())
        if any(item.get("id") == expected for item in models.get("data", [])):
            raise SystemExit(0)
        last_error = f"model {expected!r} is not served by {models_url}"
    except Exception as error:  # endpoint is external; bounded readiness only
        last_error = f"{type(error).__name__}: {error}"
    time.sleep(1.0)
raise SystemExit(f"external Qwen endpoint failed verification: {last_error}")
PY

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

gpu_used_mib() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v wanted="$1" '$1 + 0 == wanted {gsub(/ /, "", $2); print $2}'
}

used_mib="$(gpu_used_mib "$graphics_adapter")"
if [[ -z "$used_mib" ]]; then
  echo "UE GPU index is unavailable: $graphics_adapter" >&2
  exit 2
fi
allow_shared_gpu="${PIXEL_GOAL_ALLOW_SHARED_UE_GPU:-0}"
if [[ "$allow_shared_gpu" != "0" && "$allow_shared_gpu" != "1" ]]; then
  echo "PIXEL_GOAL_ALLOW_SHARED_UE_GPU must be 0 or 1" >&2
  exit 2
fi
if (( used_mib > 4096 )) && [[ "$allow_shared_gpu" != "1" ]]; then
  echo "UE GPU $graphics_adapter is using ${used_mib} MiB; refusing to interfere" >&2
  exit 2
fi
# An explicitly authorised shared-machine run may coexist only with a modest
# pre-existing allocation.  This keeps the normal fail-closed ownership gate
# intact and still refuses heavily occupied cards even when the override is on.
# The shared ceiling is configurable for hosts whose cards carry other
# people's steady-state allocations: a 24 GB card with 9 GiB in use still has
# room for the editor's ~5 GiB. The default is unchanged.
shared_limit_mib="${PIXEL_GOAL_SHARED_UE_GPU_LIMIT_MIB:-8192}"
if [[ ! "$shared_limit_mib" =~ ^[0-9]+$ ]]; then
  echo "PIXEL_GOAL_SHARED_UE_GPU_LIMIT_MIB must be an integer number of MiB" >&2
  exit 2
fi
if (( used_mib > shared_limit_mib )); then
  echo "UE GPU $graphics_adapter is using ${used_mib} MiB; shared limit is ${shared_limit_mib} MiB" >&2
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
export PIXEL_GOAL_UE_LOG="$editor_log"
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

# How long the launched process may take to exec the editor before the
# identity check calls it a mismatch. One second is right for an idle host;
# on a loaded one the backgrounding shell can take longer, and the check
# then sees bash where it expects the editor -- a refusal that has nothing
# to do with the build's identity. Raise it per host, never lower it.
identity_timeout_s="${PIXEL_GOAL_IDENTITY_TIMEOUT_S:-1.0}"
if ! "$python_bin" "$identity_helper" verify-process \
    --pid "$editor_pid" \
    --expected-editor "$editor_bin" \
    --timeout-s "$identity_timeout_s" >/dev/null; then
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

# PIXEL_GOAL_RUNNER_PY runs another script against the same booted, verified
# engine with the same arguments: an engine probe, a harness experiment. The
# delivery runner is the default and the only one whose reports are audited.
runner_py="$repo_root/tools/run_pixel_goal_front_rear_delivery.py"
if [[ -n "${PIXEL_GOAL_RUNNER_PY:-}" ]]; then
  runner_py="$PIXEL_GOAL_RUNNER_PY"
fi
"$python_bin" "$runner_py" \
  --simworld-root "$simworld_root" \
  --citycore-content "$citycore_source" \
  --spear-config "$runtime_config" \
  --launch-mode attach \
  --shutdown-attached-editor \
  --model-endpoint "$qwen_endpoint" \
  --model "$qwen_model_name" \
  --model-image-capacity "$model_image_capacity" \
  --output "$output_dir" \
  "$@"

echo "Front/rear Pixel Goal delivery artifacts: $output_dir"
