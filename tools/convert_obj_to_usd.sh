#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  tools/convert_obj_to_usd.sh [options] INPUT.obj [OUTPUT.usd] [Isaac Lab AppLauncher args]

Examples:
  tools/convert_obj_to_usd.sh assets/objects/model.obj
  tools/convert_obj_to_usd.sh assets/objects/model.obj assets/objects/model.usd
  tools/convert_obj_to_usd.sh --decimate-ratio 0.3 assets/objects/model.obj
  tools/convert_obj_to_usd.sh --target-faces 10000 assets/objects/model.obj

Options:
  --decimate-ratio RATIO   Collapse decimation ratio after planar cleanup. Default: 0.3.
  --planar-angle-deg DEG   Dissolve near-coplanar faces before collapse. Default: 0 for texture safety.
  --target-faces COUNT     Reduce visual mesh to at most COUNT faces. Overrides ratio if stricter.
  --no-preserve-boundaries Allow collapse across UV/material/seam/sharp/normal boundaries.
  --no-decimate            Convert the original OBJ without reducing faces.
  --keep-decimated-obj     Keep the intermediate decimated OBJ next to the output USD.
  --collision-approx MODE  USD collision approximation: disabled, triangle-mesh,
                           mesh-simplification, convex-hull, convex-decomposition,
                           or bounding-cube. Default: disabled.
  --collision-simplification-metric VALUE
                           Optional PhysX simplification accuracy for mesh-simplification.
  --unlit-textures         Display diffuse textures through emission so scene lights do not shade the mesh.
  --make-instanceable      Make the converted geometry instanceable.
  --blender-exe PATH       Blender executable. Default: BLENDER_EXE, PATH, or /home/sicheng/blender/blender-4.2.1-linux-x64/blender.

This runs Blender texture-preserving collapse decimation first, then Isaac Lab's MeshConverter with collision disabled.
Set ISAACLAB_PATH if Isaac Lab is not installed at /home/sicheng/IsaacLab.
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-/home/sicheng/IsaacLab}"
ISAACLAB_CONDA_ENV="${ISAACLAB_CONDA_ENV:-env_isaaclab}"
ISAACLAB_SH="${ISAACLAB_PATH}/isaaclab.sh"
CONVERTER_SCRIPT="${SCRIPT_DIR}/convert_obj_to_usd.py"
DECIMATE_SCRIPT="${SCRIPT_DIR}/decimate_obj_blender.py"
decimate_ratio="${DECIMATE_RATIO:-0.3}"
planar_angle_deg="${PLANAR_ANGLE_DEG:-0}"
target_faces="${TARGET_FACES:-}"
preserve_boundaries=true
keep_decimated_obj=false
blender_exe="${BLENDER_EXE:-}"

if [[ ! -x "${ISAACLAB_SH}" ]]; then
    echo "Could not find executable Isaac Lab launcher: ${ISAACLAB_SH}" >&2
    echo "Set ISAACLAB_PATH to your Isaac Lab checkout, for example:" >&2
    echo "  ISAACLAB_PATH=/path/to/IsaacLab tools/convert_obj_to_usd.sh INPUT.obj OUTPUT.usd" >&2
    exit 1
fi

positional_args=()
app_args=()
converter_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --decimate-ratio)
            decimate_ratio="$2"
            shift 2
            ;;
        --decimate-ratio=*)
            decimate_ratio="${1#*=}"
            shift
            ;;
        --no-decimate)
            decimate_ratio="1.0"
            planar_angle_deg="0"
            target_faces=""
            shift
            ;;
        --no-preserve-boundaries)
            preserve_boundaries=false
            shift
            ;;
        --planar-angle-deg)
            planar_angle_deg="$2"
            shift 2
            ;;
        --planar-angle-deg=*)
            planar_angle_deg="${1#*=}"
            shift
            ;;
        --target-faces)
            target_faces="$2"
            shift 2
            ;;
        --target-faces=*)
            target_faces="${1#*=}"
            shift
            ;;
        --keep-decimated-obj)
            keep_decimated_obj=true
            shift
            ;;
        --collision-approx)
            converter_args+=("$1" "$2")
            shift 2
            ;;
        --collision-approx=*)
            converter_args+=("$1")
            shift
            ;;
        --collision-simplification-metric)
            converter_args+=("$1" "$2")
            shift 2
            ;;
        --collision-simplification-metric=*)
            converter_args+=("$1")
            shift
            ;;
        --unlit-textures)
            converter_args+=("$1")
            shift
            ;;
        --make-instanceable)
            converter_args+=("$1")
            shift
            ;;
        --blender-exe)
            blender_exe="$2"
            shift 2
            ;;
        --blender-exe=*)
            blender_exe="${1#*=}"
            shift
            ;;
        --)
            shift
            app_args+=("$@")
            break
            ;;
        -*)
            app_args+=("$@")
            break
            ;;
        *)
            positional_args+=("$1")
            shift
            ;;
    esac
done

if [[ ${#positional_args[@]} -lt 1 || ${#positional_args[@]} -gt 2 ]]; then
    usage
    exit 1
fi

input_path="${positional_args[0]}"

if [[ ${#positional_args[@]} -eq 2 ]]; then
    output_path="${positional_args[1]}"
else
    output_path="${input_path%.*}.usd"
fi

if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
    export TERM=xterm-256color
fi

current_python="${CONDA_PREFIX:-}/bin/python"
if [[ ! -x "${current_python}" ]] || ! "${current_python}" -c "import isaaclab" >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
        source "${conda_base}/etc/profile.d/conda.sh"
        if conda env list | awk '{print $1}' | grep -qx "${ISAACLAB_CONDA_ENV}"; then
            conda activate "${ISAACLAB_CONDA_ENV}"
        fi
    fi
fi

conversion_input="${input_path}"
cleanup_dir=""
cleanup() {
    if [[ -n "${cleanup_dir}" ]]; then
        rm -rf "${cleanup_dir}"
    fi
}
trap cleanup EXIT

ratio_mode="$(
    python - "${decimate_ratio}" <<'PY'
import sys

try:
    ratio = float(sys.argv[1])
except ValueError:
    print("error")
    sys.exit(0)

if 0.0 < ratio < 1.0:
    print("decimate")
elif ratio == 1.0:
    print("skip")
else:
    print("error")
PY
)"

if [[ "${ratio_mode}" == "error" ]]; then
    echo "--decimate-ratio must be a number in (0, 1], got: ${decimate_ratio}" >&2
    exit 1
fi

planar_mode="$(
    python - "${planar_angle_deg}" <<'PY'
import sys

try:
    angle = float(sys.argv[1])
except ValueError:
    print("error")
    sys.exit(0)

if angle < 0.0:
    print("error")
elif angle == 0.0:
    print("skip")
else:
    print("planar")
PY
)"
if [[ "${planar_mode}" == "error" ]]; then
    echo "--planar-angle-deg must be a number >= 0, got: ${planar_angle_deg}" >&2
    exit 1
fi

if [[ -n "${target_faces}" ]]; then
    if ! python - "${target_faces}" <<'PY' >/dev/null 2>&1
import sys
target = int(sys.argv[1])
sys.exit(0 if target >= 4 else 1)
PY
    then
        echo "--target-faces must be an integer >= 4, got: ${target_faces}" >&2
        exit 1
    fi
fi

if [[ "${ratio_mode}" == "decimate" || "${planar_mode}" == "planar" || -n "${target_faces}" ]]; then
    if [[ -z "${blender_exe}" ]]; then
        blender_exe="$(command -v blender || true)"
    fi
    if [[ -z "${blender_exe}" && -x "/home/sicheng/blender/blender-4.2.1-linux-x64/blender" ]]; then
        blender_exe="/home/sicheng/blender/blender-4.2.1-linux-x64/blender"
    fi
    if [[ ! -x "${blender_exe}" ]]; then
        echo "Could not find executable Blender for --decimate-ratio ${decimate_ratio}." >&2
        echo "Set BLENDER_EXE or pass --blender-exe PATH, or use --no-decimate." >&2
        exit 1
    fi

    input_base="$(basename "${input_path%.*}")"
    if [[ "${keep_decimated_obj}" == true ]]; then
        if [[ -n "${target_faces}" ]]; then
            decimated_obj="$(dirname "${output_path}")/${input_base}_decimated_f${target_faces}.obj"
        else
            decimated_obj="$(dirname "${output_path}")/${input_base}_decimated_r${decimate_ratio//./p}.obj"
        fi
    else
        cleanup_dir="$(mktemp -d)"
        decimated_obj="${cleanup_dir}/${input_base}_decimated.obj"
    fi
    blender_args=(
        "${input_path}" "${decimated_obj}"
        "--ratio" "${decimate_ratio}"
        "--planar-angle-deg" "${planar_angle_deg}"
    )
    if [[ "${preserve_boundaries}" == false ]]; then
        blender_args+=("--no-preserve-boundaries")
    fi
    if [[ -n "${target_faces}" ]]; then
        blender_args+=("--target-faces" "${target_faces}")
    fi
    "${blender_exe}" --background --python "${DECIMATE_SCRIPT}" -- "${blender_args[@]}"
    conversion_input="${decimated_obj}"
else
    echo "Skipping OBJ decimation (ratio=${decimate_ratio}, planar_angle_deg=${planar_angle_deg})."
fi

"${ISAACLAB_SH}" -p "${CONVERTER_SCRIPT}" "${conversion_input}" "${output_path}" "${converter_args[@]}" --headless "${app_args[@]}"
