#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  tools/build_or_room_assets.sh [halo|pulm|all] [options] [-- Isaac Lab AppLauncher args]

Options:
  --skip-usd           Only bake OBJ/MTL files; skip OBJ -> USD conversion.
  --blender-exe PATH   Blender executable. Default: BLENDER_EXE, PATH, or /home/sicheng/blender/blender-4.2.1-linux-x64/blender.
  --origin-mode MODE   keep_xy or center_xy. Default: center_xy.
  --floor-to-z Z       Ground the baked mesh to this Z value. Default: 0.
  --collision-approx MODE
                      Collision approximation for generated USD. Default: mesh-simplification.
  --collision-simplification-metric VALUE
                      Optional PhysX simplification accuracy for mesh-simplification.
  --lit-textures      Keep OR room textures as regular lit PBR materials.
                      Default: patch textures to emissive/unlit so the room displays texture color directly.

Examples:
  tools/build_or_room_assets.sh all
  tools/build_or_room_assets.sh halo --skip-usd
  tools/build_or_room_assets.sh pulm -- --/app/window/enabled=false

This bakes OR-room scale and rotation in Blender, exports normalized OBJ assets,
then converts those OBJ files to USD with identity scale/rotation/translation.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OR_MODEL_DIR="${PROJECT_ROOT}/assets/objects/OR/Model"
BAKE_SCRIPT="${SCRIPT_DIR}/bake_or_scene_blender.py"
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_obj_to_usd.sh"

target="${1:-all}"
if [[ "${target}" == "-h" || "${target}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ "${target}" == "halo" || "${target}" == "pulm" || "${target}" == "all" ]]; then
    shift || true
else
    target="all"
fi

skip_usd=false
origin_mode="center_xy"
floor_to_z="0"
collision_approx="mesh-simplification"
collision_simplification_metric=""
unlit_textures=true
blender_exe="${BLENDER_EXE:-}"
converter_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-usd)
            skip_usd=true
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
        --origin-mode)
            origin_mode="$2"
            shift 2
            ;;
        --origin-mode=*)
            origin_mode="${1#*=}"
            shift
            ;;
        --floor-to-z)
            floor_to_z="$2"
            shift 2
            ;;
        --floor-to-z=*)
            floor_to_z="${1#*=}"
            shift
            ;;
        --collision-approx)
            collision_approx="$2"
            shift 2
            ;;
        --collision-approx=*)
            collision_approx="${1#*=}"
            shift
            ;;
        --collision-simplification-metric)
            collision_simplification_metric="$2"
            shift 2
            ;;
        --collision-simplification-metric=*)
            collision_simplification_metric="${1#*=}"
            shift
            ;;
        --lit-textures)
            unlit_textures=false
            shift
            ;;
        --)
            shift
            converter_args+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

case "${origin_mode}" in
    keep_xy|center_xy)
        ;;
    *)
        echo "--origin-mode must be keep_xy or center_xy, got: ${origin_mode}" >&2
        exit 1
        ;;
esac

case "${collision_approx}" in
    disabled|triangle-mesh|mesh-simplification|convex-hull|convex-decomposition|bounding-cube)
        ;;
    *)
        echo "--collision-approx must be disabled, triangle-mesh, mesh-simplification, convex-hull, convex-decomposition, or bounding-cube; got: ${collision_approx}" >&2
        exit 1
        ;;
esac

if [[ -z "${blender_exe}" ]]; then
    blender_exe="$(command -v blender || true)"
fi
if [[ -z "${blender_exe}" && -x "/home/sicheng/blender/blender-4.2.1-linux-x64/blender" ]]; then
    blender_exe="/home/sicheng/blender/blender-4.2.1-linux-x64/blender"
fi
if [[ ! -x "${blender_exe}" ]]; then
    echo "Could not find executable Blender." >&2
    echo "Set BLENDER_EXE or pass --blender-exe PATH." >&2
    exit 1
fi

bake_and_convert() {
    local room="$1"
    local scale output_dir output_obj output_usd
    local rot=()
    local inputs=()

    case "${room}" in
        halo)
            scale="4.8"
            rot=(0.6721136569976807 0.2843584716320038 0.3799154460430145 -0.5683904886245728)
            inputs=(
                "${OR_MODEL_DIR}/halo_room/halo_hole_fix_final_a.obj"
                "${OR_MODEL_DIR}/halo_room/halo_hole_fix_final_b.obj"
            )
            output_dir="${OR_MODEL_DIR}/halo_room_baked"
            output_obj="${output_dir}/halo_room_baked.obj"
            output_usd="${output_dir}/halo_room_baked.usd"
            ;;
        pulm)
            scale="5.1"
            rot=(0.33943048119544983 0.5507918000221252 0.2742823660373688 -0.7114664912223816)
            inputs=(
                "${OR_MODEL_DIR}/pulm_room/pulm_room_1_a_clean.obj"
                "${OR_MODEL_DIR}/pulm_room/pulm_room_1_b_clean.obj"
            )
            output_dir="${OR_MODEL_DIR}/pulm_room_baked"
            output_obj="${output_dir}/pulm_room_baked.obj"
            output_usd="${output_dir}/pulm_room_baked.usd"
            ;;
        *)
            echo "Unknown room: ${room}" >&2
            exit 1
            ;;
    esac

    mkdir -p "${output_dir}"
    echo "Baking ${room} OR room OBJ..."
    "${blender_exe}" --background --python "${BAKE_SCRIPT}" -- \
        "${output_obj}" "${inputs[@]}" \
        --scale "${scale}" \
        --rot-quat "${rot[@]}" \
        --floor-to-z "${floor_to_z}" \
        --origin-mode "${origin_mode}"

    if [[ "${skip_usd}" == false ]]; then
        echo "Converting ${room} OR room OBJ to USD..."
        local collision_args=("--collision-approx" "${collision_approx}")
        if [[ -n "${collision_simplification_metric}" ]]; then
            collision_args+=("--collision-simplification-metric" "${collision_simplification_metric}")
        fi
        if [[ "${unlit_textures}" == true ]]; then
            collision_args+=("--unlit-textures")
        fi
        "${CONVERT_SCRIPT}" --no-decimate "${output_obj}" "${output_usd}" "${collision_args[@]}" "${converter_args[@]}"
    fi
}

case "${target}" in
    halo)
        bake_and_convert halo
        ;;
    pulm)
        bake_and_convert pulm
        ;;
    all)
        bake_and_convert halo
        bake_and_convert pulm
        ;;
esac
