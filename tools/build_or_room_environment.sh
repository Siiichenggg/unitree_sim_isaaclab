#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  tools/build_or_room_environment.sh [halo|pulm|all] [options] [-- Isaac Lab AppLauncher args]

Options:
  --skip-usd           Only bake OBJ/MTL files; skip OBJ -> USD conversion.
  --blender-exe PATH   Blender executable. Default: BLENDER_EXE, PATH, or /home/sicheng/blender/blender-4.2.1-linux-x64/blender.
  --origin-mode MODE   keep_xy or center_xy. Default: center_xy.
  --floor-to-z Z       Ground the baked mesh to this Z value. Default: 0.
  --collision-approx MODE
                      Collision approximation for generated USD. Default: mesh-simplification.
  --collision-simplification-metric VALUE
                      Optional PhysX simplification accuracy for mesh-simplification.
  --lowpoly-collision Generate final USD as high-poly visual plus hidden low-poly
                      collision mesh instead of putting collision on the visual mesh.
  --collision-target-faces COUNT
                      Low-poly collision target face count. Default: 25000.
  --collision-planar-angle-deg DEG
                      Planar dissolve angle before collision decimation. Default: 5.
  --flatten-collision Flatten the hidden low-poly collision reference into the final
                      room USD. Use with --lowpoly-collision to produce two self-contained USDs.
  --filter-low-collision-surfaces
                      Remove broad near-horizontal collision faces slightly above
                      the floor before attaching the hidden low-poly collision mesh.
                      Enabled by default with --lowpoly-collision.
  --no-filter-low-collision-surfaces
                      Disable low collision surface filtering.
  --low-collision-z-range MIN MAX
                      Z range for filtered low surfaces. Default: 0.04 0.20.
  --low-collision-min-area AREA
                      Minimum face area for low-surface filtering. Default: 0.04.
  --low-collision-min-xy-span SPAN
                      Minimum X/Y span for low-surface filtering. Default: 0.60.
  --low-collision-max-slope-deg DEG
                      Maximum slope from horizontal for low-surface filtering. Default: 15.
  --embed-environment Add room environment lights into the final USD.
  --lit-textures      Keep OR room textures as regular lit PBR materials.
                      Default: patch textures to emissive/unlit so the room displays texture color directly.

Examples:
  tools/build_or_room_environment.sh all
  tools/build_or_room_environment.sh all --lowpoly-collision --flatten-collision --embed-environment
  tools/build_or_room_environment.sh halo --skip-usd
  tools/build_or_room_environment.sh pulm -- --/app/window/enabled=false

This bakes OR-room scale and rotation in Blender, exports normalized OBJ assets,
then converts those OBJ files to USD with identity scale/rotation/translation.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OR_MODEL_DIR="${PROJECT_ROOT}/assets/objects/OR/Model"
BAKE_SCRIPT="${SCRIPT_DIR}/bake_or_scene_blender.py"
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_obj_to_usd.sh"
ATTACH_COLLISION_SCRIPT="${SCRIPT_DIR}/attach_lowpoly_collision_usd.py"
FILTER_COLLISION_SCRIPT="${SCRIPT_DIR}/filter_low_collision_surfaces_usd.py"
CONFIGURE_ENV_SCRIPT="${SCRIPT_DIR}/configure_or_room_environment_usd.py"
IMPORT_BLEND_CAMERAS_SCRIPT="${SCRIPT_DIR}/import_blender_cameras_to_or_usd.py"
ISAACLAB_PATH="${ISAACLAB_PATH:-/home/sicheng/IsaacLab}"
ISAACLAB_SH="${ISAACLAB_PATH}/isaaclab.sh"

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
lowpoly_collision=false
collision_target_faces="25000"
collision_planar_angle_deg="5"
flatten_collision=false
filter_low_collision_surfaces=true
low_collision_min_z="0.04"
low_collision_max_z="0.20"
low_collision_min_area="0.04"
low_collision_min_xy_span="0.60"
low_collision_max_slope_deg="15"
embed_environment=false
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
        --lowpoly-collision)
            lowpoly_collision=true
            shift
            ;;
        --collision-target-faces)
            collision_target_faces="$2"
            shift 2
            ;;
        --collision-target-faces=*)
            collision_target_faces="${1#*=}"
            shift
            ;;
        --collision-planar-angle-deg)
            collision_planar_angle_deg="$2"
            shift 2
            ;;
        --collision-planar-angle-deg=*)
            collision_planar_angle_deg="${1#*=}"
            shift
            ;;
        --flatten-collision)
            flatten_collision=true
            shift
            ;;
        --filter-low-collision-surfaces)
            filter_low_collision_surfaces=true
            shift
            ;;
        --no-filter-low-collision-surfaces)
            filter_low_collision_surfaces=false
            shift
            ;;
        --low-collision-z-range)
            low_collision_min_z="$2"
            low_collision_max_z="$3"
            shift 3
            ;;
        --low-collision-min-area)
            low_collision_min_area="$2"
            shift 2
            ;;
        --low-collision-min-area=*)
            low_collision_min_area="${1#*=}"
            shift
            ;;
        --low-collision-min-xy-span)
            low_collision_min_xy_span="$2"
            shift 2
            ;;
        --low-collision-min-xy-span=*)
            low_collision_min_xy_span="${1#*=}"
            shift
            ;;
        --low-collision-max-slope-deg)
            low_collision_max_slope_deg="$2"
            shift 2
            ;;
        --low-collision-max-slope-deg=*)
            low_collision_max_slope_deg="${1#*=}"
            shift
            ;;
        --embed-environment)
            embed_environment=true
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

if [[ "${lowpoly_collision}" == false && "${flatten_collision}" == true ]]; then
    echo "--flatten-collision requires --lowpoly-collision." >&2
    exit 1
fi

if [[ "${lowpoly_collision}" == true || "${embed_environment}" == true ]]; then
    if [[ ! -x "${ISAACLAB_SH}" ]]; then
        echo "Could not find executable Isaac Lab launcher: ${ISAACLAB_SH}" >&2
        echo "Set ISAACLAB_PATH to your Isaac Lab checkout." >&2
        exit 1
    fi
fi

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
    local scale output_dir output_obj output_usd collision_usd blend_scene blend_source_root
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
            collision_usd="${output_dir}/halo_room_collision_low.usd"
            blend_scene="${PROJECT_ROOT}/assets/objects/OR/humanoid_demo.blend"
            blend_source_root="halo_hole_fix_final_a"
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
            collision_usd="${output_dir}/pulm_room_collision_low.usd"
            blend_scene="${PROJECT_ROOT}/assets/objects/OR/pulm_room_SFV1.blend"
            blend_source_root="pulm_room"
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
        local texture_args=()
        if [[ "${unlit_textures}" == true ]]; then
            texture_args+=("--unlit-textures")
        fi

        if [[ "${lowpoly_collision}" == true ]]; then
            echo "Converting ${room} OR room visual OBJ to collision-free USD..."
            "${CONVERT_SCRIPT}" --no-decimate "${output_obj}" "${output_usd}" \
                --collision-approx disabled "${texture_args[@]}" "${converter_args[@]}"

            echo "Generating ${room} low-poly collision USD..."
            "${CONVERT_SCRIPT}" \
                --target-faces "${collision_target_faces}" \
                --planar-angle-deg "${collision_planar_angle_deg}" \
                --no-preserve-boundaries \
                --keep-decimated-obj \
                "${output_obj}" "${collision_usd}" \
                --collision-approx triangle-mesh \
                "${converter_args[@]}"

            if [[ "${filter_low_collision_surfaces}" == true ]]; then
                echo "Filtering ${room} low, broad collision surfaces..."
                TERM=xterm-256color "${ISAACLAB_SH}" -p "${FILTER_COLLISION_SCRIPT}" \
                    "${collision_usd}" \
                    --min-z "${low_collision_min_z}" \
                    --max-z "${low_collision_max_z}" \
                    --min-area "${low_collision_min_area}" \
                    --min-xy-span "${low_collision_min_xy_span}" \
                    --max-slope-deg "${low_collision_max_slope_deg}"
            fi

            echo "Attaching ${room} hidden low-poly collision to final USD..."
            local attach_args=()
            if [[ "${flatten_collision}" == true ]]; then
                attach_args+=("--flatten")
            fi
            TERM=xterm-256color "${ISAACLAB_SH}" -p "${ATTACH_COLLISION_SCRIPT}" \
                "${output_usd}" "${collision_usd}" "${attach_args[@]}"
        else
            echo "Converting ${room} OR room OBJ to USD..."
            local collision_args=("--collision-approx" "${collision_approx}")
            if [[ -n "${collision_simplification_metric}" ]]; then
                collision_args+=("--collision-simplification-metric" "${collision_simplification_metric}")
            fi
            "${CONVERT_SCRIPT}" --no-decimate "${output_obj}" "${output_usd}" \
                "${collision_args[@]}" "${texture_args[@]}" "${converter_args[@]}"
        fi

        if [[ "${embed_environment}" == true ]]; then
            echo "Embedding ${room} OR room lights into final USD..."
            TERM=xterm-256color "${ISAACLAB_SH}" -p "${CONFIGURE_ENV_SCRIPT}" "${output_usd}"

            echo "Importing ${room} Blender cameras into final USD..."
            "${blender_exe}" --background "${blend_scene}" --python "${IMPORT_BLEND_CAMERAS_SCRIPT}" -- \
                "${output_usd}" "${inputs[@]}" \
                --scale "${scale}" \
                --rot-quat "${rot[@]}" \
                --floor-to-z "${floor_to_z}" \
                --origin-mode "${origin_mode}" \
                --source-root-object "${blend_source_root}"
        fi
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
