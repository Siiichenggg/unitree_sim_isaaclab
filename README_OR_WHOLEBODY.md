# OR Wholebody Control README

This file records the commands for running the G1 Dex1 wholebody policy inside
the operating-room scenes.

## Quick Start: Complete Terminal Commands

Use two terminals. Do not run A* navigation and keyboard control at the same
time because both publish to `rt/run_command/cmd`.

Terminal 1, start Halo OR in Isaac Lab:

```bash
conda activate unitree_sim_env
cd /home/sicheng/unitree_sim_isaaclab

python sim_main.py \
  --task Isaac-OR-HaloRoom-G129-Dex1-Wholebody \
  --robot_type g129 \
  --enable_dex1_dds \
  --enable_cameras \
  --camera_include world_camera \
  --camera_exclude "" \
  --device cuda:0
```

Terminal 1, start Pulm OR instead:

```bash
conda activate unitree_sim_env
cd /home/sicheng/unitree_sim_isaaclab

python sim_main.py \
  --task Isaac-OR-PulmRoom-G129-Dex1-Wholebody \
  --robot_type g129 \
  --enable_dex1_dds \
  --enable_cameras \
  --camera_include world_camera \
  --camera_exclude "" \
  --device cuda:0
```

Terminal 2, start A* navigation for Halo OR:

```bash
conda activate unitree_sim_env
cd /home/sicheng/unitree_sim_isaaclab

python tools/astar_nav_dds.py \
  --scene halo \
  --goal 0.0 0.0
```

Terminal 2, start A* navigation for Pulm OR:

```bash
conda activate unitree_sim_env
cd /home/sicheng/unitree_sim_isaaclab

python tools/astar_nav_dds.py \
  --scene pulm \
  --goal 2.5 -1.2
```

Optional, check the Halo A* path without moving the robot:

```bash
conda activate unitree_sim_env
cd /home/sicheng/unitree_sim_isaaclab

python tools/astar_nav_dds.py \
  --scene halo \
  --start 1.35 -1.45 \
  --goal 0.0 0.0 \
  --plan-only
```

Optional, use manual keyboard control instead of A*:

```bash
conda activate unitree_sim_env
cd /home/sicheng/unitree_sim_isaaclab

python send_commands_keyboard.py --mode terminal
```

## 1. Start Isaac Lab

Open terminal 1:

```bash
conda activate unitree_sim_env
cd /home/sicheng/unitree_sim_isaaclab
```

Halo OR:

```bash
python sim_main.py \
  --task Isaac-OR-HaloRoom-G129-Dex1-Wholebody \
  --robot_type g129 \
  --enable_dex1_dds \
  --enable_cameras \
  --camera_include world_camera \
  --camera_exclude "" \
  --device cuda:0
```

Pulm OR:

```bash
python sim_main.py \
  --task Isaac-OR-PulmRoom-G129-Dex1-Wholebody \
  --robot_type g129 \
  --enable_dex1_dds \
  --enable_cameras \
  --camera_include world_camera \
  --camera_exclude "" \
  --device cuda:0
```

The `Wholebody` task name is important. It makes `sim_main.py` switch to the
Unitree wholebody policy action provider and load:

```bash
assets/model/policy.onnx
```

## 2. Start Keyboard Control

Open terminal 2:

```bash
conda activate unitree_sim_env
cd /home/sicheng/unitree_sim_isaaclab
python send_commands_keyboard.py --mode terminal
```

The expected log is:

```text
keyboard listener started in terminal mode...
focus this terminal and press W/A/S/D/Z/X/C directly; do not press Enter
```

Keep focus on terminal 2 and press keys directly. Do not type `w` then Enter.

Controls:

```text
W/S: forward/backward
A/D: left/right
Z/X: turn left/right
C: crouch
Q: quit keyboard control
```

When the keyboard input works, terminal 2 should print lines like:

```text
[KEY] W: press
commands: [0.05, -0.0, -0.0, 0.8]
```

## 3. Optional Gamepad Control

```bash
conda activate unitree_sim_env
cd /home/sicheng/unitree_sim_isaaclab
python send_commands_8bit.py
```

## 4. A* Navigation

Start Isaac Lab first with a `Wholebody` OR task. Then open another terminal:

```bash
conda activate unitree_sim_env
cd /home/sicheng/unitree_sim_isaaclab
python tools/astar_nav_dds.py --scene halo --goal 0.0 0.0
```

Pulm OR:

```bash
python tools/astar_nav_dds.py --scene pulm --goal 2.5 -1.2
```

The script subscribes to `rt/sim_state` for the current robot pose, plans on a
2D A* grid generated from the baked OR OBJ, and publishes velocity commands to:

```text
rt/run_command/cmd
```

Check the path without moving the robot:

```bash
python tools/astar_nav_dds.py --scene halo --start 1.35 -1.45 --goal 0.0 0.0 --plan-only
```

Useful options:

```text
--goal X Y                         target position in world coordinates
--start X Y                        planning start if sim_state is unavailable
--rect xmin,ymin,xmax,ymax         add an obstacle rectangle
--resolution 0.10                  A* grid resolution
--robot-radius 0.20                obstacle inflation radius
--clearance 0.05                   extra obstacle clearance
--max-speed 0.35                   max forward command
--max-yaw-rate 0.8                 max yaw command
--allow-strafe                     allow lateral velocity commands
```

## 5. Common Issues

If Isaac says a camera was spawned without `--enable_cameras`, use the launch
commands above with `--enable_cameras`.

If keyboard input has no effect, make sure:

- You are running `python send_commands_keyboard.py --mode terminal`.
- Terminal 2 has focus.
- You press keys directly without Enter.
- You see `[KEY] ...` and `commands: ...` in terminal 2.

If imports such as `pxr` or `omni` fail, make sure you are using:

```bash
conda activate unitree_sim_env
```

If the robot does not use the locomotion policy, make sure the task id contains
`Wholebody`, for example:

```bash
Isaac-OR-HaloRoom-G129-Dex1-Wholebody
```

If A* fails to find a path, try:

```bash
python tools/astar_nav_dds.py --scene halo --goal 0.0 0.0 --robot-radius 0.15 --clearance 0.02
```

or add/remove coarse obstacle rectangles with `--rect`.

## 6. Rebuild OR Assets

The OR meshes are baked in Blender and converted to USD offline:

```bash
tools/build_or_room_assets.sh all
```
