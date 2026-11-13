# Alloy workbench for LiveKit Portal

The robot publishes synchronized RGB frames from the overhead, left wrist, and right wrist cameras with a 14-field joint state. An operator can send positions in the same field order. The glass-water preset runs physics and rendering concurrently on one NVIDIA GPU.

## Install

Use Python 3.11 or 3.12 in a virtual environment. From the checkout root:

```sh
python -m pip install -e '.[import]'
python -m pip install -r examples/livekit/requirements.txt
```

For the GPU preset, use Linux with a supported NVIDIA driver, CUDA MPS, and a CUDA toolkit compatible with the driver. Build the OptiX helper after accepting the NVIDIA OptiX SDK license:

```sh
python -m alloy.camera.optix build --accept-nvidia-optix-license
```

Use `CUDA_HOME` and `CXX` to select the toolkit and host compiler if needed. If the driver rejects the compiled OptiX IR, choose a toolkit supported by the installed driver and rebuild with `--force`.

## Try the workbench locally

```sh
python -m examples.livekit.yam_portal smoke --out /tmp/alloy-yam
python -m examples.livekit.yam_portal viewer --port 8086
```

See [the shared workbench scene](../scenes/yam_workbench/README.md) for the bundled assets, hardware profile, and scene helpers.

## Run the three-camera fluid robot

Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` or pass a dotenv file. For a deployment with pre-minted credentials, set `LIVEKIT_URL` and `LIVEKIT_TOKEN`. Use a distinct token for each participant in the same room.

```sh
bash examples/livekit/run_fluid_mps_linux.sh --env-file /path/to/livekit.env --room alloy-yam
```

The launcher uses the active Python or `ALLOY_PYTHON`, starts an isolated MPS server, and targets three 640×480 streams at 30 FPS. The default preview uses 6 mm water spacing, one sample per pixel with OptiX denoising, and 30 ms snapshot delay. Pass `--rig my.rig.json` or `--look my.look.json` to override the hardware or materials. Pass `--liquid-accuracy default` for 4 mm water or `--snapshot-delay-ms 120` to give interpolation more time during an impact. Interactive coupling trades pressure and contact accuracy for speed. Publishing FPS and simulated seconds per wall second are reported separately.

The standard-coupling, single-process alternative uses two samples per pixel and overlaps denoising and readback with physics. It advances an average of 3.75 ms of simulation per video frame.

```sh
ALLOY_CAMERA_WF_SPEC=0 python -m examples.livekit.yam_portal robot \
  --device cuda:0 --scene glass-water --cup-collision decompose \
  --width 640 --height 480 --fps 30 --spp 2 --control-dt .0025 --steps-per-frame 1.5 \
  --camera-backend optix --denoise optix --joined-cameras --pipeline-render \
  --env-file /path/to/livekit.env --room alloy-yam
```

For the dry workbench, use the `robot` module command without the fluid and GPU options. For a movable mug, beaker, and catch basin, pass `--pourable-cup` to the MPS launcher. `--household-objects` adds the sponge, spoon, tomato, and bottle.

## Verify reception

Run a separate operator in the same room. The command below checks all three decoded tracks, their resolution, and the synchronized observation rate. `--move` also verifies that joint targets are reaching the simulation. Omit it for a receive-only check.

```sh
python -m examples.livekit.yam_portal operator --env-file /path/to/livekit.env --room alloy-yam \
  --width 640 --height 480 --fps 30 --seconds 30 --min-received-fps 29.5 --move
```

`--demo-splash-after 15` on the MPS publisher adds a block drop to a test run. Use a separate room when testing. To benchmark local physics and rendering independently of Portal, run `python -m examples.livekit.benchmark_yam --help`.

The wire fields are `left_joint_1.pos` through `left_joint_6.pos`, `left_gripper.pos`, then the equivalent right-arm fields. Arm positions are radians. Gripper values are normalized from closed (0) to open (1). The VP8 tracks are `overhead_camera`, `left_wrist_camera`, and `right_wrist_camera`. The `sim.motion` RPC with payload `reset` restores the settled scene and releases control. Use `python -m examples.livekit.yam_portal reset` with the same connection settings.
