# SPDX-License-Identifier: Apache-2.0
"""Dual YAM simulation, viewer, and LiveKit Portal robot/operator smoke test."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from fractions import Fraction
import math
import json
import os
import sys
from pathlib import Path
import time

from alloy.workbench import FIELDS, VIDEOS
from alloy.workbench.simulation import Simulation



def config(cls, args):
    from livekit.portal import DType, VideoCodec
    cfg = cls(args.room)
    for name in VIDEOS:
        cfg.add_video(name, codec=VideoCodec.VP8)
    schema = [(field, DType.F32) for field in FIELDS]
    cfg.add_state_typed(schema)
    cfg.add_action_typed(schema)
    cfg.set_fps(args.fps)
    return cfg


def credentials(args, identity):
    from livekit import api
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    url = os.environ.get("LIVEKIT_URL")
    token = os.environ.get("LIVEKIT_TOKEN")
    if not url:
        raise ValueError("Set LIVEKIT_URL (or pass --env-file)")
    if token:
        return url, token
    token = (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
             .with_identity(identity).with_ttl(timedelta(hours=1))
             .with_grants(api.VideoGrants(room_join=True, room=args.room, can_publish=True,
                                         can_subscribe=True, can_publish_data=True,
                                         can_update_own_metadata=True)).to_jwt())
    return url, token


async def robot_main(args):
    from livekit.portal import Robot, RobotConfig, RpcError
    url, token = credentials(args, "alloy-yam-robot")
    loop = asyncio.get_running_loop()
    simulation_cls = Simulation
    if args.split_render:
        from alloy.workbench.split_render import SplitSimulation

        simulation_cls = SplitSimulation
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="alloy") as pool:
        async def work(fn, *values):
            return await loop.run_in_executor(pool, fn, *values)
        sim = await work(simulation_cls, args)
        try:
            robot = Robot(config(RobotConfig, args))
        except BaseException:
            await work(sim.close)
            raise
        latest, received, count = None, 0., 0
        pending_reset = None
        reset_cutoff = 0

        def invalidate(_=None):
            nonlocal latest, received
            latest, received = None, 0.

        def action_cb(action):
            nonlocal latest, received, count
            if pending_reset is not None or action.sender != robot.active_operator():
                return
            anchor = action.in_reply_to_ts_us
            if action.timestamp_us < reset_cutoff or (anchor is not None and anchor < reset_cutoff):
                return
            try:
                latest = sim.validate(action.values)
            except ValueError as exc:
                print(f"Rejected action: {exc}", flush=True)
                return
            received, count = time.monotonic(), count + 1

        async def request_motion(data):
            nonlocal pending_reset
            if data.payload != "reset":
                raise RpcError.Error(code=1500, message="Alloy sim.motion expects reset", data=None)
            if pending_reset is not None:
                raise RpcError.Error(code=1500, message="A reset is already in progress", data=None)
            pending_reset = loop.create_future()
            invalidate()
            error = await asyncio.shield(pending_reset)
            if error is not None:
                raise RpcError.Error(code=1500, message=error, data=None)
            return "reset complete; control released"

        robot.on_action(action_cb)
        robot.on_active_operator_changed(invalidate)
        robot.on_operator_left(lambda who: invalidate() if who == robot.active_operator() else None)
        robot.register_rpc_method("sim.motion", request_motion)
        try:
            if args.prewarm_splash and not args.split_render:
                await work(sim.prewarm_splash)
            await work(sim.tick, None, 1)  # Warm renderer before joining the room.
            await work(sim.check)
            await robot.connect(url, token)
            if args.split_render:
                sim.start()
            print(f"Connected robot in room {args.room}; waiting for operator", flush=True)
            start, report = time.monotonic(), time.monotonic()
            next_frame = start
            frame_period = 1 / args.fps
            action_timeout = max(1.0, 3.0 / args.fps)
            step_remainder = frames_since_report = 0
            control_hz = round(1 / sim.sim.dt)
            step_rate = (Fraction(str(args.steps_per_frame)) if args.steps_per_frame is not None
                         else Fraction(control_hz, args.fps))
            last_sim_time = sim.sim.t
            pending_timestamp = None
            while not args.seconds or time.monotonic() - start < args.seconds:
                tick = time.monotonic()
                reset = pending_reset
                if reset is not None:
                    await robot.set_active_operator(None)
                    await work(sim.reset)
                    pending_timestamp = None
                command = latest if tick - received < action_timeout else None
                if latest is not None and command is None:
                    invalidate()
                    await robot.set_active_operator(None)
                # Preserve the average step count when it is fractional.
                steps, step_remainder = divmod(step_rate.numerator + step_remainder, step_rate.denominator)
                rendering = None
                if args.pipeline_render and reset is None:
                    state, observation = await work(sim.advance_pipeline, command, steps)
                    rendering = loop.run_in_executor(pool, sim.enqueue_frame, state)
                    timestamp = pending_timestamp
                else:
                    tick_args = (command, steps, received + action_timeout) if args.split_render else (command, steps)
                    observation = await work(sim.tick, *tick_args)
                    timestamp = robot.now_us()
                if reset is not None:
                    if observation is None:
                        raise RuntimeError("Alloy reset did not produce an observation")
                    # Discard claims made during reset and reject actions anchored to older frames.
                    await robot.set_active_operator(None)
                    invalidate()
                    timestamp = reset_cutoff = robot.now_us()
                try:
                    if observation is not None:
                        state, frames = observation
                        for name, rgb in frames.items():
                            if name in VIDEOS:
                                robot.send_video_frame(name, rgb, timestamp_us=timestamp)
                        robot.send_state(state, timestamp_us=timestamp)
                        frames_since_report += 1
                finally:
                    if rendering is not None:
                        await rendering
                        pending_timestamp = robot.now_us()
                if reset is not None:
                    pending_reset = None
                    reset.set_result(None)
                now = time.monotonic()
                if now - report >= 5:
                    elapsed = now - report
                    stages = " ".join(f"{name}={value:.1f}" for name, value in sim.timings.items())
                    print(f"t={sim.sim.t:.2f}s actions={count} publisher_fps={frames_since_report/elapsed:.2f} "
                          f"sim_rate={(sim.sim.t-last_sim_time)/elapsed:.3f} frame_cycle_ms={(now-tick)*1000:.0f} "
                          f"{stages}", flush=True)
                    report, last_sim_time, frames_since_report = now, sim.sim.t, 0
                    await work(sim.check)
                next_frame += frame_period
                now = time.monotonic()
                if next_frame < now - frame_period:
                    next_frame = now
                await asyncio.sleep(max(0., next_frame - now))
            await work(sim.check)
            print(f"Robot finished: accepted_actions={count}", flush=True)
        finally:
            if pending_reset is not None:
                pending_reset.set_result("The simulator stopped before reset completed")
            try:
                await robot.disconnect()
            finally:
                robot.close()
                await work(sim.close)


async def operator_main(args):
    from livekit.portal import Operator, OperatorConfig, frame_bytes_to_numpy_rgb
    url, token = credentials(args, "alloy-yam-test-operator")
    op = Operator(config(OperatorConfig, args))
    seen, baseline, lo, hi = 0, None, math.inf, -math.inf
    first_observation, last_observation = None, None
    incomplete = 0
    expected = set(VIDEOS)
    per_camera = dict.fromkeys(VIDEOS, 0)
    sizes = {name: set() for name in VIDEOS}
    start = time.monotonic()

    def observe(obs):
        nonlocal seen, baseline, lo, hi, first_observation, last_observation, incomplete
        names = set(obs.frames)
        for name in names & expected:
            frame = obs.frames[name]
            per_camera[name] += 1
            sizes[name].add((frame.width, frame.height))
        if names != expected:
            incomplete += 1
            return
        last_observation = time.monotonic()
        if first_observation is None:
            first_observation = last_observation
        seen += 1
        if baseline is None:
            baseline = dict(obs.state)
        q = obs.state["left_joint_1.pos"]
        lo, hi = min(lo, q), max(hi, q)
        if seen == 1 and args.out:
            from PIL import Image
            args.out.mkdir(parents=True, exist_ok=True)
            for name, frame in obs.frames.items():
                Image.fromarray(frame_bytes_to_numpy_rgb(bytes(frame.data), frame.width, frame.height)).save(args.out / f"received_{name}.png")
        if args.move:
            values = dict(baseline)
            values["left_joint_1.pos"] += .12 * math.sin((time.monotonic() - start) * .7)
            values["right_joint_1.pos"] -= .12 * math.sin((time.monotonic() - start) * .7)
            op.send_action(values, in_reply_to_ts_us=obs.timestamp_us)
        if seen % args.fps == 0:
            print(f"observations={seen} cameras={len(obs.frames)} left_joint_1={q:.4f}", flush=True)

    op.on_observation(observe)
    try:
        await op.connect(url, token)
        if args.move:
            for attempt in range(20):
                try:
                    await op.set_active_operator(op.local_identity())
                    break
                except Exception:
                    if attempt == 19:
                        raise
                    await asyncio.sleep(.5)
        print(f"Operator connected; motion={'on' if args.move else 'off'}", flush=True)
        await asyncio.sleep(args.seconds or 30)
        elapsed = (last_observation - first_observation) if seen > 1 else 0.
        rate = (seen - 1) / elapsed if elapsed else 0.
        print(f"RESULT observations={seen} observation_elapsed_s={elapsed:.3f} received_fps={rate:.2f} "
              f"measured_joint_span_rad={hi-lo:.5f}", flush=True)
        for name in VIDEOS:
            print(f"CAMERA {name} received={per_camera[name]} sizes={sorted(sizes[name])}", flush=True)
        if args.min_received_fps is not None and (rate < args.min_received_fps or incomplete
                or any(sizes[name] != {(args.width, args.height)} for name in VIDEOS)):
            raise RuntimeError(f"Portal reception failed: {rate:.2f} FPS, {incomplete} incomplete observations; "
                               f"expected at least {args.min_received_fps:g} FPS with three {args.width}×{args.height} cameras")
        if not seen or (args.move and hi - lo < .03):
            raise RuntimeError("No synchronized observations or commanded motion not observed")
    finally:
        try:
            if args.move and op.active_operator() == op.local_identity():
                await op.set_active_operator(None)
        finally:
            await op.disconnect()
            op.close()


async def reset_main(args):
    from livekit.portal import Operator, OperatorConfig

    url, token = credentials(args, "alloy-yam-reset")
    op = Operator(config(OperatorConfig, args))
    try:
        await op.connect(url, token)
        print(await op.perform_rpc("sim.motion", "reset"), flush=True)
    finally:
        try:
            await op.disconnect()
        finally:
            op.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "viewer", "robot", "operator", "reset"))
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--rig", type=Path, help="dual-arm model, joint mapping, home pose and camera profile")
    parser.add_argument("--room", default="alloy-yam")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=int, default=10, help="Video frame target (1–50)")
    parser.add_argument("--control-dt", type=float, default=0.01,
                        choices=(0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0005),
                        help="Seconds of simulated time per control step")
    parser.add_argument("--steps-per-frame", type=float,
                        help="Average physics steps per video frame, e.g. 1 or 1.5. Defaults to real-time scheduling.")
    parser.add_argument("--physics-budget-ms", type=float,
                        help="Adjust the physics step count to fit this per-frame budget, with at least one step")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--spp", type=int, default=1)
    parser.add_argument("--render-quality", choices=("auto", "fast", "full"), default="auto",
                        help="fast: diffuse colours/textures and studio lighting; full: PBR look materials")
    parser.add_argument("--denoise", choices=("off", "oidn", "optix", "learned", "auto"), default="off",
                        help="Denoise camera frames before encoding; oidn requires its cached CUDA library")
    parser.add_argument("--scene", choices=("workbench", "glass-water"), default="workbench")
    parser.add_argument("--liquid-cfl", type=float, help="Maximum liquid travel per step in particle spacings")
    parser.add_argument("--liquid-accuracy", choices=("default", "super", "fast"), default="default",
                        help="Liquid particle spacing: default 4 mm, super 2 mm, fast 6 mm")
    parser.add_argument("--prewarm-splash", action="store_true",
                        help="Warm CUDA graphs for a block drop before going live with the glass-water scene")
    parser.add_argument("--cup-collision", choices=("sdf", "decompose", "surface"), default="sdf",
                        help="Collision geometry for glass-water. Surface uses the glass mesh for particle contacts")
    parser.add_argument("--pourable-cup", action="store_true",
                        help="Use a movable glass mug, an empty glass beaker, and a catch basin")
    parser.add_argument("--household-objects", action="store_true",
                        help="Add a movable kitchen sponge, spoon, tomato, and water bottle")
    parser.add_argument("--joined-cameras", action="store_true", help="Trace the full-material views in one ray batch")
    parser.add_argument("--pipeline-render", action="store_true",
                        help="Overlap the current video frame's OptiX denoising with the next physics step")
    parser.add_argument("--split-render", action="store_true",
                        help="Step physics continuously in a second CUDA process under MPS")
    parser.add_argument("--realtime-fluid", action="store_true",
                        help="Use the interactive fluid solver and pace split physics against wall time")
    parser.add_argument("--snapshot-delay-ms", type=float, default=120.,
                        help="Render interpolated split snapshots this far behind the physics worker")
    parser.add_argument("--demo-splash-after", type=float, help="Drop the workbench block into the cup after this delay")
    parser.add_argument("--camera-backend", choices=("warp", "optix", "auto", "wavefront-warp"),
                        help="Ray tracing backend; optix uses RT cores after its helper is built")
    parser.add_argument("--look", type=Path, help="Full-material look JSON; also adds workbench scenery")
    parser.add_argument("--seconds", type=float, default=0, help="Wall time for Portal; 0 runs robot indefinitely")
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--move", action="store_true", help="Operator sends a small joint-1 sine wave to each simulated arm")
    parser.add_argument("--min-received-fps", type=float,
                        help="Operator test: fail below this rate or if any synchronized camera is missing")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("Video width and height must be positive and even")
    if args.seconds < 0 or args.spp < 1:
        parser.error("seconds must be nonnegative and spp must be positive")
    if args.min_received_fps is not None and (
            args.mode != "operator" or not math.isfinite(args.min_received_fps) or args.min_received_fps <= 0):
        parser.error("min-received-fps needs operator mode and a positive finite rate")
    if not 1 <= args.fps <= 50:
        parser.error("fps must be between 1 and 50")
    if args.steps_per_frame is not None and (not math.isfinite(args.steps_per_frame) or args.steps_per_frame < 1):
        parser.error("steps-per-frame must be finite and at least 1")
    if args.joined_cameras and (args.render_quality == "fast" or (args.scene != "glass-water"
                                and not args.look and args.render_quality != "full")):
        parser.error("joined-cameras requires the full-material Alloy renderer")
    if args.pipeline_render and (args.mode != "robot" or not args.joined_cameras or args.denoise != "optix"
                                 or args.viewer or not args.device.startswith("cuda")):
        parser.error("pipeline-render requires robot mode, CUDA, --joined-cameras --denoise optix --no-viewer")
    if args.physics_budget_ms is not None and (not args.pipeline_render or not math.isfinite(args.physics_budget_ms)
                                                or args.physics_budget_ms <= 0):
        parser.error("physics-budget-ms needs --pipeline-render and a finite positive budget")
    if args.liquid_cfl is not None and (args.scene != "glass-water" or not 0 < args.liquid_cfl <= 1):
        parser.error("liquid-cfl needs the glass-water scene and a value in (0, 1]")
    if args.pourable_cup and (args.scene != "glass-water" or args.cup_collision != "surface"):
        parser.error("pourable-cup needs glass-water with --cup-collision surface")
    if args.scene == "glass-water" and args.render_quality == "fast":
        parser.error("glass-water needs full materials for transparent glass and water")
    if args.split_render:
        if (args.mode != "robot" or args.scene != "glass-water" or not args.joined_cameras
                or args.camera_backend != "optix" or args.denoise != "optix" or args.viewer
                or not args.device.startswith("cuda") or args.pipeline_render or args.steps_per_frame is not None):
            parser.error("split-render needs a CUDA glass-water robot with joined OptiX cameras and no viewer")
        if sys.platform != "linux" or not os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
            parser.error("split-render needs a Linux CUDA MPS session. Use run_fluid_mps_linux.sh")
        if not math.isfinite(args.snapshot_delay_ms) or args.snapshot_delay_ms < 0:
            parser.error("snapshot-delay-ms must be finite and nonnegative")
    if args.realtime_fluid and (not args.split_render or args.cup_collision != "surface"):
        parser.error("realtime-fluid needs --split-render and --cup-collision surface")
    if args.prewarm_splash and (args.scene != "glass-water" or not (args.pipeline_render or args.split_render)):
        parser.error("prewarm-splash needs the glass-water scene and split or pipelined rendering")
    if args.demo_splash_after is not None and (not args.split_render or not math.isfinite(args.demo_splash_after)
                                               or args.demo_splash_after < 0):
        parser.error("demo-splash-after needs --split-render and a finite, nonnegative delay")
    if args.mode in ("robot", "operator", "reset"):
        asyncio.run({"robot": robot_main, "operator": operator_main, "reset": reset_main}[args.mode](args))
        return
    sim = Simulation(args)
    try:
        if args.mode == "viewer":
            sim.view.run(seconds=args.seconds or None)
        else:
            t = time.perf_counter()
            state, frames = sim.tick(sim.target, 10)
            print(f"14 joints: {state}; first camera batch: {time.perf_counter()-t:.2f}s", flush=True)
            if args.out:
                from PIL import Image
                args.out.mkdir(parents=True, exist_ok=True)
                for name, rgb in frames.items():
                    Image.fromarray(rgb).save(args.out / f"{name}.png")
                if sim.fluid_scene:
                    import alloy
                    detail = alloy.Camera("cup_detail", width=640, height=480, spp=args.spp,
                                          look_at=((.70, -.23, .17), (.52, 0, .035)), fov=38,
                                          look=sim.look, quality="full", backend=args.camera_backend,
                                          denoise=args.denoise)
                    Image.fromarray(sim.sim.render(detail)[0]).save(args.out / "cup_detail.png")
            t = time.perf_counter()
            sim.tick(sim.target, 10)
            print(f"Warm physics + {len(frames)} cameras: {time.perf_counter()-t:.3f}s", flush=True)
            print(f"Stage timings: {sim.timings}", flush=True)
        sim.check()
    finally:
        sim.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
