# SPDX-License-Identifier: Apache-2.0
"""Reusable Alloy dual-arm workbench simulator."""

from __future__ import annotations

import math
import time

import numpy as np

from . import FIELDS, VIDEOS, Rig, add_workbench, apply_fast_look, load_look
from .glassware import in_cup


class Simulation:
    """All Alloy calls stay on one worker; the network event loop never steps physics."""

    def __init__(self, args, *, physics_only=False, render_only=False):
        import alloy
        self.args = args
        self.rig = Rig.load(getattr(args, "rig", None))
        self.look = load_look(args.look) if args.look or args.scene == "glass-water" else None
        self.fluid_scene = args.scene == "glass-water"
        self.pourable_cup = args.pourable_cup
        quality = "full" if self.fluid_scene else args.render_quality
        mjcf, self.limits = self.rig.mjcf(interactive=args.realtime_fluid)
        self.joints = {field: f"yam/{name}" for field, name in self.rig.joints.items()}
        scene = alloy.Scene(dt=args.control_dt, liquid_cfl=args.liquid_cfl, liquid_accuracy=args.liquid_accuracy)
        scene.add_robot("yam", mjcf=mjcf)
        self.cup_origin = add_workbench(scene, self.look, scenery=self.look is not None,
                                        cup_collision=args.cup_collision if self.fluid_scene else None,
                                        pourable=self.pourable_cup, household=args.household_objects)
        if quality == "fast" and self.look:
            apply_fast_look(scene, self.look)
        print(f"Building dual YAM on {args.device}; first use may compile kernels for several minutes", flush=True)
        self.sim = alloy.build(scene, device=args.device,
                               liquid_coupling="interactive" if args.realtime_fluid else "standard",
                               contact_capacity=64 if args.realtime_fluid else None)
        if not render_only:
            home = self.validate(self.rig.home)
            targets = self.to_targets(home)
            self.sim.run(2., record=[], targets={k: (0., v) for k, v in targets.items()})
            self.sim.set_targets(targets)
            self.sim.step(round(0.3 / self.sim.dt))
            print("Home pose ready", flush=True)
        self.target = self.read()
        self.sim.set_targets(self.to_targets(self.target))
        if not render_only:
            self.sim.settle(0)  # Reset to the settled workbench and home pose.
        self.view = None
        self.pending_frame = None
        self.physics_budget_ms = getattr(args, "physics_budget_ms", None)
        self.step_estimate_ms = None
        if physics_only:
            return
        common = dict(width=args.width, height=args.height, spp=args.spp,
                      look=None if quality == "fast" else self.look, quality=quality,
                      denoise=args.denoise, backend=args.camera_backend)
        self.cameras = {name: alloy.Camera(name, **common, **self.rig.cameras[name]) for name in VIDEOS}
        self.camera_batch = None
        if args.joined_cameras:
            from alloy.camera import PbrCameraBatch

            self.camera_batch = PbrCameraBatch(self.cameras)
        if args.viewer or args.mode == "viewer":
            import alloy.viewer
            self.view = alloy.viewer.launch(self.sim, host="127.0.0.1", port=args.port,
                                           joints=args.mode == "viewer", contacts=False)
            print(f"Viewer: {self.view.url}", flush=True)

    def read(self):
        state = self.sim.state(list(self.joints.values()), world=0)
        return {field: float(state[f"{joint}.q"]) / self.rig.scale(field)
                for field, joint in self.joints.items()}

    def to_targets(self, values):
        return {self.joints[k]: v * self.rig.scale(k) for k, v in values.items()}

    def validate(self, values):
        if set(values) != set(FIELDS):
            raise ValueError("Action must contain exactly the 14 YAM fields")
        if not all(math.isfinite(float(v)) for v in values.values()):
            raise ValueError("Non-finite joint command")
        return {k: float(np.clip(v, *self.limits[k])) for k, v in values.items()}

    def hold(self):
        """Hold the measured pose when control changes hands."""
        self.target = self.validate(self.read())
        self.sim.set_targets(self.to_targets(self.target))

    def tick(self, action, steps, render=True, budget_ms=None):
        start = time.perf_counter()
        desired = self.validate(action) if action is not None else self.target
        # Hold the last ramped target after timeout; ramp new commands at 1 rad/s or opening/s.
        slew = self.sim.dt
        for index in range(steps):
            if budget_ms is not None:
                step_start = time.perf_counter()
                if index and (step_start - start) * 1000 + self.step_estimate_ms >= budget_ms:
                    break
            changed = {
                field: current + min(slew, max(-slew, desired[field] - current))
                for field, current in self.target.items()
                if desired[field] != current
            }
            if changed:
                self.sim.set_targets(self.to_targets(changed))
                self.target.update(changed)
            self.sim.step(1)
            if budget_ms is not None:
                step_ms = (time.perf_counter() - step_start) * 1000
                # React immediately to an expensive step and restore extra steps gradually.
                self.step_estimate_ms = max(step_ms, 0.85 * (self.step_estimate_ms or step_ms))
        state = self.read()
        physics_done = time.perf_counter()
        frames = self.render() if render else {}
        render_done = time.perf_counter()
        if self.view:
            self.view.sync()
        self.timings = {"physics_ms": (physics_done - start) * 1000,
                        "render_ms": (render_done - physics_done) * 1000,
                        "viewer_ms": (time.perf_counter() - render_done) * 1000}
        return state, frames

    def advance_pipeline(self, action, steps):
        """Step while the preceding frame denoises, then collect its matching image and state."""
        state, _ = self.tick(action, steps, render=False, budget_ms=self.physics_budget_ms)
        return state, self.finish_frame()

    def reset(self):
        self.finish_frame()  # Discard any pre-reset frame still in the denoiser.
        self.sim.reset(keep_coupled_graphs=self.fluid_scene)
        self.target = self.read()
        self.sim.set_targets(self.to_targets(self.target))

    def prewarm_splash(self):
        """Warm the GPU graphs for a representative splash, then restore the settled scene."""
        heights = (.16, .14, .18) if self.args.realtime_fluid else (.16,)
        for height in heights:
            try:
                self.sim.set_pose("block", pos=(*self.cup_origin[:2], height), world=0)
                self.sim.step(round((.48 if self.args.realtime_fluid else .31) / self.sim.dt))
            finally:
                self.reset()
        print(f"Splash warmup finished; GPU graph resets kept={self.sim.report()['graph'].get('reset_kept', 0)}",
              flush=True)

    def enqueue_frame(self, state):
        start = time.perf_counter()
        self.pending_frame = state, self.camera_batch.enqueue(self.sim)
        self.timings["render_ms"] = (time.perf_counter() - start) * 1000

    def finish_frame(self):
        if self.pending_frame is None:
            return None
        state, finish = self.pending_frame
        frames = {name: np.ascontiguousarray(rgb[0]) for name, rgb in finish().items()}
        self.pending_frame = None
        return state, frames

    def render(self):
        from alloy.camera import render_cameras
        frames = (self.camera_batch.render(self.sim) if self.camera_batch is not None
                  else render_cameras(self.sim, self.cameras))
        return {name: np.ascontiguousarray(rgb[0]) for name, rgb in frames.items()}

    def check(self):
        started = time.monotonic()
        result = self.sim.check()
        check_ms = 1000 * (time.monotonic() - started)
        print(f"Simulation t={self.sim.t:.2f}s health_ok={result['ok']} "
              f"check_ms={check_ms:.1f} problems={result['problems']}", flush=True)
        if self.fluid_scene:
            state = self.sim.state(["water", "cup"] if self.pourable_cup else ["water"], world=0)
            points = state["water.points"]
            inside = in_cup(points, state.get("cup.pos", self.cup_origin),
                            state.get("cup.quat_xyzw", (0., 0., 0., 1.)))
            print(f"Water: particles={len(points)} volume_ml={self.sim.liquid_volume('water')*1e6:.1f} "
                  f"in_cup={inside.mean():.3f} retired={self.sim.health()['retired_particles']}", flush=True)
        if not result["ok"]:
            raise RuntimeError("Alloy health check failed")

    def close(self):
        try:
            self.finish_frame()
            if self.view:
                self.view.close()
        finally:
            self.sim.close()
