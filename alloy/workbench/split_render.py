# SPDX-License-Identifier: Apache-2.0
"""A CUDA MPS worker for physics and a separate process for the Portal cameras."""
from __future__ import annotations

import ctypes
import multiprocessing
import time
from types import SimpleNamespace

import numpy as np
import warp as wp

_RING_SIZE = 32
_MAX_CATCHUP_SECONDS = .10
_STATE_FIELDS = ("body_q", "particle_q", "particle_qd", "particle_flags")


@wp.kernel
def _interpolate_bodies(a: wp.array(dtype=wp.transform), b: wp.array(dtype=wp.transform),
                        mix: float, out: wp.array(dtype=wp.transform)):
    i = wp.tid()
    p = wp.lerp(wp.transform_get_translation(a[i]), wp.transform_get_translation(b[i]), mix)
    q = wp.quat_slerp(wp.transform_get_rotation(a[i]), wp.transform_get_rotation(b[i]), mix)
    out[i] = wp.transform(p, q)


@wp.kernel
def _interpolate_particles(pa: wp.array(dtype=wp.vec3), va: wp.array(dtype=wp.vec3),
                           fa: wp.array(dtype=wp.int32), pb: wp.array(dtype=wp.vec3),
                           vb: wp.array(dtype=wp.vec3), fb: wp.array(dtype=wp.int32), mix: float,
                           points: wp.array(dtype=wp.vec3), speeds: wp.array(dtype=wp.vec3),
                           flags: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if fa[i] == fb[i]:
        points[i] = wp.lerp(pa[i], pb[i], mix)
        speeds[i] = wp.lerp(va[i], vb[i], mix)
        flags[i] = fa[i]
    elif mix < 0.5:
        points[i] = pa[i]
        speeds[i] = va[i]
        flags[i] = fa[i]
    else:
        points[i] = pb[i]
        speeds[i] = vb[i]
        flags[i] = fb[i]


def _arrays(sim, flags):
    state = sim.engine.state
    return state.body_q, state.particle_q, state.particle_qd, flags.current()


class _Snapshot:
    """Copy one robot and fluid snapshot between CUDA and shared memory."""

    def __init__(self, arrays):
        self.host = [wp.empty_like(a, device="cpu", pinned=True) for a in arrays]
        self.bytes = [a.numpy().view(np.uint8).reshape(-1) for a in self.host]
        self.size = sum(a.size for a in self.bytes)

    def copy_to(self, buffer):
        offset = 0
        for array in self.bytes:
            buffer[offset:offset + array.size] = array
            offset += array.size

    def copy_from(self, buffer):
        offset = 0
        for array in self.bytes:
            array[:] = buffer[offset:offset + array.size]
            offset += array.size


def _physics_worker(args, shared):
    # Set this before importing Alloy or initializing a CUDA context in this process.
    realtime = getattr(args, "realtime_fluid", False)
    from alloy.camera.liquid_scene import DeviceFlags
    from . import FIELDS
    from .simulation import Simulation

    runner = Simulation(args, physics_only=True)
    try:
        sim = runner.sim
        if args.prewarm_splash:
            runner.prewarm_splash()
        runner.check()
        flags = DeviceFlags(sim.engine)
        snapshot = _Snapshot(_arrays(sim, flags))
        buffer = np.frombuffer(shared.payload, dtype=np.uint8).reshape(_RING_SIZE, -1)
        if buffer.shape[1] != snapshot.size:
            raise RuntimeError("Physics and render snapshot layouts differ")
        joint_buffer = np.frombuffer(shared.joints, dtype=np.float32).reshape(_RING_SIZE, len(FIELDS))
        stream = wp.get_stream(args.device)

        def publish(joints, deadline=None, *, reset=False):
            for host, device in zip(snapshot.host, _arrays(sim, flags)):
                wp.copy(host, device, stream=stream)
            wp.synchronize_stream(stream)
            with shared.snapshot_lock:
                slot = shared.serial.value % _RING_SIZE
                snapshot.copy_to(buffer[slot])
                joint_buffer[slot] = [joints[name] for name in FIELDS]
                shared.sim_times[slot] = sim.t
                ready_at = time.monotonic()
                # Keep the paced timeline while on schedule. If physics falls behind,
                # interpolate against when its snapshots actually become available.
                shared.wall_times[slot] = ready_at if deadline is None else max(ready_at, deadline)
                shared.serial.value += 1
                if reset:
                    shared.first_serial.value = shared.serial.value

        publish(runner.target)
        shared.ready.set()
        while not shared.started.wait(.1):
            if shared.stop.is_set():
                return
        start = time.monotonic()
        clock_origin = start
        initial_time = sim.t
        if realtime:
            with shared.snapshot_lock:
                shared.wall_times[0] = start
        clock_report = start
        peak_lag = 0.
        rebased = 0.
        report = start
        splash_after = args.demo_splash_after
        while not shared.stop.is_set():
            if shared.reset_requested.is_set():
                runner.reset()
                publish(runner.read(), reset=True)
                clock_origin = clock_report = report = time.monotonic()
                initial_time = sim.t
                peak_lag = rebased = 0.
                shared.reset_requested.clear()
                shared.reset_done.set()
            now = time.monotonic()
            if splash_after is not None and now - start >= splash_after:
                sim.set_pose("block", pos=(*runner.cup_origin[:2], .16), world=0)
                print(f"Splash drop: simulated_t={sim.t:.6f}", flush=True)
                splash_after = None
            with shared.command_lock:
                command = dict(zip(FIELDS, shared.command)) if now < shared.command_until.value else None
                hold = shared.hold_requested.value
                shared.hold_requested.value = False
            if hold:
                runner.hold()
            joints, _ = runner.tick(command, 1, render=False)
            deadline = clock_origin + (sim.t - initial_time) if realtime else None
            publish(joints, deadline)
            if deadline is not None:
                finished = time.monotonic()
                lag = finished - deadline
                peak_lag = max(peak_lag, lag)
                # Keep all physics steps, but stop chasing old wall time during sustained contact.
                if lag > _MAX_CATCHUP_SECONDS:
                    clock_origin += lag
                    rebased += lag
                    deadline = finished
                if finished - now > .05:
                    stats = sim.report()
                    exchange = stats["coupling"].dt_x * 1000
                    print(f"Slow physics step: t={sim.t:.3f}s wall_ms={1000 * (finished - now):.1f} "
                          f"exchange_ms={exchange:g} captures={stats['graph']['captures']} "
                          f"rollbacks={stats['rollbacks']}", flush=True)
                if finished - clock_report >= 5.:
                    print(f"Physics clock: lag_ms={1000 * max(0., finished - deadline):.1f} "
                          f"peak_lag_ms={1000 * peak_lag:.1f} rebased_ms={1000 * rebased:.1f}", flush=True)
                    clock_report, peak_lag, rebased = finished, 0., 0.
                shared.stop.wait(max(0., deadline - finished))
            if now - report >= 15.:
                if realtime:
                    # The full contact report constructs a probe and can stall the snapshot stream.
                    # Keep checking live solver health and run the detailed report before and after.
                    health = sim.health()
                    if not health["ok"]:
                        runner.check()
                else:
                    runner.check()
                report = now
        runner.check()
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()


class SplitSimulation:
    """Render fresh snapshots on the existing Portal worker while physics runs continuously."""

    def __init__(self, args):
        from alloy.camera.liquid_scene import DeviceFlags
        from . import FIELDS
        from .simulation import Simulation

        self.renderer = Simulation(args, render_only=True)
        self.physics = None
        self.validate = self.renderer.validate
        self.timings = {}
        self.previous = None
        self.delay = args.snapshot_delay_ms / 1000
        try:
            sim = self.renderer.sim
            self.sim = SimpleNamespace(t=sim.t, dt=sim.dt)
            arrays = _arrays(sim, DeviceFlags(sim.engine))
            self.snapshots = [_Snapshot(arrays), _Snapshot(arrays)]
            self.upload = [[wp.empty_like(a) for a in arrays] for _ in self.snapshots]
            self.interpolated = [wp.empty_like(a) for a in arrays]
            self.first_state = SimpleNamespace(**dict(zip(_STATE_FIELDS, self.upload[0])))
            self.mixed_state = SimpleNamespace(**dict(zip(_STATE_FIELDS, self.interpolated)))
            wp.load_module(module=__name__, device=args.device)
            ctx = multiprocessing.get_context("spawn")
            self.shared = SimpleNamespace(
                payload=ctx.RawArray(ctypes.c_uint8, _RING_SIZE * self.snapshots[0].size),
                snapshot_lock=ctx.Lock(),
                joints=ctx.RawArray(ctypes.c_float, _RING_SIZE * len(FIELDS)),
                sim_times=ctx.RawArray(ctypes.c_double, _RING_SIZE),
                wall_times=ctx.RawArray(ctypes.c_double, _RING_SIZE),
                serial=ctx.RawValue(ctypes.c_uint64),
                first_serial=ctx.RawValue(ctypes.c_uint64, 1),
                command_lock=ctx.Lock(),
                command=ctx.RawArray(ctypes.c_double, len(FIELDS)),
                command_until=ctx.RawValue(ctypes.c_double),
                hold_requested=ctx.RawValue(ctypes.c_bool),
                ready=ctx.Event(),
                started=ctx.Event(),
                stop=ctx.Event(),
                reset_requested=ctx.Event(),
                reset_done=ctx.Event(),
            )
            self.buffer = np.frombuffer(self.shared.payload, dtype=np.uint8).reshape(_RING_SIZE, -1)
            self.joints = np.frombuffer(self.shared.joints, dtype=np.float32).reshape(_RING_SIZE, len(FIELDS))
            self.physics = ctx.Process(target=_physics_worker, args=(args, self.shared), name="alloy-physics")
            self.physics.start()
            while not self.shared.ready.wait(.25):
                self.check()
        except BaseException:
            self.close()
            raise

    def start(self):
        self.shared.started.set()

    def reset(self):
        with self.shared.command_lock:
            self.shared.command_until.value = 0.
            self.shared.hold_requested.value = False
        self.shared.reset_done.clear()
        self.shared.reset_requested.set()
        while not self.shared.reset_done.wait(.1):
            self.check()
        self.previous = None

    def tick(self, action, _steps, command_until=0., *, hold=False):
        from . import FIELDS

        start = time.perf_counter()
        self.check()
        with self.shared.command_lock:
            self.shared.command_until.value = command_until if action is not None else 0.
            if hold:
                self.shared.hold_requested.value = True
            if action is not None:
                self.shared.command[:] = [action[name] for name in FIELDS]
        with self.shared.snapshot_lock:
            newest = self.shared.serial.value
            before = after = newest
            view_time = time.monotonic() - self.delay
            # Never interpolate across a reset, even while the camera delay spans it.
            oldest = max(self.shared.first_serial.value, newest - _RING_SIZE + 1)
            for before in range(newest, oldest - 1, -1):
                if self.shared.wall_times[(before - 1) % _RING_SIZE] <= view_time:
                    break
                after = before
            left, right = (before - 1) % _RING_SIZE, (after - 1) % _RING_SIZE
            mix = 0.
            if left != right:
                first, last = self.shared.wall_times[left], self.shared.wall_times[right]
                mix = (view_time - first) / (last - first)
            key = before, after, mix
            if key == self.previous:
                return None
            self.snapshots[0].copy_from(self.buffer[left])
            simulated = self.shared.sim_times[left]
            joints = self.joints[left].copy()
            if left != right:
                self.snapshots[1].copy_from(self.buffer[right])
                simulated += mix * (self.shared.sim_times[right] - simulated)
                joints += mix * (self.joints[right] - joints)
        device = self.renderer.args.device
        stream = wp.get_stream(device)
        endpoints = 1 if left == right else 2
        for dst, snapshot in zip(self.upload[:endpoints], self.snapshots):
            for out, host in zip(dst, snapshot.host):
                wp.copy(out, host, stream=stream)
        state = self.first_state
        if left != right:
            a, b = self.upload
            body, points, _, _ = self.interpolated
            wp.launch(_interpolate_bodies, dim=body.size, inputs=[a[0], b[0], mix],
                      outputs=[body], stream=stream)
            wp.launch(_interpolate_particles, dim=points.size, inputs=[*a[1:], *b[1:], mix],
                      outputs=self.interpolated[1:], stream=stream)
            state = self.mixed_state
        batch = self.renderer.camera_batch.render(self.renderer.sim, state=state)
        self.sim.t = simulated
        self.previous = key
        self.timings = {"render_ms": (time.perf_counter() - start) * 1000}
        return dict(zip(FIELDS, map(float, joints))), {name: rgb[0] for name, rgb in batch.items()}

    def check(self):
        if not self.physics.is_alive():
            raise RuntimeError(f"Alloy physics worker exited with code {self.physics.exitcode}")

    def close(self):
        try:
            if self.physics is not None and self.physics.pid is not None:
                self.shared.stop.set()
                self.physics.join(timeout=5)
                if self.physics.is_alive():
                    self.physics.terminate()
                    self.physics.join(timeout=5)
                if self.physics.is_alive():
                    self.physics.kill()
                    self.physics.join()
                self.physics.close()
        finally:
            self.renderer.close()
