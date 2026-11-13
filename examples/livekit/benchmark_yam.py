# SPDX-License-Identifier: Apache-2.0
"""Benchmark warm camera batches and physics separately, without connecting to LiveKit."""
from argparse import ArgumentParser
import math
from pathlib import Path
from statistics import median
import time

from .yam_portal import Simulation


def measure(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(1000 * (time.perf_counter() - start))
    return round(median(samples), 2)


def measure_splash(runner):
    """Measure the moving-arm workbench before impact, during the splash and in recovery."""
    sim = runner.sim
    runner.prewarm_splash()
    home = dict(runner.target)
    sim.set_pose("block", pos=(*runner.cup_origin[:2], .16), world=0)
    start = sim.t
    timings = {"fall": [], "splash": [], "recovery": []}
    for _ in range(round(.5 / sim.dt)):
        elapsed = sim.t - start
        target = dict(home)
        offset = .1 * math.sin(7 * elapsed)
        target["left_joint_1.pos"] += offset
        target["right_joint_1.pos"] -= offset
        phase = "fall" if elapsed < .15 else "splash" if elapsed < .31 else "recovery"
        now = time.perf_counter()
        runner.tick(target, 1, render=False)
        timings[phase].append(time.perf_counter() - now)
    for phase, samples in timings.items():
        rate = len(samples) * sim.dt / sum(samples)
        print(f"{phase} steps={len(samples)} median_ms={1000*median(samples):.2f} "
              f"simulated_seconds_per_wall_second={rate:.3f}", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--scene', choices=('workbench', 'glass-water'), default='workbench')
    parser.add_argument('--cup-collision', choices=('sdf', 'decompose', 'surface'), default='sdf')
    parser.add_argument('--rig', type=Path)
    parser.add_argument('--camera-backend', default='warp', choices=('warp', 'optix', 'wavefront-warp'))
    parser.add_argument('--render-quality', choices=('auto', 'fast', 'full'), default='auto')
    parser.add_argument('--look', type=Path, default=Path(__file__).resolve().parents[1] / 'scenes/yam_workbench/workbench.look.json')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--spp', type=int, nargs='+', default=[1, 4, 8, 16])
    parser.add_argument('--denoise', choices=('off', 'oidn', 'optix', 'learned', 'auto'), default='off')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--steps', type=int, nargs='+', default=[2, 5, 10, 50])
    parser.add_argument('--liquid-accuracy', choices=('default', 'super', 'fast'), default='default')
    parser.add_argument('--joined-cameras', action='store_true')
    parser.add_argument('--physics-only', action='store_true', help='Skip camera warmup and timing')
    parser.add_argument('--control-dt', type=float, default=.01, choices=(.00125, .0025, .005, .01))
    parser.add_argument('--liquid-cfl', type=float, help='Maximum liquid travel per step in particle spacings')
    parser.add_argument('--splash', action='store_true', help='Time a block drop instead of the settled step batches')
    parser.add_argument('--realtime-fluid', action='store_true', help='Measure the interactive liquid solver')
    args = parser.parse_args()
    samples = args.spp
    if args.repeats < 1 or min(samples) < 1 or min(args.steps) < 1:
        parser.error('repeats, spp, and steps must be positive')
    if args.splash and (args.scene != 'glass-water' or not args.device.startswith('cuda')):
        parser.error('splash needs the CUDA glass-water scene')
    args.spp, args.viewer, args.mode, args.port = samples[0], False, 'smoke', 8086
    args.pourable_cup = args.household_objects = False
    sim = Simulation(args, physics_only=args.physics_only)
    try:
        sim.tick(None, 10, render=not args.physics_only)  # Exclude compilation and initial graph capture.
        if args.splash:
            measure_splash(sim)
        else:
            for steps in args.steps:
                ms = measure(lambda: sim.tick(None, steps, render=False), args.repeats)
                rate = steps * sim.sim.dt / (ms / 1000)
                print(f'control_steps={steps} median_ms={ms} simulated_seconds_per_wall_second={rate:.3f}', flush=True)
        if args.scene == 'glass-water':
            report = sim.sim.report()
            coupling = report['coupling']
            print(f'contact_pipeline={report["native"]} exchange_ms={coupling.dt_x * 1000:g} '
                  f'substeps_per_exchange={coupling.substeps}', flush=True)
        render_samples = [] if args.physics_only else samples
        for spp in render_samples:
            for camera in sim.cameras.values():
                camera.spp = spp
            sim.render()  # Allow backend buffers to grow before timing.
            ms = measure(sim.render, args.repeats)
            settings = f'backend={args.camera_backend} quality={args.render_quality} spp={spp} denoise={args.denoise}'
            print(f'{settings} views=3 resolution={args.width}x{args.height} '
                  f'render_median_ms={ms}', flush=True)
        sim.check()
    finally:
        sim.close()


if __name__ == '__main__':
    main()
