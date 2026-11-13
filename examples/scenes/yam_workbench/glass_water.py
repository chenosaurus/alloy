# SPDX-License-Identifier: Apache-2.0
"""A standalone water containment and rendering diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import alloy
import numpy as np

from alloy.workbench.glassware import LOOK, add_water_cup, in_cup


def make_scene(accuracy='default'):
    scene = alloy.Scene(dt=.005, liquid_accuracy=accuracy)
    scene.add_box('table', size=(.6, .6, .04), pos=(0, 0, -.02), static=True)
    add_water_cup(scene, kinematic=True)
    # Colour markers behind the cup make refraction and the fill line easier to see.
    scene.add_box('blue_marker', size=(.12, .008, .018), pos=(0, .075, .025), static=True)
    scene.add_box('red_marker', size=(.018, .018, .085), pos=(-.065, .055, .0425), static=True)
    return scene


def metrics(sim):
    state = sim.state(['water', 'cup'], world=0)
    points = state['water.points']
    inside = in_cup(points, state['cup.pos'], state['cup.quat_xyzw'])
    return {'t': float(sim.t), 'particles': len(points),
            'water_ml': float(sim.liquid_volume('water') * 1e6),
            'fraction_in_cup': float(inside.mean()),
            'retired_particles': sim.health()['retired_particles'],
            'health': sim.check()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--accuracy', choices=('default', 'super', 'fast'), default='default')
    parser.add_argument('--motion', choices=('still', 'slosh'), default='slosh')
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--out', type=Path, default=Path('glass-water-output'))
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--animate', action='store_true', help='Render a 10 fps preview GIF as well as stills')
    parser.add_argument('--backend', choices=('warp', 'optix', 'wavefront-warp'), default='warp')
    parser.add_argument('--spp', type=int, default=32)
    parser.add_argument('--viewer', action='store_true', help='Keep a geometry/particle viewer open after the test')
    parser.add_argument('--port', type=int, default=8087)
    args = parser.parse_args()
    if args.seconds <= 0 or args.spp < 1:
        parser.error('seconds and spp must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    scene = make_scene(args.accuracy)
    (args.out / 'scene.json').write_text(scene.to_json())
    (args.out / 'look.json').write_text(json.dumps(LOOK, indent=2))
    print('Building glass/water test; first liquid kernels may take several minutes', flush=True)
    started = time.perf_counter()
    sim = alloy.build(scene, device=args.device)
    view = None
    images, records = [], []
    try:
        sim.settle(.4)
        print('Settled:', json.dumps(metrics(sim)), flush=True)
        camera = alloy.Camera('glass_water', width=640, height=480, spp=args.spp,
                              look_at=((.17, -.23, .18), (0, 0, .05)), fov=38,
                              clip=(.002, 10), look=LOOK, quality='full',
                              backend=args.backend, denoise=None)
        def capture(name):
            from PIL import Image
            image = Image.fromarray(np.asarray(sim.render(camera)[0]))
            image.save(args.out / f'{name}.png')
            if args.animate:
                images.append(image)
        if args.render or args.animate:
            capture('settled')
        count = max(1, round(args.seconds / .1))
        for i in range(count):
            if args.motion == 'slosh':
                # Smooth translation (amplitude 10 mm, period 1 s); no tilt or teleport.
                t = i * .1
                x0 = .01 * (1 - np.cos(2 * np.pi * t))
                x1 = .01 * (1 - np.cos(2 * np.pi * (t + .1)))
                sim.drive('cup', velocity=((x1 - x0) / .1, 0, 0), duration=.1)
            records.append(sim.run(.1, record=['water', 'cup'], every=20).world(0))
            report = metrics(sim)
            print(json.dumps({k: report[k] for k in ('t', 'water_ml', 'fraction_in_cup', 'retired_particles')}
                             | {'health_ok': report['health']['ok']}), flush=True)
            if not report['health']['ok']:
                raise RuntimeError('Fluid health check failed')
            if args.animate:
                capture(f'frame_{i:04d}')
        sim.stop_drives()
        if args.render:
            capture('final')
        trajectory = {key: np.concatenate([r[key] if i == 0 else r[key][1:]
                                          for i, r in enumerate(records)], axis=0)
                      for key in records[0]}
        np.savez_compressed(args.out / 'trajectory.npz', **trajectory,
                            particle_volume_m3=sim.particle_volume('water'))
        report = metrics(sim)
        report['wall_seconds_including_build_and_render'] = time.perf_counter() - started
        (args.out / 'report.json').write_text(json.dumps(report, indent=2))
        if images:
            images[0].save(args.out / 'preview.gif', save_all=True, append_images=images[1:],
                           duration=100, loop=0)
        print('Saved:', args.out.resolve(), flush=True)
        if args.viewer:
            from alloy import viewer
            view = viewer.launch(sim, host='127.0.0.1', port=args.port, contacts=False)
            print(f'Geometry / particle viewer: {view.url}', flush=True)
            view.run()
    finally:
        if view:
            view.close()
        sim.close()


if __name__ == '__main__':
    main()
