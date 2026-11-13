# SPDX-License-Identifier: Apache-2.0
"""A glass cup of simulated water, with optional gentle sloshing and rendered previews."""
from __future__ import annotations

from pathlib import Path

import numpy as np

RADIUS, DEPTH, WALL = .04, .10, .004
FIXED_CUP_ORIGIN = (.52, 0., -.01)
POUR_CUP_ORIGIN = (.43, 0., -.01)
BEAKER_ORIGIN = (.43, .122, -.01)
RECEIVER_ORIGIN = (.60, -.04, -.01)
RECEIVER_INNER = (.17, .22, .05)
ASSETS = Path(__file__).with_name("assets")
GLASS = {"preset": "glass_clear", "roughness": 0.025, "ior": 1.5}
LOOK = {
    'materials': {
        'glass': GLASS,
        'table': {'preset': 'plastic_matte', 'base_color': [.35, .39, .44], 'roughness': .65},
        'red': {'preset': 'plastic_matte', 'base_color': [.8, .06, .025]},
        'blue': {'preset': 'plastic_matte', 'base_color': [.035, .25, .7]},
    },
    'assign': [{'match': '^cup', 'material': 'glass'}, {'match': '^table', 'material': 'table'},
               {'match': '^red', 'material': 'red'}, {'match': '^blue', 'material': 'blue'}],
    'lights': [
        {'type': 'area', 'eye': [.15, -.25, .45], 'target': [0, 0, .04],
         'size_m': [.22, .3], 'watts': 12, 'kelvin': 5500},
        {'type': 'area', 'eye': [-.2, .18, .3], 'target': [0, 0, .05],
         'size_m': [.2, .25], 'watts': 8, 'kelvin': 6500},
    ],
    'world': {'strength': .5},
    'render': {'view': 'AgX', 'exposure': 0.0},
}


def cup_mesh(segments=96):
    """Closed glass solid with an open cavity; outward-wound outer/inner surfaces."""
    vertices = []
    for radius, z in ((RADIUS + WALL, 0), (RADIUS + WALL, DEPTH + WALL),
                      (RADIUS, DEPTH + WALL), (RADIUS, WALL)):
        vertices.extend((radius * np.cos(a), radius * np.sin(a), z)
                        for a in np.arange(segments) * (2 * np.pi / segments))
    bottom, floor = len(vertices), len(vertices) + 1
    vertices.extend(((0, 0, 0), (0, 0, WALL)))
    faces = []
    for ring in range(3):
        for i in range(segments):
            j = (i + 1) % segments
            a, b = ring * segments + i, ring * segments + j
            c, d = (ring + 1) * segments + i, (ring + 1) * segments + j
            faces.extend(((a, b, d), (a, d, c)))
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend(((bottom, j, i), (floor, 3 * segments + i, 3 * segments + j)))
    return np.asarray(vertices), np.asarray(faces)


def add_water_cup(scene, origin=(0., 0., 0.), kinematic=False, collision='sdf', *, pourable=False):
    """Use one glass surface with a choice of collision geometry."""
    x, y, z = origin
    if pourable:
        if kinematic or collision != 'surface':
            raise ValueError('The pourable cup needs surface collision and must be free to move')
        mesh = {'path': ASSETS / 'pour-cup.obj'}
    elif collision in ('decompose', 'surface'):
        # The OBJ matches cup_mesh(); its sidecar contains the base and 24 wall sections.
        mesh = {'path': ASSETS / 'glass-cup.obj'}
    else:
        vertices, faces = cup_mesh()
        mesh = {'vertices': vertices, 'faces': faces}
    scene.add_mesh('cup', **mesh, collision='decompose' if collision == 'surface' else collision,
                   particle_surface=collision == 'surface', material='glass_soda_lime',
                   pos=origin, rigid=True, static=not (kinematic or pourable), kinematic=kinematic,
                   **({'friction': .9} if pourable else {}))
    scene.add_liquid('water', shape='cylinder', size=(.068, .068, .05),
                     pos=(x, y, z + WALL + .002 + .025), material='water')


def add_beaker(scene):
    """Place an empty, movable beaker to the mug's left in the overhead view."""
    scene.add_mesh('beaker', ASSETS / 'glass-beaker.obj',
                   collision='decompose', particle_surface=True, material='glass_soda_lime',
                   pos=BEAKER_ORIGIN, rigid=True, friction=.9)


def in_cup(points, origin, quat_xyzw):
    """Count particles in the cup's local cavity as it moves and tilts."""
    from scipy.spatial.transform import Rotation

    relative = Rotation.from_quat(quat_xyzw).inv().apply(points - origin)
    return ((np.linalg.norm(relative[:, :2], axis=1) < RADIUS + .002)
            & (relative[:, 2] > -.002) & (relative[:, 2] < DEPTH + WALL))
