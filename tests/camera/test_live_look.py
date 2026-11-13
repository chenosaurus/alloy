# SPDX-License-Identifier: Apache-2.0
"""Persistent full-material cameras match snapshots and follow current simulation state."""
import numpy as np
import pytest

import alloy


@pytest.mark.parametrize('look', [None, {
    'materials': {'red': {'preset': 'plastic_matte', 'base_color': [0.8, 0.03, 0.02]}},
    'assign': [{'match': 'block', 'material': 'red'}],
}])
def test_live_full_camera_matches_snapshot_after_motion(look):
    scene = alloy.Scene()
    scene.add_ground()
    scene.add_box('block', size=(.2, .2, .2), pos=(0, 0, .1), mass=1)
    sim = alloy.build(scene, device='cpu')
    view = ((.7, -.7, .5), (0, 0, .1))
    options = dict(width=24, height=18, fov=50, spp=2, quality='full', look=look)
    camera = alloy.Camera('live', look_at=view, **options)
    try:
        before = sim.render(camera)[0].copy()
        reference = sim.render(look_at=view, **options)
        np.testing.assert_array_equal(before, reference)
        sim.set_pose('block', pos=(.3, 0, .1))
        after = sim.render(camera)[0].copy()
        reference = sim.render(look_at=view, **options)
        np.testing.assert_array_equal(after, reference)
        assert np.count_nonzero(before != after) > 50
    finally:
        sim.close()
