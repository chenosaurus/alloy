# SPDX-License-Identifier: Apache-2.0
"""Persistent full-material cameras match snapshots and follow current simulation state."""
import numpy as np
import pytest

import alloy
from alloy.camera import CameraError, PbrCameraBatch


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


def test_joined_batch_tracks_each_view_after_body_motion():
    scene = alloy.Scene()
    scene.add_ground()
    scene.add_box("block", size=(0.2, 0.2, 0.2), pos=(0, 0, 0.1), mass=1)
    sim = alloy.build(scene, device="cpu")
    options = dict(width=24, height=18, spp=2, quality="full")
    cameras = {
        "static": alloy.Camera("static", look_at=((.7, -.7, .5), (0, 0, .1)), **options),
        "attached": alloy.Camera("attached", attach_to="block",
                                 look_at=((-.6, .7, .6), (0, 0, 0)), **options),
    }
    batch = PbrCameraBatch(cameras)
    try:
        before = batch.render(sim)
        assert all(camera._bound[1].rgb is None for camera in cameras.values())
        sim.set_pose("block", pos=(.25, 0, .1))
        after = batch.render(sim)
        np.testing.assert_array_equal(after["static"], sim.render(cameras["static"]))
        attached = sim.render(cameras["attached"])
        error = np.mean(np.abs(after["attached"].astype(float) - attached))
        # The second image uses a different sample seed when traced in a batch.
        assert error < 12
        assert np.any(before["static"] != after["static"])
        assert not np.array_equal(after["static"], after["attached"])
    finally:
        sim.close()


@pytest.mark.gpu
def test_enqueued_batch_retains_the_traced_state():
    from alloy.camera.optix import available

    if not available("cuda:0")[0]:
        pytest.skip("OptiX helper not built")
    scene = alloy.Scene()
    scene.add_ground()
    scene.add_box("block", size=(.2, .2, .2), pos=(0, 0, .1), mass=1)
    sim = alloy.build(scene, device="cuda:0")
    opts = dict(width=48, height=32, spp=2, quality="full", backend="optix", denoise="optix")
    batch = PbrCameraBatch({
        "first": alloy.Camera("first", look_at=((.7, -.7, .5), (0, 0, .1)), **opts),
        "second": alloy.Camera("second", look_at=((-.7, .7, .5), (0, 0, .1)), **opts),
    })
    try:
        expected = batch.render(sim)
        finish = batch.enqueue(sim)
        try:
            with pytest.raises(CameraError, match="still rendering"):
                batch.render(sim)
            sim.set_pose("block", pos=(.25, 0, .1))
            sim.step()
        finally:
            before = finish()
        after = batch.enqueue(sim)()
        for name in expected:
            np.testing.assert_array_equal(before[name], expected[name])
            assert np.any(after[name] != before[name])
    finally:
        sim.close()
