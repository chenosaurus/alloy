# SPDX-License-Identifier: Apache-2.0
"""The workbench assets and rig can be reused without LiveKit."""
import hashlib
import json
import xml.etree.ElementTree as ET

import alloy

from examples.scenes.yam_workbench import FIELDS, ROOT, VIDEOS, Rig, add_workbench, glassware


def test_rig_loads_a_model_relative_to_its_profile(tmp_path):
    stock = Rig.load()
    assert set(stock.joints) == set(FIELDS)
    assert set(stock.cameras) == set(VIDEOS)

    model = tmp_path / "custom.xml"
    model.write_text("<mujoco><worldbody><body name='unrelated'/>"
                     + "".join(
                         f"<body name='{body}'>"
                         + "".join(f"<joint name='{joint}' range='-90 90'/>"
                                   for field, joint in stock.joints.items() if field.startswith(side))
                         + "</body>" for side, body in stock.roots.items()
                     )
                     + "</worldbody></mujoco>")
    profile = tmp_path / "custom.rig.json"
    data = json.loads((ROOT / "yam.rig.json").read_text())
    profile.write_text(json.dumps(data | {"model": model.name, "gripper_travel": {"left": .02, "right": .03}}))

    rig = Rig.load(profile)
    mjcf, limits = rig.mjcf()
    assert rig.model == model
    assert rig.scale("right_gripper.pos") == .03
    assert abs(limits["left_joint_1.pos"][1] - 1.57079632679) < 1e-9
    assert {body.get("name") for body in ET.fromstring(mjcf).findall("./worldbody/body")} == set(stock.roots.values())


def test_glassware_assets_and_empty_custom_look_are_reusable():
    scene = alloy.Scene(liquid_accuracy="fast")
    look = {}
    origin = add_workbench(scene, look, scenery=False, cup_collision="surface", pourable=True)
    assert origin == glassware.POUR_CUP_ORIGIN
    assert look["liquid_supports"] == ["table", "pour_basin_floor"]
    assert scene.items["cup"]["shape"]["mesh"]["particle_surface"]
    for hull in glassware.ASSETS.glob("*.hulls.obj"):
        mesh = hull.with_name(hull.name.replace(".hulls.obj", ".obj"))
        with hull.open() as file:
            assert file.readline().split()[-1] == hashlib.sha256(mesh.read_bytes()).hexdigest()
