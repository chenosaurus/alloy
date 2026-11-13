# SPDX-License-Identifier: Apache-2.0
"""Reusable Alloy scene and rig configuration for a dual-arm workbench."""
from __future__ import annotations

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import alloy
import numpy as np
from alloy.cards.library import default_root

from . import glassware

ROOT = Path(__file__).parent
VIDEOS = ("overhead_camera", "left_wrist_camera", "right_wrist_camera")
SIDES = ("left", "right")
FIELDS = tuple(f"{side}_{name}.pos" for side in SIDES
               for name in (*(f"joint_{i}" for i in range(1, 7)), "gripper"))


@dataclass(frozen=True)
class Rig:
    model: Path
    roots: dict[str, str]
    joints: dict[str, str]
    gripper_travel: dict[str, float]
    home: dict[str, float]
    cameras: dict[str, dict]

    @classmethod
    def load(cls, path: Path | None = None) -> Rig:
        """Read a dual-arm hardware profile. Model paths resolve next to the profile."""
        path = path or ROOT / "yam.rig.json"
        data = json.loads(path.read_text())
        model = data.get("model")
        model_path = path.parent / model if model else default_root() / "robots/yam_bench/robot.xml"
        fields = set(FIELDS)
        travel = {side: float(value) for side, value in data["gripper_travel"].items()}
        home = {name: float(value) for name, value in data["home"].items()}
        if (set(data["joints"]) != fields or set(home) != fields or set(data["roots"]) != set(SIDES)
                or set(travel) != set(SIDES) or set(data["cameras"]) != set(VIDEOS)):
            raise ValueError("rig needs both arm roots, all 14 joint and home fields, both gripper travels and three cameras")
        if not all(np.isfinite(value) and value > 0 for value in travel.values()):
            raise ValueError("gripper travel must be positive and finite")
        if not all(np.isfinite(value) for value in home.values()):
            raise ValueError("home commands must be finite")
        return cls(model_path.resolve(), data["roots"], data["joints"], travel, home, data["cameras"])

    def scale(self, field: str) -> float:
        return self.gripper_travel[field.split("_", 1)[0]] if field.endswith("_gripper.pos") else 1.0

    def mjcf(self, *, interactive: bool = False) -> tuple[str, dict[str, tuple[float, float]]]:
        """Keep both robots and their assets, replacing the room geometry with Alloy primitives."""
        root = ET.parse(self.model).getroot()
        world = root.find("worldbody")
        kept = set(self.roots.values())
        if world is None or kept - {body.get("name") for body in world.findall("body")}:
            raise ValueError("rig roots must name direct children of the MJCF worldbody")
        for child in list(world):
            if child.tag != "body" or child.get("name") not in kept:
                world.remove(child)
        compiler = root.find("compiler")
        radians = compiler is not None and compiler.get("angle") == "radian"
        asset_dirs = {
            "mesh": compiler.get("meshdir", compiler.get("assetdir", "")) if compiler is not None else "",
            "texture": compiler.get("texturedir", compiler.get("assetdir", "")) if compiler is not None else "",
        }
        for asset in root.findall("asset/*"):
            if asset.get("file"):
                asset.set("file", str((self.model.parent / asset_dirs.get(asset.tag, "") / asset.get("file")).resolve()))
        if interactive:
            option = root.find("option")
            if option is None:
                option = ET.SubElement(root, "option")
            option.attrib.update(iterations="20", tolerance="1e-6", noslip_iterations="4")
        by_name = {joint.get("name"): joint for joint in world.iter("joint")}
        for body in world.iter("body"):
            body.set("gravcomp", "1")
        limits = {}
        for field, name in self.joints.items():
            joint = by_name.get(name)
            if joint is None or "range" not in joint.attrib:
                raise ValueError(f"rig field {field} needs an MJCF joint named {name!r} with a range")
            joint.set("actuatorgravcomp", "true")
            if field.endswith("_gripper.pos"):
                limits[field] = (0., 1.)
            else:
                low, high = map(float, joint.get("range").split())
                if joint.get("type", "hinge") == "hinge" and not radians:
                    low, high = np.deg2rad((low, high))
                limits[field] = (low, high)
        return ET.tostring(root, encoding="unicode"), limits


def load_look(path: Path | None = None) -> dict:
    return json.loads((path or ROOT / "workbench.look.json").read_text())


def add_workbench(scene: alloy.Scene, look: dict | None, *, scenery: bool = True,
                  cup_collision: str | None = None, pourable: bool = False,
                  household: bool = False) -> tuple[float, float, float] | None:
    """Place shared workbench props and optional glassware in an existing scene."""
    scene.add_box("table", size=(1.2, 1.3, .08), pos=(.25, 0, -.05), static=True)
    scene.add_box("block", size=(.04, .04, .04), pos=(.36, .14, .02) if pourable else (.4, 0, .02), mass=.1)
    if scenery:
        scene.add_box("floor", size=(6., 6., .1), pos=(0., 0., -.9), static=True)
        scene.add_box("back_wall", size=(.1, 6., 3.), pos=(-1.5, 0., .6), static=True)
        for xi, x in enumerate((-.25, .75)):
            for yi, y in enumerate((-.5, .5)):
                scene.add_box(f"table_leg_{xi}_{yi}", size=(.05, .05, .76), pos=(x, y, -.47), static=True)
        scene.add_open_box("parts_tray", inner=(.18, .14, .035), wall=.008, pos=(.43, .32, -.01))
        scene.add_cylinder("metal_part", radius=.025, height=.035, pos=(.40, -.16, .0075), mass=.12)
    if household:
        add_household_objects(scene, look)
    if cup_collision is None:
        return None
    origin = glassware.POUR_CUP_ORIGIN if pourable else glassware.FIXED_CUP_ORIGIN
    glassware.add_water_cup(scene, origin=origin, collision=cup_collision, pourable=pourable)
    if pourable:
        glassware.add_beaker(scene)
        scene.add_open_box("pour_basin", inner=glassware.RECEIVER_INNER, wall=.005, pos=glassware.RECEIVER_ORIGIN)
    if look is not None:
        look["materials"]["cup_glass"] = dict(glassware.GLASS)
        look["assign"].insert(0, {"match": "^(cup|beaker)", "material": "cup_glass"})
        if pourable:
            look["materials"]["pour_basin"] = {"preset": "plastic_matte", "base_color": [.07, .24, .32]}
            look["assign"].insert(0, {"match": "^pour_basin", "material": "pour_basin"})
        look["liquid_supports"] = [name for name in ("table", "parts_tray_floor", "pour_basin_floor")
                                   if name in scene.items]
    return origin


def apply_fast_look(scene: alloy.Scene, look: dict) -> None:
    """Use the look's colors and color maps in Alloy's diffuse tracer."""
    rules = look.get("assign", [])
    for name, item in scene.items.items():
        material = next((rule["material"] for rule in rules if re.search(rule["match"], name)), None)
        if material:
            item["look"] = {"materials": look["materials"], "assign": rules, "default": material}


def add_household_objects(scene: alloy.Scene, look: dict | None) -> None:
    meshes = default_root() / "meshes"
    scene.add_mesh("kitchen_sponge", meshes / "sponge_kitchen.obj", pos=(.25, -.195, .0035),
                   rpy=(0., 0., .18), material="pu_foam_sponge", mass=.05, rigid=True, friction=.9)
    scene.add_mesh("steel_spoon", glassware.ASSETS / "steel-spoon.obj", pos=(.275, .24, -.009),
                   rpy=(0., 0., 2.), collision="decompose", material="stainless_304", mass=.045, friction=.85,
                   card_extra={"features": {"min_wall": .0001}})
    scene.add_sphere("tomato", radius=.026, pos=(.51, .185, .017), material="fruit_tomato_whole",
                     mass=.075, rigid=True, restitution=.05)
    scene.add_mesh("water_bottle", meshes / "pet_bottle_500ml.obj", scale=.8, pos=(.49, -.25, -.009),
                   material="pet", mass=.06, rigid=True, card_extra={"features": {"min_wall": .00024}})
    if look is not None:
        materials = {
            "kitchen_sponge": {"preset": "plastic_matte", "base_color": [.94, .68, .06], "roughness": .85},
            "steel_spoon": {"preset": "steel_brushed", "roughness": .42},
            "tomato": {"preset": "plastic_matte", "base_color": [.72, .045, .022], "roughness": .45},
            "water_bottle": {"preset": "plastic_matte", "base_color": [.15, .56, .8], "roughness": .28},
        }
        look["materials"].update(materials)
        look["assign"][:0] = [{"match": f"^{name}", "material": name} for name in materials]
