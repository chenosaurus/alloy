# Reuse the dual-arm workbench

This example owns the glassware and spoon OBJ meshes, the matching precomputed convex hulls, the generators, and the default material and light setup. Keep each `.hulls.obj` beside its mesh. The bundled model supplies the YAM geometry, and Alloy's library supplies the sponge and bottle meshes. The default look needs no downloaded textures.

Build an Alloy scene without Portal:

```python
import alloy
from examples.scenes.yam_workbench import Rig, add_workbench, load_look

rig = Rig.load()
look = load_look()
scene = alloy.Scene(liquid_accuracy="default")
mjcf, limits = rig.mjcf()
scene.add_robot("yam", mjcf=mjcf)
add_workbench(scene, look, cup_collision="surface", pourable=True)
cameras = {name: alloy.Camera(name, width=640, height=480, look=look, **view)
           for name, view in rig.cameras.items()}
```

For another setup, copy [the stock rig profile](yam.rig.json) and set `model` to an absolute or profile-relative path to a self-contained dual-arm MJCF. The `roots` name the two top-level arm bodies. The `joints`, `home`, and `gripper_travel` map the fixed Portal wire schema onto that rig. The three entries under `cameras` accept Alloy camera pose, attachment, and lens options. Use names such as `yam/right_d405` for camera attachments. Leave `model` out to use the bundled YAM. Set Portal's `--rig` to your profile path, or pass it to `Rig.load()`.

Pass `cup_collision=None` to reuse just the table and props. The helpers also work in a scene with no robot. Run the standalone slosh diagnostic with `python -m examples.scenes.yam_workbench.glass_water --device cpu`. The OBJ files are also usable directly by another simulator. The generators in this directory document their dimensions and decomposition.
