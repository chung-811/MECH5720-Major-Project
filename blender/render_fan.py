import argparse
import math
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

# one blade seen close up, hub above the frame. metres. placeholders until the fan is designed
R0, R1 = 0.030, 0.080             # blade inner and outer edge from the rotation centre
W0, W1 = 0.015, 0.030             # blade width at R0 (narrow end, toward the hub) and at R1
LETTERS, TEXT_H = "ABC", 0.008    # stacked down the blade
VIEW_H = 0.055                    # frame height: the 50 mm blade roughly fills it
WHITE, BLACK = 0.8, 0.03

ap = argparse.ArgumentParser()
ap.add_argument("--rpm", type=float, default=0.35)
ap.add_argument("--t_end", type=float, default=0.95)
ap.add_argument("--dt", type=float, default=1 / 120)
ap.add_argument("--width", type=int, default=728)    # half the IMX296 1456 x 1088
ap.add_argument("--height", type=int, default=544)
ap.add_argument("--steps_per_rev", type=float, default=0, help="0 = smooth, 2038 = 28BYJ-48 full steps")
ap.add_argument("--texture", default="", help="grey image printed on the blade, values taken as albedo")
ap.add_argument("--out", default="renders/blade.npz")
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

s = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)


def mat(v):
    m = bpy.data.materials.new(f"m{v}")
    m.diffuse_color = (v, v, v, 1)
    return m


hub = bpy.data.objects.new("hub", None)       # rotation centre at the world origin
s.collection.objects.link(hub)

me = bpy.data.meshes.new("blade")             # pointing down (-y) at angle 0
me.from_pydata([(-W0 / 2, -R0, 0), (-W1 / 2, -R1, 0), (W1 / 2, -R1, 0), (W0 / 2, -R0, 0)], [], [[0, 1, 2, 3]])
me.materials.append(mat(WHITE))
if a.texture:                                 # image across the blade, u across, v along it
    me.uv_layers.new().data.foreach_set("uv", [0, 1, 0, 0, 1, 0, 1, 1])
    m = me.materials[0]
    m.use_nodes = True
    tex = m.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(Path(a.texture).resolve()))
    tex.image.colorspace_settings.name = "Non-Color"           # pixel values are albedo, no sRGB decode
    m.node_tree.links.new(tex.outputs["Color"], m.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    m.node_tree.nodes.active = tex
blade = bpy.data.objects.new("blade", me)
s.collection.objects.link(blade)
blade.parent = hub

for i, ch in enumerate(LETTERS):
    txt = bpy.data.curves.new(f"t{i}", "FONT")
    txt.body, txt.size = ch, TEXT_H
    txt.align_x, txt.align_y = "CENTER", "CENTER"
    txt.materials.append(mat(BLACK))
    t = bpy.data.objects.new(f"t{i}", txt)
    t.location = (0, -(R0 + (R1 - R0) * (i + 1) / (len(LETTERS) + 1)), 0.001)
    s.collection.objects.link(t)
    t.parent = hub

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.type, cam.data.sensor_fit, cam.data.ortho_scale = "ORTHO", "VERTICAL", VIEW_H
cam.location = (0, -(R0 + R1) / 2, 1)
s.collection.objects.link(cam)
s.camera = cam

s.world = bpy.data.worlds.new("w")
s.world.color = (0, 0, 0)
s.render.engine = "BLENDER_WORKBENCH"
s.display.shading.light = "FLAT"
s.display.shading.color_type = "TEXTURE" if a.texture else "MATERIAL"
s.display.shading.background_type = "WORLD"
s.display.shading.show_specular_highlight = False
s.display.render_aa = "32"
s.view_settings.view_transform = "Standard"
s.render.resolution_x, s.render.resolution_y = a.width, a.height
s.render.resolution_percentage = 100
s.render.image_settings.file_format = "OPEN_EXR"
s.render.image_settings.color_depth = "32"

tmp = Path(tempfile.mkdtemp()) / "f.exr"
s.render.filepath = str(tmp)


def render(theta):
    hub.rotation_euler.z = theta
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(tmp), check_existing=False)
    px = np.empty(a.width * a.height * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(a.height, a.width, 4)[::-1, :, 1].copy()  # blender stores bottom row first


omega = a.rpm * 2 * math.pi / 60
t = (np.arange(math.ceil(a.t_end / a.dt)) + 0.5) * a.dt
theta = omega * t
if a.steps_per_rev:                          # stop-start: hold each step until the next is due
    step = 2 * math.pi / a.steps_per_rev
    theta = np.floor(theta / step) * step
frames = np.stack([render(th) for th in theta])
p = VIEW_H / a.height                        # metres per pixel
hub_rc = (a.height / 2 - 0.5 - (R0 + R1) / 2 / p, a.width / 2 - 0.5)   # pixel centres at index + 0.5
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, frames=frames, t=t, dt=a.dt, omega=omega, ref=render(0.0), hub=hub_rc, px_m=p,
         r0_px=R0 / p, r1_px=R1 / p, half_angle=max(math.atan2(W0 / 2, R0), math.atan2(W1 / 2, R1)))
print(f"{len(t)} frames {a.width}x{a.height}, {1e-3 / p:.1f} px/mm, {a.rpm} rpm, "
      f"outer edge {R1 * omega * a.dt / p:.2f} px per frame -> {out}")
