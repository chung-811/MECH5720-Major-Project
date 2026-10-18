"""Render a spinning fan as sharp sub-frames. Runs inside Blender:

    blender -b -P blender/render_fan.py -- --rpm 2 --t_end 0.9 --dt 0.008 --out renders/fan.npz

Frame k is the fan at t = (k + 0.5) dt, angle = omega t, counter-clockwise on screen.
`ref` is the fan at t = 0. Images are linear albedo, top row first, hub at the image centre.
Workbench flat shading: no lighting, no noise. Light and noise are added later in flutter.py.
"""
import argparse
import math
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

# fan geometry in metres. placeholders until the real fan is designed
HUB_R = 0.025
BLADE_R0, BLADE_R1, BLADE_W = 0.02, 0.10, 0.035
TEXT_H = 0.014
LABELS = ["ABCD", "EFGH", "KLMN"]
VIEW_W = 0.22                     # width of the square view
WHITE, GREY, BLACK = 0.8, 0.3, 0.03

ap = argparse.ArgumentParser()
ap.add_argument("--rpm", type=float, default=2.0)
ap.add_argument("--t_end", type=float, default=0.9)
ap.add_argument("--dt", type=float, default=1 / 120)
ap.add_argument("--size", type=int, default=400)
ap.add_argument("--out", default="renders/fan.npz")
a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

s = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)


def mat(v):
    m = bpy.data.materials.new(f"m{v}")
    m.diffuse_color = (v, v, v, 1)
    return m


def mesh(name, verts, v, parent):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], [list(range(len(verts)))])
    me.materials.append(mat(v))
    o = bpy.data.objects.new(name, me)
    s.collection.objects.link(o)
    o.parent = parent
    return o


hub = bpy.data.objects.new("hub", None)
s.collection.objects.link(hub)
mesh("disc", [(HUB_R * math.cos(a), HUB_R * math.sin(a), 0.001)
              for a in np.linspace(0, 2 * math.pi, 64, endpoint=False)], GREY, hub)

for i, label in enumerate(LABELS):
    ang = 2 * math.pi * i / len(LABELS)
    mid = (BLADE_R0 + BLADE_R1) / 2
    c, sn = math.cos(ang), math.sin(ang)
    corners = [(BLADE_R0, -BLADE_W / 2), (BLADE_R1, -BLADE_W / 2), (BLADE_R1, BLADE_W / 2), (BLADE_R0, BLADE_W / 2)]
    mesh(f"blade{i}", [(x * c - y * sn, x * sn + y * c, 0) for x, y in corners], WHITE, hub)

    txt = bpy.data.curves.new(f"t{i}", "FONT")
    txt.body, txt.size = label, TEXT_H
    txt.align_x, txt.align_y = "CENTER", "CENTER"
    txt.materials.append(mat(BLACK))
    t = bpy.data.objects.new(f"t{i}", txt)
    t.location = (mid * math.cos(ang), mid * math.sin(ang), 0.002)
    t.rotation_euler.z = ang
    s.collection.objects.link(t)
    t.parent = hub

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.type, cam.data.ortho_scale = "ORTHO", VIEW_W
cam.location = (0, 0, 1)
s.collection.objects.link(cam)
s.camera = cam

s.world = bpy.data.worlds.new("w")
s.world.color = (0, 0, 0)
s.render.engine = "BLENDER_WORKBENCH"
s.display.shading.light = "FLAT"
s.display.shading.color_type = "MATERIAL"
s.display.shading.background_type = "WORLD"
s.display.shading.show_specular_highlight = False
s.display.render_aa = "32"
s.view_settings.view_transform = "Standard"
s.render.resolution_x = s.render.resolution_y = a.size
s.render.resolution_percentage = 100
s.render.image_settings.file_format = "OPEN_EXR"
s.render.image_settings.color_depth = "32"

tmp = Path(tempfile.mkdtemp()) / "f.exr"
s.render.filepath = str(tmp)


def render(theta):
    hub.rotation_euler.z = theta
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(tmp), check_existing=False)
    px = np.empty(a.size * a.size * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(a.size, a.size, 4)[::-1, :, 1].copy()  # blender stores bottom row first


omega = a.rpm * 2 * math.pi / 60
t = (np.arange(math.ceil(a.t_end / a.dt)) + 0.5) * a.dt
frames = np.stack([render(omega * tk) for tk in t])
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, frames=frames, t=t, dt=a.dt, omega=omega, ref=render(0.0),
         px_m=VIEW_W / a.size, rim_px=BLADE_R1 / (VIEW_W / a.size))
print(f"{len(t)} frames {a.size}px, {a.rpm} rpm, rim {BLADE_R1 * omega * a.dt / (VIEW_W / a.size):.2f} px per frame -> {out}")
