"""Render a SAMSON path (for example a bridge slow_growth run) as a clip.

    python samson_clip.py "<path name prefix>" OUTDIR [--fps F] [--size WxH] [--title T]

Needs the bridge running in SAMSON with python.exec allowed. Run it with any Python
that has numpy and Pillow; everything goes through the bridge:

1. Read every frame of the path (python.exec steps it, structure.get reads it).
2. Align the frames for viewing: centre of mass on the camera target and overall
   rotation removed (mass-weighted Kabsch against the first frame), with the first
   frame's nucleophile→leaving-group axis along the screen x of SAMSON's front
   view. Only the tumbling of the free molecule is removed.
3. Add a carbon–nucleophile bond (one undo step, removed at the end), then for each
   frame write the aligned positions into SAMSON without an undo step, show one
   carbon–halogen bond (to whichever halogen is less stretched relative to its
   equilibrium bond length; a drawing convention, not a bond order), capture the
   viewport, and label it with ξ and both distances.

Each frame is its own request, so SAMSON keeps processing events. Writes
OUTDIR/frames/*.png, clip.gif and clip.webp, and leaves the path on its first frame.
The nucleophile and leaving group must be different elements.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from samson_mlip_visualizer.remote.client import SamsonClient  # noqa: E402

MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Cl": 35.453,
          "Br": 79.904, "I": 126.904}
BOND_TO_CARBON = {"F": 1.38, "Cl": 1.79, "Br": 1.94, "I": 2.14, "O": 1.43, "N": 1.47}

parser = argparse.ArgumentParser()
parser.add_argument("path_prefix")
parser.add_argument("outdir")
parser.add_argument("--fps", type=float, default=15)
parser.add_argument("--size", default="960x600")
parser.add_argument("--fs-per-frame", type=float, default=20.0)
parser.add_argument("--carbon", type=int, default=0)
parser.add_argument("--nucleophile", type=int, default=5)
parser.add_argument("--leaving", type=int, default=4)
parser.add_argument("--title", default="F⁻ + CH₃Cl, fine-tuned AIMNet2, slow growth at 300 K")
args = parser.parse_args()
width, height = (int(v) for v in args.size.split("x"))
out = Path(args.outdir)
(out / "frames").mkdir(parents=True, exist_ok=True)
client = SamsonClient.from_connection_file(timeout=120)


def execute(code):
    return client.call("python.exec", code=code)["result"]


find = ("p = [n for n in SAMSON.getNodes('node.type path') "
        f"if n.name.startswith({args.path_prefix!r})][0]\n")
symbols = client.call("structure.get")["symbols"]
nucleophile, leaving = symbols[args.nucleophile], symbols[args.leaving]
if nucleophile == leaving:
    raise SystemExit("the nucleophile and the leaving group must be different elements")
m = np.array([MASSES[s] for s in symbols])
steps = int(execute(find + "int(p.numberOfSteps)"))
frames = []
for k in range(steps):
    execute(find + f"p.currentStep = {k}")
    frames.append(np.asarray(client.call("structure.get")["positions"]))
print(f"read {steps} frames")

# SAMSON's front view looks along +y with +z up, so screen x is world x.
execute("SAMSON.getActiveCamera().frontView()")
target = np.array([float(v.split()[0]) / 100 for v in  # pm → Å
                   execute("str(SAMSON.getActiveCamera().getTargetPosition())")
                   .strip("'()").split(",")])


def centred(x):
    return x - (m[:, None] * x).sum(0) / m.sum()


ref = centred(frames[0])
axis = ref[args.leaving] - ref[args.nucleophile]
axis /= np.linalg.norm(axis)
v, c = np.cross(axis, [1.0, 0.0, 0.0]), axis[0]
vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
turn = np.diag([-1.0, -1.0, 1.0]) if c < -0.999 else np.eye(3) + vx + vx @ vx / (1 + c)
ref = ref @ turn.T  # Rodrigues: axis → +x
aligned = []
for x in frames:
    x = centred(x)
    u, _, vt = np.linalg.svd((m[:, None] * x).T @ ref)
    rotation = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    aligned.append(x @ rotation.T + target)

bonds = (f"atoms = list(SAMSON.getNodes('node.type atom'))\n"
         f"carbon, nucleophile = atoms[{args.carbon}], atoms[{args.nucleophile}]\n")
execute(bonds + "SAMSON.beginHolding('Clip: carbon-nucleophile bond')\n"
        "bond = samson.SBBond(carbon, nucleophile)\nbond.create()\n"
        "list(SAMSON.getNodes('node.type structuralModel'))[0]"
        ".getStructuralRoot().addChild(bond)\nSAMSON.endHolding()")
show_bonds = ("for b in SAMSON.getNodes('node.type bond'):\n"
              "    ends = {str(b.leftAtom.elementSymbol), str(b.rightAtom.elementSymbol)}\n"
              f"    if ends == {{'C', {leaving!r}}}: b.visibilityFlag = LEAVING\n"
              f"    if ends == {{'C', {nucleophile!r}}}: b.visibilityFlag = NUCLEOPHILE\n")
write = ("from samson_mlip_visualizer.samson_bridge import extract_structure, sync_positions\n"
         "sync_positions(extract_structure(SAMSON), np.array(POSITIONS), samson=SAMSON, "
         "process_events=False)\n" + show_bonds)
font = ImageFont.truetype("segoeui.ttf", 22)
small_font = ImageFont.truetype("segoeui.ttf", 17)
images = []
try:
    for k, x in enumerate(aligned):
        d_leaving = np.linalg.norm(x[args.carbon] - x[args.leaving])
        d_nucleophile = np.linalg.norm(x[args.carbon] - x[args.nucleophile])
        swapped = bool(d_nucleophile / BOND_TO_CARBON[nucleophile]
                       < d_leaving / BOND_TO_CARBON[leaving])
        execute(write.replace("POSITIONS", json.dumps(np.round(x, 5).tolist()))
                .replace("NUCLEOPHILE", str(swapped)).replace("LEAVING", str(not swapped)))
        png = out / "frames" / f"frame_{k:04d}.png"
        # Capture 1.5× larger and keep the centre: closer, without the navigation cube.
        client.call("view.capture", path=str(png), width=width * 3 // 2,
                    height=height * 3 // 2)
        image = Image.open(png).convert("RGB").crop(
            (width // 4, height // 4, width // 4 + width, height // 4 + height))
        draw = ImageDraw.Draw(image)
        draw.text((18, 14), f"ξ = {d_leaving - d_nucleophile:+.2f} Å    "
                  f"d(C–{nucleophile}) = {d_nucleophile:.2f} Å    "
                  f"d(C–{leaving}) = {d_leaving:.2f} Å", fill=(20, 20, 20), font=font)
        draw.text((18, 44), f"t = {k * args.fs_per_frame / 1000:.2f} ps    {args.title}",
                  fill=(60, 60, 60), font=small_font)
        image.save(png)
        images.append(image)
        if k % 20 == 0:
            print(f"frame {k:4d}  ξ {d_leaving - d_nucleophile:+.3f}", flush=True)
finally:
    execute(bonds + "SAMSON.beginHolding('Clip: remove the carbon-nucleophile bond')\n"
            + show_bonds.replace("LEAVING", "True").replace("NUCLEOPHILE", "True")
            + "for b in list(SAMSON.getNodes('node.type bond')):\n"
            f"    if {{str(b.leftAtom.elementSymbol), str(b.rightAtom.elementSymbol)}} == "
            f"{{'C', {nucleophile!r}}}: b.erase()\nSAMSON.endHolding()\n"
            + find + "p.currentStep = 0")

duration = int(1000 / args.fps)
small = [im.resize((width * 2 // 3, height * 2 // 3), Image.LANCZOS) for im in images]
small[0].save(out / "clip.gif", save_all=True, append_images=small[1:], duration=duration,
              loop=0, optimize=True)
images[0].save(out / "clip.webp", save_all=True, append_images=images[1:],
               duration=duration, loop=0, quality=85)
print(f"wrote {out / 'clip.gif'} and {out / 'clip.webp'}")
