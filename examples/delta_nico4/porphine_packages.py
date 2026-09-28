"""The scale-up for LONI: Ni(II) porphine (NiP, 37 atoms), labeled with ORCA.
Nothing is submitted from here.

PBE0/def2-TZVP gradients on 37 atoms take minutes each even with RIJCOSX, too
much for the laptop for a campaign, so this writes two ORCA packages:

- ``porphine_smoke/`` (also copied to
  ``D:\\MLIP_Work_Folder\\hpc_smoke_tests\\06_orca_ni_porphine``): 3 frames — the
  GFN2-xTB minimum, a rattled copy (σ 0.03 Å), and a 600 K MD frame. It checks
  ORCA on a transition-metal complex (closed-shell singlet Ni(II), d⁸ square
  planar), the collector, and the cost per frame;
- ``porphine_campaign/``: 150 frames of GFN2-xTB MD at 300 and 600 K plus
  rattled minima, for the same learning curves as Ni(CO)₄.

The structure is built from standard porphyrin bond geometry (D4h, Ni–N 1.96 Å)
and relaxed with GFN2-xTB. Usage: ``porphine_packages.py [smoke|campaign ...]``.
"""

import shutil
import sys

import numpy as np
from ase import Atoms, units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from common import WORK, xtb

from samson_mlip_visualizer.finetune import Selection, write_selection
from samson_mlip_visualizer.labeling import SlurmSettings, write_label_package

LEVEL = "PBE0 def2-TZVP def2/J RIJCOSX"
SMOKE_COPY = r"D:\MLIP_Work_Folder\hpc_smoke_tests\06_orca_ni_porphine"
which = sys.argv[1:] or ["smoke", "campaign"]


def ni_porphine():
    """Ni at the origin, the four pyrrole N on the axes, meso carbons on the diagonals."""
    pyrrole = [("N", 2.00, 0.00), ("C", 2.84, 1.10), ("C", 2.84, -1.10), ("C", 4.22, 0.68),
               ("C", 4.22, -0.68), ("H", 5.10, 1.32), ("H", 5.10, -1.32)]
    symbols, positions = ["Ni"], [[0.0, 0.0, 0.0]]
    for k in range(4):
        c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        for symbol, x, y in pyrrole:
            symbols.append(symbol)
            positions.append([c * x - s * y, s * x + c * y, 0.0])
        meso = np.array([np.cos((k + 0.5) * np.pi / 2), np.sin((k + 0.5) * np.pi / 2), 0.0])
        symbols += ["C", "H"]
        positions += [3.42 * meso, 4.50 * meso]
    return Atoms(symbols, positions=positions)


gfn2 = xtb("gfn2")
minimum = ni_porphine()
minimum.calc = gfn2
BFGS(minimum, logfile=None).run(fmax=0.005, steps=500)
d = minimum.get_all_distances()
print(f"GFN2-xTB minimum: Ni–N {d[0, 1]:.3f} Å, {len(minimum)} atoms", flush=True)
rng = np.random.default_rng(5)


def plain(atoms):
    return Atoms(atoms.get_chemical_symbols(), positions=atoms.positions)


def md(temperature, count, seed):
    atoms = minimum.copy()
    atoms.calc = gfn2
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=np.random.default_rng(seed))
    dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=temperature, friction=0.02,
                   rng=np.random.default_rng(seed + 1))
    dyn.run(200)
    out = []
    for _ in range(count):
        dyn.run(40)
        out.append(plain(atoms))
    return out


rattled = [Atoms(minimum.get_chemical_symbols(),
                 positions=minimum.positions + rng.normal(0, 0.03, minimum.positions.shape))
           for _ in range(10)]
frames = {"smoke": [plain(minimum), rattled[0]] + md(600, 1, seed=1)}
if "campaign" in which:
    frames["campaign"] = [plain(minimum)] + rattled + md(300, 70, seed=2) + md(600, 69, seed=3)
cpu = {"smoke": SlurmSettings(cpus=16, memory_gb=32, time="04:00:00", max_parallel=None),
       "campaign": SlurmSettings(cpus=16, memory_gb=32, time="04:00:00", max_parallel=20)}
for name in which:
    chosen = Selection()
    for index in range(len(frames[name])):
        chosen.add(index, "smoke-test" if name == "smoke" else "campaign", None, float("inf"))
    frames_path, manifest_path = write_selection(
        WORK / f"_porphine_{name}_frames", frames[name], chosen,
        source="Ni(II) porphine, GFN2-xTB minimum and MD", model="GFN2-xTB")
    target = WORK / f"porphine_{name}"
    if target.exists():
        raise SystemExit(f"{target} exists; remove it to write the package again")
    write_label_package(target, frames_path, manifest_path, code="orca", level=LEVEL,
                        slurm=cpu[name])
    print(f"{name}: {len(frames[name])} frames -> {target}", flush=True)
    if name == "smoke":
        if shutil.os.path.exists(SMOKE_COPY):
            raise SystemExit(f"{SMOKE_COPY} exists; remove it to copy the smoke test again")
        shutil.copytree(target, SMOKE_COPY)
        write(target / "minimum_gfn2.xyz", minimum)
        print(f"smoke: copied to {SMOKE_COPY}", flush=True)
