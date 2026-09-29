"""Step 2 (laptop): the LONI packages.  Nothing is submitted from here.

Pristine (``frames.extxyz``, from ``make_frames.py``):

- ``smoke/`` (also copied to ``D:\\MLIP_Work_Folder\\hpc_smoke_tests\\05_vasp_bbvo``):
  3 frames — the PBE+U primitive cell itself, one rattled primitive cell, and
  one 40-atom MD frame. It checks the VASP inputs, the PBE+U → HSE06 restart, the
  collector, and the cost of a 40-atom HSE06 frame. On the primitive cell, the
  energies should come out near the earlier runs of this structure on the same
  5×5×5 mesh: HSE06 −84.208 eV (PRECFOCK = Fast there, Normal here), PBE+U
  −66.071 eV (7×7×7 and LREAL = Auto there, so a few meV off).
- ``campaign/``: every frame of ``frames.extxyz``. Heavy: HSE06 on 40-atom cells
  takes node-hours per frame; run the smoke test first.

V-site substitution series (``doped_frames.extxyz``, from ``doping.py``):

- ``doped_smoke/`` (also ``hpc_smoke_tests\\07_vasp_bbvo_doped``): 3 relaxed
  40-atom cells — the two end members Ba₂BiNbO₆ and Ba₂BiTaO₆, and Nb at
  x = 0.5. It checks the Nb_pv / Ta_pv POTCARs, that PBE+U puts U on V only, and
  the cost of the doped cells;
- ``doped_campaign/``: every doped frame, plus the relaxed pristine cell. Heavy.

Dilute dopant (``dilute_frames.extxyz``, from ``dilute.py``):

- ``dilute/``: one Nb or Ta in the 80-atom cell (x = 12.5 %), relaxed and at
  600 K, 4 frames, all held out: do corrections trained on 40-atom cells transfer?
  HSE06 on 80 atoms: the heaviest frames here.

Usage: ``make_packages.py [smoke|campaign|doped_smoke|doped_campaign|dilute ...]``
(default: all whose frames exist).
"""

import shutil
import sys

from ase.io import read
from common import KSPACING, WORK

from samson_mlip_visualizer.finetune import Selection, write_selection
from samson_mlip_visualizer.vasp_labeling import VaspSlurmSettings, write_vasp_package

SMOKE_COPIES = {"smoke": r"D:\MLIP_Work_Folder\hpc_smoke_tests\05_vasp_bbvo",
                "doped_smoke": r"D:\MLIP_Work_Folder\hpc_smoke_tests\07_vasp_bbvo_doped"}
sources = {name: WORK / file for name, file in (
    ("smoke", "frames.extxyz"), ("campaign", "frames.extxyz"),
    ("doped_smoke", "doped_frames.extxyz"), ("doped_campaign", "doped_frames.extxyz"),
    ("dilute", "dilute_frames.extxyz"))}
which = sys.argv[1:] or [name for name, path in sources.items()
                         if path.exists() and not (WORK / name).exists()]


def choose(name, frames):
    groups = [f.info["group"] for f in frames]
    if name == "smoke":
        strain = [i for i, g in enumerate(groups) if g == "strain"]
        exact = next(i for i, f in enumerate(frames) if f.info.get("pbe_u_geometry"))
        return [exact, strain[-1], groups.index("md300")]
    if name == "doped_smoke":
        return [groups.index("nb1_min"), groups.index("ta1_min"), groups.index("nb0.5_min")]
    return list(range(len(frames)))


# One node per frame; KPAR splits k-points. QB4: 64 cores per node, 72 h at most.
# Measured on QB4 (smoke tests 05 and 08): HSE06 takes 3.3 h on the 10-atom cell
# and 9.1-14.8 h on 40 atoms, PBE+U under 2 min. The 12 h of the first doped
# smoke test killed all three of its HSE06 runs, so every 40-atom package gets
# 36 h; the 80-atom dilute cells the maximum.
slurm = {name: VaspSlurmSettings(account="loni_perovsk27", partition="workq",
                                 potcar_dir="/home/lgutsev/pot/potpaw_PBE", tasks_per_node=64,
                                 modules=("vasp6/6.5.1-cpu",),
                                 setup=("export SINGULARITYENV_OMP_NUM_THREADS=1",),
                                 run="srun vasp_std",
                                 time="72:00:00" if name == "dilute" else "36:00:00",
                                 max_parallel=None if "smoke" in name else 10)
         for name in sources}
extra = {"pbe_u": {"NCORE": 4, "KPAR": 4}, "hse06": {"NCORE": 4, "KPAR": 4}}
for name in which:
    frames = read(sources[name], ":")
    selection = Selection()
    for index in choose(name, frames):
        selection.add(index, "smoke-test" if "smoke" in name else "campaign", None,
                      float("inf"))
    frames_path, manifest_path = write_selection(
        WORK / f"_{name}_frames", frames, selection, source="MACE-MP-0 frames of Ba2BiVO6"
        + (" with V-site Nb/Ta" if name != "smoke" and name != "campaign" else ""),
        model="MACE-MP-0 small",
        notes={"groups": [frames[i].info["group"] for i in selection.indices]})
    target = WORK / name
    if target.exists():
        raise SystemExit(f"{target} exists; remove it to write the package again")
    write_vasp_package(target, frames_path, manifest_path, kspacing=KSPACING,
                       incar_extra=extra, slurm=slurm[name])
    print(f"{name}: {len(selection)} frames -> {target}")
    if name in SMOKE_COPIES:
        copy = SMOKE_COPIES[name]
        if shutil.os.path.exists(copy):
            raise SystemExit(f"{copy} exists; remove it to copy the smoke test again")
        shutil.copytree(target, copy)
        print(f"{name}: copied to {copy}")
