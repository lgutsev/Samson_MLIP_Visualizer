"""How good is the PBE0 reference? DLPNO-CCSD(T) on the same frames, for LONI.
Nothing is submitted from here.

PBE0 is the reference of this example, and not the last word for Ni–CO bonding.
This writes ORCA single points (``job="energy"``: DLPNO-CCSD(T) has no analytic
gradients in ORCA) at DLPNO-CCSD(T)/def2-TZVP with TightPNO, on frames that
already have PBE0 labels:

- ``cc_smoke/`` (also ``hpc_smoke_tests\\09_orca_nico4_dlpno``): 3 points of the
  CO pull (Ni–C 1.80, 2.70, 5.00 Å); the CCSD(T) pull energy against PBE0's
  1.10 eV is the first answer;
- ``cc_campaign/``: the whole pull (15 points) and the held-out 650 K MD frames (15).

Back on the desktop, ``collect_labels`` gives ``REF_energy`` per frame. If
PBE0's pull energy or its energies along the MD differ from CCSD(T) by more than
~0.1 eV, a second correction (CCSD(T) − PBE0, energies only, on these 30 points)
is the next step.
"""

import shutil
import sys

from ase.io import read
from common import POOL, TEST_SET, WORK

from samson_mlip_visualizer.finetune import Selection, write_selection
from samson_mlip_visualizer.labeling import SlurmSettings, write_label_package

LEVEL = "DLPNO-CCSD(T) def2-TZVP def2-TZVP/C TightPNO"
SMOKE_COPY = r"D:\MLIP_Work_Folder\hpc_smoke_tests\09_orca_nico4_dlpno"
frames = read(POOL, ":") + read(TEST_SET, ":")
scan = sorted([f for f in frames if f.info["group"] == "scan"], key=lambda f: f.info["r_nic"])
md = [f for f in frames if f.info["group"] == "md650"]


def nearest(r):
    return min(scan, key=lambda f: abs(f.info["r_nic"] - r))


choices = {"smoke": [nearest(1.80), nearest(2.70), nearest(5.00)], "campaign": scan + md}
for name in sys.argv[1:] or ["smoke", "campaign"]:
    chosen = choices[name]
    selection = Selection()
    for index in range(len(chosen)):
        selection.add(index, "reference-check", None, float("inf"))
    frames_path, manifest_path = write_selection(
        WORK / f"_cc_{name}_frames", chosen, selection,
        source="Ni(CO)4 frames with PBE0/def2-TZVP labels", model="PBE0/def2-TZVP (Psi4)",
        notes={"r_nic": [f.info["r_nic"] for f in chosen],
               "group": [f.info["group"] for f in chosen],
               "pbe0_energy_eV": [f.info["REF_energy"] for f in chosen]})
    target = WORK / f"cc_{name}"
    if target.exists():
        sys.exit(f"{target} exists; remove it to write the package again")
    write_label_package(target, frames_path, manifest_path, code="orca", level=LEVEL,
                        job="energy",
                        slurm=SlurmSettings(cpus=16, memory_gb=64, time="06:00:00",
                                            max_parallel=None if name == "smoke" else 15))
    print(f"{name}: {len(chosen)} frames -> {target}")
    if name == "smoke":
        if shutil.os.path.exists(SMOKE_COPY):
            sys.exit(f"{SMOKE_COPY} exists; remove it to copy the smoke test again")
        shutil.copytree(target, SMOKE_COPY)
        print(f"smoke: copied to {SMOKE_COPY}")
