"""Write the HPC smoke tests: small labeling and fine-tuning packages that check
each piece of the fine-tuning workflow on a real cluster before a real campaign.

    python make_smoke_tests.py [OUTPUT_DIR]      (default D:\\MLIP_Work_Folder\\hpc_smoke_tests)

1. 01_label_gaussian: Gaussian Force on 3 HCN/HNC geometries (PBEPBE/def2TZVP).
2. 02_label_orca: the same with ORCA EnGrad (PBE def2-TZVP def2/J).
3. 03_train_plain: plain fine-tuning of MACE-MP-0 small, 1 seed, 5 epochs, GPU.
4. 04_train_multihead_mp: multihead fine-tuning with 500 Materials Project
   replay structures, 1 seed, 3 epochs, GPU (download step on a login node).

Tests 1-2 check that the collectors read real outputs; the three geometries
also have Psi4 PBE/def2-TZVP labels (computed here, if Psi4 is installed) to
compare against. Tests 3-4 check that mace-torch, the GPU, and the replay data
work on the cluster. Back on the desktop, check_smoke_results.py checks
everything that was copied back.
"""

import json
import shutil
import sys
from pathlib import Path

from ase import Atoms

from samson_mlip_visualizer.finetune import Selection, write_selection
from samson_mlip_visualizer.labeling import SlurmSettings, write_label_package
from samson_mlip_visualizer.paths import foundation_model
from samson_mlip_visualizer.training import GpuSlurmSettings, TrainingSpec, write_training_package

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\MLIP_Work_Folder\hpc_smoke_tests")
HERE = Path(__file__).parent
FOUNDATION = foundation_model()
HCN_DATA = HERE / "hcn_training_87_structures.extxyz"
# LONI QB4: ORCA 6.1.1 and OpenMPI 4.1.8 from the project folder; Gaussian from its
# module; a conda env with mace-torch 0.3.16 for the GPU jobs (see the README).
ORCA_QB4 = dict(modules=(),  # the user's OpenMPI 4.1.8, the one ORCA 6.1.1 is built with
                setup=("OMPI_DIR=/ddnB/project/ramu/lgutsev/openmpi-4.1.8",
                       "ORCA_DIR=/ddnB/project/ramu/lgutsev/Orca_6_1_1",
                       'export PATH="$ORCA_DIR:$OMPI_DIR/bin:$PATH"',
                       'export LD_LIBRARY_PATH="$ORCA_DIR/lib:$OMPI_DIR/lib:${LD_LIBRARY_PATH:-}"'))
GPU_QB4 = dict(account="loni_perovsk27", partition="gpu2", modules=(),
               activate=("source /home/lgutsev/miniforge3/etc/profile.d/conda.sh && "
                         "conda activate /project/lgutsev/env/mace_env"),
               cache_dir="/project/lgutsev/cache")  # not ~/.cache: home is 10 GB

# HCN, the transition state, and HNC from the MACE-MP-0 IRC (atom order C, N, H).
GEOMETRIES = {
    "HCN": [[0.0, 0.190947, 0.098346], [0.0, -0.117311, 1.220035], [0.0, 0.478946, -0.951011]],
    "TS": [[0.0, -0.000156, -0.000053], [0.0, 0.000263, 1.203814], [0.0, 1.122285, 0.446894]],
    "HNC": [[0.0, -0.14585, -0.023794], [0.0, 0.174106, 1.105805], [0.0, 0.452125, 2.087416]],
}


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"{OUT} is not empty; remove it or give another folder")
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [Atoms("CNH", positions=p) for p in GEOMETRIES.values()]
    chosen = Selection()
    for index in range(len(frames)):
        chosen.add(index, "smoke-test", None, float("inf"))
    frames_path, manifest_path = write_selection(
        OUT / "_frames", frames, chosen, source="HCN <-> HNC IRC (MACE-MP-0 small)",
        model="MACE-MP-0 small", notes={"names": list(GEOMETRIES)},
    )
    # LONI: the 'single' partition takes an 8-core Gaussian job; GPU jobs go to 'gpu2'.
    cpu = SlurmSettings(account="loni_perovsk27", partition="single", cpus=8, memory_gb=16,
                        time="00:30:00", max_parallel=None, **ORCA_QB4)
    gaussian_cpu = SlurmSettings(account="loni_perovsk27", partition="single", cpus=8,
                                 memory_gb=16, time="00:30:00", max_parallel=None,
                                 modules=("gaussian/g16-c01",))
    write_label_package(OUT / "01_label_gaussian", frames_path, manifest_path, code="gaussian",
                        slurm=gaussian_cpu)
    write_label_package(OUT / "02_label_orca", frames_path, manifest_path, code="orca", slurm=cpu)
    reference_psi4(frames)

    gpu = GpuSlurmSettings(time="00:30:00", **GPU_QB4)
    card = {"reference": "PBE/def2-TZVP (Psi4)", "scope": "HPC smoke test only; not a model"}
    write_training_package(
        TrainingSpec(name="smoke_plain", foundation=str(FOUNDATION), train_file=str(HCN_DATA),
                     seeds=(1,), epochs=5, card=card),
        OUT / "03_train_plain", slurm=gpu,
    )
    write_training_package(
        TrainingSpec(name="smoke_multihead_mp", foundation=str(FOUNDATION),
                     train_file=str(HCN_DATA), mode="multihead", replay="mp",
                     replay_samples=500, seeds=(1,), epochs=3, card=card),
        OUT / "04_train_multihead_mp", slurm=gpu,
    )
    shutil.rmtree(OUT / "_frames")
    shutil.copyfile(HERE / "check_smoke_results.py", OUT / "check_smoke_results.py")
    shutil.copyfile(HERE / "README.md", OUT / "README.md")
    print(f"Wrote the smoke tests to {OUT}")


def reference_psi4(frames) -> None:
    """PBE/def2-TZVP energies and forces for the three geometries, if Psi4 is here."""
    from samson_mlip_visualizer.psi4_backend import Psi4Calculator, find_psi4

    python = find_psi4()
    if python is None:
        print("Psi4 not found: no reference file; the check will skip the comparison")
        return
    calc = Psi4Calculator(python, method="pbe", basis="def2-tzvp")
    reference = {}
    for name, frame in zip(GEOMETRIES, frames, strict=True):
        atoms = frame.copy()
        atoms.calc = calc
        reference[name] = {"energy_ev": atoms.get_potential_energy(),
                           "forces_ev_per_A": atoms.get_forces().tolist()}
    calc.close()
    (OUT / "reference_psi4_pbe_def2tzvp.json").write_text(json.dumps(reference, indent=1))


if __name__ == "__main__":
    main()
