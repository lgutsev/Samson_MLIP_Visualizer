"""UMA worker process for :mod:`.uma_backend`.

This file runs in the Python that has ``fairchem-core`` (its own environment), not
in SAMSON's, so it imports nothing from this package: only the standard library,
NumPy, ASE, and fairchem. Protocol as in :mod:`.worker_process`.

Request: ``{"model" (.pt path or fairchem name), "task", "device", "charge",
"multiplicity", "numbers", "positions" (Å), "cell" (or null), "pbc"}``.
Reply: ``{"energy" (eV), "forces" (eV/Å)}`` or ``{"error"}``.
"""

import json
import os
import sys

PREFIX = "@@SAMSON "
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # torch.compile needs a C++ compiler


def main() -> None:
    protocol = sys.stdout
    sys.stdout = sys.stderr  # stray Python prints must not reach the protocol stream

    def reply(message):
        protocol.write(PREFIX + json.dumps(message) + "\n")
        protocol.flush()

    try:
        from importlib.metadata import version

        import numpy as np
        from ase import Atoms
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        reply({"error": f"Could not import fairchem: {type(exc).__name__}: {exc}"})
        return
    reply({"ready": True, "version": version("fairchem-core")})

    units = {}  # (model, device) -> predict unit; loading the checkpoint is the slow part
    calculators = {}  # (model, device, task) -> FAIRChemCalculator
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            unit_key = (request["model"], request.get("device", "cpu"))
            if unit_key not in units:
                model, device = unit_key
                if model.lower().endswith(".pt"):
                    units[unit_key] = pretrained_mlip.load_predict_unit(model, device=device)
                else:
                    units[unit_key] = pretrained_mlip.get_predict_unit(model, device=device)
            key = (*unit_key, request.get("task", "omol"))
            if key not in calculators:
                calculators[key] = FAIRChemCalculator(units[unit_key], task_name=key[2])
            cell = request.get("cell")
            atoms = Atoms(numbers=request["numbers"], positions=request["positions"],
                          cell=cell, pbc=request.get("pbc", False) if cell else False)
            if key[2] == "omol":
                atoms.info.update(charge=int(request.get("charge", 0)),
                                  spin=int(request.get("multiplicity", 1)))
            atoms.calc = calculators[key]
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            reply({"energy": energy, "forces": forces.tolist()})
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            reply({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
