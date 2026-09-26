"""AIMNet2 worker process for :mod:`.aimnet2_backend`.

This file runs in the Python that has the ``aimnet`` package (its own
environment), not in SAMSON's, so it imports nothing from this package: only
the standard library, NumPy, and aimnet. Protocol as in :mod:`.worker_process`.

Request: ``{"model" (registry name or .pt path), "device", "charge",
"multiplicity", "numbers", "positions" (Å)}``.
Reply: ``{"energy" (eV), "forces" (eV/Å)}`` or ``{"error"}``.
"""

import json
import os
import sys

PREFIX = "@@SAMSON "
# aimnet's neighbor list is wrapped in torch.compile, which needs a C++ compiler
# (MSVC on Windows) that is usually absent; eager mode gives the same numbers.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def main() -> None:
    protocol = sys.stdout
    sys.stdout = sys.stderr  # stray Python prints must not reach the protocol stream

    def reply(message):
        protocol.write(PREFIX + json.dumps(message) + "\n")
        protocol.flush()

    try:
        from importlib.metadata import version

        import numpy as np
        from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator
        from ase import Atoms
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        reply({"error": f"Could not import aimnet: {type(exc).__name__}: {exc}"})
        return
    reply({"ready": True, "version": version("aimnet")})

    models = {}  # (model, device) -> AIMNet2Calculator; loading is the slow part
    calculators = {}  # (model, device, charge, multiplicity) -> AIMNet2ASE
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            model_key = (request["model"], request.get("device", "cpu"))
            if model_key not in models:
                models[model_key] = AIMNet2Calculator(model_key[0], device=model_key[1])
            key = (*model_key, int(request.get("charge", 0)), int(request.get("multiplicity", 1)))
            if key not in calculators:
                calculators[key] = AIMNet2ASE(models[model_key], charge=key[2], mult=key[3])
            atoms = Atoms(numbers=request["numbers"], positions=request["positions"])
            atoms.calc = calculators[key]
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            reply({"energy": energy, "forces": forces.tolist()})
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            reply({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
