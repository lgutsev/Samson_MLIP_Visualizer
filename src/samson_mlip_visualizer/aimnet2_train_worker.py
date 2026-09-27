"""Fine-tune one AIMNet2 model; run in the aimnet environment by :mod:`.active_learning`.

    python aimnet2_train_worker.py job.json

This file imports nothing from this package (it runs in another environment):
only the standard library, NumPy, torch, ASE, and aimnet. The job file holds:

- ``base_model``: an AIMNet2 ``.pt`` (registry format 2) to start from;
- ``out_model``: where to write the tuned model (same format, so the AIMNet2
  backend loads it as ``aimnet_model``);
- ``data``: an extxyz with ``REF_energy_raw`` (reference energy, eV),
  ``REF_forces``, ``info["charge"]``, and optionally ``info["weight"]``;
- ``epochs``, ``lr``, ``e_scale`` (eV), ``f_scale`` (eV/Å), ``seed``, ``log``.

Training goes through ``AIMNet2Calculator(train=True)``, the exact inference
path (network + embedded Coulomb + D3), so nothing has to be kept consistent by
hand; ``aimnet train --load`` would silently start from random weights when
given the registry wrapper. Energies: per-element offsets between the reference
and the model, fitted once by least squares on the starting model, cancel in
every reaction energy. Loss: weighted mean (ΔE/e_scale)² + mean (ΔF/f_scale)²,
full batch (one batch per atom count). The state with the lowest training loss
(checked every 10 epochs) is kept, which guards against a late Adam spike.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # torch.compile needs MSVC on Windows

import numpy as np  # noqa: E402
import torch  # noqa: E402
from aimnet.calculators import AIMNet2Calculator  # noqa: E402
from ase.io import read  # noqa: E402

ELEMENTS = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53)  # AIMNet2 wB97M-D3 species


def batches(images):
    groups = []
    for n in sorted({len(a) for a in images}):
        index = [k for k, a in enumerate(images) if len(a) == n]
        chosen = [images[k] for k in index]
        groups.append({
            "index": torch.tensor(index),
            "coord": torch.tensor(np.array([a.positions for a in chosen]), dtype=torch.float32),
            "numbers": torch.tensor(np.array([a.numbers for a in chosen])),
            "charge": torch.tensor([float(a.info.get("charge", 0)) for a in chosen]),
            "forces": torch.tensor(np.array([a.arrays["REF_forces"] for a in chosen]),
                                   dtype=torch.float32),
            "weight": torch.tensor([float(a.info.get("weight", 1.0)) for a in chosen]),
        })
    return groups


def predict(calc, groups, count):
    energies = torch.zeros(count, dtype=torch.float64)
    forces = []
    for group in groups:
        out = calc.eval({"coord": group["coord"], "numbers": group["numbers"],
                         "charge": group["charge"]}, forces=True)
        energies = energies.index_put((group["index"],), out["energy"].double().reshape(-1))
        forces.append((out["forces"].reshape(group["forces"].shape), group))
    return energies, forces


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    torch.manual_seed(int(job.get("seed", 0)))
    images = read(job["data"], ":")
    groups = batches(images)
    e_ref = torch.tensor([a.info["REF_energy_raw"] for a in images], dtype=torch.float64)
    weights = torch.tensor([float(a.info.get("weight", 1.0)) for a in images],
                           dtype=torch.float64)
    counts = torch.tensor([[int((a.numbers == z).sum()) for z in ELEMENTS] for a in images],
                          dtype=torch.float64)
    calc = AIMNet2Calculator(job["base_model"], device="cpu", train=True)
    for parameter in calc.model.parameters():
        parameter.requires_grad_(True)
    energy, _ = predict(calc, groups, len(images))
    offsets = torch.tensor(np.linalg.lstsq(counts.numpy(), (e_ref - energy.detach()).numpy(),
                                           rcond=None)[0], dtype=torch.float64)
    e_scale, f_scale = float(job.get("e_scale", 0.010)), float(job.get("f_scale", 0.050))
    optimizer = torch.optim.Adam(calc.model.parameters(), lr=float(job.get("lr", 5e-5)))
    history, best, best_state, start = [], float("inf"), None, time.perf_counter()
    for epoch in range(1, int(job["epochs"]) + 1):
        optimizer.zero_grad()
        energy, forces = predict(calc, groups, len(images))
        de = energy + counts @ offsets - e_ref
        energy_loss = (weights * (de / e_scale).pow(2)).sum() / weights.sum()
        force_sum = sum((g["weight"][:, None, None] * ((f - g["forces"]) / f_scale).pow(2)).sum()
                        for f, g in forces)
        force_count = sum((g["weight"][:, None, None] * torch.ones_like(g["forces"])).sum()
                          for _, g in forces)
        loss = energy_loss + (force_sum / force_count).double()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            value = float(loss.detach())
            rmse_e = float(de.detach().pow(2).mean().sqrt())
            history.append({"epoch": epoch, "loss": value, "energy_rmse_ev": rmse_e})
            if value < best:
                best, best_state = value, {k: v.detach().clone()
                                           for k, v in calc.model.state_dict().items()}
    calc.model.load_state_dict(best_state)
    checkpoint = torch.load(job["base_model"], map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = {k: v.cpu() for k, v in best_state.items()}
    Path(job["out_model"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, job["out_model"])
    Path(job["log"]).write_text(json.dumps({
        "training_s": round(time.perf_counter() - start, 1), "best_loss": best,
        "offsets_ev": dict(zip(map(str, ELEMENTS), offsets.tolist(), strict=True)),
        "history": history}, indent=1))


if __name__ == "__main__":
    main()
