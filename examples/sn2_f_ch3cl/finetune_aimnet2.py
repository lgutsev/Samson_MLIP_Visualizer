"""Fine-tune AIMNet2 (ωB97M-D3, member 0) to ωB97X-D/def2-TZVPD for F- + CH3Cl.

Run with the Python of the aimnet environment (CPU is enough):

    python finetune_aimnet2.py [--fragments] [work folder]

Data: the 93 labels of the MACE fine-tune (``finetune/train.extxyz``: 31
AIMNet2 IRC frames + 2 rattled copies each, raw ωB97X-D energies and forces).
With ``--fragments``, also the 17 structures of ``fragment_data.py``
(``finetune_aimnet2/fragments.extxyz``): free F- and Cl-, CH3Cl and CH3F, and
the complexes pulled apart. Without them the model learns the complexes but not
their energy relative to the separated fragments.

Why a small loop here rather than ``aimnet train --load``: the registry file is
a format-2 wrapper (``state_dict`` next to the long-range Coulomb and D3
settings the calculator adds outside the network). ``aimnet train`` loads
weights with ``strict=False``, so handing it the wrapper silently trains from
random weights, and its data pipeline would need the D3 part removed from the
targets. Training through ``AIMNet2Calculator(train=True)`` uses the exact
inference path (network + Coulomb + D3), so nothing has to be kept consistent
by hand.

- Energies: ωB97X-D and AIMNet2 absolute energies differ by per-element
  constants, fitted once by least squares on the untuned model and then held
  fixed. They cancel in every reaction energy (the same atoms on both sides);
  with a single composition they reduce to one constant shift.
- Loss: weighted mean of (ΔE / 10 meV)² plus mean (ΔF / 50 meV/Å)²; the fragment
  structures (17 of 110) weigh 3×.
- Validation: 3 of the 31 IRC frames with their rattled copies (split by frame,
  so no near-duplicate leaks), and with fragments also the pulled-apart
  structures at r(C-F) 4.0 Å and r(C-Cl) 4.2 Å; the best validation state is kept.
- Adam, lr 5e-5, full batch (one batch per atom count), up to 800 epochs.

Writes ``D:\\MLIP_Work_Folder\\cache\\aimnet\\finetuned\\<name>.pt`` (same format
as the registry file, loadable by the AIMNet2 backend as ``aimnet_model``),
a model card ``<name>.pt.json``, and ``finetune_aimnet2/log[_fragments].json``.
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

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FRAGMENTS = "--fragments" in sys.argv
WORK = Path(ARGS[0] if ARGS else r"D:\MLIP_Work_Folder\sn2_F_CH3Cl")
BASE = Path(r"D:\MLIP_Work_Folder\cache\aimnet\aimnet2_wb97m_d3_0.pt")
NAME = "aimnet2_wb97m_d3_0_SN2-F-CH3Cl_wB97XD-def2TZVPD" + ("_fragments" if FRAGMENTS else "")
OUT_MODEL = BASE.parent / "finetuned" / f"{NAME}.pt"
OUT = WORK / "finetune_aimnet2"
LOG = OUT / ("log_fragments.json" if FRAGMENTS else "log.json")
EPOCHS, LR, E_SCALE, F_SCALE = 800, 5e-5, 0.010, 0.050
VALIDATION_FRAMES, VALIDATION_TAGS, FRAGMENT_WEIGHT = (5, 15, 25), {"pulled_F_4.0",
                                                                   "pulled_Cl_4.2"}, 3.0
ELEMENTS = (1, 6, 9, 17)


class Batches:
    """Structures grouped by atom count (one tensor batch each), with weights."""

    def __init__(self, images, weights):
        self.images, self.weights = images, np.asarray(weights, float)
        self.groups = []
        for n in sorted({len(a) for a in images}):
            index = [k for k, a in enumerate(images) if len(a) == n]
            chosen = [images[k] for k in index]
            self.groups.append({
                "index": index,
                "coord": torch.tensor(np.array([a.positions for a in chosen]), dtype=torch.float32),
                "numbers": torch.tensor(np.array([a.numbers for a in chosen])),
                "charge": torch.tensor([float(a.info.get("charge", -1)) for a in chosen]),
                "forces": torch.tensor(np.array([a.arrays["REF_forces"] for a in chosen]),
                                       dtype=torch.float32),
            })
        self.e_ref = torch.tensor([a.info["raw_energy"] for a in images], dtype=torch.float64)
        self.counts = torch.tensor([[int((a.numbers == z).sum()) for z in ELEMENTS]
                                    for a in images], dtype=torch.float64)

    def predict(self, calc):
        """Energies (float64, in image order) and per-group (forces, reference forces)."""
        energies = torch.zeros(len(self.images), dtype=torch.float64)
        forces = []
        for group in self.groups:
            out = calc.eval({"coord": group["coord"], "numbers": group["numbers"],
                             "charge": group["charge"]}, forces=True)
            energies = energies.index_put((torch.tensor(group["index"]),),
                                          out["energy"].double().reshape(-1))
            forces.append((out["forces"].reshape(group["forces"].shape), group["forces"],
                           torch.tensor(self.weights[group["index"]], dtype=torch.float32)))
        return energies, forces


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    data = read(WORK / "finetune" / "train.extxyz", ":")
    irc = [k for k, a in enumerate(data) if a.info["tag"] == "irc"]
    rattled = [k for k, a in enumerate(data) if a.info["tag"] == "rattle"]
    # Two rattled copies per IRC frame, written in frame order.
    groups = {n: [k, rattled[2 * n], rattled[2 * n + 1]] for n, k in enumerate(irc)}
    held = {k for frame in VALIDATION_FRAMES for k in groups[frame]}
    train = [a for k, a in enumerate(data) if k not in held]
    valid = [a for k, a in enumerate(data) if k in held]
    if FRAGMENTS:
        for image in read(OUT / "fragments.extxyz", ":"):
            (valid if image.info["tag"] in VALIDATION_TAGS else train).append(image)

    def weight(image):
        return 1.0 if image.info.get("tag") in ("irc", "rattle") else FRAGMENT_WEIGHT

    sets = {name: Batches(images, [weight(a) for a in images])
            for name, images in (("train", train), ("valid", valid))}
    calc = AIMNet2Calculator(str(BASE), device="cpu", train=True)
    for parameter in calc.model.parameters():
        parameter.requires_grad_(True)

    # Per-element offsets between ωB97X-D and AIMNet2 energies, from the untuned model.
    energy, _ = sets["train"].predict(calc)
    residual = (sets["train"].e_ref - energy.detach()).numpy()
    offsets = torch.tensor(np.linalg.lstsq(sets["train"].counts.numpy(), residual,
                                           rcond=None)[0], dtype=torch.float64)

    def errors(name):
        batch = sets[name]
        energy, forces = batch.predict(calc)
        de = energy + batch.counts @ offsets - batch.e_ref
        return de, forces, batch

    def metrics(name):
        de, forces, _ = errors(name)
        squared = torch.cat([(f.detach() - r).pow(2).reshape(-1) for f, r, _ in forces])
        return {"energy_rmse_ev": float(de.detach().pow(2).mean().sqrt()),
                "force_rmse_ev_per_A": float(squared.mean().sqrt())}

    log = {"fragments": FRAGMENTS, "offsets_ev": dict(zip(map(str, ELEMENTS), offsets.tolist(),
                                                          strict=True)),
           "train": len(train), "valid": len(valid), "validation_frames": list(VALIDATION_FRAMES),
           "before": {"train": metrics("train"), "valid": metrics("valid")}, "epochs": []}
    print("before:", log["before"], flush=True)
    optimizer = torch.optim.Adam(calc.model.parameters(), lr=LR)
    best, best_state, start = float("inf"), None, time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        de, forces, batch = errors("train")
        weights = torch.tensor(batch.weights)
        energy_loss = (weights * (de / E_SCALE).pow(2)).sum() / weights.sum()
        force_sum = sum((w[:, None, None] * ((f - r) / F_SCALE).pow(2)).sum() for f, r, w in forces)
        force_count = sum((w[:, None, None] * torch.ones_like(r)).sum() for _, r, w in forces)
        loss = energy_loss + (force_sum / force_count).double()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            valid_metrics = metrics("valid")
            score = valid_metrics["energy_rmse_ev"] / E_SCALE + \
                valid_metrics["force_rmse_ev_per_A"] / F_SCALE
            if score < best:
                best = score
                best_state = {k: v.detach().clone() for k, v in calc.model.state_dict().items()}
                log["best_epoch"] = epoch
            log["epochs"].append({"epoch": epoch, "loss": float(loss.detach()), **valid_metrics})
            if epoch % 50 == 0:
                print(f"epoch {epoch}: loss {float(loss):.3f} valid {valid_metrics}", flush=True)
    log["training_s"] = round(time.perf_counter() - start, 1)
    calc.model.load_state_dict(best_state)
    log["after"] = {"train": metrics("train"), "valid": metrics("valid")}
    print("after (best epoch", log["best_epoch"], "):", log["after"], flush=True)

    checkpoint = torch.load(BASE, map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = {k: v.cpu() for k, v in best_state.items()}
    torch.save(checkpoint, OUT_MODEL)
    trained_on = (f"{len(train)} structures: [CH3FCl]- along the AIMNet2 IRC of F- + CH3Cl -> "
                  "CH3F + Cl- (the MACE fine-tune's labels)")
    if FRAGMENTS:
        trained_on += ", plus free F-, Cl-, CH3Cl, CH3F and the complexes pulled apart"
    Path(str(OUT_MODEL) + ".json").write_text(json.dumps({
        "fine_tuned_from": BASE.name + " (AIMNet2 wB97M-D3, ensemble member 0)",
        "fine_tuning": "all weights, energies and forces through AIMNet2Calculator(train=True): "
                       "network + embedded Coulomb + D3 as at inference",
        "trained_on": trained_on + f"; {len(valid)} held out for validation",
        "reference": "wB97X-D/def2-TZVPD (Psi4); charges -1 (anions) and 0 (molecules)",
        "best_epoch": log["best_epoch"],
        "scope": "F- + CH3Cl SN2 specialist; keeps AIMNet2's charge input, other chemistry "
                 "not re-validated",
    }, indent=1))
    LOG.write_text(json.dumps(log, indent=1))
    print(OUT_MODEL)


if __name__ == "__main__":
    main()
