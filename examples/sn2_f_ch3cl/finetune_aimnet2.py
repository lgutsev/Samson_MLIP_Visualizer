"""Fine-tune AIMNet2 (ωB97M-D3, member 0) to ωB97X-D/def2-TZVPD for F- + CH3Cl.

Run with the Python of the aimnet environment (CPU is enough):

    python finetune_aimnet2.py [work folder]

Same 93 labels as the MACE fine-tune (``finetune/train.extxyz``: 31 AIMNet2
IRC frames + 2 rattled copies each, raw ωB97X-D energies and forces).

Why a small loop here rather than ``aimnet train --load``: the registry file is
a format-2 wrapper (``state_dict`` next to the long-range Coulomb and D3
settings the calculator adds outside the network). ``aimnet train`` loads
weights with ``strict=False``, so handing it the wrapper silently trains from
random weights, and its data pipeline would need the D3 part removed from the
targets. Training through ``AIMNet2Calculator(train=True)`` uses the exact
inference path (network + Coulomb + D3), so nothing has to be kept consistent
by hand.

- Loss: (ΔE / 10 meV)² + mean (ΔF / 50 meV/Å)², energies with one constant
  shift (every label is the same composition, so relative energies, including
  those against the separated fragments, are unchanged by it).
- Validation: 3 of the 31 IRC frames with their rattled copies (split by frame,
  so no near-duplicate leaks); the best validation state is kept.
- Adam, lr 5e-5, full batch, up to 800 epochs (~0.4 s each on the laptop CPU).

Writes ``D:\\MLIP_Work_Folder\\cache\\aimnet\\finetuned\\<name>.pt`` (same format
as the registry file, loadable by the AIMNet2 backend as ``aimnet_model``),
a model card ``<name>.pt.json``, and ``finetune_aimnet2/log.json``.
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

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\MLIP_Work_Folder\sn2_F_CH3Cl")
BASE = Path(r"D:\MLIP_Work_Folder\cache\aimnet\aimnet2_wb97m_d3_0.pt")
NAME = "aimnet2_wb97m_d3_0_SN2-F-CH3Cl_wB97XD-def2TZVPD"
OUT_MODEL = BASE.parent / "finetuned" / f"{NAME}.pt"
OUT = WORK / "finetune_aimnet2"
EPOCHS, LR, E_SCALE, F_SCALE, VALIDATION_FRAMES = 800, 5e-5, 0.010, 0.050, (5, 15, 25)


def tensors(images):
    return (
        torch.tensor(np.array([a.positions for a in images]), dtype=torch.float32),
        torch.tensor(np.array([a.numbers for a in images])),
        torch.full((len(images),), -1.0),
        torch.tensor([a.info["raw_energy"] for a in images], dtype=torch.float64),
        torch.tensor(np.array([a.arrays["REF_forces"] for a in images]), dtype=torch.float32),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    data = read(WORK / "finetune" / "train.extxyz", ":")
    irc = [k for k, a in enumerate(data) if a.info["tag"] == "irc"]
    rattled = [k for k, a in enumerate(data) if a.info["tag"] == "rattle"]
    # Two rattled copies per IRC frame, written in frame order.
    groups = {frame: [k, rattled[2 * n], rattled[2 * n + 1]] for n, (frame, k)
              in enumerate(zip(range(len(irc)), irc, strict=True))}
    held = {k for frame in VALIDATION_FRAMES for k in groups[frame]}
    train = [a for k, a in enumerate(data) if k not in held]
    valid = [a for k, a in enumerate(data) if k in held]

    calc = AIMNet2Calculator(str(BASE), device="cpu", train=True)
    for parameter in calc.model.parameters():
        parameter.requires_grad_(True)
    sets = {"train": tensors(train), "valid": tensors(valid)}

    def evaluate(name):
        # Autograd stays on even for metrics: the forces are a gradient themselves.
        coord, numbers, charge, e_ref, f_ref = sets[name]
        out = calc.eval({"coord": coord, "numbers": numbers, "charge": charge}, forces=True)
        return out["energy"].double(), out["forces"], e_ref, f_ref

    energy, _, e_ref, _ = evaluate("train")
    shift = float((e_ref - energy.detach()).mean())

    def metrics(name):
        energy, forces, e_ref, f_ref = evaluate(name)
        de = (energy.detach() + shift - e_ref)
        de = de - de.mean()  # relative energies: the constant is not the model's business
        return {"energy_rmse_ev": float(de.pow(2).mean().sqrt()),
                "force_rmse_ev_per_A": float((forces.detach() - f_ref).pow(2).mean().sqrt())}

    log = {"shift_ev": shift, "train": len(train), "valid": len(valid),
           "validation_frames": list(VALIDATION_FRAMES), "before": {
               "train": metrics("train"), "valid": metrics("valid")}, "epochs": []}
    print("before:", log["before"], flush=True)
    optimizer = torch.optim.Adam(calc.model.parameters(), lr=LR)
    best, best_state, start = float("inf"), None, time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        energy, forces, e_ref, f_ref = evaluate("train")
        loss = ((energy + shift - e_ref) / E_SCALE).pow(2).mean() + (
            (forces - f_ref) / F_SCALE).pow(2).mean().double()
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
    Path(str(OUT_MODEL) + ".json").write_text(json.dumps({
        "fine_tuned_from": BASE.name + " (AIMNet2 wB97M-D3, ensemble member 0)",
        "fine_tuning": "all weights, energies and forces through AIMNet2Calculator(train=True): "
                       "network + embedded Coulomb + D3 as at inference",
        "trained_on": f"{len(train)} configurations of [CH3FCl]- along the AIMNet2 IRC of "
                      "F- + CH3Cl -> CH3F + Cl- (the MACE fine-tune's labels, "
                      f"{len(valid)} held out for validation)",
        "reference": "wB97X-D/def2-TZVPD (Psi4), charge -1",
        "best_epoch": log["best_epoch"],
        "scope": "F- + CH3Cl SN2 specialist; keeps AIMNet2's charge input, other chemistry "
                 "not re-validated",
    }, indent=1))
    (OUT / "log.json").write_text(json.dumps(log, indent=1))
    print(OUT_MODEL)


if __name__ == "__main__":
    main()
