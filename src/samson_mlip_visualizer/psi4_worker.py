"""Psi4 worker process for :mod:`.psi4_backend`.

This file runs in the Python that has Psi4 (its own conda environment), not in
SAMSON's, so it imports nothing from this package: only the standard library,
NumPy, and Psi4. It reads one JSON request per line from stdin and answers each
with one line on stdout that starts with ``@@SAMSON `` (anything else a library
prints is ignored by the parent). Psi4 stays imported between requests, so only
the first calculation pays for the import.

Request: ``{"symbols", "positions" (Å), "method", "basis", "charge",
"multiplicity", "threads", "memory_mb", "options"}``.
Reply: ``{"energy" (Eh), "gradient" (Eh/bohr)}`` or ``{"error"}``.

With ``"task": "matrices"`` (:mod:`.hamiltonian`) the reply is instead the AO
``"overlap"`` and ``"shells"`` layout, plus (unless ``"overlap_only"``) the
converged ``"hamiltonian"`` (Kohn-Sham/Fock matrix, Eh), ``"eigenvalues"``,
``"n_occupied"``, and ``"energy"``.
"""

import json
import os
import sys
import tempfile

PREFIX = "@@SAMSON "


def main() -> None:
    protocol = sys.stdout
    sys.stdout = sys.stderr  # stray Python prints must not reach the protocol stream

    def reply(message):
        protocol.write(PREFIX + json.dumps(message) + "\n")
        protocol.flush()

    # Psi4 writes scratch files (timer.dat, ...) into the working directory from
    # import on, so move to a private directory first.
    workdir = tempfile.mkdtemp(prefix="samson-psi4-")
    os.chdir(workdir)
    try:
        import numpy as np
        import psi4
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        reply({"error": f"Could not import Psi4: {type(exc).__name__}: {exc}"})
        return
    psi4.core.set_output_file(os.path.join(workdir, "psi4.out"), False)
    reply({"ready": True, "version": psi4.__version__, "workdir": workdir})

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            psi4.set_memory(f"{int(request.get('memory_mb', 2000))} MB", quiet=True)
            psi4.set_num_threads(int(request.get("threads", 4)), quiet=True)
            multiplicity = int(request.get("multiplicity", 1))
            psi4.core.clean_options()
            psi4.set_options(
                {
                    "basis": request["basis"],
                    "reference": "uhf" if multiplicity > 1 else "rhf",
                    "scf_type": "df",
                    **request.get("options", {}),
                }
            )
            atoms = "\n".join(
                f"{symbol} {x:.10f} {y:.10f} {z:.10f}"
                for symbol, (x, y, z) in zip(request["symbols"], request["positions"], strict=True)
            )
            molecule = psi4.geometry(
                f"{int(request.get('charge', 0))} {multiplicity}\n{atoms}\n"
                "units angstrom\nsymmetry c1\nno_reorient\nno_com\n"
            )
            if request.get("task") == "matrices":
                reply(_matrices(psi4, np, request, molecule))
                continue
            gradient, wavefunction = psi4.gradient(
                request["method"], molecule=molecule, return_wfn=True
            )
            reply({"energy": wavefunction.energy(), "gradient": np.asarray(gradient).tolist()})
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            reply({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            psi4.core.clean()


def _matrices(psi4, np, request, molecule):
    """The AO overlap and (unless ``overlap_only``) the converged Kohn-Sham/Fock
    matrix, both in Hartree units and Psi4's AO order, with the shell layout
    (atom index and angular momentum of every shell) needed to reorder them."""
    if request.get("overlap_only"):
        wavefunction = None
        basis = psi4.core.BasisSet.build(molecule, "ORBITAL", request["basis"])
    else:
        _, wavefunction = psi4.energy(request["method"], molecule=molecule, return_wfn=True)
        basis = wavefunction.basisset()
    if not basis.has_puream():
        raise ValueError("Hamiltonian export needs a spherical (pure) basis set")
    shells = [[basis.shell_to_center(i), basis.shell(i).am] for i in range(basis.nshell())]
    overlap = np.asarray(psi4.core.MintsHelper(basis).ao_overlap())
    answer = {"shells": shells, "overlap": overlap.tolist()}
    if wavefunction is not None:
        if wavefunction.nalpha() != wavefunction.nbeta():
            raise ValueError("Hamiltonian export supports closed-shell (RHF/RKS) runs only")
        answer.update(
            energy=wavefunction.energy(),
            hamiltonian=np.asarray(wavefunction.Fa()).tolist(),
            eigenvalues=np.asarray(wavefunction.epsilon_a()).tolist(),
            n_occupied=wavefunction.nalpha(),
        )
    return answer


if __name__ == "__main__":
    main()
