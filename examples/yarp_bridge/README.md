# YARP enumeration → MLIP transition states

A prototype that lets [YARP](https://github.com/Savoie-Research-Group/yarp)
propose the reactions and the visualizer characterize them. YARP enumerates
products of a reactant on the bond graph (break *n*, form *m*); each
reactant/product pair then goes through the usual protocol with an MLIP:
relax both ends, QST2, frequencies, IRC.

Only YARP's enumeration and its force-field clean-up are used. Its xTB, CREST,
and ORCA stages, and its EGAT barrier model, are not.

## Two environments, two scripts

YARP needs RDKit and Open Babel, which must stay out of SAMSON's Python, so the
two halves exchange files:

1. `enumerate_yarp.py` (YARP environment) writes `reactant.xyz`, one
   `rxn_NNN/{reactant,product}.xyz` per candidate, and `candidates.json` with
   the bonds each candidate breaks and forms.
2. `run_qst.py` (visualizer environment) reads that folder and writes
   `ts.xyz`, the relaxed end points, and `results.json`.

```
python enumerate_yarp.py "CC=O" --n-break 2 --n-form 2 --out runs/acetaldehyde --yarp path/to/yarp
python run_qst.py runs/acetaldehyde --threads 4
```

YARP is imported from a clone (`--yarp` or `YARP_HOME`), tag `v3.0.1`. Do not
copy its code into this repository: its license forbids commercial use without
the Savoie group's permission. The environment used here: Python 3.10 with
`scikit-learn=1.3.0 rdkit openbabel scipy networkx pyyaml ase h5py pandas psutil
ipython` from conda-forge. `enumerate_yarp.py` sets up the DLL path and
`BABEL_DATADIR` itself, so the environment does not have to be activated; YARP is
not installed with pip and runs natively on Windows for this part.

## What makes the hand-off work

- **Atom order.** YARP builds each product by editing the reactant's bond
  matrix, so product and reactant list the same atoms in the same order, which
  `qst` requires. An `.xyz` reactant is reordered by YARP's canonicalization;
  take the reactant from the output folder, not from the input file.
- **Intended bonding.** The broken and formed bonds in `candidates.json` are
  the test for everything downstream: a relaxed end point must still be the
  species YARP meant, and an IRC end is called reactant or product by those
  bonds (bonded: closer than 1.3 × the summed covalent radii).
- **Confirmed** means one imaginary mode and an IRC that reaches both minima.
  Anything else is reported with the reason and should be read as "not this
  reaction", not as a barrier.

## Results (MACE-MP-0 small, float64, CPU, 2026-11-18)

| Reactant | Product | Barrier | Reaction | Imaginary mode | Status |
|---|---|---|---|---|---|
| HCN | HNC | 2.63 eV | +0.63 eV | 987i cm⁻¹ | confirmed |
| acetaldehyde | vinyl alcohol | 2.16 eV | +0.56 eV | 1520i cm⁻¹ | confirmed |
| acetaldehyde | oxirane | (2.65 eV) | +1.24 eV | 548i cm⁻¹ | not confirmed: IRC reaches oxirane and something that is not acetaldehyde |
| acetaldehyde | ketene + H₂ | (4.74 eV) | +1.06 eV | 279i cm⁻¹ | not confirmed: neither IRC end is acetaldehyde or ketene + H₂ |

The HCN numbers reproduce the hand-built search in the main README (2.63 eV,
+0.63 eV) with no guess supplied. These are MACE-MP-0 values, not reference
barriers: for HCN the model is 0.63 eV above PBE.

For the two unconfirmed rows the NEB did not converge in 500 steps before the
P-RFO refinement, so the saddle found is likely on another path. The IRC end
points were not saved, so what they are has not been looked at.

## Limits found

- **HCN needs non-default settings.** Concerted break-1/form-1 cannot produce a
  1,2-shift, and YARP's default formal-charge filter drops HNC ([C-]#[NH+]):
  `--n-break 1 --n-form 1 --mode sequential --lewis-cutoff 5 --fc-cutoff 3`.
  That also lets through a bare C≡N⁺ + H⁻ candidate, which is not worth running.
- **Formaldehyde gives nothing** at break-2/form-2 concerted (H₂ + CO is not
  enumerated).
- **Cost.** On 4 CPU threads the HCN pair took about 10 minutes and the three
  acetaldehyde pairs about 5.5 hours, nearly all of it in the NEB. A screen of
  more than a few candidates needs a GPU or a cheaper first pass.
- **No ranking.** Every candidate that survives YARP's filters is run. YARP's
  EGAT barrier estimate, which would order them, was not tried.
- **Charge and spin** are recorded in `candidates.json` but not passed to the
  calculator; MACE-MP-0 ignores them.
