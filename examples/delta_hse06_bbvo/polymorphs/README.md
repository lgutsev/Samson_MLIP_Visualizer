# Ba₂BiVO₆ polymorph candidates (inputs to `polymorph_package.py`)

`candidates.json` lists every structure for package 19, with its source. The structures here are:

| Folder | Structure | Source | License |
|---|---|---|---|
| `oqmd_1344250/` | Cmc2₁, 20 atoms: Ba₂[BiO₂][VO₄] with isolated VO₄ tetrahedra (**not a perovskite**). OQMD reports 0.0155 eV/atom above its hull and a 2.978 eV gap | OQMD entry 1344250 (calculation 3402125), OPTIMADE id 5740924, retrieved 2026-09-30. `meta.json` keeps the database fields and the raw-response SHA-256. The raw responses and the retry log (HTTP 502 for about 1.7 h before success) are archived in the audit repository under `data/oqmd/`; `*.log` files are gitignored here | OQMD, CC BY 4.0 (cite Saal et al., JOM 2013; Kirklin et al., npj Comput. Mater. 2015) |
| `mp_mp-1522665/` | Pn‑3, 40 atoms (a tilted rock-salt double perovskite); E_hull 0.146 eV/atom | Materials Project via OPTIMADE, retrieved 2026-09-29 | MP, CC BY 4.0 (cite Jain et al., APL Mater. 2013) |
| `alexandria_pbe_agm002175356/` | C2/m, 10 atoms (tilted perovskite); hull 0.138 eV/atom; PBE gap 1.18 eV | Alexandria PBE via OPTIMADE, retrieved 2026-09-29 | Alexandria, CC BY 4.0 (cite Schmidt et al.) |
| `pbeu08_fixedcell/` | PBE+U relaxation at the cubic cell from a 0.05 Å rattle (smoke test 08, `CONTCAR.1_ions`); symmetrises to R3 and matches OQMD entry 1286152 | our LONI run | — |
| `pbeu08_cellrelaxed/` | the same, continued with the cell free (`CONTCAR.2_cell`; V five-coordinate) | our LONI run | — |

The cubic reference, the MACE-MP-0 structures and the Nb/Ta cells come from `../poscars/`.

The structures were compared, and duplicates identified, in the BBVO audit repository (`lgutsev/bbvo_analysis_review`, `13_STRUCTURAL_SHORTLIST.md`):
- the cubic cell = mp‑1522771 = OQMD 1310613;
- the R3 structure = OQMD 1286152;
- the Alexandria C2/m = OQMD 1297416;
- the MACE VO₄ structure ≠ Cmc2₁.

Database energies come from other settings. Package 19 recomputes the relative polymorph energies at one consistent PBE+U level. Those are **not hull energies**.
