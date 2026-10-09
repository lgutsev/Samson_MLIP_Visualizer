"""A VASP package in the dispatcher's ``run_vasp.slurm`` layout, whose levels form a chain.

``submit_smokes.sh`` counts a ``run_vasp.slurm`` frame done when every
``inputs/frame_NNNN/INCAR.<level>`` has ``outputs/frame_NNNN/<level>/vasprun.xml`` ending in
``</modeling>``; array task N runs frame N. The labeling script starts every level from the
frame's POSCAR. Here each level can instead start from the previous level's CONTCAR (a
relaxation continued, then a static), optionally rattled first (a Gaussian displacement of every
atom, in Å, with a fixed seed), so relax -> relax -> static -> rattle-relax chains fit the same
layout. The script is plain bash and awk, with no Python on the cluster.

Per level, ``chain[level]`` is one of:
- ``"poscar"``: start from inputs/frame_NNNN/POSCAR (a single point or the first relaxation);
- ``"contcar"``: start from the previous level's CONTCAR;
- ``"last"``: the previous level's CONTCAR if the frame ran an earlier level, else its POSCAR
  (frames that relax first and frames that do not can share the later levels);
- ``("rattle", sigma, seed)``: the previous level's CONTCAR, rattled by ``sigma`` Å.

Optional, per level (``write_script`` keywords; packages that do not use them are unchanged):
- ``level_kpoints``: ``inputs/frame_NNNN/KPOINTS.<level>``, if present, replaces the frame's
  KPOINTS for that level (explicit lists with zero-weight band points);
- ``exe``: {level: executable} instead of ``vasp_std`` (``vasp_ncl`` for spin-orbit coupling);
- ``wavecar``: levels that start from the previous level's WAVECAR (that WAVECAR is kept until
  they finish);
- ``keep``: extra files copied back to outputs/ (e.g. EIGENVAL);
- ``resume``: levels whose timed-out attempt continues from its own last CONTCAR in runs/
  instead of the chain's start (relaxations; a rattle level is not re-rattled);
- ``inplane_guard``: levels after which a frame whose INCAR.<level> sets IOPTCELL stops (exit 4)
  if lattice rows 1-2 moved by more than 1e-4 Å from the input POSCAR (IOPTCELL is a VTST tag;
  VASP 6.6.1 applied ``0 0 0 0 0 0 0 0 1`` inverted in package 31).

A level without an INCAR.<level> in the frame is skipped (the next level chains from the last one
that exists), and a level whose vasprun.xml is already complete is not rerun, so a resubmitted task
picks up after the last finished level.
"""

import json
from pathlib import Path

from samson_mlip_visualizer.vasp_labeling import (
    incar,
    kpoint_mesh,
    potcar_names,
    species_order,
    write_poscar,
)

# New packages run VASP 6.6.1 (personal license; smoke 20: bit-identical to 6.5.1 on 08's cubic
# frame). Label sets that started on 6.5.1 (05/07/08 and their campaigns) stay on 6.5.1.
LONI = dict(account="loni_perovsk27", partition="workq", module="vasp6/6.6.1-cpu",
            potcars="/home/lgutsev/pot/potpaw_PBE", run="srun vasp_std")

# Cartesian Gaussian rattle of a VASP5 CONTCAR/POSCAR in direct coordinates (awk; Box-Muller)
RATTLE_AWK = r'''
function gauss() { do { u = rand() } while (u == 0); return sqrt(-2 * log(u)) * cos(6.283185307179586 * rand()) }
BEGIN { srand(seed) }
NR == 2 { s = $1 }
NR >= 3 && NR <= 5 { for (j = 1; j <= 3; j++) A[NR - 2, j] = $j * s }
NR == 7 { n = 0; for (j = 1; j <= NF; j++) n += $j }
NR == 8 && $1 !~ /^[Dd]/ { print "rattle.awk: only Direct coordinates are handled" > "/dev/stderr"; exit 2 }
NR <= 7 { print; next }
NR == 8 { print "Direct";
  det = A[1,1]*(A[2,2]*A[3,3]-A[2,3]*A[3,2]) - A[1,2]*(A[2,1]*A[3,3]-A[2,3]*A[3,1]) + A[1,3]*(A[2,1]*A[3,2]-A[2,2]*A[3,1])
  # B = A^-1 (rows of A are lattice vectors; f = r B)
  B[1,1] =  (A[2,2]*A[3,3]-A[2,3]*A[3,2])/det; B[1,2] = -(A[1,2]*A[3,3]-A[1,3]*A[3,2])/det; B[1,3] =  (A[1,2]*A[2,3]-A[1,3]*A[2,2])/det
  B[2,1] = -(A[2,1]*A[3,3]-A[2,3]*A[3,1])/det; B[2,2] =  (A[1,1]*A[3,3]-A[1,3]*A[3,1])/det; B[2,3] = -(A[1,1]*A[2,3]-A[1,3]*A[2,1])/det
  B[3,1] =  (A[2,1]*A[3,2]-A[2,2]*A[3,1])/det; B[3,2] = -(A[1,1]*A[3,2]-A[1,2]*A[3,1])/det; B[3,3] =  (A[1,1]*A[2,2]-A[1,2]*A[2,1])/det
  next }
NR > 8 && k < n { k++; for (j = 1; j <= 3; j++) r[j] = sigma * gauss()
  for (j = 1; j <= 3; j++) f[j] = $j + r[1]*B[1,j] + r[2]*B[2,j] + r[3]*B[3,j]
  printf "%16.10f %16.10f %16.10f\n", f[1], f[2], f[3] }
'''  # noqa: E501


def write_frame(root, index, atoms, levels, spacing, mesh=None, title=""):
    """inputs/frame_NNNN with POSCAR, KPOINTS, POTCAR.names and one INCAR.<level> per level
    (``levels``: {level: extra INCAR tags}, PBE+U with Materials Project U)."""
    folder = Path(root) / "inputs" / f"frame_{index:04d}"
    folder.mkdir(parents=True)
    write_poscar(folder / "POSCAR", atoms)
    mesh = mesh or kpoint_mesh(atoms, spacing)
    (folder / "KPOINTS").write_text(f"Gamma-centered {title}\n0\nGamma\n"
                                    f"{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n",
                                    newline="\n")
    (folder / "POTCAR.names").write_text(" ".join(potcar_names(species_order(atoms))) + "\n",
                                         newline="\n")
    for level, tags in levels.items():
        text = incar("pbe_u", atoms, extra=tags).replace("single point", f"{title} {level}".strip())
        (folder / f"INCAR.{level}").write_text(text, newline="\n")
    return tuple(mesh)


def write_script(root, count, chain, *, name, time, throttle=None, cfg=LONI, exe=None,
                 wavecar=(), keep=(), level_kpoints=False, resume=(), inplane_guard=()):
    """run_vasp.slurm: array 0..count-1, the levels of ``chain`` (ordered dict) in order."""
    levels = list(chain)
    exe = exe or {}
    wavecar = set(wavecar)
    resume, inplane_guard = set(resume), set(inplane_guard)
    # a level's WAVECAR is kept when the next level of the chain reads it
    feeds = {levels[i] for i in range(len(levels) - 1) if levels[i + 1] in wavecar}
    copied = " ".join(["vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR", "vasp.out", "POSCAR",
                       *keep])
    arr = f"0-{count - 1}" + (f"%{throttle}" if throttle else "")
    lines = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={cfg['account']}
#SBATCH --partition={cfg['partition']}
#SBATCH --array={arr}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time={time}
#SBATCH --output=logs/%x_%A_%a.out
# Written by samson-mlip-visualizer (examples/delta_hse06_bbvo/loni_chain.py). Nothing was submitted.
# One array task per frame; levels in order: {' -> '.join(levels)}.
set -eo pipefail
# The container module writes its wrappers (~/.local/bin/modules/<module>/vasp_std) on the compute
# node when it loads. Packages 05/07/08 ran 3 tasks; batch04 started 14-34 at once and every rank
# failed with "execve(): .../vasp_std: Exec format error", as if one task ran a wrapper another was
# rewriting. Stagger the loads, then check the wrapper is a complete script before srun uses it.
sleep $(( (SLURM_ARRAY_TASK_ID % 8) * 20 ))
module purge
module load {cfg['module']}
set -u
export SINGULARITYENV_OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
frame=$(printf "frame_%04d" "$SLURM_ARRAY_TASK_ID")
work="runs/$frame"
mkdir -p "$work" "outputs/$frame"
vasp=$(command -v vasp_std || true)
case "$vasp" in
  "") echo "no vasp_std after module load {cfg['module']}" >&2; exit 2 ;;
  "$HOME"/bin/*) echo "vasp_std resolves to $vasp, not the module; check PATH" >&2; exit 2 ;;
esac
wrapper_ok() {{ [ -s "$vasp" ] && [ -x "$vasp" ] && [ "$(head -c 2 "$vasp")" = "#!" ]; }}
for try in 1 2 3 4 5 6 7 8 9 10 11 12; do wrapper_ok && break; echo "vasp_std wrapper not ready (try $try)" >&2; sleep 15; done
{{ echo "vasp_std: $vasp"; ls -l --time-style=full-iso "$vasp"; head -n 3 "$vasp"; echo "LOADEDMODULES: ${{LOADEDMODULES:-}}"; date; hostname; }} > "outputs/$frame/vasp_binary.txt" 2>&1
wrapper_ok || {{ echo "vasp_std at $vasp is not a complete script after 3 min; see outputs/$frame/vasp_binary.txt" >&2; exit 2; }}
POTCARS="{cfg['potcars']}"
: > "$work/POTCAR"
for name in $(cat "inputs/$frame/POTCAR.names"); do cat "$POTCARS/$name/POTCAR" >> "$work/POTCAR"; done
grep TITEL "$work/POTCAR" | awk '{{print $4}}' > "outputs/$frame/POTCAR.used"
""".splitlines()  # noqa: E501
    if inplane_guard:
        lines += r"""# Frames that fix a and b with IOPTCELL (biaxial strain): stop if VASP moved them anyway, i.e.
# if this build ignores the (VTST) tag. Compares lattice rows 1-2 of the input and the CONTCAR.
inplane_kept() {  # input POSCAR, CONTCAR
  awk 'FNR==2{s=$1} FNR==3||FNR==4{for(i=1;i<=3;i++) v[FILENAME,FNR,i]=$i*s; f[FNR]=1}
       END{for(r=3;r<=4;r++) for(i=1;i<=3;i++){d=v[ARGV[1],r,i]-v[ARGV[2],r,i]; if(d<0)d=-d; if(d>1e-4) bad=1}; exit bad}' "$1" "$2"
}
uses_ioptcell() { grep -qiE '^[[:space:]]*IOPTCELL' "inputs/$frame/INCAR.$1" 2>/dev/null; }""".splitlines()  # noqa: E501
    lines += ['done_level() { [ -f "outputs/$frame/$1/vasprun.xml" ] && tail -n 3 '
              '"outputs/$frame/$1/vasprun.xml" | grep -q "</modeling>"; }',
              "previous="]
    for level in levels:
        how = chain[level]
        lines += [f"# --- {level} ---", f'if [ ! -f "inputs/$frame/INCAR.{level}" ]; then',
                  f'  echo "{level}: not in this frame, skipped"',
                  f"elif ! done_level {level}; then",
                  f'  mkdir -p "$work/{level}" "outputs/$frame/{level}"',
                  f'  cp "inputs/$frame/INCAR.{level}" "$work/{level}/INCAR"',
                  f'  cp "inputs/$frame/KPOINTS" "$work/POTCAR" "$work/{level}/"']
        if level_kpoints:
            lines += [f'  if [ -f "inputs/$frame/KPOINTS.{level}" ]; then '
                      f'cp "inputs/$frame/KPOINTS.{level}" "$work/{level}/KPOINTS"; fi']
        # a resumed level continues a timed-out attempt from its own last geometry
        own = (f'if [ -s "$work/{level}/CONTCAR" ]; then '
               f'cp "$work/{level}/CONTCAR" "$work/{level}/POSCAR"')
        if how == "poscar":
            start = f'cp "inputs/$frame/POSCAR" "$work/{level}/POSCAR"'
            lines += [f"  {own}; else {start}; fi" if level in resume else f"  {start}"]
        elif how == "last":
            lines += ['  if [ -n "$previous" ]; then',
                      '    [ -s "outputs/$frame/$previous/CONTCAR" ] || '
                      f'{{ echo "{level}: no CONTCAR from $previous" >&2; exit 1; }}',
                      f'    cp "outputs/$frame/$previous/CONTCAR" "$work/{level}/POSCAR"',
                      f'  else cp "inputs/$frame/POSCAR" "$work/{level}/POSCAR"; fi']
        else:
            src = '"outputs/$frame/$previous/CONTCAR"'
            lines += [f'  [ -s {src} ] || '
                      f'{{ echo "{level}: no CONTCAR from $previous" >&2; exit 1; }}']
            if how == "contcar":
                start = f'cp {src} "$work/{level}/POSCAR"'
                lines += [f"  {own}; else {start}; fi" if level in resume else f"  {start}"]
            else:
                _, sigma, seed = how
                start = (f'awk -v sigma={sigma} -v seed={seed} -f rattle.awk {src} '
                         f'> "$work/{level}/POSCAR"')
                lines += ([f"  {own}", f"  else {start}; fi"] if level in resume
                          else [f"  {start}"])
        if level in wavecar:
            lines += ['  if [ -n "$previous" ] && [ -s "$work/$previous/WAVECAR" ]; then '
                      f'cp "$work/$previous/WAVECAR" "$work/{level}/"; '
                      f'else echo "{level}: no WAVECAR from ${{previous:-none}}, '
                      'starting from scratch"; fi']
        run = cfg["run"]
        if level in exe:
            run = run.replace("vasp_std", exe[level])
            # refuse a copy under ~/bin, as for vasp_std (desk, 8a59fab)
            lines += [f'  x=$(command -v {exe[level]} || true)',
                      '  case "$x" in',
                      f'    "") echo "{level}: no {exe[level]} after module load '
                      f'{cfg["module"]}" >&2; exit 2 ;;',
                      f'    "$HOME"/bin/*) echo "{level}: {exe[level]} resolves to $x, '
                      'not the module; check PATH" >&2; exit 2 ;;',
                      '  esac']
        lines += ['  wrapper_ok || { sleep 30; wrapper_ok || '
                  '{ echo "vasp_std wrapper broken before srun" >&2; exit 2; }; }',
                  f'  (cd "$work/{level}" && {run} > vasp.out 2>&1) || true',
                  f"  for f in {copied}; do",
                  f'    if [ -f "$work/{level}/$f" ]; then '
                  f'cp "$work/{level}/$f" "outputs/$frame/{level}/"; fi',
                  "  done",
                  ('  rm -f' if level in feeds else '  rm -f "$work/' + level + '/WAVECAR"')
                  + f' "$work/{level}/CHG" "$work/{level}/CHGCAR"',
                  f'  done_level {level} || {{ echo "{level} did not finish" >&2; exit 1; }}']
        if level in wavecar:
            lines += [f'  rm -f "$work/{level}/WAVECAR"',
                      '  if [ -n "$previous" ]; then rm -f "$work/$previous/WAVECAR"; fi']
        if level in inplane_guard:
            lines += [f'  if uses_ioptcell {level} && ! inplane_kept "inputs/$frame/POSCAR" '
                      f'"outputs/$frame/{level}/CONTCAR"; then',
                      f'    echo "{level}: a or b changed although IOPTCELL fixes them: this VASP '
                      'ignores IOPTCELL; strain lost, stopping" >&2; exit 4',
                      "  fi"]
        lines += ["fi", f'if [ -f "inputs/$frame/INCAR.{level}" ]; then previous={level}; fi']
    Path(root, "run_vasp.slurm").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    Path(root, "rattle.awk").write_text(RATTLE_AWK.lstrip(), encoding="utf-8", newline="\n")
    # tracked, so a plain sbatch from a fresh clone finds logs/ (Slurm does not create it)
    (Path(root) / "logs").mkdir(exist_ok=True)
    Path(root, "logs", "README.txt").write_text(
        "SLURM logs of run_vasp.slurm (Slurm does not create this folder).\n", newline="\n")


def write_manifest(root, frames, meta):
    Path(root, "package.json").write_text(json.dumps({**meta, "frames": frames}, indent=1),
                                          encoding="utf-8")
