#!/bin/bash
# Pack the finished smoke-test results into one archive for the desktop. On a LONI login node:
#   bash export_results.sh             # everything finished since the last export
#   bash export_results.sh --all       # everything finished, exported before or not
#   bash export_results.sh --dry-run   # list what would go in, write nothing
#   bash export_results.sh 05 13       # only the folders starting with 05 or 13
# It writes exports/results_<date>.tar.gz. Copy that one file into the smoke-test folder
# on the desktop (D:\MLIP_Work_Folder\hpc_smoke_tests) and unpack it there, in PowerShell or
# Git Bash:
#   tar -xzf results_<date>.tar.gz
# Every file lands where the desktop checkers look (check_smoke_results.py, collect_labels,
# install_models, 13's parse_results.py).
#
# "Finished" is what submit_smokes.sh calls done. Per finished unit it takes:
#   Gaussian / ORCA   outputs/frame_*.log, or .out and .engrad
#   VASP              outputs/frame_*/ (vasprun.xml, OUTCAR, OSZICAR, vasp.out, POTCAR.used)
#   VASP relax (08)   OUTCAR.*, OSZICAR.*, CONTCAR.*, vasp_*.out, relax_*.out
#   MACE training     runs/seed<N>/ without checkpoints/ and the *_compiled.model copy
#   UMA (12)          outputs/<checkpoint>/
#   triplets (13)     inputs/<level>/<id>/ without ORCA's .gbw and .tmp files
# plus logs/ of every package it takes anything from, and submissions.log. Units exported
# before are listed in exports/exported.txt and skipped unless --all.
set -o pipefail
SMOKES_LIB=1
source "$(dirname "${BASH_SOURCE[0]}")/submit_smokes.sh" || exit 1
cd "$SMOKES_ROOT" || exit 1

ALL=0 DRY=0
for arg in "$@"; do
  case $arg in
    --all) ALL=1 ;;
    -n|--dry-run) DRY=1 ;;
    -h|--help) sed -n '2,/^set -o/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $arg (see --help)" >&2; exit 2 ;;
    *) ONLY+=("${arg%/}") ;;
  esac
done
load_queue || true  # only to say which packages are still running

mkdir -p exports
touch exports/exported.txt
PATHS=()  # paths relative to the smoke-test folder
N_NEW_PKG=0
unknown_package() { report "${1#./}" SKIPPED "$2"; }

finish() {  # folder, script, kind: add the finished units not exported yet
  local dir=$1 kind=$3 rel=${1#./} path new=0 old=0 note=
  for path in "${DONE_PATHS[@]}"; do
    if [ $ALL = 0 ] && grep -qxF "$rel/$path" exports/exported.txt; then old=$((old + 1)); continue; fi
    PATHS+=("$rel/$path"); new=$((new + 1))
  done
  # packages with something new; nothing new anywhere means no archive
  [ $new -gt 0 ] && N_NEW_PKG=$((N_NEW_PKG + 1))
  if [ $new -gt 0 ] || [ ${#FAILED[@]} -gt 0 ]; then
    [ -d "$dir/logs" ] && PATHS+=("$rel/logs")
  fi
  [ -n "$(queued_in "$(cd "$dir" && pwd -P)")" ] && note=" (jobs still in the queue)"
  if [ $new -gt 0 ]; then report "$rel" EXPORT "$new new result paths; $(tally_line "$([ "$kind" = triplet ] && echo counts)")$note"
  elif [ $N_DONE -gt 0 ]; then report "$rel" EXPORTED "all $old finished result paths exported before$note"
  else report "$rel" NOTHING "$(tally_line "$([ "$kind" = triplet ] && echo counts)")$note"; fi
}

scan_all
echo
if [ $N_NEW_PKG -eq 0 ]; then
  echo "nothing new to export$([ $ALL = 0 ] && echo ' (--all exports everything finished again)')"
  exit 0
fi
[ -f submissions.log ] && PATHS+=(submissions.log)

stamp=$(date '+%Y%m%d_%H%M')
list="exports/results_$stamp.txt" archive="exports/results_$stamp.tar.gz"
if [ $DRY = 1 ]; then
  printf '%s\n' "${PATHS[@]}"
  echo "(dry run: would write $archive)"
  exit 0
fi
printf '%s\n' "${PATHS[@]}" > "$list"
# the list goes in too, so the desktop knows what arrived
if tar -czf "$archive" --exclude=checkpoints --exclude='*_compiled.model' \
     --exclude='*.gbw' --exclude='*.tmp' --exclude='WAVECAR' --exclude='CHG' --exclude='CHGCAR' \
     -T "$list" "$list"; then
  grep -v '/logs$' "$list" | grep -vx submissions.log >> exports/exported.txt
  echo "wrote $SMOKES_ROOT/$archive ($(du -h "$archive" | cut -f1))"
  echo "copy it into D:\\MLIP_Work_Folder\\hpc_smoke_tests and run there: tar -xzf results_$stamp.tar.gz"
else
  rm -f "$archive"
  echo "tar failed; nothing was marked as exported" >&2
  exit 1
fi
