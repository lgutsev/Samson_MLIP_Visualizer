#!/bin/bash
# Submit every smoke test that is not done yet. Run it on a LONI login node, from anywhere:
#   bash submit_smokes.sh                  # status of every test; sbatch whatever is not done
#   bash submit_smokes.sh --dry-run        # status only, submit nothing
#   bash submit_smokes.sh --retry-failed   # also resubmit what ran but did not finish
#   bash submit_smokes.sh 05 07            # only the folders starting with 05 or 07
# Run it as often as you like: it never submits a test whose results are there, or one that
# has a job in the queue (squeue, matched by the folder the job was submitted from; the job
# names repeat across tests). Array packages get only their missing tasks (--array=1,2).
#
# "Done" is read from the files each job writes:
#   Gaussian / ORCA   outputs/frame_*.log / .out ending in a normal termination
#   VASP              outputs/frame_*/<level>/vasprun.xml complete for every level (pbe_u, hse06)
#   VASP relax (08)   OUTCAR.2_cell finished
#   MACE training     runs/seed<N>/*.model
#   UMA (12)          outputs/<checkpoint>/results.json
#   triplets (13)     every step's .out, for every job in inputs/jobs.txt, terminated normally
# A unit that left output behind without finishing (crash, walltime) is FAILED and is
# resubmitted only with --retry-failed: read its logs/ first. A script still holding a
# <PLACEHOLDER> is skipped. Each submission is appended to submissions.log.
# export_results.sh uses the same checks to pack the finished results for the desktop.
set -o pipefail
SMOKES_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---- checks shared with export_results.sh (which sources this file) ----

QUEUE=""  # queued and running jobs: "jobid<TAB>physical submit folder" per line
load_queue() {
  local raw id where
  command -v squeue >/dev/null 2>&1 || return 1
  raw=$(squeue -u "$USER" -h -o "%i %Z") || { echo "squeue failed" >&2; exit 1; }
  while read -r id where; do
    [ -n "$id" ] && QUEUE+="$id"$'\t'"$(readlink -f "$where" 2>/dev/null || echo "$where")"$'\n'
  done <<< "$raw"
}
queued_in() { printf '%s' "$QUEUE" | awk -F'\t' -v d="$1" '$2 == d {printf "%s%s", s, $1; s = ","}'; }
tail_has() { [ -f "$1" ] && tail -n 50 "$1" | grep -q "$2"; }
has_placeholder() { grep -v '^# ' "$1" | grep -qE '<[A-Z][A-Z0-9_]*>'; }
array_ids() {  # the task ids of a script's #SBATCH --array line (1-3,5 and %throttle forms)
  local spec part
  spec=$(grep -m1 '^#SBATCH --array=' "$1" | cut -d= -f2)
  for part in ${spec//,/ }; do
    part=${part%%%*}
    if [[ $part == *-* ]]; then seq "${part%-*}" "${part#*-}"; else echo "$part"; fi
  done
}

# Each package scanner tallies its units, then calls finish folder script kind, which the
# controller (below) and export_results.sh define differently.
RETRY=0 ONLY=()
begin() { N_DONE=0; FAILED=(); TODO=(); IDX=(); DONE_PATHS=(); }
tally() {  # unit, done|failed|todo, array task id (or job) to rerun it, result paths if done
  local name=$1 state=$2 index=$3
  shift 3
  case $state in
    done) N_DONE=$((N_DONE + 1)); DONE_PATHS+=("$@"); return ;;
    failed) FAILED+=("$name"); [ $RETRY = 1 ] || return ;;
    todo) TODO+=("$name") ;;
  esac
  IDX+=("$index")
}
tally_line() {
  local total=$((N_DONE + ${#FAILED[@]} + ${#TODO[@]})) line
  if [ "$1" = counts ]; then  # 13: the unit lists are too long to print
    echo "$N_DONE/$total jobs done, ${#FAILED[@]} failed, ${#TODO[@]} not run"; return
  fi
  line="$N_DONE/$total done"
  [ ${#FAILED[@]} -gt 0 ] && line+="; failed: ${FAILED[*]}"
  [ ${#TODO[@]} -gt 0 ] && line+="; not run: ${TODO[*]}"
  echo "$line"
}

label_package() {  # folder, script, gaussian|orca|vasp
  local dir=$1 script=$2 per_task frame name n state= level ok out paths
  # packed scripts run several frames per array task: index = task * per_task + k
  per_task=$(grep -oE 'SLURM_ARRAY_TASK_ID \* [0-9]+' "$dir/$script" | head -1 | awk '{print $3}')
  per_task=${per_task:-1}
  begin
  for frame in "$dir"/inputs/frame_*; do
    [ -e "$frame" ] || continue
    name=$(basename "$frame"); name=${name%.*}
    n=$((10#${name#frame_}))
    case $3 in
      gaussian) out="$dir/outputs/$name.log"; paths=("outputs/$name.log")
        tail_has "$out" "Normal termination" && state=done ;;
      orca) out="$dir/outputs/$name.out"; paths=("outputs/$name.out")
        [ -f "$dir/outputs/$name.engrad" ] && paths+=("outputs/$name.engrad")
        tail_has "$out" "ORCA TERMINATED NORMALLY" && state=done ;;
      vasp) out="$dir/outputs/$name"; paths=("outputs/$name"); ok=1
        for level in "$frame"/INCAR.*; do
          tail_has "$out/${level##*INCAR.}/vasprun.xml" "</modeling>" || ok=0
        done
        [ $ok = 1 ] && state=done ;;
    esac
    [ "$state" = done ] || { [ -e "$out" ] && state=failed || state=todo; }
    tally "$name" "$state" $((n / per_task)) "${paths[@]}"
    state=
  done
  finish "$dir" "$script" array
}

train_package() {
  local dir=$1 script=$2 seed
  begin
  for seed in $(array_ids "$dir/$script"); do
    if compgen -G "$dir/runs/seed$seed/*.model" >/dev/null; then tally "seed$seed" done "$seed" "runs/seed$seed"
    elif [ -d "$dir/runs/seed$seed" ]; then tally "seed$seed" failed "$seed"
    else tally "seed$seed" todo "$seed"; fi
  done
  finish "$dir" "$script" array
}

uma_package() {
  local dir=$1 script=$2 task checkpoints name
  read -ra checkpoints <<< "$(grep -m1 '^CHECKPOINTS=(' "$dir/$script" | sed 's/.*(\(.*\)).*/\1/')"
  begin
  for task in $(array_ids "$dir/$script"); do
    name=${checkpoints[$((task - 1))]%.pt}
    if [ -f "$dir/outputs/$name/results.json" ]; then tally "$name" done "$task" "outputs/$name"
    elif [ -d "$dir/outputs/$name" ]; then tally "$name" failed "$task"
    else tally "$name" todo "$task"; fi
  done
  finish "$dir" "$script" array
}

relax_package() {
  local dir=$1 script=$2 f paths=()
  begin
  if tail_has "$dir/OUTCAR.2_cell" "General timing"; then
    for f in "$dir"/{OUTCAR,OSZICAR,CONTCAR}.* "$dir"/vasp_*.out "$dir"/relax_*.out; do
      [ -f "$f" ] && paths+=("${f##*/}")
    done
    tally relax done 0 "${paths[@]}"
  elif compgen -G "$dir/relax_*.out" >/dev/null; then tally relax failed 0
  else tally relax todo 0; fi
  finish "$dir" "$script" single
}

triplet_package() {  # 13: every ORCA step of every job in inputs/jobs.txt
  local dir=$1 script=$2 job inp ok started
  begin
  while read -r job; do
    job=${job%$'\r'}  # the list may come from Windows
    [ -n "$job" ] || continue
    ok=1 started=0
    for inp in "$dir/inputs/$job"/*.inp; do
      [ -f "${inp%.inp}.out" ] && started=1
      tail_has "${inp%.inp}.out" "ORCA TERMINATED NORMALLY" || ok=0
    done
    if [ $ok = 1 ]; then tally "$job" done "$job" "inputs/$job"
    elif [ $started = 1 ]; then tally "$job" failed "$job"
    else tally "$job" todo "$job"; fi
  done < "$dir/inputs/jobs.txt"
  finish "$dir" "$script" triplet
}

wanted() {
  local rel=${1#./} prefix
  [ ${#ONLY[@]} -eq 0 ] && return 0
  for prefix in "${ONLY[@]}"; do [[ $rel == "$prefix"* ]] && return 0; done
  return 1
}

scan_all() {  # every package under the smoke-test folder (or the ones named in ONLY)
  local path dir script
  cd "$SMOKES_ROOT" || exit 1
  while read -r path; do
    dir=$(dirname "$path") script=$(basename "$path")
    wanted "$dir" || continue
    case $script in
      run_gaussian.slurm) label_package "$dir" "$script" gaussian ;;
      run_orca.slurm) label_package "$dir" "$script" orca ;;
      run_vasp.slurm) label_package "$dir" "$script" vasp ;;
      run_train.slurm) train_package "$dir" "$script" ;;
      run_uma.slurm) uma_package "$dir" "$script" ;;
      run_relax.slurm) relax_package "$dir" "$script" ;;
      submit_array.slurm) if [ -f "$dir/inputs/jobs.txt" ]; then triplet_package "$dir" "$script"
                          else unknown_package "$dir" "$script: no inputs/jobs.txt"; fi ;;
      *) unknown_package "$dir" "$script: not a script this knows" ;;
    esac
  done < <(find . -name '*.slurm' -not -path '*/inputs/*' -not -path '*/outputs/*' -not -path '*/runs/*' | sort)
}

report() { printf '%-42s %-9s %s\n' "$1" "$2" "$3"; }

[ "${SMOKES_LIB:-}" = 1 ] && return 0

# ---- the controller ----

DRY=0
N_SUBMITTED=0 N_QUEUED=0 N_DONE_PKG=0 N_FAILED_PKG=0 N_SETUP=0
unknown_package() { report "${1#./}" UNKNOWN "$2; submit it yourself"; }

submit() {  # folder, script, sbatch options...; prints the job id
  local dir=$1 script=$2 out
  shift 2
  mkdir -p "$dir/logs"
  # stdout only: sbatch's warnings go to the terminal, not into the job id
  out=$(cd "$dir" && sbatch --parsable "$@" "$script") || return 1
  out=${out%%;*}
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${dir#./}" "$out" "$*" >> submissions.log
  echo "$out"
}

finish() {  # folder, script, array|single|triplet: report the tally, submit what is missing
  local dir=$1 script=$2 kind=$3 rel=${1#./} line jobs
  line=$(tally_line "$([ "$kind" = triplet ] && echo counts)")
  jobs=$(queued_in "$(cd "$dir" && pwd -P)")
  if [ -n "$jobs" ]; then
    N_QUEUED=$((N_QUEUED + 1)); report "$rel" QUEUED "$line (jobs $jobs)"; return
  fi
  if [ ${#IDX[@]} -eq 0 ]; then
    if [ ${#FAILED[@]} -gt 0 ]; then
      N_FAILED_PKG=$((N_FAILED_PKG + 1)); report "$rel" FAILED "$line (logs in $rel/logs; --retry-failed resubmits)"
    else
      N_DONE_PKG=$((N_DONE_PKG + 1)); report "$rel" DONE "$line"
    fi
    return
  fi
  if has_placeholder "$dir/$script"; then
    N_SETUP=$((N_SETUP + 1))
    report "$rel" SETUP "$line; fill in $(grep -v '^# ' "$dir/$script" | grep -oE '<[A-Z][A-Z0-9_]*>' | sort -u | tr '\n' ' ')in $script"
    return
  fi
  if [ "$kind" = triplet ]; then submit_triplet "$dir" "$script" "$line"; return; fi
  local opts=() ids job
  if [ "$kind" = array ]; then
    ids=$(printf '%s\n' "${IDX[@]}" | sort -nu | paste -sd, -)
    opts=(--array="$ids")
  fi
  if [ $DRY = 1 ]; then
    N_SUBMITTED=$((N_SUBMITTED + 1)); report "$rel" WOULD "$line -> sbatch ${opts[*]} $script"; return
  fi
  if job=$(submit "$dir" "$script" "${opts[@]}"); then
    N_SUBMITTED=$((N_SUBMITTED + 1)); report "$rel" SUBMITTED "$line -> job $job ${opts[*]}"
  else
    report "$rel" ERROR "$line; sbatch failed (message above)"
  fi
}

submit_triplet() {  # 13: array tasks over job lists, L2 after L1, as in its submit.sh
  local dir=$1 script=$2 line=$3 rel=${1#./} job per_node stage1=() stage2=() j1= j2=
  per_node=$(grep -oE '^PER_NODE=[0-9]+' "$dir/$script" | cut -d= -f2)
  per_node=${per_node:-4}
  for job in "${IDX[@]}"; do
    if [[ $job == L2* ]]; then stage2+=("$job"); else stage1+=("$job"); fi
  done
  line+=" -> stage 1: ${#stage1[@]} jobs on $(( (${#stage1[@]} + per_node - 1) / per_node )) nodes,"
  line+=" stage 2: ${#stage2[@]} on $(( (${#stage2[@]} + per_node - 1) / per_node ))"
  if [ $DRY = 1 ]; then N_SUBMITTED=$((N_SUBMITTED + 1)); report "$rel" WOULD "$line"; return; fi
  if [ ${#stage1[@]} -gt 0 ]; then
    printf '%s\n' "${stage1[@]}" > "$dir/inputs/jobs_todo_stage1.txt"
    j1=$(submit "$dir" "$script" --array=0-$(( (${#stage1[@]} + per_node - 1) / per_node - 1 )) \
         --export=ALL,JOBLIST=inputs/jobs_todo_stage1.txt) || { report "$rel" ERROR "$line; sbatch failed"; return; }
  fi
  if [ ${#stage2[@]} -gt 0 ]; then
    printf '%s\n' "${stage2[@]}" > "$dir/inputs/jobs_todo_stage2.txt"
    # afterany, as in submit.sh: an L2 job needs only its own L1 geometry
    j2=$(submit "$dir" "$script" ${j1:+--dependency=afterany:$j1} \
         --array=0-$(( (${#stage2[@]} + per_node - 1) / per_node - 1 )) \
         --export=ALL,JOBLIST=inputs/jobs_todo_stage2.txt) || { report "$rel" ERROR "$line; stage 2 sbatch failed"; return; }
  fi
  N_SUBMITTED=$((N_SUBMITTED + 1))
  report "$rel" SUBMITTED "$line (jobs ${j1:-none} ${j2:+then $j2})"
}

for arg in "$@"; do
  case $arg in
    -n|--dry-run) DRY=1 ;;
    --retry-failed) RETRY=1 ;;
    -h|--help) sed -n '2,/^set -o/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $arg (see --help)" >&2; exit 2 ;;
    *) ONLY+=("${arg%/}") ;;
  esac
done
if ! load_queue; then
  if [ $DRY = 1 ]; then echo "(no squeue here: the queue is not checked)"
  else echo "no squeue on this machine: run this on the cluster, or use --dry-run" >&2; exit 1; fi
fi
scan_all
echo
echo "$N_DONE_PKG done, $N_QUEUED in the queue, $N_SUBMITTED $([ $DRY = 1 ] && echo 'to submit (dry run)' || echo submitted)," \
     "$N_FAILED_PKG failed, $N_SETUP need setup"
