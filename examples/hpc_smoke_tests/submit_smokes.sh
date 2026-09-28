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
set -o pipefail
cd "$(dirname "$0")" || exit 1

DRY=0 RETRY=0 ONLY=()
for arg in "$@"; do
  case $arg in
    -n|--dry-run) DRY=1 ;;
    --retry-failed) RETRY=1 ;;
    -h|--help) sed -n '2,/^set -o/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $arg (see --help)" >&2; exit 2 ;;
    *) ONLY+=("${arg%/}") ;;
  esac
done

# queued and running jobs: "jobid<TAB>physical submit folder" per line
QUEUE=""
if command -v squeue >/dev/null 2>&1; then
  raw=$(squeue -u "$USER" -h -o "%i %Z") || { echo "squeue failed" >&2; exit 1; }
  while read -r id where; do
    [ -n "$id" ] && QUEUE+="$id"$'\t'"$(readlink -f "$where" 2>/dev/null || echo "$where")"$'\n'
  done <<< "$raw"
elif [ $DRY = 1 ]; then
  echo "(no squeue here: the queue is not checked)"
else
  echo "no squeue on this machine: run this on the cluster, or use --dry-run" >&2
  exit 1
fi

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

# per-package tally, reset by begin
begin() { N_DONE=0; FAILED=(); TODO=(); IDX=(); }
tally() {  # unit name, done|failed|todo, array task id to rerun it
  case $2 in
    done) N_DONE=$((N_DONE + 1)); return ;;
    failed) FAILED+=("$1"); [ $RETRY = 1 ] || return ;;
    todo) TODO+=("$1") ;;
  esac
  IDX+=("$3")
}

N_SUBMITTED=0 N_QUEUED=0 N_DONE_PKG=0 N_FAILED_PKG=0 N_SETUP=0
report() { printf '%-42s %-9s %s\n' "$1" "$2" "$3"; }

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

finish() {  # folder, script: report the tally, submit the missing array tasks
  local dir=$1 script=$2 rel=${1#./} total ids line jobs job
  total=$((N_DONE + ${#FAILED[@]} + ${#TODO[@]}))
  line="$N_DONE/$total done"
  [ ${#FAILED[@]} -gt 0 ] && line+="; failed: ${FAILED[*]}"
  [ ${#TODO[@]} -gt 0 ] && line+="; not run: ${TODO[*]}"
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
  ids=$(printf '%s\n' "${IDX[@]}" | sort -nu | paste -sd, -)
  local opts=()
  grep -q '^#SBATCH --array=' "$dir/$script" && opts=(--array="$ids")
  if [ $DRY = 1 ]; then
    N_SUBMITTED=$((N_SUBMITTED + 1)); report "$rel" WOULD "$line -> sbatch ${opts[*]} $script"; return
  fi
  if job=$(submit "$dir" "$script" "${opts[@]}"); then
    N_SUBMITTED=$((N_SUBMITTED + 1)); report "$rel" SUBMITTED "$line -> job $job ${opts[*]}"
  else
    report "$rel" ERROR "$line; sbatch failed (message above)"
  fi
}

label_package() {  # folder, script, gaussian|orca|vasp
  local dir=$1 script=$2 per_task frame name n state= level ok out
  # packed scripts run several frames per array task: index = task * per_task + k
  per_task=$(grep -oE 'SLURM_ARRAY_TASK_ID \* [0-9]+' "$dir/$script" | head -1 | awk '{print $3}')
  per_task=${per_task:-1}
  begin
  for frame in "$dir"/inputs/frame_*; do
    [ -e "$frame" ] || continue
    name=$(basename "$frame"); name=${name%.*}
    n=$((10#${name#frame_}))
    case $3 in
      gaussian) out="$dir/outputs/$name.log"; tail_has "$out" "Normal termination" && state=done ;;
      orca) out="$dir/outputs/$name.out"; tail_has "$out" "ORCA TERMINATED NORMALLY" && state=done ;;
      vasp)
        out="$dir/outputs/$name"; ok=1
        for level in "$frame"/INCAR.*; do
          tail_has "$out/${level##*INCAR.}/vasprun.xml" "</modeling>" || ok=0
        done
        [ $ok = 1 ] && state=done ;;
    esac
    [ "$state" = done ] || { [ -e "$out" ] && state=failed || state=todo; }
    tally "$name" "$state" $((n / per_task))
    state=
  done
  finish "$dir" "$script"
}

train_package() {
  local dir=$1 script=$2 seed
  begin
  for seed in $(array_ids "$dir/$script"); do
    if compgen -G "$dir/runs/seed$seed/*.model" >/dev/null; then tally "seed$seed" done
    elif [ -d "$dir/runs/seed$seed" ]; then tally "seed$seed" failed "$seed"
    else tally "seed$seed" todo "$seed"; fi
  done
  finish "$dir" "$script"
}

uma_package() {
  local dir=$1 script=$2 task checkpoints name
  read -ra checkpoints <<< "$(grep -m1 '^CHECKPOINTS=(' "$dir/$script" | sed 's/.*(\(.*\)).*/\1/')"
  begin
  for task in $(array_ids "$dir/$script"); do
    name=${checkpoints[$((task - 1))]%.pt}
    if [ -f "$dir/outputs/$name/results.json" ]; then tally "$name" done
    elif [ -d "$dir/outputs/$name" ]; then tally "$name" failed "$task"
    else tally "$name" todo "$task"; fi
  done
  finish "$dir" "$script"
}

relax_package() {
  local dir=$1 script=$2
  begin
  if tail_has "$dir/OUTCAR.2_cell" "General timing"; then tally relax done
  elif compgen -G "$dir/relax_*.out" >/dev/null; then tally relax failed 0
  else tally relax todo 0; fi
  finish "$dir" "$script"
}

triplet_package() {  # 13: array tasks over job lists, L2 after L1; see its submit.sh
  local dir=$1 script=$2 rel=${1#./} job inp ok started stage1=() stage2=() per_node jobs line j1 j2 n
  per_node=$(grep -oE '^PER_NODE=[0-9]+' "$dir/$script" | cut -d= -f2)
  per_node=${per_node:-4}
  begin
  while read -r job; do
    job=${job%$'\r'}  # the list may come from Windows
    [ -n "$job" ] || continue
    ok=1 started=0
    for inp in "$dir/inputs/$job"/*.inp; do
      [ -f "${inp%.inp}.out" ] && started=1
      tail_has "${inp%.inp}.out" "ORCA TERMINATED NORMALLY" || ok=0
    done
    if [ $ok = 1 ]; then tally "$job" done; continue; fi
    if [ $started = 1 ]; then tally "$job" failed "$job"; else tally "$job" todo "$job"; fi
    [ $started = 1 ] && [ $RETRY = 0 ] && continue
    if [[ $job == L2* ]]; then stage2+=("$job"); else stage1+=("$job"); fi
  done < "$dir/inputs/jobs.txt"
  # the per-job lists are long here: counts only
  n=$((N_DONE + ${#FAILED[@]} + ${#TODO[@]}))
  line="$N_DONE/$n jobs done, ${#FAILED[@]} failed, ${#TODO[@]} not run"
  jobs=$(queued_in "$(cd "$dir" && pwd -P)")
  if [ -n "$jobs" ]; then N_QUEUED=$((N_QUEUED + 1)); report "$rel" QUEUED "$line (jobs $jobs)"; return; fi
  if [ ${#stage1[@]} -eq 0 ] && [ ${#stage2[@]} -eq 0 ]; then
    if [ ${#FAILED[@]} -gt 0 ]; then
      N_FAILED_PKG=$((N_FAILED_PKG + 1)); report "$rel" FAILED "$line (logs/<level>_<id>.log; --retry-failed resubmits)"
    else
      N_DONE_PKG=$((N_DONE_PKG + 1)); report "$rel" DONE "$line"
    fi
    return
  fi
  line+=" -> stage 1: ${#stage1[@]} jobs on $(( (${#stage1[@]} + per_node - 1) / per_node )) nodes,"
  line+=" stage 2: ${#stage2[@]} on $(( (${#stage2[@]} + per_node - 1) / per_node ))"
  if [ $DRY = 1 ]; then N_SUBMITTED=$((N_SUBMITTED + 1)); report "$rel" WOULD "$line"; return; fi
  j1=
  if [ ${#stage1[@]} -gt 0 ]; then
    printf '%s\n' "${stage1[@]}" > "$dir/inputs/jobs_todo_stage1.txt"
    j1=$(submit "$dir" "$script" --array=0-$(( (${#stage1[@]} + per_node - 1) / per_node - 1 )) \
         --export=ALL,JOBLIST=inputs/jobs_todo_stage1.txt) || { report "$rel" ERROR "$line; sbatch failed"; return; }
  fi
  j2=
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

wanted() {
  local rel=${1#./} prefix
  [ ${#ONLY[@]} -eq 0 ] && return 0
  for prefix in "${ONLY[@]}"; do [[ $rel == "$prefix"* ]] && return 0; done
  return 1
}

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
    submit_array.slurm) [ -f "$dir/inputs/jobs.txt" ] && triplet_package "$dir" "$script" \
                          || report "${dir#./}" UNKNOWN "$script: no inputs/jobs.txt" ;;
    *) report "${dir#./}" UNKNOWN "$script: not a script this knows; submit it yourself" ;;
  esac
done < <(find . -name '*.slurm' -not -path '*/inputs/*' -not -path '*/outputs/*' -not -path '*/runs/*' | sort)

echo
echo "$N_DONE_PKG done, $N_QUEUED in the queue, $N_SUBMITTED $([ $DRY = 1 ] && echo 'to submit (dry run)' || echo submitted)," \
     "$N_FAILED_PKG failed, $N_SETUP need setup"
