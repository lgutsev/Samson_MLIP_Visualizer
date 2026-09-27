"""Run the SN2 free-energy tasks in parallel, one CPU thread each.

    python launch.py [stage] [system]

Stage ``main`` (default): slow growth both ways, the blue-moon windows and free MD
for both systems, and metadynamics for F⁻ + CH₃Cl.
Stage ``ts``: the constrained run at ξ* for each system, after ``analyze.py`` has
found ξ* from the blue-moon profile (``<work>/<system>/xi_star.json``).
A system (``F`` or ``Cl``) limits the run to it. Finished tasks are skipped. Logs go
to ``<work>/logs``.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import WORK, window_grid

HERE = Path(__file__).parent


def tasks(stage):
    if stage == "main":
        jobs = [("F", "metadynamics")]
        for system in ("F", "Cl"):
            jobs += [(system, "slow_growth", "forward"), (system, "slow_growth", "reverse"),
                     (system, "free_md")]
            jobs += [(system, "window", f"{xi:.3f}") for xi in window_grid(system)]
        return jobs
    jobs = []
    for system in ("F", "Cl"):
        path = WORK / system / "xi_star.json"
        if path.exists():
            jobs.append((system, "ts_velocity", f"{json.loads(path.read_text())['xi']:.4f}"))
    return jobs


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "main"
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    running = []
    only = sys.argv[2] if len(sys.argv) > 2 else None
    for job in tasks(stage):
        if only and job[0] != only:
            continue
        log = open(logs / ("_".join(job) + ".log"), "w")  # noqa: SIM115
        running.append((job, subprocess.Popen([sys.executable, "-u", str(HERE / "run.py"), *job],
                                              stdout=log, stderr=subprocess.STDOUT,
                                              env=environment, cwd=HERE), log))
    start = time.perf_counter()
    for job, process, log in running:
        code = process.wait()
        log.close()
        print(f"{' '.join(job):32s} exit {code}  ({time.perf_counter() - start:.0f} s)",
              flush=True)


if __name__ == "__main__":
    main()
