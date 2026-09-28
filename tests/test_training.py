"""Fine-tuning runs: arguments, local runs (with a fake trainer), HPC packages, installing."""

import json
import sys

import pytest
from ase.build import molecule
from ase.io import write

from samson_mlip_visualizer import training
from samson_mlip_visualizer.training import (
    GpuSlurmSettings,
    TrainingSpec,
    final_errors,
    install_models,
    train_arguments,
    train_local,
    write_training_package,
)

TABLE = """+---------------+---------------------+------------------+-------------------+
|  config_type  | RMSE E / meV / atom | RMSE F / meV / A | relative F RMSE % |
+---------------+---------------------+------------------+-------------------+
| train_Default |            3.8      |         38.4     |          1.59     |
| valid_Default |            3.9      |         25.4     |          1.02     |
+---------------+---------------------+------------------+-------------------+
"""

# Records its arguments and environment, writes <model_dir>/<name>.model and the table.
FAKE_TRAINER = f'''
import json, os, sys
args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
open(os.path.join(args["model_dir"], args["name"] + ".model"), "w").write("model")
json.dump({{"argv": sys.argv[1:], "cache": os.environ.get("XDG_CACHE_HOME"),
           "train_exists": os.path.exists(args["train_file"])}},
          open(os.path.join(args["model_dir"], "call.json"), "w"))
print({TABLE!r})
'''


@pytest.fixture
def data(tmp_path):
    # "config" in a name makes ASE guess DL_POLY even for .extxyz; training copies
    # the data to train.extxyz so mace-torch never sees such a name.
    path = tmp_path / "configs_with_a_tricky_name.extxyz"
    frames = [molecule("H2O") for _ in range(4)]
    for frame in frames:
        frame.info["REF_energy"] = -14.0
    write(path, frames, format="extxyz")
    foundation = tmp_path / "foundation.model"
    foundation.write_text("weights")
    return path, foundation


def spec(data, **changes):
    train_file, foundation = data
    return TrainingSpec(name="water_test", foundation=str(foundation),
                        train_file=str(train_file), **changes)


def test_arguments_for_plain_and_multihead(data):
    plain = train_arguments(spec(data), 2, "work")
    assert "--multiheads_finetuning=False" in plain
    assert "--foundation_model_elements=True" in plain  # never drop elements silently
    assert "--seed=2" in plain and "--name=water_test_seed2" in plain
    multihead_spec = spec(data, mode="multihead", replay="mp", replay_samples=500)
    multihead = train_arguments(multihead_spec, 1, "w")
    assert "--multiheads_finetuning=True" in multihead
    assert "--foundation_model_elements=True" in multihead
    assert "--pt_train_file=mp" in multihead and "--num_samples_pt=500" in multihead
    with pytest.raises(ValueError, match="replay"):
        spec(data, mode="multihead")
    with pytest.raises(ValueError, match="multihead"):
        spec(data, replay="mp")
    with pytest.raises(ValueError, match="mode"):
        spec(data, mode="lora")


def test_final_error_table():
    errors = final_errors("INFO: Training complete\n" + TABLE)
    assert errors == {
        "train_Default": {"rmse_e_mev_per_atom": 3.8, "rmse_f_mev_per_A": 38.4},
        "valid_Default": {"rmse_e_mev_per_atom": 3.9, "rmse_f_mev_per_A": 25.4},
    }


def test_train_local_runs_every_seed_and_installs(data, tmp_path, monkeypatch):
    trainer = tmp_path / "fake_train.py"
    trainer.write_text(FAKE_TRAINER)
    monkeypatch.setattr(training, "mace_run_train", lambda: sys.executable)
    run_spec = spec(data, seeds=(1, 2), card={"reference": "PBE/def2-TZVP", "scope": "water"})
    original = training.train_arguments
    monkeypatch.setattr(training, "train_arguments",
                        lambda *a, **k: [str(trainer), *original(*a, **k)])
    runs = train_local(run_spec, tmp_path / "run", cache_dir=tmp_path / "cache")
    assert [run.ok for run in runs] == [True, True]
    call = json.loads((tmp_path / "run" / "seed1" / "call.json").read_text())
    assert call["cache"] == str((tmp_path / "cache").resolve())  # downloads stay there
    assert call["train_exists"] and any("run" in a and "train.extxyz" in a for a in call["argv"])
    installed = install_models(tmp_path / "run", destination=tmp_path / "finetuned")
    assert [path.name for path in installed] == ["water_test_seed1.model", "water_test_seed2.model"]
    card = json.loads((installed[0].parent / "water_test_seed1.model.json").read_text())
    assert card["fine_tuning"] == "plain" and card["keeps_foundation_elements"]
    assert card["training_errors_meV"]["valid_Default"]["rmse_f_mev_per_A"] == 25.4
    assert card["scope"] == "water" and card["seed"] == 1


def test_training_package_for_the_hpc(data, tmp_path):
    package = write_training_package(
        spec(data, mode="multihead", replay="mp", replay_samples=500, seeds=(1, 2, 3), epochs=3),
        tmp_path / "package",
    )
    script = (package / "run_train.slurm").read_text()
    assert "#SBATCH --array=1,2,3" in script and "#SBATCH --gres=gpu:1" in script
    assert 'cd "runs/seed${SEED}"' in script
    assert '"--train_file=../../data/train.extxyz"' in script
    assert '"--foundation_model=../../foundation/foundation.model"' in script
    assert '"--name=water_test_seed${SEED}"' in script and '"--seed=${SEED}"' in script
    assert '"--pt_train_file=mp"' in script
    assert (package / "data" / "train.extxyz").is_file() and (package / "logs").is_dir()
    assert "mp_traj_combined.xyz" in (package / "download_mp_replay.sh").read_text()
    info = json.loads((package / "spec.json").read_text())
    assert info["placeholders_left"] == [
        "<ACCOUNT>", "<ACTIVATE_PYTHON_ENV_WITH_MACE>", "<CUDA_MODULE>", "<GPU_PARTITION>"]
    readme = (package / "README.md").read_text()
    assert "download_mp_replay.sh" in readme and "install_models" in readme
    # Models copied back from the cluster install like local ones.
    (package / "runs" / "seed2").mkdir(parents=True)
    (package / "runs" / "seed2" / "water_test_seed2.model").write_text("model")
    (installed,) = install_models(package, destination=tmp_path / "finetuned")
    assert installed.name == "water_test_seed2.model"
    assert "replay 'mp'" in json.loads(installed.with_suffix(".model.json").read_text())[
        "fine_tuning"]


def test_package_keeps_caches_out_of_a_small_home(data, tmp_path):
    package = write_training_package(
        spec(data, mode="multihead", replay="mp"), tmp_path / "package",
        slurm=GpuSlurmSettings(cache_dir="/project/me/cache"))
    script = (package / "run_train.slurm").read_text()
    # after set -u, before training: mace-torch reads the replay set from there
    assert script.index("set -u") < script.index('export XDG_CACHE_HOME="/project/me/cache"') \
        < script.index("mace_run_train")
    download = (package / "download_mp_replay.sh").read_text()
    assert 'cache="/project/me/cache/mace"' in download and "HOME" not in download
    assert "/project/me/cache/mace" in (package / "README.md").read_text()


def test_package_with_an_own_replay_file(data, tmp_path):
    replay = tmp_path / "my_replay.extxyz"
    write(replay, [molecule("CH4")])
    package = write_training_package(spec(data, mode="multihead", replay=str(replay)),
                                     tmp_path / "own", copy_foundation=False)
    script = (package / "run_train.slurm").read_text()
    assert '"--pt_train_file=../../data/replay.extxyz"' in script
    assert not (package / "download_mp_replay.sh").exists()
    assert (package / "data" / "replay.extxyz").is_file()


def test_scratch_mode_for_a_delta_correction(data, tmp_path):
    train_file, _ = data
    scratch = TrainingSpec(name="delta", foundation="", train_file=str(train_file),
                           mode="scratch", e0s={"1": -0.5, 8: 2.0},
                           architecture={"r_max": 4.0}, energy_key="DELTA_energy",
                           forces_key="DELTA_forces")
    assert scratch.e0s == {1: -0.5, 8: 2.0} and not scratch.keep_foundation_elements
    arguments = train_arguments(scratch, 1, "w")
    assert "--model=MACE" in arguments and "--E0s={1:-0.5000000000,8:2.0000000000}" in arguments
    assert "--r_max=4.0" in arguments and "--hidden_irreps=32x0e+32x1o" in arguments
    assert "--energy_key=DELTA_energy" in arguments
    assert "--loss=weighted" in arguments and "--compute_stress=True" not in arguments
    stressed = train_arguments(TrainingSpec.from_json({**scratch.to_json(), "stress_weight": 10}),
                               1, "w")
    # mace-torch ignores stress under the "weighted" loss
    assert "--loss=stress" in stressed and "--compute_stress=True" in stressed
    assert "--stress_weight=10" in stressed
    assert not any(a.startswith(("--foundation_model", "--multiheads")) for a in arguments)
    assert TrainingSpec.from_json(json.loads(json.dumps(scratch.to_json()))) == scratch
    with pytest.raises(ValueError, match="e0s"):
        TrainingSpec(name="x", foundation="", train_file="t", mode="scratch")
    with pytest.raises(ValueError, match="scratch"):
        TrainingSpec(name="x", foundation="f", train_file="t", e0s={1: 0.0})
    # Installed like any other run, with a card that says it was not fine-tuned.
    (tmp_path / "run" / "seed1").mkdir(parents=True)
    (tmp_path / "run" / "spec.json").write_text(json.dumps({**scratch.to_json(), "seeds": [1]}))
    (tmp_path / "run" / "seed1" / "delta_seed1.model").write_text("model")
    (installed,) = install_models(tmp_path / "run", destination=tmp_path / "out")
    card = json.loads((installed.parent / "delta_seed1.model.json").read_text())
    assert "fine_tuned_from" not in card and card["trained_from_scratch"]["r_max"] == 4.0
