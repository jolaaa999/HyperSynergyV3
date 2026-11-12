#!/usr/bin/env bash
# Run the existing V2 settings, first pretraining and then meta-training.
set -Eeuo pipefail
project_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
python_bin=/data/jyk/miniconda3/envs/hyper/bin/python
if ! [[ -v CUDA_VISIBLE_DEVICES ]]; then export CUDA_VISIBLE_DEVICES=0; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$project_dir"

# Torch 1.7/CUDA 11's JIT compiler does not recognize the RTX 4090 target.
# Disable GPU JIT fusion in the launched process; keep the model operators,
# training scripts, hyperparameters, and shared Python environment unchanged.
run_training_stage() {
    "$python_bin" -B -u - "$1" <<'PY'
import runpy
import sys
import torch
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)
print('Runtime compatibility: GPU JIT fusion disabled for Torch 1.7/CUDA 11 on RTX 4090', flush=True)
stage_script = sys.argv[1]
sys.argv = [stage_script]
runpy.run_path(stage_script, run_name='__main__')
PY
}

exec 9>"$project_dir/.v2_training.lock"
flock -n 9 || { echo 'V2 training is already running.' >&2; exit 1; }
"$python_bin" -B - <<'PY'
import torch
import torch_geometric
from encoders import SampleFeatureExtractor
if not torch.cuda.is_available():
    raise SystemExit('BLOCKED: CUDA is unavailable here; training has not started. Run this launcher in a GPU-enabled server terminal.')
x = torch.ones(1, device='cuda:0')
assert (x + x).item() == 2
print('GPU:', torch.cuda.get_device_name(0))
print('torch:', torch.__version__, 'torch_geometric:', torch_geometric.__version__)
print('PASS: CUDA and encoder imports')
PY

pretrained_file="$project_dir/pretrain_representation_model/pretrain_representation_few_zero_setting_change_encoders.model"
checkpoint_dir="$project_dir/saved_model/saved_model_few_shot_setting_change_encoders"
if [[ -e "$pretrained_file" ]]; then
    echo "Refusing to overwrite existing V2 weights: $pretrained_file" >&2
    exit 1
fi
if compgen -G "$checkpoint_dir/*.pth" > /dev/null; then
    echo "Existing V2 checkpoints found: $checkpoint_dir" >&2
    exit 1
fi

mkdir -p "$project_dir/runs" "$project_dir/pretrain_representation_model" "$project_dir/saved_model"
run_dir="$(mktemp -d "$project_dir/runs/v2_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
stage=setup
trap 'exit_code=$?; printf "%s stage=%s exit_code=%s\n" "$(date -Is)" "$stage" "$exit_code" >> "$run_dir/status.log"' EXIT

# Snapshot unchanged V2 Python files; keep data and weight paths in V2.
cp "$project_dir"/*.py "$run_dir/"
mkdir "$run_dir/encoders"
cp "$project_dir"/encoders/*.py "$run_dir/encoders/"
cp "$project_dir/run_v2_training.sh" "$run_dir/run_v2_training.snapshot.sh"
ln -s "$project_dir/data" "$run_dir/data"
ln -s "$project_dir/pretrain_representation_model" "$run_dir/pretrain_representation_model"
ln -s "$project_dir/saved_model" "$run_dir/saved_model"
cd "$run_dir"
sha256sum ./*.py ./encoders/*.py ./run_v2_training.snapshot.sh > source.sha256
printf '%s\n' "Run directory: $run_dir" "Pretrained weights: $pretrained_file" "Meta checkpoints: $checkpoint_dir"
printf '%s\n' "$run_dir" > "$project_dir/runs/latest_v2_run.txt"
printf '%s\n' "$$" > launcher.pid

stage=pretraining
printf '%s stage=%s started\n' "$(date -Is)" "$stage" >> status.log
run_training_stage representation_learning.py > representation_learning.log 2>&1

stage=verify_pretrained
"$python_bin" -B - <<'PY' > checkpoint_check.log 2>&1
import torch
from encoders import SampleFeatureExtractor
path = 'pretrain_representation_model/pretrain_representation_few_zero_setting_change_encoders.model'
state = torch.load(path, map_location='cpu')
SampleFeatureExtractor().load_state_dict(state, strict=True)
for name, tensor in state.items():
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise RuntimeError('Non-finite pretrained tensor: ' + name)
print('PASS: new extractor weights load strictly and are finite')
PY

stage=meta_training
printf '%s stage=%s started\n' "$(date -Is)" "$stage" >> status.log
run_training_stage train_few_shot.py > train_few_shot.log 2>&1
stage=completed
printf '%s stage=%s\n' "$(date -Is)" "$stage" >> status.log
echo "V2 pretraining and meta-training completed. Logs: $run_dir"
