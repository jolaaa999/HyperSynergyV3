# [Innovation-1: Biological Graph Cell Encoder]
"""隔离三阶段输出，执行原训练函数，禁止偷偷更改学习率/shot/训练计划。"""
import argparse
import json
import os
import random
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--stage', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--checkpoint', help='Stage 3 的本实验 Stage 2 checkpoint')
    parser.add_argument('--rng-state', help='复用同阶段对照组初始 RNG 状态，不引入新 seed')
    parser.add_argument('--resume-stage1', action='store_true', help='仅恢复同配置同源代码的 Stage 1 epoch checkpoint')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.resume_stage1 and (args.stage != 1 or args.rng_state):
        raise ValueError('--resume-stage1 only applies to Stage 1 and restores checkpoint RNG directly')
    import numpy as np
    import torch
    from encoders.cell_factory import load_encoder_config
    from encoders.biological_graph_data import load_biological_graph, sha256
    from innovation_checkpoint import load_extractor, load_meta_model
    config = load_encoder_config(str(Path(args.config).resolve()))
    if config['cell_encoder']['type'] == 'biological_graph':
        _, audit = load_biological_graph(config['biological_graph'])
        print('Verified biological graph: ' + audit['fingerprint'], flush=True)
    destination = Path(args.run_dir).resolve()
    if EXPERIMENT not in destination.parents or destination == EXPERIMENT:
        raise ValueError('run-dir must be a new subdirectory of ' + str(EXPERIMENT))
    scripts = {1: 'representation_learning.py', 2: 'train_few_shot.py', 3: 'test_few_shot.py'}
    pretrain = destination / 'checkpoints/stage1/pretrain_representation_few_zero_setting_change_encoders.model'
    if args.stage > 1:
        if not pretrain.is_file():
            raise FileNotFoundError('Stage 1 of this run has not completed: ' + str(pretrain))
        from encoders import SampleFeatureExtractor
        extractor = SampleFeatureExtractor(cell_encoder_config=config)
        load_extractor(extractor, torch.load(str(pretrain), map_location='cpu'))
        del extractor
    if args.stage == 3:
        if not args.checkpoint:
            raise ValueError('--checkpoint is required for Stage 3')
        checkpoint = Path(args.checkpoint).resolve()
        if not checkpoint.is_file() or checkpoint.parent != destination / 'checkpoints/stage2':
            raise ValueError('Select a Stage 2 checkpoint belonging to this experiment run')
        from model_few import HyperSynergy
        checked = HyperSynergy(5, 40, cell_encoder_config=config)
        load_meta_model(checked, torch.load(str(checkpoint), map_location='cpu'))
        del checked
    if args.check_only:
        print('PASS: data/config/checkpoint preflight; no training started')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Original HyperSynergy adaptation requires CUDA; no CPU algorithm substitution')
    # 沿用当前 V2 启动器针对 Torch 1.7 / RTX 4090 的兼容设置。
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination / 'configs/encoder.json'
    for name in ('configs', 'logs', 'results/result_few_shot', 'checkpoints/stage1',
                 'checkpoints/stage2', 'saved_model'):
        (destination / name).mkdir(parents=True, exist_ok=True)
    if resolved.exists() and json.loads(resolved.read_text()) != config:
        raise ValueError('Existing run has a different encoder config')
    if args.stage == 1 and pretrain.exists():
        raise FileExistsError('Completed Stage 1 exists; refusing overwrite')
    if args.stage == 2 and list((destination / 'checkpoints/stage2').glob('*.pth')):
        raise FileExistsError('Stage 2 weights already exist; start a separate run')
    stage_log = destination / 'logs/{}.log'.format(scripts[args.stage][:-3])
    if args.resume_stage1:
        resume = destination / 'checkpoints/stage1/pretrain_representation_few_zero_setting_change_encoders.training.pt'
        if not resume.is_file() or not resolved.is_file():
            raise FileNotFoundError('No resumable Stage 1 checkpoint/config in this run')
        index = 1
        while (destination / 'logs/representation_learning.resume_{}.log'.format(index)).exists():
            index += 1
        stage_log = destination / 'logs/representation_learning.resume_{}.log'.format(index)
    elif stage_log.exists():
        raise FileExistsError('Stage log already exists; refusing to mix runs: ' + str(stage_log))
    resolved.write_text(json.dumps(config, indent=2) + '\n')
    for link, target in ((destination / 'data', ROOT / 'data'),
                         (destination / 'pretrain_representation_model', destination / 'checkpoints/stage1'),
                         (destination / 'saved_model/saved_model_few_shot_setting_change_encoders',
                          destination / 'checkpoints/stage2')):
        if link.exists():
            if link.resolve() != target.resolve():
                raise ValueError('Unexpected run output/data link: ' + str(link))
        else:
            link.symlink_to(target, target_is_directory=True)
    snapshot = destination / 'source'
    files = list(ROOT.glob('*.py')) + list((ROOT / 'encoders').glob('*.py'))
    if snapshot.exists():
        for source in files:
            copy = snapshot / source.relative_to(ROOT)
            if not copy.exists() or sha256(copy) != sha256(source):
                raise ValueError('Run source differs from current source: ' + str(source))
    else:
        for source in files:
            copy = snapshot / source.relative_to(ROOT)
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(copy))
    (destination / 'configs/source_sha256.json').write_text(json.dumps(
        {str(p.relative_to(ROOT)): sha256(p) for p in files}, indent=2))
    # 同阶段可恢复相同初始 RNG；保持旧脚本内部 numpy seed=1500 及抽样实现不变。
    if args.rng_state:
        state = torch.load(str(Path(args.rng_state).resolve()), map_location='cpu')
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device count mismatch')
        random.setstate(state['python']); np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])
    rng = dict(python=random.getstate(), numpy=np.random.get_state(),
               torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
    if not args.resume_stage1:
        torch.save(rng, str(destination / 'configs/stage{}_initial_rng.pt'.format(args.stage)))
    os.environ['HYPERSYNERGY_ENCODER_CONFIG'] = str(resolved)
    os.environ['HYPERSYNERGY_RUN_DIR'] = str(destination)
    if args.stage == 3:
        os.environ['HYPERSYNERGY_TEST_CHECKPOINT'] = str(checkpoint)
    os.chdir(str(destination))
    sys.path.insert(0, str(snapshot))
    print('Executing original stage {}. Log: {}'.format(args.stage, stage_log), flush=True)
    # 独立日志不写基线；不添加 epoch/shot/learning-rate 等隐式覆盖。
    with stage_log.open('x', buffering=1) as log:
        sys.stdout = log
        sys.stderr = log
        try:
            runpy.run_path(str(snapshot / scripts[args.stage]), run_name='__main__')
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == '__main__':
    main()
