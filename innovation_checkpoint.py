# [Innovation-1: Biological Graph Cell Encoder]
"""保持旧 checkpoint 兼容，同时显式核验新编码器配置、图版本和所有参数键。"""
import json
import torch
import os
from pathlib import Path


def guard_experiment_directory():
    """显式新配置只允许隔离启动，防止直接运行脚本覆盖旧 checkpoint。"""
    if not os.environ.get('HYPERSYNERGY_ENCODER_CONFIG'):
        return
    root = os.environ.get('HYPERSYNERGY_RUN_DIR')
    if (not root or Path.cwd().resolve() != Path(root).resolve()
            or not (Path(root) / 'configs/encoder.json').is_file()
            or not (Path(root) / 'checkpoints/stage1').is_dir()):
        raise RuntimeError('Use experiments/innovation_1_biograph/run_stage.py for isolated configured runs')


def encoder_metadata(extractor):
    cell = extractor.cell_encoder
    if not hasattr(cell, 'innovation_config'):
        return None
    return {'format': 'innovation_1_cell_encoder_v1',
            'config': cell.innovation_config,
            'graph': getattr(cell, 'graph_audit', None),
            'output_dim': extractor.normf.num_features - 2 * extractor.drug_encoder.fc.out_features,
            'cell_parameters': sum(p.numel() for p in cell.parameters())}


def _check_metadata(extractor, metadata):
    expected = encoder_metadata(extractor)
    if expected != metadata:
        raise ValueError('Cell encoder checkpoint metadata/config/graph mismatch; refusing cross-encoder loading')


def strict_load(module, state):
    """加载前报告 missing/unexpected/shape；关键图 buffer 不允许被 checkpoint 偷换。"""
    current = module.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    shapes = [k for k in set(current) & set(state)
              if not torch.is_tensor(state[k]) or tuple(current[k].shape) != tuple(state[k].shape)]
    report = dict(loaded_keys=len(current), missing_keys=missing, unexpected_keys=unexpected, shape_mismatches=shapes)
    print('CHECKPOINT_AUDIT ' + json.dumps(report, sort_keys=True), flush=True)
    if missing or unexpected or shapes:
        raise ValueError('Strict checkpoint parameter mismatch: ' + json.dumps(report))
    immutable = ('edge_index', 'gene_positions', 'padding_positions', 'graph_fingerprint')
    for key, tensor in state.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite checkpoint tensor: ' + key)
        if key.split('.')[-1] in immutable and not torch.equal(tensor.cpu(), current[key].cpu()):
            raise ValueError('Checkpoint graph/order buffer mismatch: ' + key)
    result = module.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError('Unexpected strict load failure')
    return report


def extractor_payload(extractor):
    metadata = encoder_metadata(extractor)
    if metadata is None:
        return extractor.state_dict()
    return dict(format='innovation_1_stage1_extractor_v1',
                encoder_metadata=metadata, state_dict=extractor.state_dict())


def load_extractor(extractor, payload):
    """Stage 2/3 只接收相应 Stage 1 编码器，药物参数也必须完整匹配。"""
    if isinstance(payload, dict) and payload.get('format') == 'innovation_1_stage1_extractor_v1':
        _check_metadata(extractor, payload.get('encoder_metadata'))
        return strict_load(extractor, payload['state_dict'])
    if encoder_metadata(extractor) is not None:
        # 显式 SE-CNN 对照配置可以读取原 SE-CNN；BioGraph 绝不能接收旧权重。
        if hasattr(extractor.cell_encoder, 'graph_audit'):
            raise ValueError('BioGraph requires a BioGraph Stage 1 checkpoint with verified metadata')
    return strict_load(extractor, payload)


def add_meta_metadata(state, extractor):
    metadata = encoder_metadata(extractor)
    if metadata is not None:
        state['encoder_metadata'] = metadata
    return state


def load_meta_model(model, payload):
    metadata = payload.get('encoder_metadata')
    if hasattr(model.extractor.cell_encoder, 'graph_audit') or metadata is not None:
        _check_metadata(model.extractor, metadata)
    return strict_load(model, payload['state_dict'])


def add_resume_metadata(config, extractor):
    metadata = encoder_metadata(extractor)
    if metadata is not None:
        config['encoder_metadata'] = metadata
    return config
