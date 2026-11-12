# [Innovation-1: Biological Graph Cell Encoder]
"""可选 JSON 配置；缺省完全沿用 SE-CNN，不读取任何 PPI 文件。"""
import copy
import json
import os
from pathlib import Path

from .cell_secnn import CellSEConv


def load_encoder_config(config=None):
    if config is None:
        config = os.environ.get('HYPERSYNERGY_ENCODER_CONFIG')
    if not config:
        return None
    if isinstance(config, (str, Path)):
        with open(str(config)) as handle:
            config = json.load(handle)
    config = copy.deepcopy(config)
    if set(config) - {'cell_encoder', 'biological_graph'}:
        raise ValueError('Encoder config must not override training or task settings')
    # [Innovation-1: Biological Graph Cell Encoder]
    # 文件路径统一相对 V3 项目根目录，独立实验 cwd 不影响真实数据定位。
    root = Path(__file__).resolve().parents[1]
    for key, value in config.get('biological_graph', {}).items():
        if key.endswith('_path') and not Path(value).is_absolute():
            config['biological_graph'][key] = str(root / value)
    return config


def build_configured_cell_encoder(out_dim=256, config=None):
    """只替换 Cell Encoder，输出维度由当前 SampleFeatureExtractor 接口传入。"""
    config = load_encoder_config(config)
    if config is None:
        return CellSEConv(out_dim=out_dim)
    options = dict(config.get('cell_encoder', {}))
    name = options.pop('type', 'se_cnn')
    if name in ('se_cnn', 'secnn'):
        if options:
            raise ValueError('SE-CNN baseline does not accept BioGraph parameters')
        encoder = CellSEConv(out_dim=out_dim)
    elif name == 'biological_graph':
        from .biological_graph_data import load_biological_graph
        from .cell_biological_graph import BiologicalGraphCellEncoder
        graph_config = dict(config.get('biological_graph', {}))
        if graph_config.get('dynamic_edge_gate', False):
            raise ValueError('Dynamic edge gate is outside Innovation-1')
        allowed = {'feature_path', 'codes_path', 'gene_index_path', 'expression_reference_path',
                   'edge_path', 'source_manifest_path', 'gene_id_namespace', 'directed',
                   'add_self_loops', 'normalize_expression', 'dynamic_edge_gate'}
        if set(graph_config) - allowed:
            raise ValueError('Unknown biological_graph options: ' + str(set(graph_config) - allowed))
        edges, audit = load_biological_graph(graph_config)
        encoder = BiologicalGraphCellEncoder(edges, audit, out_dim=out_dim,
            normalize_expression=graph_config.get('normalize_expression', True), **options)
    else:
        raise ValueError('Unknown cell encoder: ' + str(name))
    encoder.innovation_config = config
    return encoder
