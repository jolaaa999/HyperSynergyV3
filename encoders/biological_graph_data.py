# [Innovation-1: Biological Graph Cell Encoder]
"""真实表达、基因顺序和 PPI 的可追溯校验；失败时禁止构造训练模型。"""
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _id(value):
    text = str(value).strip()
    return text[:-2] if text.endswith('.0') else text


def read_features(feature_path):
    """读取原始 cg 张量，返回原数据帧和 [细胞数, 输入维度]，不增加特征。"""
    with open(str(feature_path), 'rb') as handle:
        frame = pickle.load(handle)
    if not {'CELL_ID', 'cg'}.issubset(frame.columns):
        raise ValueError('Expression pickle requires CELL_ID and cg')
    ids = [_id(x) for x in frame.CELL_ID]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate expression CELL_ID')
    vectors = [torch.as_tensor(x).detach().cpu() for x in frame.cg]
    if not vectors or any(x.dim() != 1 for x in vectors):
        raise ValueError('Expected nonempty one-dimensional expression vectors')
    if len(set(x.numel() for x in vectors)) != 1:
        raise ValueError('Inconsistent expression dimensions')
    matrix = torch.stack(vectors)
    if not matrix.is_floating_point() or not torch.isfinite(matrix).all():
        raise ValueError('Expression must be finite floating-point values')
    return frame, matrix


def validate_expression(feature_path, gene_index_path, reference_path, codes_path=None):
    """逐维对照带基因表头的真实表达和原 pickle；padding 必须显式标记且全零。

    reference CSV 的每一行均与同 CELL_ID 的原 cg 比较；不借用其他 CCLE 顺序。
    允许部分细胞参考，但明确记录覆盖率，不声称已核验缺失的原始行。
    """
    frame, matrix = read_features(feature_path)
    with open(str(gene_index_path), newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['feature_index', 'gene_id', 'is_padding']:
            raise ValueError('gene_index columns must be feature_index,gene_id,is_padding')
        records = list(reader)
    width = matrix.size(1)
    if len(records) != width or [int(r['feature_index']) for r in records] != list(range(width)):
        raise ValueError('gene_index must cover every expression position in exact order')
    if any(r['is_padding'] not in ('0', '1') for r in records):
        raise ValueError('is_padding must be 0 or 1')
    positions = [i for i, r in enumerate(records) if r['is_padding'] == '0']
    padding = [i for i, r in enumerate(records) if r['is_padding'] == '1']
    genes = [records[i]['gene_id'].strip() for i in positions]
    if not genes or any(not g for g in genes) or len(genes) != len(set(genes)):
        raise ValueError('Real gene identifiers must be nonempty and unique')
    if any(records[i]['gene_id'].strip() for i in padding):
        raise ValueError('Padding positions must not have biological gene identifiers')
    if padding and torch.count_nonzero(matrix[:, padding]).item():
        raise ValueError('Declared padding contains nonzero expression values')
    with open(str(reference_path), newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError('Duplicate reference column names')
    reference = pd.read_csv(str(reference_path), dtype={'CELL_ID': str})
    if 'CELL_ID' not in reference.columns:
        raise ValueError('Reference CSV requires CELL_ID')
    named_columns = [c for c in reference.columns if c not in ('CELL_ID', 'CELL_NAME')]
    if named_columns != genes:
        raise ValueError('Gene index order differs from actual named reference expression columns')
    reference_ids = [_id(x) for x in reference.CELL_ID]
    if len(reference_ids) < 2 or len(reference_ids) != len(set(reference_ids)):
        raise ValueError('At least two distinct real reference cells are required')
    lookup = {_id(cell): i for i, cell in enumerate(frame.CELL_ID)}
    for row_index, cell in enumerate(reference_ids):
        if cell not in lookup:
            raise ValueError('Reference CELL_ID absent from original features: ' + cell)
        observed = matrix[lookup[cell], positions]
        expected = torch.tensor(reference.iloc[row_index][genes].astype(float).values, dtype=matrix.dtype)
        if not torch.isfinite(expected).all() or not torch.equal(observed, expected):
            bad = torch.nonzero(observed != expected).flatten().tolist()
            first = bad[0] if bad else 0
            raise ValueError('Expression alignment mismatch: CELL_ID={} feature_index={} gene_id={}'.format(
                cell, positions[first], genes[first]))
    if codes_path is not None:
        with open(str(codes_path), 'rb') as handle:
            codes = pickle.load(handle)['cells']
        for key, index in codes.item2idx.items():
            if index not in frame.index or _id(frame.loc[index, 'CELL_ID']) != _id(key):
                raise ValueError('codes_cell / expression row mismatch: {}'.format(key))
        if set(_id(x) for x in codes.item2idx) != set(lookup):
            raise ValueError('codes_cell does not cover expression CELL_IDs exactly')
    report = dict(input_dim=width, num_genes=len(genes), padding_indices=padding,
                  num_cells=len(frame), reference_cells=reference_ids,
                  reference_coverage='{}/{}'.format(len(reference_ids), len(frame)),
                  comparison='exact equality after original float32 conversion',
                  zero_expression_cells=[_id(frame.iloc[i].CELL_ID) for i in range(len(frame))
                                         if not torch.count_nonzero(matrix[i, positions]).item()],
                  all_zero_columns=torch.nonzero((matrix == 0).all(0)).flatten().tolist(),
                  genes=genes, gene_positions=positions,
                  feature_sha256=sha256(feature_path), gene_index_sha256=sha256(gene_index_path),
                  reference_sha256=sha256(reference_path))
    if codes_path is not None:
        report['codes_sha256'] = sha256(codes_path)
    return report


def load_biological_graph(config):
    """按真实基因 ID 建图，过滤集合外边；排序去重后明确双向化及加自环。"""
    required = ('feature_path', 'gene_index_path', 'expression_reference_path', 'codes_path',
                'edge_path', 'source_manifest_path')
    missing = [str(config.get(k, '<missing {}>'.format(k))) for k in required
               if not config.get(k) or not Path(config[k]).is_file()]
    if missing:
        raise FileNotFoundError('BioGraph BLOCKED: required real graph/expression resources missing: ' + ', '.join(missing))
    audit = validate_expression(config['feature_path'], config['gene_index_path'],
                                config['expression_reference_path'], config['codes_path'])
    with open(config['source_manifest_path']) as handle:
        source = json.load(handle)
    for field in ('source_name', 'source_url', 'version', 'gene_id_namespace', 'edge_sha256'):
        if not isinstance(source.get(field), str) or not source[field].strip():
            raise ValueError('PPI provenance missing ' + field)
    if source['gene_id_namespace'] != config.get('gene_id_namespace'):
        raise ValueError('PPI namespace differs from expression gene identifiers')
    edge_hash = sha256(config['edge_path'])
    if source['edge_sha256'] != edge_hash:
        raise ValueError('PPI edge file SHA256 differs from source manifest')
    node_map = {g: i for i, g in enumerate(audit['genes'])}
    edges, raw, retained, external, raw_loops = set(), 0, 0, 0, 0
    directed = config.get('directed', False)
    with open(config['edge_path'], newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {'source', 'target'}.issubset(reader.fieldnames):
            raise ValueError('PPI CSV requires source,target identifiers in the configured namespace')
        for row in reader:
            raw += 1
            a, b = row['source'].strip(), row['target'].strip()
            if not a or not b:
                raise ValueError('Empty PPI endpoint at row {}'.format(raw + 1))
            if a not in node_map or b not in node_map:
                external += 1
                continue
            a, b = node_map[a], node_map[b]
            if a == b:
                raw_loops += 1
                continue
            retained += 1
            edges.add((a, b) if directed else tuple(sorted((a, b))))
    if not edges:
        raise ValueError('No valid biological non-self edges overlap the expression genes')
    biological_edges = len(edges)
    connected = set(i for e in edges for i in e)
    isolates = sorted(set(range(len(node_map))) - connected)
    if not directed:
        edges |= {(b, a) for a, b in list(edges)}
    arcs = len(edges)
    if config.get('add_self_loops', True):
        edges |= {(i, i) for i in range(len(node_map))}
    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    audit.update(source=source, raw_edges=raw, retained_input_rows=retained,
                 filtered_external_rows=external, discarded_input_self_loops=raw_loops,
                 biological_edges=biological_edges, directed_arcs_without_loops=arcs,
                 final_edges=edge_index.size(1), isolated_nodes=len(isolates),
                 isolated_gene_ids=[audit['genes'][i] for i in isolates],
                 directed=directed, add_self_loops=config.get('add_self_loops', True),
                 edge_sha256=edge_hash, source_manifest_sha256=sha256(config['source_manifest_path']))
    fingerprint = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    audit['fingerprint'] = fingerprint
    return edge_index, audit
