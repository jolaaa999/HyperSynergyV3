# [Innovation-1: Biological Graph Cell Encoder]
"""仅从实际一致的带表头表达生成索引；不下载或伪造 PPI。"""
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from encoders.biological_graph_data import read_features, validate_expression, load_biological_graph
from encoders.cell_factory import load_encoder_config


def prepare_index(feature_path, reference_path, destination, codes_path, padding=()):
    """padding 只能显式指定；真实 CSV 按原顺序逐值通过才发布索引。"""
    _, matrix = read_features(feature_path)
    with open(str(reference_path), newline='') as handle:
        header = next(csv.reader(handle))
    if len(set(header)) != len(header):
        raise ValueError('Duplicate reference headers')
    genes = [g for g in header if g not in ('CELL_ID', 'CELL_NAME')]
    padding = set(padding)
    if any(i < 0 or i >= matrix.size(1) for i in padding):
        raise ValueError('Padding position out of bounds')
    if len(genes) + len(padding) != matrix.size(1):
        raise ValueError('Named genes do not cover expression width; explicit verified padding required')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(destination.parent), suffix='.csv')
    try:
        with os.fdopen(fd, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['feature_index', 'gene_id', 'is_padding'])
            iterator = iter(genes)
            for i in range(matrix.size(1)):
                writer.writerow([i, '' if i in padding else next(iterator), int(i in padding)])
        report = validate_expression(feature_path, temporary, reference_path, codes_path)
        if destination.exists():
            if destination.read_bytes() != Path(temporary).read_bytes():
                raise FileExistsError('Refusing to overwrite a different gene index: ' + str(destination))
        else:
            os.replace(temporary, str(destination))
        return report
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).parent / 'configs/biological_graph.json'))
    parser.add_argument('--prepare-index', action='store_true')
    parser.add_argument('--padding-index', type=int, action='append', default=[])
    parser.add_argument('--report', default=str(Path(__file__).parent / 'results/data_audit.json'))
    args = parser.parse_args()
    config = load_encoder_config(args.config)['biological_graph']
    if args.prepare_index:
        prepare_index(config['feature_path'], config['expression_reference_path'],
                      config['gene_index_path'], config['codes_path'], args.padding_index)
    report = {'expression': validate_expression(config['feature_path'], config['gene_index_path'],
                config['expression_reference_path'], config['codes_path'])}
    try:
        _, graph = load_biological_graph(config)
        report.update(status='ready', graph=graph)
    except (ValueError, FileNotFoundError) as exc:
        report.update(status='blocked', graph_error=str(exc))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'expression'}, indent=2))
    print('Expression:', {k: v for k, v in report['expression'].items() if k not in ('genes', 'gene_positions')})
    return 0 if report['status'] == 'ready' else 2


if __name__ == '__main__':
    sys.exit(main())
