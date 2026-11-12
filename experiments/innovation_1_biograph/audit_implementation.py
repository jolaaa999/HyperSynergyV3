# [Innovation-1: Biological Graph Cell Encoder]
"""记录本次实际 diff 和保护文件检查，不将原有未提交修改归为本次改动。"""
import ast
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = json.loads((EXP/'results/before_source_sha256.json').read_text())
    modified = [name for name, value in before.items() if digest(ROOT/name) != value]
    added_core = ['encoders/biological_graph_data.py', 'encoders/cell_biological_graph.py',
                  'encoders/cell_factory.py', 'innovation_checkpoint.py']
    protected = json.loads((EXP/'results/protected_files_before.json').read_text())
    touched = []
    for name, metadata in protected.items():
        path = ROOT/name
        if (not path.exists() or path.stat().st_size != metadata['size']
                or path.stat().st_mtime_ns != metadata['mtime_ns']):
            touched.append(name)
    files = list(ROOT.glob('*.py')) + list((ROOT/'encoders').glob('*.py'))
    files += list(EXP.glob('*.py')) + list((EXP/'tests').glob('*.py'))
    for path in files:
        compile(path.read_text(), str(path), 'exec')
    from encoders import SampleFeatureExtractor
    model = SampleFeatureExtractor(cell_encoder_config={'cell_encoder': {'type': 'se_cnn'}})
    cell_dim = model.cell_encoder.fc.out_features
    drug_dim = model.drug_encoder.fc.out_features
    config = json.loads((EXP/'configs/biological_graph.json').read_text())['cell_encoder']
    expression = json.loads((EXP/'results/data_audit.json').read_text())['expression']
    g, h, e, layers = expression['num_genes'], config['hidden_dim'], config['gene_embedding_dim'], config['num_layers']
    count = g*e + h*(e+2) + layers*(2*h*h+6*h) + h+1 + h*cell_dim+cell_dim
    source_checks = {name: digest(ROOT/name) == before[name] for name in
                     ('encoders/drug_transformerconv.py', 'encoders/cell_secnn.py', 'data_syn.py', 'utils_syn.py')}
    report = dict(modified_existing_sources=modified, added_core_sources=added_core,
        protected_artifacts_checked=len(protected), protected_artifacts_changed=touched,
        protected_source_hashes_match=source_checks, syntax_files_checked=len(files),
        dimensions=dict(input=expression['input_dim'], real_genes=g, drug=drug_dim, cell=cell_dim, sample=2*drug_dim+cell_dim),
        parameters=dict(se_cnn_cell=sum(p.numel() for p in model.cell_encoder.parameters()),
            biological_graph_cell=count,
            biological_graph_count_method='exact structure formula; graph-independent; not a trained model',
            biological_graph_formula='G*E + H*(E+2) + L*(2*H*H+6*H) + (H+1) + (H*D+D)',
            baseline_extractor=sum(p.numel() for p in model.parameters()),
            biological_graph_extractor=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in model.cell_encoder.parameters())+count,
            capacity_matched=False),
        compute=dict(biograph_measured=False, reason='missing real PPI',
                     leading_order_per_cell='O(L*(G*H^2 + E_graph*H))'),
        known_protocol_limits=['Stage 2 uses get_test_batch for validation/checkpoint selection',
            'Original seed=1500 covers NumPy Stage 1 sampling only; no global Python/Torch seed',
            'Named original expression values available for 3 of 125 cells only'])
    (EXP/'results/implementation_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (EXP/'results/modified_source_sha256.json').write_text(json.dumps(
        {name: digest(ROOT/name) for name in modified+added_core}, indent=2) + '\n')
    patches = []
    for name in modified:
        patches.extend(difflib.unified_diff((EXP/'originals'/name).read_text().splitlines(True),
            (ROOT/name).read_text().splitlines(True), fromfile='a/'+name, tofile='b/'+name))
    for name in added_core:
        patches.extend(difflib.unified_diff([], (ROOT/name).read_text().splitlines(True),
            fromfile='/dev/null', tofile='b/'+name))
    (EXP/'results/innovation1_only.patch').write_text(''.join(patches))
    (EXP/'results/git_status_after.txt').write_bytes(subprocess.check_output(['git','status','--short'], cwd=str(ROOT)))
    print(json.dumps(report, indent=2))
    if touched or not all(source_checks.values()):
        raise RuntimeError('Protected files changed; investigate before declaring completion')


if __name__ == '__main__':
    main()
