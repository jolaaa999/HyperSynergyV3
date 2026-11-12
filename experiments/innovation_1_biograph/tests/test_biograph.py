# [Innovation-1: Biological Graph Cell Encoder]
"""单元夹具仅用于验证算子，绝不作为生物图或真实训练数据。"""
import ast
import copy
import csv
import importlib.util
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'experiments/innovation_1_biograph'
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from encoders import SampleFeatureExtractor
from encoders.cell_factory import load_encoder_config, build_configured_cell_encoder
from encoders.cell_biological_graph import BiologicalGraphCellEncoder
from encoders.biological_graph_data import validate_expression, load_biological_graph, sha256
from innovation_checkpoint import (strict_load, extractor_payload, load_extractor,
                                   add_meta_metadata, load_meta_model, add_resume_metadata)
from model_rep import EMBNet
from model_few import HyperSynergy
from pretrain_checkpoint import save_training_checkpoint, load_training_checkpoint
from utilities import Mapping
torch.set_num_threads(2)
# 与现有启动器完全相同的 Torch 1.7 / RTX 4090 兼容设置，仅当前测试进程。
if torch.cuda.is_available():
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)


def fixture_encoder(**kwargs):
    # 手工极小算子夹具：0<->2，节点1无生物邻居；不是 PPI，不进入任何真实训练。
    audit = dict(input_dim=5, num_genes=3, gene_positions=[0, 2, 3],
                 padding_indices=[1, 4], fingerprint='00' * 32, fixture_only=True)
    edges = torch.tensor([[0, 2, 0, 1, 2], [2, 0, 0, 1, 2]], dtype=torch.long)
    return BiologicalGraphCellEncoder(edges, audit, dropout=0, hidden_dim=8,
                                      gene_embedding_dim=4, **kwargs)


class BioGraphOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)  # 仅测试局部 RNG；不改变生产训练设置。
        self.model = fixture_encoder()
        self.x = torch.tensor([[1., 0., 2., 4., 0.], [3., 0., 1., 2., 0.]])

    def test_shapes_batch_one_and_many(self):
        self.assertEqual(tuple(self.model(self.x[:1]).shape), (1, 256))
        self.assertEqual(tuple(self.model(self.x).shape), (2, 256))

    def test_no_cross_graph_edges(self):
        edges, batch = self.model.batched_topology(3)
        self.assertTrue(torch.equal(batch[edges[0]], batch[edges[1]]))
        self.assertEqual(edges.size(1), self.model.edge_index.size(1) * 3)
        self.assertEqual(batch.tolist(), [0]*3 + [1]*3 + [2]*3)

    def test_pooling_is_per_cell_and_batch_equivalent(self):
        self.model.eval()
        joint = self.model(self.x)
        separate = torch.cat([self.model(x[None]) for x in self.x])
        self.assertTrue(torch.allclose(joint, separate, atol=1e-6))
        self.assertFalse(torch.allclose(joint[0], joint[1]))

    def test_isolated_node_and_zero_expression_finite(self):
        values = torch.zeros_like(self.x)
        out = self.model(values)
        self.assertTrue(torch.isfinite(out).all())
        # 全零输入在共享 identity 下不可区分，明确测试而不是掩盖该局限。
        self.assertTrue(torch.allclose(out[0], out[1], atol=1e-6))

    def test_padding_never_a_graph_node(self):
        self.assertEqual(self.model.gene_embedding.num_embeddings, 3)
        self.assertEqual(self.model.gene_positions.tolist(), [0, 2, 3])
        self.x[0, 1] = 1
        with self.assertRaisesRegex(ValueError, 'padding'):
            self.model(self.x)

    def test_gradients_and_parameter_update(self):
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model(self.x).pow(2).mean().backward()
        for prefix in ('gene_embedding.', 'layers.0.', 'layers.1.', 'pool.'):
            grads = [p.grad for n, p in self.model.named_parameters() if n.startswith(prefix)]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        optimizer.step()
        for prefix in ('gene_embedding.', 'layers.0.', 'layers.1.', 'pool.'):
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in self.model.named_parameters()
                                if n.startswith(prefix)))

    def test_frozen_extractor_policy(self):
        for p in self.model.parameters():
            p.requires_grad = False
        self.assertFalse(self.model(self.x).requires_grad)

    def test_default_architecture_parameter_formula(self):
        model = BiologicalGraphCellEncoder(self.model.edge_index, self.model.graph_audit)
        g, h, e, d, layers = 3, 128, 16, 256, 2
        expected = g*e+h*(e+2)+layers*(2*h*h+6*h)+h+1+h*d+d
        self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
        self.assertEqual(len(model.layers), 2)
        self.assertEqual(model.output_dim, 256)

    def test_strict_roundtrip_and_graph_tamper(self):
        clone = fixture_encoder()
        strict_load(clone, self.model.state_dict())
        self.assertTrue(torch.equal(clone(self.x), self.model(self.x)))
        wrong = copy.deepcopy(self.model.state_dict())
        wrong['edge_index'][0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'graph/order'):
            strict_load(clone, wrong)

    def test_missing_extra_and_wrong_shape_rejected(self):
        for kind in ('missing', 'extra', 'shape'):
            wrong = copy.deepcopy(self.model.state_dict())
            if kind == 'missing':
                wrong.pop('gene_embedding.weight')
            elif kind == 'extra':
                wrong['extra'] = torch.zeros(1)
            else:
                wrong['gene_embedding.weight'] = torch.zeros(1)
            with self.assertRaises(ValueError):
                strict_load(self.model, wrong)

    def test_input_shape_nonfinite_and_chunking(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(2, 1, 5))
        with self.assertRaises(ValueError):
            self.model(torch.full((2, 5), float('nan')))
        other = fixture_encoder(batch_chunk_size=1)
        other.load_state_dict(self.model.state_dict(), strict=True)
        self.assertTrue(torch.allclose(other(self.x), self.model(self.x), atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_buffers_follow_cuda(self):
        self.model.cuda()
        self.assertTrue(self.model.edge_index.is_cuda)
        self.assertTrue(self.model.gene_positions.is_cuda)
        self.assertTrue(torch.isfinite(self.model(self.x.cuda())).all())


class DataValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        # 显式单元夹具，包括 padding 和孤立节点，不冒充真实数据。
        frame = pd.DataFrame({'CELL_ID': [1, 2], 'cg': [torch.tensor([1., 0., 2., 4.]),
                                                       torch.tensor([3., 0., 1., 2.])]})
        with (self.path / 'features.p').open('wb') as handle:
            pickle.dump(frame, handle)
        with (self.path / 'codes.p').open('wb') as handle:
            pickle.dump({'cells': Mapping([1, 2])}, handle)
        (self.path / 'reference.csv').write_text('CELL_ID,A,B,C\n1,1,2,4\n2,3,1,2\n')
        (self.path / 'genes.csv').write_text('feature_index,gene_id,is_padding\n0,A,0\n1,,1\n2,B,0\n3,C,0\n')
        (self.path / 'edges.csv').write_text('source,target\nA,C\nC,A\nA,A\nEXTERNAL,B\n')
        source = dict(source_name='UNIT_TEST_FIXTURE_NOT_BIOLOGICAL', source_url='unit-test://fixture',
                      version='fixture', gene_id_namespace='fixture', edge_sha256=sha256(self.path / 'edges.csv'))
        (self.path / 'source.json').write_text(json.dumps(source))
        self.config = dict(feature_path=str(self.path/'features.p'), codes_path=str(self.path/'codes.p'),
            expression_reference_path=str(self.path/'reference.csv'), gene_index_path=str(self.path/'genes.csv'),
            edge_path=str(self.path/'edges.csv'), source_manifest_path=str(self.path/'source.json'),
            gene_id_namespace='fixture', directed=False, add_self_loops=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_padding_filtered_edges_loops_isolates(self):
        edges, audit = load_biological_graph(self.config)
        self.assertEqual(audit['num_genes'], 3)
        self.assertEqual(audit['padding_indices'], [1])
        self.assertEqual(audit['raw_edges'], 4)
        self.assertEqual(audit['biological_edges'], 1)
        self.assertEqual(audit['final_edges'], 5)
        self.assertEqual(audit['isolated_gene_ids'], ['B'])
        self.assertEqual(audit['filtered_external_rows'], 1)
        self.assertTrue(torch.equal(edges, load_biological_graph(self.config)[0]))

    def test_directed_no_selfloops(self):
        self.config.update(directed=True, add_self_loops=False)
        edges, audit = load_biological_graph(self.config)
        self.assertEqual(audit['final_edges'], 2)
        model = BiologicalGraphCellEncoder(edges, audit)
        self.assertTrue(torch.isfinite(model(torch.tensor([[1., 0., 2., 4.]]))).all())

    def test_permuted_gene_order_rejected(self):
        path = self.path/'genes.csv'
        path.write_text(path.read_text().replace('0,A,0', '0,C,0').replace('3,C,0', '3,A,0'))
        with self.assertRaisesRegex(ValueError, 'order'):
            load_biological_graph(self.config)

    def test_real_value_mismatch_rejected(self):
        (self.path/'reference.csv').write_text('CELL_ID,A,B,C\n1,99,2,4\n2,3,1,2\n')
        with self.assertRaisesRegex(ValueError, 'CELL_ID=1 feature_index=0 gene_id=A'):
            load_biological_graph(self.config)

    def test_wrong_hash_or_missing_ppi_rejected(self):
        (self.path/'edges.csv').write_text('source,target\nA,B\n')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            load_biological_graph(self.config)
        (self.path/'edges.csv').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'BLOCKED'):
            load_biological_graph(self.config)

    def test_invalid_ppi_schema_and_zero_overlap_rejected(self):
        for text in ('bad,header\nx,y\n', 'source,target\nOUTSIDE,OTHER\n'):
            (self.path/'edges.csv').write_text(text)
            source = json.loads((self.path/'source.json').read_text())
            source['edge_sha256'] = sha256(self.path/'edges.csv')
            (self.path/'source.json').write_text(json.dumps(source))
            with self.assertRaises(ValueError):
                load_biological_graph(self.config)

    def test_nonzero_padding_rejected(self):
        path = self.path/'features.p'
        with path.open('rb') as handle:
            frame = pickle.load(handle)
        frame.cg.iloc[0][1] = 7
        with path.open('wb') as handle:
            pickle.dump(frame, handle)
        with self.assertRaisesRegex(ValueError, 'padding'):
            load_biological_graph(self.config)

    def test_configured_stage1_stage2_stage3_checkpoint_roundtrip_fixture_only(self):
        config = dict(cell_encoder=dict(type='biological_graph', hidden_dim=8, dropout=0),
                      biological_graph=self.config)
        stage1 = EMBNet(cell_encoder_config=config)
        payload = extractor_payload(stage1.extractor)
        destination = self.path/'extractor.pt'; torch.save(payload, str(destination))
        stage2 = HyperSynergy(5, 40, cell_encoder_config=config)
        load_extractor(stage2.extractor, torch.load(str(destination), map_location='cpu'))
        for p in stage2.extractor.parameters():
            p.requires_grad = False
        self.assertTrue(all(not p.requires_grad for p in stage2.extractor.parameters()))
        meta = add_meta_metadata(dict(step=0, state_dict=stage2.state_dict()), stage2.extractor)
        stage3 = HyperSynergy(5, 40, cell_encoder_config=config)
        load_meta_model(stage3, meta)
        self.assertTrue(torch.equal(stage1.extractor.cell_encoder.gene_embedding.weight,
                                    stage3.extractor.cell_encoder.gene_embedding.weight))
        with self.assertRaises(ValueError):
            load_extractor(stage2.extractor, SampleFeatureExtractor().state_dict())
        wrong = copy.deepcopy(payload); wrong['encoder_metadata']['graph']['fingerprint'] = 'bad'
        with self.assertRaises(ValueError):
            load_extractor(stage2.extractor, wrong)
        optimizer = torch.optim.Adam(stage1.parameters(), lr=.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=1.)
        resume_config = add_resume_metadata(dict(num_epochs=500), stage1.extractor)
        resume = str(self.path/'resume.pt')
        rows = np.zeros((2, 4))  # 仅序列化夹具，不执行训练。
        save_training_checkpoint(resume, stage1, optimizer, scheduler, 0, rows, rows, resume_config)
        epoch, train, val = load_training_checkpoint(resume, stage1, optimizer, scheduler, resume_config)
        self.assertEqual(epoch, 0); self.assertTrue(np.array_equal(train, rows))


class ExistingDataAndBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # [Innovation-1: Biological Graph Cell Encoder]
        # 仅测试入口接受已审计候选配置；不覆盖任何基线或生产配置。
        cls.config = load_encoder_config(os.environ.get(
            'BIOGRAPH_TEST_CONFIG', str(EXP/'configs/biological_graph.json')))

    def test_real_125_cell_900_column_alignment(self):
        c = self.config['biological_graph']
        report = validate_expression(c['feature_path'], c['gene_index_path'],
                                     c['expression_reference_path'], c['codes_path'])
        self.assertEqual(report['num_cells'], 125)
        self.assertEqual(report['reference_cells'], ['45', '96', '207'])
        self.assertEqual(report['num_genes'] + len(report['padding_indices']), report['input_dim'])
        self.assertEqual(report['zero_expression_cells'], [])

    def test_real_alignment_rejects_wrong_gene_index(self):
        c = self.config['biological_graph']
        with tempfile.TemporaryDirectory() as tmp:
            lines = Path(c['gene_index_path']).read_text().splitlines()
            left, right = lines[1].split(','), lines[2].split(',')
            left[1], right[1] = right[1], left[1]
            lines[1], lines[2] = ','.join(left), ','.join(right)
            path = Path(tmp)/'wrong.csv'; path.write_text('\n'.join(lines)+'\n')
            with self.assertRaisesRegex(ValueError, 'order'):
                validate_expression(c['feature_path'], path, c['expression_reference_path'], c['codes_path'])

    def test_baseline_exact_initialization_and_output(self):
        path = EXP/'originals/encoders/sample_feature.py'
        spec = importlib.util.spec_from_file_location('encoders._original_sample', str(path))
        original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
        torch.manual_seed(123); before = original.SampleFeatureExtractor()
        torch.manual_seed(123); after = SampleFeatureExtractor()
        self.assertEqual(list(before.state_dict()), list(after.state_dict()))
        for k, v in before.state_dict().items():
            self.assertTrue(torch.equal(v, after.state_dict()[k]), k)
        before.eval(); after.eval()
        with (ROOT/'data/sample_features/cell_feature_900.p').open('rb') as handle:
            features = pickle.load(handle)
        x = torch.stack(features.cg.tolist()[:3])
        self.assertTrue(torch.equal(before.cell_encoder(x), after.cell_encoder(x)))
        torch.manual_seed(123); explicit = SampleFeatureExtractor(cell_encoder_config={'cell_encoder': {'type': 'se_cnn'}})
        for k, v in before.state_dict().items():
            self.assertTrue(torch.equal(v, explicit.state_dict()[k]), k)

    def test_baseline_no_ppi_dependency_and_legacy_load(self):
        encoder = build_configured_cell_encoder(config={'cell_encoder': {'type': 'se_cnn'},
            'biological_graph': {'edge_path': '/NONEXISTENT/ppi.csv'}})
        self.assertEqual(tuple(encoder(torch.zeros(1, 900)).shape), (1, 256))
        model = SampleFeatureExtractor()
        legacy_path = ROOT/'pretrain_representation_model/pretrain_representation_few_zero_setting_change_encoders.model'
        if not legacy_path.is_file():
            self.skipTest('Optional legacy checkpoint is not distributed with the source repository')
        payload = torch.load(str(legacy_path), map_location='cpu')
        load_extractor(model, payload)
        self.assertEqual(list(extractor_payload(model)), list(model.state_dict()))

    def test_protected_sources_and_meta_algorithms_unchanged(self):
        expected = json.loads((EXP/'results/before_source_sha256.json').read_text())
        for name in ('encoders/drug_transformerconv.py', 'encoders/cell_secnn.py',
                     'data_syn.py', 'utils_syn.py', 'feature_direction.py', 'feature_ generation.py'):
            self.assertEqual(sha256(ROOT/name), expected[name], name)
        old = ast.parse((EXP/'originals/model_few.py').read_text())
        new = ast.parse((ROOT/'model_few.py').read_text())
        oldclasses = {n.name: n for n in old.body if isinstance(n, ast.ClassDef)}
        newclasses = {n.name: n for n in new.body if isinstance(n, ast.ClassDef)}
        for name in ('Inference_model', 'Weight_generstive_netwrok'):
            self.assertEqual(ast.dump(oldclasses[name]), ast.dump(newclasses[name]))
        oldmethods = {n.name: n for n in oldclasses['HyperSynergy'].body if isinstance(n, ast.FunctionDef)}
        newmethods = {n.name: n for n in newclasses['HyperSynergy'].body if isinstance(n, ast.FunctionDef)}
        for name in oldmethods:
            if name != '__init__':
                self.assertEqual(ast.dump(oldmethods[name]), ast.dump(newmethods[name]), name)

    def test_stage1_training_and_stage3_evaluation_body_unchanged(self):
        for filename, names in [('representation_learning.py', ['train', 'predicting']),
                                ('train_few_shot.py', ['train']), ('test_few_shot.py', ['test'])]:
            oldtree = ast.parse((EXP/'originals'/filename).read_text())
            newtree = ast.parse((ROOT/filename).read_text())
            for name in names:
                old = next(n for n in oldtree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                new = next(n for n in newtree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                if filename == 'test_few_shot.py':
                    # 仅加载入口不同；从 model.eval() 起完全相同，含20轮聚合和全部指标。
                    start = lambda f: next(i for i, n in enumerate(f.body) if isinstance(n, ast.Expr) and
                        isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and
                        n.value.func.attr == 'eval')
                    old.body = old.body[start(old):]; new.body = new.body[start(new):]
                if filename == 'train_few_shot.py':
                    class RemoveMetadata(ast.NodeTransformer):
                        def visit_Expr(self, node):
                            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'add_meta_metadata':
                                return None
                            return self.generic_visit(node)
                    new = RemoveMetadata().visit(new)
                self.assertEqual(ast.dump(old), ast.dump(new), filename)

    def test_training_constants_unchanged(self):
        keys = {'TRAIN_BATCH_SIZE', 'VAL_BATCH_SIZE', 'LR', 'NUM_EPOCHS', 'seed', 'lr',
                'total_epochs', 'batch_size', 'samples_support', 'samples_query',
                'total_val_batches', 'total_test_batches', 'data_path'}
        def constants(path):
            found = {}
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    key = node.targets[0].id
                    if key in keys:
                        found[key] = ast.dump(node.value)
            return found
        for filename in ('representation_learning.py', 'train_few_shot.py', 'test_few_shot.py'):
            self.assertEqual(constants(EXP/'originals'/filename), constants(ROOT/filename), filename)

    def test_real_biograph_forward_backward_when_resources_available(self):
        c = self.config['biological_graph']
        if not Path(c['edge_path']).is_file() or not Path(c['source_manifest_path']).is_file():
            self.skipTest('BLOCKED: no real PPI/source manifest; never substitute fixture graph for real training')
        model = build_configured_cell_encoder(config=self.config)
        with open(c['feature_path'], 'rb') as handle:
            frame = pickle.load(handle)
        x = torch.stack(frame.cg.tolist()[:2])
        out = model(x)
        self.assertEqual(tuple(out.shape), (2, 256))
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        out.pow(2).mean().backward(); optimizer.step()
        self.assertTrue(torch.isfinite(out).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
