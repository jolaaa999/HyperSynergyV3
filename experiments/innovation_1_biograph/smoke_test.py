# [Innovation-1: Biological Graph Cell Encoder]
"""真实数据短测：一批 Stage1、一个 Stage2 训练任务、一个原5+40测试任务。

仅任务数量/批大小缩小以诊断接口，原 split/shot/15步适应/损失/更新公式保留；
结果绝不作为正式性能指标。缺少真实 PPI 时 BioGraph 直接阻塞，不能替换夹具。
"""
import argparse
import contextlib
import copy
import glob
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


@contextlib.contextmanager
def checkpoint_scope(destination):
    """短测权重只写全新目录，并明确标识不能作为正式实验权重。"""
    if destination:
        path = Path(destination).resolve()
        path.mkdir(parents=True, exist_ok=False)
        (path/'INTERFACE_TEST_ONLY.txt').write_text(
            'One small Stage 1 batch and one meta update only. NOT formal pretrained weights.\n')
        yield path
    else:
        with tempfile.TemporaryDirectory(prefix='innovation1_smoke_') as tmp:
            yield Path(tmp)


def parameter_evidence(module, before, prefixes):
    """记录真实反向梯度及更新；每个核心参数组必须有限且收到非零梯度。"""
    import torch
    values = {}
    for name, p in module.named_parameters():
        if not torch.isfinite(p).all():
            raise ValueError('Nonfinite parameter: ' + name)
        values[name] = dict(changed=not torch.equal(p, before[name]),
            grad_finite=p.grad is not None and bool(torch.isfinite(p.grad).all()),
            grad_l1=None if p.grad is None else float(p.grad.abs().sum().item()))
    for prefix in prefixes:
        group = [v for k, v in values.items() if k.startswith(prefix)]
        if (not group or not all(v['grad_finite'] for v in group)
                or sum(v['grad_l1'] for v in group) <= 0 or not any(v['changed'] for v in group)):
            raise ValueError('No valid gradient/update for ' + prefix)
    return values


def audited_task(dataset, split):
    """旁观原采样器的随机索引，不改变抽样：验证先support后query及原始行互斥。"""
    import numpy as np
    original = np.random.choice
    choices = []
    def observe(*args, **kwargs):
        selected = original(*args, **kwargs)
        choices.append(np.asarray(selected).copy())
        return selected
    with mock.patch.object(np.random, 'choice', side_effect=observe):
        support, query = dataset.get_batch(split, augment=False)
    if len(choices) != 2 or dataset.batch_size != 1:
        raise ValueError('Diagnostic expects exactly one original sampled task')
    cell_index, indexes = int(choices[0][0]), choices[1]
    source = dataset.datasets[split][cell_index].values
    combined = np.concatenate([support, query], axis=1)[0]
    if not np.array_equal(source[indexes], combined) or len(set(indexes)) != len(indexes):
        raise ValueError('Support/query row provenance or disjointness failed')
    ns = dataset.samples_support
    def pairs(rows):
        return {tuple(sorted((int(r[1]), int(r[2])))) for r in rows}
    evidence = dict(split=split, cell_id=int(combined[0, 0]), source_cell_index=cell_index,
        support_row_indices=indexes[:ns].tolist(), query_row_indices=indexes[ns:].tolist(),
        support_query_row_overlap=[], exact_original_row_order=True,
        drug_pair_overlap=[list(p) for p in sorted(pairs(combined[:ns]) & pairs(combined[ns:]))],
        support_rows=np.asarray(support)[0].tolist(), query_rows=np.asarray(query)[0].tolist())
    return support, query, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--checkpoint-dir', help='New directory for retained interface-only checkpoints')
    parser.add_argument('--legacy-extractor', help='Read-only compatibility check; never initializes BioGraph')
    args = parser.parse_args()
    report_path = Path(args.report).resolve()
    if report_path.exists():
        raise FileExistsError('Refusing to overwrite smoke report: ' + str(report_path))
    os.chdir(str(ROOT))
    import numpy as np
    import pandas as pd
    import torch
    from encoders.cell_factory import load_encoder_config, build_configured_cell_encoder
    from encoders.biological_graph_data import load_biological_graph, sha256
    config = load_encoder_config(args.config)
    kind = config['cell_encoder']['type']
    report = dict(encoder=kind, purpose='diagnostic smoke only; not experiment metrics', stages={})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'biological_graph':
        try:
            _, report['graph_audit'] = load_biological_graph(config['biological_graph'])
        except (FileNotFoundError, ValueError) as exc:
            report.update(status='blocked', reason=str(exc))
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            print(report['reason'])
            return 2
    if not torch.cuda.is_available():
        raise RuntimeError('Original HyperSynergy run_batch needs CUDA')
    torch.set_num_threads(2)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)
    from model_rep import EMBNet
    from model_few import HyperSynergy
    import representation_learning as stage1
    import train_few_shot as stage2
    import data_syn
    from utils_syn import TestbedDataset, TestbedDataset1, DataLoader, mse, spearman, r2_score
    from innovation_checkpoint import extractor_payload, load_extractor, add_meta_metadata, load_meta_model, strict_load
    feature_dir = ROOT/'data/sample_features'
    def read(name):
        with (feature_dir/name).open('rb') as handle:
            return pickle.load(handle)
    codes, drugs, cells = read('codes_cell.p'), read('drug_feature_cell.p'), read('cell_feature_900.p')
    def batch(rows):
        a = next(iter(DataLoader(TestbedDataset(rows, drugs, cells, codes), batch_size=len(rows), shuffle=False)))
        b = next(iter(DataLoader(TestbedDataset1(rows, drugs, cells, codes), batch_size=len(rows), shuffle=False)))
        expected = torch.stack([cells.loc[codes['cells'].item2idx[int(r[0])], 'cg'] for r in rows])
        if not torch.equal(a.c, expected) or not torch.equal(a.y, torch.tensor(rows[:, 3], dtype=torch.float32)):
            raise ValueError('Batch expression/label ordering mismatch')
        return a.cuda(), b.cuda()
    train_cells, test_cells = set(), set()
    for name, into in [('train', train_cells), ('test', test_cells)]:
        for path in glob.glob(str(ROOT/'data/data_few_zero_rich'/name/'*.txt')):
            into.update(pd.read_csv(path, sep='\t').CELL_ID.astype(int).tolist())
    if train_cells & test_cells:
        raise ValueError('Train/test cell overlap: ' + str(train_cells & test_cells))
    report['split_check'] = dict(train_cells=len(train_cells), test_cells=len(test_cells), overlap=[],
        train_ids=sorted(train_cells), test_ids=sorted(test_cells), expression_rows=len(cells),
        validation_files=len(glob.glob(str(ROOT/'data/data_few_zero_rich/val/*.txt'))))
    representation = EMBNet(cell_encoder_config=config).cuda()
    if args.legacy_extractor:
        from encoders import SampleFeatureExtractor
        legacy = torch.load(args.legacy_extractor, map_location='cpu')
        baseline = SampleFeatureExtractor(cell_encoder_config={'cell_encoder': {'type': 'se_cnn'}})
        baseline_audit = load_extractor(baseline, legacy)
        drug = copy.deepcopy(representation.extractor.drug_encoder).cpu()
        drug_state = {k[len('drug_encoder.'):]: v for k, v in legacy.items() if k.startswith('drug_encoder.')}
        drug_audit = strict_load(drug, drug_state)
        assert all(torch.equal(v, drug.state_dict()[k]) for k, v in baseline.drug_encoder.state_dict().items())
        rejected = None
        if kind == 'biological_graph':
            try:
                load_extractor(representation.extractor, legacy)
            except ValueError as exc:
                rejected = str(exc)
            else:
                raise AssertionError('Legacy SE-CNN checkpoint must not load into BioGraph')
        report['legacy_checkpoint'] = dict(path=str(Path(args.legacy_extractor).resolve()),
            sha256=sha256(args.legacy_extractor), baseline=baseline_audit, drug=drug_audit,
            drug_output_dim=drug.fc.out_features, biograph_rejection=rejected,
            reused_for_training=False, note='Shared drug subtree is reusable; smoke trains fresh initialization')
        del baseline, drug, legacy
    # 真实表达GPU检查：所有900位置保留；批内图无跨细胞边，266孤立点各有自环。
    encoder = representation.extractor.cell_encoder.eval()
    x = torch.stack(cells.cg.tolist()[:8]).cuda()
    with torch.no_grad():
        single, multiple = encoder(x[:1]), encoder(x[:3])
        separate = torch.cat([encoder(row[None]) for row in x[:3]])
    assert tuple(single.shape) == (1, 256) and tuple(multiple.shape) == (3, 256)
    assert torch.isfinite(multiple).all() and torch.allclose(multiple, separate, atol=1e-5, rtol=1e-5)
    report['real_forward'] = dict(single_shape=list(single.shape), multi_shape=list(multiple.shape),
        per_cell_pooling_equal=True, device=str(single.device), finite=True)
    if kind == 'biological_graph':
        assert encoder.gene_positions.tolist() == list(range(x.size(1)))
        assert encoder.padding_positions.numel() == 0
        edges, assignments = encoder.batched_topology(3)
        assert torch.equal(assignments[edges[0]], assignments[edges[1]])
        loops = encoder.edge_index[0, encoder.edge_index[0] == encoder.edge_index[1]]
        assert sorted(loops.tolist()) == list(range(encoder.num_genes))
        assert all(b.is_cuda for _, b in encoder.named_buffers())
        assert encoder.graph_audit['isolated_nodes'] == 266
        report['real_forward'].update(edge_shape=list(encoder.edge_index.shape),
            all_graph_buffers_cuda=True, all_nodes_have_one_loop=True, cross_graph_edges=0,
            isolated_nodes=266, real_gene_positions=encoder.num_genes, padding=0)
    data = data_syn.tranditional_model_data(train_rate=.9, val_rate=.1, seed=1500)
    train_rows, val_rows, _ = data.get_train_batch(augment=False)
    rows = train_rows[:8]
    if not set(rows[:, 0].astype(int)).issubset(train_cells):
        raise ValueError('Stage 1 smoke contains held-out cells')
    optimizer = torch.optim.Adam(representation.parameters(), lr=.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=1.)
    stage1.loss_fn = torch.nn.MSELoss()
    before = {n: p.detach().clone() for n, p in representation.extractor.cell_encoder.named_parameters()}
    loader_a = DataLoader(TestbedDataset(rows, drugs, cells, codes), batch_size=len(rows), shuffle=False)
    loader_b = DataLoader(TestbedDataset1(rows, drugs, cells, codes), batch_size=len(rows), shuffle=False)
    loss = stage1.train(representation, 'cuda:0', loader_a, loader_b, optimizer, scheduler)
    if not np.isfinite(loss):
        raise ValueError('Stage 1 nonfinite loss')
    changed = [n for n, p in representation.extractor.cell_encoder.named_parameters() if not torch.equal(p, before[n])]
    if not changed:
        raise ValueError('Stage 1 cell encoder did not update')
    gradients = {}
    for prefix in (('gene_embedding.', 'layers.0.', 'layers.1.', 'pool.') if kind == 'biological_graph' else ('gconv1.', 'se3.')):
        grads = [p.grad for n, p in representation.extractor.cell_encoder.named_parameters() if n.startswith(prefix)]
        if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
            raise ValueError('Missing/nonfinite gradients for ' + prefix)
        gradients[prefix] = sum(g.abs().sum().item() for g in grads)
        if gradients[prefix] <= 0:
            raise ValueError('Zero gradients for ' + prefix)
    prefixes = ('gene_embedding.', 'layers.0.', 'layers.1.', 'pool.') if kind == 'biological_graph' else ('gconv1.', 'se3.')
    parameters = parameter_evidence(representation.extractor.cell_encoder, before, prefixes)
    if kind == 'biological_graph':
        for key in ('layers.0.att', 'layers.1.att'):
            assert parameters[key]['grad_l1'] > 0 and parameters[key]['changed']
    report['stages']['stage1'] = dict(status='passed', samples=8, batch_count=1, mse=float(loss),
        cell_ids=sorted(set(int(r[0]) for r in rows)), source='representation_learning.train unchanged',
        changed_cell_parameters=len(changed), gradient_l1=gradients, parameter_evidence=parameters,
        all_parameters_finite=all(bool(torch.isfinite(p).all()) for p in representation.parameters()))
    assert report['stages']['stage1']['all_parameters_finite']
    report['parameters'] = dict(cell=sum(p.numel() for p in representation.extractor.cell_encoder.parameters()),
        drug=sum(p.numel() for p in representation.extractor.drug_encoder.parameters()),
        extractor=sum(p.numel() for p in representation.extractor.parameters()),
        representation=sum(p.numel() for p in representation.parameters()))
    # 真实输入测量编码器推理时间/显存；只比较相同 batch/device/mode 的记录。
    encoder = representation.extractor.cell_encoder.eval()
    x = torch.stack(cells.cg.tolist()[:8]).cuda()
    with torch.no_grad():
        for _ in range(3):
            encoder(x)
        torch.cuda.synchronize(); base_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
        for _ in range(20):
            output = encoder(x)
        torch.cuda.synchronize()
    report['compute'] = dict(device=torch.cuda.get_device_name(0), batch_size=8, warmups=3, repeats=20,
        forward_ms=(time.perf_counter()-start)*1000/20,
        incremental_peak_bytes=torch.cuda.max_memory_allocated()-base_memory,
        output_shape=list(output.shape))
    with checkpoint_scope(args.checkpoint_dir) as tmp:
        path = Path(tmp)/'extractor.interface_test_only.pt'
        payload = extractor_payload(representation.extractor)
        payload['usage'] = 'interface_test_only; one batch of eight; NOT formal pretrained weights'
        torch.save(payload, str(path))
        meta = HyperSynergy(50, 40, cell_encoder_config=config).cuda()
        stage1_load = load_extractor(meta.extractor, torch.load(str(path), map_location='cuda:0'))
        assert all(torch.equal(v, meta.extractor.state_dict()[k]) for k, v in representation.extractor.state_dict().items())
        meta.extractor.eval()
        with torch.no_grad():
            restored_output = meta.extractor.cell_encoder(x)
            original_output = representation.extractor.cell_encoder(x)
            forward_error = float((restored_output - original_output).abs().max().item())
            # GPU scatter浮点归约可能不逐位确定；参数仍逐位检查，输出使用明确容差。
            assert torch.allclose(restored_output, original_output, atol=1e-5, rtol=1e-5), forward_error
        report['checkpoint'] = dict(stage1=dict(path=str(path), sha256=sha256(path), load=stage1_load,
            exact_state_equal=True, cell_forward_allclose=True, forward_max_abs_error=forward_error,
            forward_atol=1e-5, forward_rtol=1e-5, retained=bool(args.checkpoint_dir)))
        del representation, payload
        for p in meta.extractor.parameters():
            p.requires_grad = False
        frozen_before = {n: p.detach().clone() for n, p in meta.extractor.named_parameters()}
        buffers_before = {n: p.detach().clone() for n, p in meta.extractor.named_buffers()}
        dataset = data_syn.MiniCellDataSet(batch_size=1, samples_support=50, samples_query=40)
        params = [p for n, p in meta.named_parameters() if n != 'inner_l_rate' and p.requires_grad]
        outer = torch.optim.Adam(params, lr=.001, weight_decay=1e-5)
        inner = torch.optim.Adam([meta.inner_l_rate], lr=.1)
        schedule = torch.optim.lr_scheduler.MultiStepLR(outer, milestones=[500,1000,2500,5000,10000,30000], gamma=.5)
        # 直接调用原Stage2 train(1)，仅任务batch=1；不复制或替换任何参数更新公式。
        stage2.drug_features, stage2.cell_features, stage2.codes = drugs, cells, codes
        stage2.batch_size = 1
        train_before = {n: p.detach().clone() for n, p in meta.named_parameters()}
        observed = {}
        original_run = meta.run_batch
        def sample_train(augment=False):
            support, query, evidence = audited_task(dataset, 'train')
            assert evidence['cell_id'] in train_cells
            observed['sampling'] = evidence
            return support, query
        def observe_run(a, b, *arguments):
            expected = np.array(observed['sampling']['support_rows'] + observed['sampling']['query_rows'])
            assert torch.equal(a.y.cpu(), torch.tensor(expected[:, 3], dtype=torch.float32))
            expected_c = torch.stack([cells.loc[codes['cells'].item2idx[int(r[0])], 'cg'] for r in expected])
            assert torch.equal(a.c.cpu(), expected_c)
            result = original_run(a, b, *arguments)
            assert torch.equal(result[5].detach().cpu().flatten(), torch.tensor(expected[50:, 3], dtype=torch.float32))
            observed['outputs'] = tuple(t.detach() for t in result)
            return result
        with mock.patch.object(dataset, 'get_train_batch', side_effect=sample_train), mock.patch.object(meta, 'run_batch', side_effect=observe_run):
            stage2.train(1, dataset, meta, outer, inner, schedule, 'cuda:0')
        outputs = observed['outputs']
        if not all(torch.isfinite(t).all() for t in outputs):
            raise ValueError('Stage 2 nonfinite output')
        if any(not torch.equal(p, frozen_before[n]) or p.grad is not None for n, p in meta.extractor.named_parameters()):
            raise ValueError('Frozen extractor parameters changed or received gradients')
        update_evidence = parameter_evidence(meta, train_before, ('inference.', 'generative.', 'inner_l_rate'))
        report['stages']['stage2'] = dict(status='passed', support=50, query=40, task_count=1,
            source='train_few_shot.train unchanged, one complete update', extractor_frozen=True,
            frozen_parameter_tensors=len(frozen_before), trainable_parameter_tensors=sum(p.requires_grad for p in meta.parameters()),
            changed_trainable_parameters=[n for n, v in update_evidence.items() if v['changed']],
            parameter_evidence={n: v for n, v in update_evidence.items() if not n.startswith('extractor.')},
            changed_frozen_module_buffers=[n for n, v in meta.extractor.named_buffers() if not torch.equal(v, buffers_before[n])],
            sampling=observed['sampling'], losses=[float(t.item()) for t in outputs[:4]],
            inner_steps=15, outer_optimizer='Adam lr=0.001 weight_decay=1e-5', inner_optimizer='Adam lr=0.1')
        metapath = Path(tmp)/'meta.interface_test_only.pt'
        torch.save(add_meta_metadata(dict(step=1, state_dict=meta.state_dict(),
            usage='interface_test_only; one meta update; NOT final experiment weights'), meta.extractor), str(metapath))
        test = HyperSynergy(5, 40, cell_encoder_config=config).cuda()
        load_extractor(test.extractor, torch.load(str(path), map_location='cuda:0'))
        meta_load = load_meta_model(test, torch.load(str(metapath), map_location='cuda:0'))
        assert all(torch.equal(v, test.state_dict()[k]) for k, v in meta.state_dict().items())
        report['checkpoint']['stage2'] = dict(path=str(metapath), sha256=sha256(metapath), load=meta_load,
            exact_state_equal=True, retained=bool(args.checkpoint_dir))
        for p in test.extractor.parameters():
            p.requires_grad = False
        test.eval()
        dataset.samples_support = 5
        support, query, evidence = audited_task(dataset, 'test')
        rows = np.concatenate([support, query], axis=1).reshape(-1, 4)
        heldout = set(rows[:, 0].astype(int))
        if heldout & train_cells or not heldout.issubset(test_cells):
            raise ValueError('Stage 3 cell is not held out')
        a, b = batch(rows)
        test_before = {n: p.detach().clone() for n, p in test.state_dict().items()}
        outputs = test.run_batch(a, b, 1, False)
        if not all(torch.isfinite(t).all() for t in outputs):
            raise ValueError('Stage 3 nonfinite output')
        if tuple(outputs[4].shape) != (1, 40):
            raise ValueError('Unexpected few-shot query shape')
        assert torch.equal(outputs[5].detach().cpu().flatten(), torch.tensor(rows[5:, 3], dtype=torch.float32))
        assert all(torch.equal(v, test_before[k]) for k, v in test.state_dict().items())
        assert all(p.grad is None for p in test.extractor.parameters())
        # 沿用原Stage3的double数组及同一指标函数，仅保留单个真实任务的诊断数值。
        predicted, labels = [t.detach().cpu().double().numpy().flatten() for t in outputs[4:6]]
        metrics = dict(MSE=float(mse(labels, predicted)), SCC=float(spearman(labels, predicted)),
                       R2=float(r2_score(labels, predicted)))
        assert all(np.isfinite(value) for value in metrics.values())
        report['stages']['stage3'] = dict(status='passed', support=5, query=40, task_count=1,
            heldout_cell_ids=sorted(int(x) for x in heldout), inner_steps=15, predictions_finite=True,
            sampling=evidence, metrics=metrics, predictions=predicted.tolist(), labels=labels.tolist(),
            query_labels_in_exact_order=True, model_state_unchanged_during_adaptation=True,
            extractor_frozen_no_gradients=True, note='one task with diagnostic weights; NOT paper metrics')
        report['checkpoint']['schema'] = dict(stage1_keys=sorted(torch.load(str(path), map_location='cpu')['state_dict']),
            stage2_keys=sorted(test.state_dict()))
    report['status'] = 'passed'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
