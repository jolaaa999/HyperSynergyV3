# Innovation-1：BioGraph Cell Encoder 实施报告

工作目录：/data/jyk/HyperSynergyV3。

**代码与接口已实现；真实 BioGraph 验证被缺失 PPI 阻塞。没有启动正式训练，没有产生 BioGraph 正式指标。**

## 1. 修改前的项目架构确认

- encoders/drug_transformerconv.py 的 DrugTransformerConv：3层 TransformerConv，原子输入78、hidden=156、heads=4，GraphNorm/GELU、首末层残差、global_add_pool，最终每药物128维。
- encoders/cell_secnn.py 的 CellSEConv：输入[B,900]转换[B,1,30,30]，四层卷积，后两层SE，输出[B,256]；原代码还接受[B,1,900]和[B,30,30]。
- encoders/sample_feature.py：两个128维药物表示与256维细胞表示拼接，原BatchNorm1d，样本输出512维。
- feature_ generation.py 从表达CSV的CELL_ID及第3列开始按原顺序提取；feature_direction.py转换float32。实际训练读取data/sample_features/cell_feature_900.p，没有新表达数据集或额外表达特征。完整原始data/gene_expression_cell.csv/json当前不存在。
- utils_syn.py的TestbedDataset通过codes_cell.p定位cg，放入Data.c。批处理输入为[B,900]。全部125个CELL_ID与codes行映射一致。
- Stage1：representation_learning.py构造EMBNet，所有编码器参与Adam/MSE训练；500 epochs，train/val batch=500/1024，lr=0.0001，原StepLR，train细胞内90%/10%切分。
- Stage2：train_few_shot.py构造HyperSynergy，加载整个extractor后冻结参数。40000 steps，task batch=50，support/query=50/40，outer Adam lr=0.001，inner_l_rate Adam lr=0.1，原MultiStepLR和更新公式。
- Stage3：test_few_shot.py加载Stage1 extractor及Stage2整模型，冻结extractor并eval；support/query=5/40，task batch=50，200 batches × 20 rounds，潜变量适应15步，原损失、MSE/SCC/R²及结果聚合。
- 旧Stage1导出为68键裸extractor state_dict：drug_encoder.*、cell_encoder.*、normf.*。训练恢复文件含model/optimizer/scheduler/config/completed_epoch及RNG。旧Stage2为{step,state_dict}、83键，包含extractor.*、inference.*、generative.*、inner_l_rate。
- 原配置主要为脚本常量和run_v2_training.sh。旧输出位于pretrain_representation_model/、saved_model/、runs/及根目录统计CSV。部分旧运行记录仍指向V2绝对路径，本次未运行这些旧启动器。
- Git HEAD=f7e23e2。任务前已有5个tracked Python文件修改，以及encoders、checkpoint helper、权重、runs等untracked内容。原状态/diff在results/git_status_before.txt和git_diff_before.patch，源码备份在originals/。

## 2. 文件修改清单

| 文件 | 本次内容 |
|---|---|
| encoders/biological_graph_data.py | 新增真实表达/padding/codes检查、PPI过滤、图来源/哈希/统计 |
| encoders/cell_biological_graph.py | 新增独立BiologicalGraphCellEncoder |
| encoders/cell_factory.py | 新增JSON配置与SE-CNN默认工厂 |
| innovation_checkpoint.py | 新增严格配置/图版本/参数键/shape/finite/buffer校验及隔离保护 |
| encoders/sample_feature.py、encoders/__init__.py | 最小编码器构造接口 |
| model_rep.py、model_few.py | 仅透传可选cell_encoder_config |
| representation_learning.py | resume config和最终导出增加可选metadata |
| pretrain_checkpoint.py | 新配置恢复时验证图buffer；默认旧恢复路径不变 |
| train_few_shot.py | Stage1加载入口、保存metadata及隔离保护 |
| test_few_shot.py | 严格加载、checkpoint路径选择及隔离保护；评估主体不变 |
| 本实验configs/ | se_cnn/BioGraph配置和真实校验的900行gene_index.csv |
| prepare_data.py、run_stage.py、smoke_test.py | 数据预检、隔离三阶段入口、真实短测 |
| tests/test_biograph.py | 单元/数据对齐/checkpoint/基线保护测试 |
| audit_implementation.py、rollback.py | 本次增量审计及保留用户原修改的回滚 |

新增核心代码含统一标记 # [Innovation-1: Biological Graph Cell Encoder]，关键形状、映射、批处理和参数更新函数有中文说明。

## 3. 完整数据流与网络结构

~~~text
原cg [B,input_dim]（维度从实际pickle读取）
 → 按真实gene_positions取值，排除显式padding
 → 每细胞跨真实基因z-score（可关闭；不拟合跨细胞统计）
 → concat(expression[...,1], shared GeneEmbedding[G,16])
 → Linear(17,128)
 → GATv2Conv(128,128,heads=1) → Residual → LayerNorm → GELU → Dropout
 → GATv2Conv(128,128,heads=1) → Residual → LayerNorm → GELU → Dropout
 → 每个细胞独立GlobalAttention pooling，Linear(128,1) gate
 → Linear(128,256)
 → [B,256] → 原512维样本表示接口
~~~

默认2层、hidden128、gene embedding16、dropout0.1、attention pooling、残差开启。层数/维度/dropout/残差可配置。

共享edge_index作为buffer随模型进入CPU/GPU并保存；第k个图的边偏移k*G，无跨细胞边。没有逐细胞构造Python Data对象。默认每32个细胞处理一块；Stage1反向仍保留全训练批次所需信息，不能据此保证正式batch500不会OOM。

没有加入Diffusion Prior、Graph Information Bottleneck、Self-Explanation、Dynamic Edge Gate或OT Flow Matching。

## 4. 基因索引、PPI来源和统计

实际数据是125个细胞×900维float32；900个有名基因ID、0个padding、0个全零列、0个全零细胞。不能套用其他项目的“880+20”。

本仓库gene_expression_example.csv包含CELL_ID=45、96、207的3行。每行全部900列按float32转换后与实际pickle完全相等，最大绝对误差0。索引严格保留该参考列序，首个基因ID为780，不从另一份CCLE文件猜测。

列名是NCBI Entrez ID，**不是gene symbol**。未猜造symbol或拿其他表格的行号当基因。完整带名原始参考只覆盖3/125个细胞，因此不能声称已完成125行原始来源逐值溯源。其余行均检查维度、finite、全零和codes映射。

| 文件 | SHA256 |
|---|---|
| cell_feature_900.p | bc25027c46199ba36fb6a1c84f981a53480c60d194751e426e6d33a8fd3d5d4a |
| 已验证gene_index.csv | cf9a7e3e169b065ac2b429ddae486dc251091e8cbd8a1480845d8f38d5709499 |
| gene_expression_example.csv | cf12c8ff58110d7513f68fbafa2193d19b87ca5e4d5b49d791907bd6f3897a6a |

**没有发现真实PPI或可复用构图实现。** 配置等待data/biological_graph/ppi_edges.csv及ppi_source.json，也可改为实际路径。节点数900；原始边数、有效边数、孤立点数和边文件哈希均尚不可得，未填0冒充完成。

真实PPI输入要求：

- CSV至少包含source,target；端点必须是ncbi_entrez ID。STRING protein ID或symbol需通过真实可追溯映射转换。
- ppi_source.json必须提供真实source_name、source_url、version、gene_id_namespace、edge_sha256。可另记录阈值、上游文件哈希及映射出处。填写manifest不能替代真实生物来源。
- 自动过滤集合外基因边；无向边按无序pair去重、排序再双向化。先移除并统计输入self-loop，再按配置为每个真实节点加一个self-loop；GAT内部不再隐式加环。
- 孤立节点数在加自环前计算。零有效生物非自环边会明确拒绝。padding不会成为节点。
- 自动记录raw、retained、external、unique、arcs、loops、isolates、来源、文件哈希、完整gene_positions及图fingerprint。

单元测试的极小手工拓扑明确标记UNIT_TEST_FIXTURE_NOT_BIOLOGICAL，仅用于算子/接口验证，没有作为真实训练PPI。

## 5. 输入输出、参数量和开销

| 项目 | SE-CNN | BioGraph默认配置 |
|---|---|---|
| 本数据输入 | [B,900] | [B,900] |
| 真实基因节点 | 不适用 | 900 |
| Cell输出 | [B,256] | [B,256] |
| 两个Drug输出 | 各[B,128] | 各[B,128] |
| Sample输出 | [B,512] | [B,512] |
| Cell参数 | 613,324 | 116,929 |
| Extractor参数 | 4,017,148 | 3,520,753 |

BioGraph参数数由实际结构公式精确计算且公式通过算子测试，不依赖边数：G*E + H*(E+2) + L*(2*H²+6*H) + H+1 + H*D+D；G900/E16/H128/L2/D256。

**未做容量匹配，不能声称控制了模型容量。** SE-CNN实测时间和显存见results/smoke_se_cnn.json，batch8、eval、3次预热、20次测量；共享GPU短测不是正式速度benchmark。BioGraph真实图耗时/显存尚未测量；主要复杂度O(B*L*(G*H²+|E_graph|*H))。补齐PPI后用相同smoke入口记录。

## 6. 三阶段接入验证

- 默认SE-CNN严格加载原68键extractor；同测试RNG下原/新构造state_dict逐参数一致，真实细胞输出一致，显式se_cnn也一致。
- 新Stage1导出{format,encoder_metadata,state_dict}；resume config包含编码器配置和图审计。无新配置时仍导出旧裸state_dict。
- Stage2必须加载对应Stage1；整个drug/cell/normf均严格匹配；冻结策略不变。整模型保留{step,state_dict}并增加metadata。
- Stage3严格校验配置/图版本后执行原适应。禁止把SE-CNN权重当BioGraph权重。
- 显式报告missing/unexpected/shape mismatch；检查finite及图buffer值，禁止形状相同但顺序不同的图被静默覆盖；没有strict=False。
- BioGraph的Stage1导出、Stage2加载、Stage3整模型恢复和训练恢复格式已用小型算子夹具验证；这不等于真实BioGraph实验完成。
- 真实SE-CNN完成Stage1八个样本的一步更新、Stage2一个50+40任务的一步原公式更新、Stage3一个5+40真实测试任务和原15步适应，输出finite，冻结参数不变。
- 真实BioGraph三阶段短测在训练前因缺失PPI明确blocked。

## 7. 保持不变及既有基线局限

DrugTransformerConv、CellSEConv、data_syn.py、utils_syn.py等通过原SHA256检查不变。Inference_model、Weight_generstive_netwrok以及HyperSynergy除构造接口外方法均AST相同。Stage1 train/predicting、Stage2 train除保存metadata、Stage3从model.eval起的主体均AST相同。训练常量、采样器、support/query、优化器、冻结策略、损失、指标均保留。

实际发现而未擅改的问题：

1. train85个细胞和test21个细胞无交集；Stage1和Stage2外层优化取train。
2. val目录无任务文件；Stage2验证使用get_test_batch，并用其loss/R²选checkpoint。**存在既有测试集参与选模风险，不能宣称严格无测试信息泄漏。** 更改该协议违反本次禁止修改划分/采样/元训练的要求。
3. Stage1仅在数据类设置NumPy seed1500；未统一固定Python random/torch，Stage2/3也未显式全局seed。新入口可保存/复用每阶段初始RNG用于配对新实验，没有擅加seed或声称旧结果可完全复现。GPU scatter非确定性未被强行改变。
4. 冻结extractor参数不等于eval；原Stage2的BatchNorm running stats/dropout策略保留。

## 8. 测试与结果

环境：/data/jyk/miniconda3/envs/hyper/bin/python；Python3.6、torch1.7.0+cu110、PyG1.7.2（含GATv2Conv）、RTX4090。GPU兼容设置沿用V2 launcher中禁用JIT fusion的做法，不改共享环境。

- logs/unit_tests.log：完整测试结果，含batch1/多batch、图间隔离、pooling、孤立点、padding、有效梯度/参数更新、buffer CPU/GPU、严格恢复、错误文件/哈希/映射、真实表达和基线保护。
- results/data_audit.json：真实表达逐值校验及PPI缺失状态。
- results/smoke_se_cnn.json和logs/smoke_se_cnn.log：真实三阶段短测、参数量、时间/显存诊断；不作为正式性能指标。
- results/smoke_biograph.json：明确blocked，没有伪造通过。
- results/implementation_audit.json：语法、保护文件、参数量和源修改范围。
- results/innovation1_only.patch：相对本任务开始时的核心增量，不混入已有Git修改。

## 9. 可运行Linux命令

下面正式训练命令仅供用户执行，本次未自动运行。必须先提供真实PPI及来源manifest。预检失败必须停止。

~~~bash
cd /data/jyk/HyperSynergyV3
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
PY=/data/jyk/miniconda3/envs/hyper/bin/python
EXP=experiments/innovation_1_biograph

# 当前因真实PPI缺失返回2；不会造边。
"$PY" -B "$EXP/prepare_data.py" --prepare-index
"$PY" -B -m unittest discover -s "$EXP/tests" -p 'test_*.py' -v
"$PY" -B "$EXP/audit_implementation.py"

# 补齐PPI后的真实短测，使用新的报告路径。
"$PY" -B "$EXP/smoke_test.py" --config "$EXP/configs/biological_graph.json"   --report "$EXP/results/smoke_biograph_with_real_ppi.json"
~~~

配对实验命令（相同每阶段初始RNG、原训练计划/shot/保存规则；容量不匹配）：

~~~bash
set -e
cd /data/jyk/HyperSynergyV3
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
PY=/data/jyk/miniconda3/envs/hyper/bin/python
EXP=experiments/innovation_1_biograph
BASE="$EXP/runs/baseline_pair1"
BIO="$EXP/runs/biograph_pair1"

"$PY" -B "$EXP/prepare_data.py"

# Stage1：500 epochs，权重写入独立run。
"$PY" -B "$EXP/run_stage.py" --stage 1 --config "$EXP/configs/se_cnn.json" --run-dir "$BASE"
"$PY" -B "$EXP/run_stage.py" --stage 1 --config "$EXP/configs/biological_graph.json" --run-dir "$BIO"   --rng-state "$BASE/configs/stage1_initial_rng.pt"

# Stage2：40000 steps，原50support/40query、task batch50。
"$PY" -B "$EXP/run_stage.py" --stage 2 --config "$EXP/configs/se_cnn.json" --run-dir "$BASE"
"$PY" -B "$EXP/run_stage.py" --stage 2 --config "$EXP/configs/biological_graph.json" --run-dir "$BIO"   --rng-state "$BASE/configs/stage2_initial_rng.pt"

# 示例选择各组最后保存的checkpoint（按step），不是按测试结果排名。
BASE_CKPT=$(find "$BASE/checkpoints/stage2" -maxdepth 1 -name '*.pth' | sort -V | tail -n 1)
BIO_CKPT=$(find "$BIO/checkpoints/stage2" -maxdepth 1 -name '*.pth' | sort -V | tail -n 1)
"$PY" -B "$EXP/run_stage.py" --stage 3 --config "$EXP/configs/se_cnn.json" --run-dir "$BASE"   --checkpoint "$BASE_CKPT"
"$PY" -B "$EXP/run_stage.py" --stage 3 --config "$EXP/configs/biological_graph.json" --run-dir "$BIO"   --checkpoint "$BIO_CKPT" --rng-state "$BASE/configs/stage3_initial_rng.pt"
~~~

原保存规则仍受Stage2测试细胞验证影响，上述选择不能消除该既有问题。Stage3保留原5/40、task batch50和200×20方案。

恢复中断Stage1（同run/config/source，从原epoch checkpoint恢复optimizer/scheduler/RNG，新建resume日志）：

~~~bash
"$PY" -B "$EXP/run_stage.py" --stage 1 --config "$EXP/configs/biological_graph.json"   --run-dir "$BIO" --resume-stage1
~~~

每个runs/<name>下有configs、checkpoints/stage1、checkpoints/stage2、logs、results、source。data链接原数据；旧脚本期待的权重路径链接该run自己的checkpoints。统计CSV由原函数继续写run根目录。无新配置时原脚本行为保持。

## 10. Git diff摘要与回滚

本次修改8个既有源文件，新增4个核心模块，另有独立实验工具/配置/测试/报告。Git diff --stat包含用户原有修改；本次范围应查看results/innovation1_only.patch及implementation_audit.json。

~~~bash
cd /data/jyk/HyperSynergyV3
# 只检查和列出，不改文件：
/data/jyk/miniconda3/envs/hyper/bin/python -B experiments/innovation_1_biograph/rollback.py
# 确定回到本次任务开始前时才执行：
/data/jyk/miniconda3/envs/hyper/bin/python -B experiments/innovation_1_biograph/rollback.py --apply
~~~

回滚先核对修改后哈希；存在后续人工编辑时拒绝覆盖。仅恢复本次修改的旧文件，删除本次新增4个核心模块；保留全部实验记录、旧权重、日志及用户原有修改。不要git reset --hard或git restore .。

## 11. 未解决问题及指标影响

1. **真实PPI缺失**：真实BioGraph前向/反向、Stage1→2→3、真实图统计和计算开销未完成，当前不属于“真实BioGraph全验证通过”。补齐后执行预检/测试/真实smoke。
2. 原始命名表达仅3/125：这3行900列完全匹配；完整原生成文件仍需补齐，才能扩展到125行来源验证。
3. 既有测试选模泄漏和种子不足：明确记录并原样保留；论文若声称无测试泄漏/完全可复现需另行处理协议，而且两组同时处理。
4. 容量不匹配、真实图开销未知：不能把效果变化归为已控制容量后的纯生物先验收益；未预报任何指标提升。
5. 全零/相同表达风险：当前125行无全零；未来全零或标准化后相同的细胞在eval下共享identity/topology，会得到相同（浮点容差内）表示。没有删掉/替换测试细胞。
6. 图阈值、覆盖和孤立点的影响须用真实图报告。未伪造数据或指标，未把完整DiffPro论文结果当当前基线成绩。

本阶段到此停止，不自动开始正式训练。
