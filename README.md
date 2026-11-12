# HyperSynergyV3

在原 HyperSynergy 基础上加入药物 TransformerConv 编码器，以及可配置的 SE-CNN / BioGraph 细胞编码器。
本仓库保留原项目 MIT LICENSE 和上游 Git 历史。原项目说明与论文引用见 docs/UPSTREAM_README.md。

## 内容

- 根目录：表征预训练、元训练、元测试与 checkpoint 加载代码。
- encoders/：药物编码器、SE-CNN、BioGraph、数据校验与配置工厂。
- experiments/innovation_1_biograph/：分阶段运行入口、配置、测试及基线审计夹具。
- data/：原项目已跟踪的数据与实际使用的 BioGraph 边表、来源清单。
- requirements.txt：当前 V3 环境导出的 51 项 Python 依赖。

训练权重、训练日志、运行快照、下载压缩包和机器本地 IDE 配置不包含在发布版本中。
原始模型参数和训练计划见源码；本仓库不宣称新的性能指标或 GPU 迁移验证结果。

## 环境

源环境为 Linux x86_64、Python 3.6.15、PyTorch 1.7.0+cu110、torch-geometric 1.7.2。
RDKit 2019.09.3、cudatoolkit 11.0.3、Boost 配套构建通过 Conda 安装。
完整精确清单、CUDA 动态库激活脚本和安装前置条件见 docs/ENVIRONMENT_MIGRATION.txt。
该文件是源服务器迁移记录，其中绝对路径和历史运行目录不一定存在于克隆副本中。

准备好 Conda 前置依赖后：

    python -m pip install -r requirements.txt
    python -m pip check

使用原有 CUDA 依赖组合时，请保留 run_stage.py 中禁用 GPU JIT fusion 的兼容设置。
完整元训练/测试需要 CUDA；CPU 测试通过不能替代目标 GPU 上的验证。

## BioGraph 数据预检

仓库默认配置使用项目相对路径，边表位于 data/biological_graph/。
PPI 数据从源项目已经校验的 staged 文件原样复制，未修改边或基因索引。
来源记录及其局限见 data/biological_graph/README.md。

    export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    EXP=experiments/innovation_1_biograph
    CFG="$EXP/configs/biological_graph.json"
    python -B "$EXP/run_stage.py" --stage 1 --config "$CFG" --run-dir "$EXP/runs/preflight" --check-only

预检不会启动训练，也不会要求已有训练 checkpoint。

## 新实验

在项目根目录、已激活的环境中执行，选择可用 GPU，并使用一个新的 run 目录：

    export CUDA_VISIBLE_DEVICES=0
    EXP=experiments/innovation_1_biograph
    CFG="$EXP/configs/biological_graph.json"
    RUN="$EXP/runs/biograph_new"
    python -B -u "$EXP/run_stage.py" --stage 1 --config "$CFG" --run-dir "$RUN"
    python -B -u "$EXP/run_stage.py" --stage 2 --config "$CFG" --run-dir "$RUN"

Stage 3 必须指定这个 run 的 checkpoints/stage2/ 下实际生成的 checkpoint：

    python -B -u "$EXP/run_stage.py" --stage 3 --config "$CFG" --run-dir "$RUN" --checkpoint /实际路径/checkpoint.pth

SE-CNN 对照组改用 configs/se_cnn.json 和另外的新 run 目录。
根目录 run_v2_training.sh 为旧 V2 流程，使用当前 python，或通过 PYTHON_BIN 指定解释器；BioGraph 使用上面的 run_stage.py。

## 测试

    python -B -m unittest discover -s experiments/innovation_1_biograph/tests -p 'test_*.py' -v

GPU 不可用时 CUDA 测试会跳过；未提供旧预训练权重时，旧 checkpoint 加载测试会明确跳过。
originals/ 和 results/before_source_sha256.json 是小型基线源码校验夹具，并非训练结果。
docs/BIOGRAPH_IMPLEMENTATION_HISTORY.md 是早期实施记录，部分“缺失 PPI”状态已经过时。

## 发布到 GitHub

目标仓库：jolaaa999/HyperSynergyV3。新建时使用 private 可见性。
本地的 origin 指向目标账号；upstream 保留原 HyperSynergy 仓库地址。
在能访问 GitHub、已安装 GitHub CLI 的终端执行：

    bash publish_to_github.sh

脚本通过 GitHub 浏览器授权登录，检查当前账号，必要时创建仓库，再普通推送 main；不会强制推送。
脚本不保存账号密码。本地准备完成不代表远端已创建，须以推送成功输出为准。
