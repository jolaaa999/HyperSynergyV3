# V3 BioGraph 实验入口

完整运行步骤见项目根目录 README.md。
默认 BioGraph 配置 configs/biological_graph.json 使用已随仓库提供的真实 PPI 数据。
configs/se_cnn.json 用于独立 SE-CNN 对照组。
run_stage.py 提供 Stage 1/2/3 和 --check-only。

originals/ 和 results/before_source_sha256.json 为源码一致性测试夹具。
audit_implementation.py / rollback.py 是早期开发审计工具，部分操作依赖未发布的历史记录；
克隆仓库验证应优先使用 tests/test_biograph.py，不要直接对新工作树执行 rollback.py --apply。
历史完整实施说明见 ../../docs/BIOGRAPH_IMPLEMENTATION_HISTORY.md。
