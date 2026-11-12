# BioGraph 输入

ppi_edges.csv 与 ppi_source.json 原样来自 V3 已验证的 staged 目录。
基因索引位于 experiments/innovation_1_biograph/configs/gene_index.csv。

来源清单记录 STRING human functional association network 12.0、score threshold 700，
以及边表和原始输入的 SHA256。该清单仅记录了源服务器上的 raw 文件 URI；
原始下载 URL 与下载时间未记录。保留这一事实，没有补写未知来源信息。
原始大体积下载文件没有包含在本源码仓库，运行使用的是已处理的边表。
源 URI 是历史来源记录，不是当前运行时需要打开的路径。
