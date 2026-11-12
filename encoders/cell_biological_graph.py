# [Innovation-1: Biological Graph Cell Encoder]
"""共享真实基因图的细胞系编码器；不包含任务先验或预测损失。"""
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GlobalAttention


class BiologicalGraphCellEncoder(nn.Module):
    def __init__(self, edge_index, graph_audit, out_dim=256, num_layers=2,
                 hidden_dim=128, gene_embedding_dim=16, dropout=0.1,
                 pooling='attention', residual=True, normalize_expression=True,
                 batch_chunk_size=32):
        super().__init__()
        if pooling != 'attention':
            raise ValueError('Innovation-1 supports attention pooling only')
        if min(num_layers, hidden_dim, gene_embedding_dim, out_dim, batch_chunk_size) < 1:
            raise ValueError('Dimensions, layers and batch_chunk_size must be positive')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must lie in [0, 1)')
        self.input_dim = graph_audit['input_dim']
        self.num_genes = graph_audit['num_genes']
        self.output_dim = out_dim
        self.graph_audit = graph_audit
        self.residual = residual
        self.normalize_expression = normalize_expression
        self.batch_chunk_size = batch_chunk_size
        if edge_index.dtype != torch.long or edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError('edge_index must be a long tensor [2, E]')
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= self.num_genes):
            raise ValueError('PPI node index outside true-gene range')
        self.register_buffer('edge_index', edge_index.clone())
        self.register_buffer('gene_positions', torch.tensor(graph_audit['gene_positions'], dtype=torch.long))
        self.register_buffer('padding_positions', torch.tensor(graph_audit['padding_indices'], dtype=torch.long))
        self.register_buffer('graph_fingerprint', torch.tensor(list(bytes.fromhex(graph_audit['fingerprint'])), dtype=torch.uint8))
        self.gene_embedding = nn.Embedding(self.num_genes, gene_embedding_dim)
        self.input_projection = nn.Linear(gene_embedding_dim + 1, hidden_dim)
        # 自环已经由数据层显式处理；禁止 PyG 在每层隐式改变图拓扑。
        self.layers = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim, heads=1,
            dropout=dropout, add_self_loops=False) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.pool = GlobalAttention(nn.Linear(hidden_dim, 1))
        self.output_projection = nn.Linear(hidden_dim, out_dim)

    def batched_topology(self, batch_size):
        """向量化偏移 k*G：边 [2,B*E]，batch [B*G]，绝无细胞间跨图边。"""
        offsets = torch.arange(batch_size, device=self.edge_index.device) * self.num_genes
        edges = (self.edge_index.unsqueeze(0) + offsets[:, None, None]).permute(1, 0, 2).reshape(2, -1)
        batch = torch.arange(batch_size, device=self.edge_index.device).repeat_interleave(self.num_genes)
        return edges, batch

    def _forward_chunk(self, expression):
        """输入 [B,G]；逐细胞标准化不拟合跨细胞统计，不读取任何标签。"""
        batch_size = expression.size(0)
        values = expression.index_select(1, self.gene_positions)
        if self.normalize_expression:
            values = values - values.mean(dim=1, keepdim=True)
            values = values / values.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        identities = self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        features = torch.cat((values.unsqueeze(-1), identities), dim=-1)
        hidden = self.input_projection(features).reshape(batch_size * self.num_genes, -1)
        edges, batch = self.batched_topology(batch_size)
        for layer, norm in zip(self.layers, self.norms):
            updated = layer(hidden, edges)
            hidden = norm(hidden + updated if self.residual else updated)
            hidden = self.dropout(self.activation(hidden))
        pooled = self.pool(hidden, batch, size=batch_size)
        return self.output_projection(pooled)

    def forward(self, gene_expression):
        """输入 [B,原表达维度]，输出 [B,256]；padding 不进入 embedding/图/pooling。

        Stage 1 全参数随原优化器更新；Stage 2/3 遵循原 extractor 冻结策略。
        分块仅降低共享图批处理内存，不构造 Python Data 对象，不缓存梯度。
        """
        if gene_expression.dim() != 2 or gene_expression.size(1) != self.input_dim:
            raise ValueError('BioGraph expects [B,{}], got {}'.format(self.input_dim, tuple(gene_expression.shape)))
        if not gene_expression.is_floating_point() or not torch.isfinite(gene_expression).all():
            raise ValueError('Expression input must contain finite floating-point values')
        if self.padding_positions.numel() and torch.count_nonzero(
                gene_expression.index_select(1, self.padding_positions)).item():
            raise ValueError('Nonzero values at verified padding positions')
        if gene_expression.size(0) == 0:
            return gene_expression.new_empty((0, self.output_dim))
        return torch.cat([self._forward_chunk(chunk)
                          for chunk in gene_expression.split(self.batch_chunk_size, dim=0)], dim=0)
