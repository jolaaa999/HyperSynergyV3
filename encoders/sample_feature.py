import torch
import torch.nn as nn

from .cell_secnn import CellSEConv
from .drug_transformerconv import DrugTransformerConv
from .cell_factory import build_configured_cell_encoder


class SampleFeatureExtractor(nn.Module):
    def __init__(self, drug_dim=128, cell_dim=256, heads=4, cell_encoder_config=None):
        super().__init__()
        self.drug_encoder = DrugTransformerConv(out_dim=drug_dim, heads=heads)
        # [Innovation-1: Biological Graph Cell Encoder]
        self.cell_encoder = build_configured_cell_encoder(out_dim=cell_dim, config=cell_encoder_config)
        self.normf = nn.BatchNorm1d(2 * drug_dim + cell_dim)

    def forward_flat(self, drug_a, drug_b):
        x1 = self.drug_encoder(drug_a.x, drug_a.edge_index, drug_a.batch)
        x2 = self.drug_encoder(drug_b.x, drug_b.edge_index, drug_b.batch)
        cell = self.cell_encoder(drug_a.c)
        return self.normf(torch.cat((x1, x2, cell), dim=1))

    def forward(self, drug_a, drug_b, task_batch=None):
        embeddings = self.forward_flat(drug_a, drug_b)
        if task_batch is None:
            return embeddings
        return embeddings.view(task_batch, -1, embeddings.size(-1))
