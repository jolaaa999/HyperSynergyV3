import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_add_pool
from torch_geometric.nn.norm import GraphNorm


class DrugTransformerConv(nn.Module):
    def __init__(self, in_dim=78, hidden=156, out_dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.conv1 = TransformerConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.norm1 = GraphNorm(hidden * heads)
        self.conv2 = TransformerConv(
            hidden * heads, hidden, heads=heads, dropout=dropout
        )
        self.norm2 = GraphNorm(hidden * heads)
        self.conv3 = TransformerConv(
            hidden * heads, hidden, heads=heads, dropout=dropout
        )
        self.norm3 = GraphNorm(hidden * heads)
        self.act = nn.GELU()
        self.fc = nn.Linear(hidden * heads, out_dim)

    def forward(self, x, edge_index, batch):
        h1 = self.act(self.norm1(self.conv1(x, edge_index)))
        h2 = self.act(self.norm2(self.conv2(h1, edge_index)))
        h3 = self.act(self.norm3(self.conv3(h2, edge_index)))
        h = h1 + h3
        g = global_add_pool(h, batch)
        return self.fc(g)
