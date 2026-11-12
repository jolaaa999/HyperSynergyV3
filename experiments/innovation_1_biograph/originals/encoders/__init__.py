import torch
import torch.nn as nn
from .cell_secnn import CellSEConv
from .drug_transformerconv import DrugTransformerConv
from .sample_feature import SampleFeatureExtractor

def build_cell_encoder(name: str, out_dim: int = 256):
    name = (name or "").lower()
    if name == "secnn":
        return CellSEConv(out_dim=out_dim)
    raise ValueError(f"Unknown cell encoder: {name}")

def build_drug_encoder(name: str, out_dim: int = 128, heads: int = 4):
    name = (name or "").lower()
    if name == "transformerconv":
        return DrugTransformerConv(out_dim=out_dim, heads=heads)
    raise ValueError(f"Unknown drug encoder: {name}")


__all__ = [
    "CellSEConv",
    "DrugTransformerConv",
    "SampleFeatureExtractor",
    "build_cell_encoder",
    "build_drug_encoder",
]
