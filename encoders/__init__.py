import torch
import torch.nn as nn
from .cell_secnn import CellSEConv
from .drug_transformerconv import DrugTransformerConv
from .sample_feature import SampleFeatureExtractor
from .cell_factory import build_configured_cell_encoder, load_encoder_config

def build_cell_encoder(name: str, out_dim: int = 256, config=None):
    name = (name or "").lower()
    if name in ("secnn", "se_cnn"):
        return CellSEConv(out_dim=out_dim)
    # [Innovation-1: Biological Graph Cell Encoder]
    if name == 'biological_graph':
        if config is None:
            raise ValueError('BioGraph requires an explicit verified graph config')
        config = load_encoder_config(config)
        if config.get('cell_encoder', {}).get('type') != name:
            raise ValueError('Requested cell encoder and config type disagree')
        return build_configured_cell_encoder(out_dim=out_dim, config=config)
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
