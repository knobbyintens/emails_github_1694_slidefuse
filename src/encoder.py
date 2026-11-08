import torch
import torch.nn as nn


class TileEncoder(nn.Module):
    def __init__(self, image_dim: int, out_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(image_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, tile_features):
        return self.projection(tile_features)
