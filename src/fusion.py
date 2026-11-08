import torch
import torch.nn as nn


class GatedCrossAttention(nn.Module):
    def __init__(self, fusion_dim: int):
        super().__init__()
        self.query = nn.Linear(fusion_dim, fusion_dim)
        self.key = nn.Linear(fusion_dim, fusion_dim)
        self.value = nn.Linear(fusion_dim, fusion_dim)
        self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, image_tokens, expression_token):
        q = self.query(expression_token).unsqueeze(1)
        k = self.key(image_tokens)
        v = self.value(image_tokens)
        attn = torch.softmax(q @ k.transpose(1, 2) / (k.size(-1) ** 0.5), dim=-1)
        attended = (attn @ v).squeeze(1)
        gate = torch.sigmoid(self.gate(torch.cat([expression_token, attended], dim=-1)))
        fused = gate * attended + (1 - gate) * expression_token
        return self.norm(fused)
