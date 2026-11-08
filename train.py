import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data import SlideExpressionDataset
from src.encoder import TileEncoder
from src.fusion import GatedCrossAttention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dataset = SlideExpressionDataset("data/slides.csv", "data/expression.csv")
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    encoder = TileEncoder(cfg["image_dim"], cfg["fusion_dim"])
    fusion = GatedCrossAttention(cfg["fusion_dim"])
    optimiser = torch.optim.Adam(list(encoder.parameters()) + list(fusion.parameters()), lr=cfg["learning_rate"])
    for epoch in range(cfg["epochs"]):
        for tiles, expr, labels in loader:
            optimiser.zero_grad()
            image_tokens = encoder(tiles)
            fused = fusion(image_tokens, expr)
            loss = fused.pow(2).mean()
            loss.backward()
            optimiser.step()
        print(f"epoch {epoch} loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
