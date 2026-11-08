import argparse
import yaml
from sklearn.metrics import accuracy_score
from src.data import SlideExpressionDataset
from src.encoder import TileEncoder
from src.fusion import GatedCrossAttention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dataset = SlideExpressionDataset("data/slides.csv", "data/expression.csv")
    encoder = TileEncoder(cfg["image_dim"], cfg["fusion_dim"])
    fusion = GatedCrossAttention(cfg["fusion_dim"])
    state = __import__("torch").load(args.checkpoint)
    encoder.load_state_dict(state["encoder"])
    fusion.load_state_dict(state["fusion"])
    predictions, labels = [], []
    for tiles, expr, label in dataset:
        image_tokens = encoder(tiles)
        fused = fusion(image_tokens, expr)
        predictions.append(int(fused.sum().item() > 0))
        labels.append(int(label))
    print(f"accuracy: {accuracy_score(labels, predictions):.3f}")


if __name__ == "__main__":
    main()
