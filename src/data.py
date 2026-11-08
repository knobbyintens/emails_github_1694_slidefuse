import pandas as pd
import torch
from torch.utils.data import Dataset


class SlideExpressionDataset(Dataset):
    def __init__(self, slide_index: str, expression_table: str):
        self.index = pd.read_csv(slide_index)
        self.expression = pd.read_csv(expression_table, index_col=0)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        tiles = torch.load(row["tile_features"])
        expr = torch.tensor(self.expression.loc[row["sample_id"]].values, dtype=torch.float32)
        label = int(row["subtype"])
        return tiles, expr, label
