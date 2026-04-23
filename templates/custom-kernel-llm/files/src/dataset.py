import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, token_ids: list[int], max_len: int, stride: int = None):
        if stride is None:
            stride = max_len
        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(token_ids) - max_len, stride):
            inp = token_ids[i : i + max_len]
            tgt = token_ids[i + 1 : i + max_len + 1]
            self.input_ids.append(torch.tensor(inp, dtype=torch.long))
            self.target_ids.append(torch.tensor(tgt, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
