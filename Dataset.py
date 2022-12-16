from torch.utils.data import Dataset
import torch


class TextData(Dataset):
    def __init__(self, text, converter):
        self.text = text
        self.converter = converter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __getitem__(self, index):
        ind_text = self.text[index]
        ids, masks = self.converter.enocde(ind_text)
        return ids.to(self.device), masks.to(self.device)

    def __len__(self):
        return len(self.text)