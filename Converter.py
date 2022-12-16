
import string
import numpy as np
import torch



class BertTokenLabelConverter(object):
    """ Convert between text-label and text-index """
    """ignore index = 0"""

    def __init__(self, is_train=True):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.SPACE = '[s]'
        self.GO = '[GO]'
        self.character = string.printable[:-6]
        self.batch_max_length = 25
        self.MASK = '[MASK]'
        self.is_train = is_train
        self.list_token = [self.GO, self.SPACE]
        self.list_token_m = [self.MASK]
        # self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(self.character) + list(self.list_token_m)

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = self.batch_max_length + len(self.list_token) + len(self.list_token_m)

    def encode(self, text):
        """ convert text-label into text-index.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        attn_mask = torch.ones(batch_text.shape)
        for i, t in enumerate(text):
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            prob = np.random.uniform()
            mask_len = round(len(list(t)) * 0.15)
            if self.is_train and mask_len > 0:
                for m in range(mask_len):
                    index = np.random.randint(1, len(t) + 1)
                    prob = np.random.uniform()
                    if prob > 0.2:
                        txt[index] = self.dict[self.MASK]
                    elif prob > 0.1:
                        char_index = np.random.randint(len(self.list_token), len(self.character))
                        txt[index] = self.dict[self.character[char_index]]

                    attn_mask[i, index] = 0

            batch_text[i][:len(txt)] = torch.LongTensor(txt)  # batch_text[:, 0] = [GO] token
        return batch_text.to(device), attn_mask.to(device)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

