""" 
PyTorch HistoGPT Model
© Manuel Tran / Helmholtz Munich
"""

import h5py
import numpy as np
import random
import re
import torch

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import BioGptTokenizer


class VLMDataset(Dataset):
    """
    Vision Language Modeling Dataset
    """
    def __init__(self, text_path, diag_path, feat_path, transform):
        """
        initiate with text_path, diag_path, feat_path, trasnsform
        """
        #self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt-large")
        self.transform = transform
        self.max_len = 64000

        self.text_path = text_path
        self.feat_path = feat_path

        with h5py.File(feat_path, 'r') as feat_file:
            keys = list(feat_file.keys())

        label_data = torch.load(diag_path).tolist()
        label_map = {i[0].split('_')[0]: i[1] for i in label_data}

        error = [uuid for uuid in keys if uuid not in label_map]
        self.uuid = sorted(list(set(sorted(keys)) - set(sorted(error))))

        self.diag = [label_map.get(key) for key in self.uuid]

    def __getitem__(self, idx):
        """
        return text, mask, feat
        """
        # get features
        with h5py.File(self.feat_path, 'r') as feat_file:
            feat = np.array(feat_file.get(self.uuid[idx]))
            feat = torch.tensor(feat)
            seq_len, _ = feat.size()

        if seq_len > self.max_len:
            indices = random.sample(range(seq_len), self.max_len)
            feat = feat[indices, :]

        # get reports
        uuid = str(self.uuid[idx]) + '_15'
        with h5py.File(self.text_path, 'r') as text_file:
            if text_file.get(uuid) is not None:
                text = [np.array(i).tolist() for i in text_file.get(uuid)]
                text = [i.decode('UTF-8') if i else '<unk>' for i in text]
                #text = np.random.choice(text)
                text = text[0]
            else:
                text = '<unk>'

        # get diagnosis
        diag = self.diag[idx]
        diag = ' Final diagnosis: ' + diag + '.'

        #text = 'Final diagnosis: ' + diag + '. ' + text

        # shuffle text
        text = text + diag
        copy = text
        try:
            text = text.split('Critical findings:')
            part = text[1].split('Final diagnosis:')
            text = [text[0], 'Critical findings:' + part[0], 'Final diagnosis:' + part[1]]
            random.shuffle(text)
            text = ''.join(text)
            text = re.sub(r'\. *', '. ', text)
        except IndexError:
            text = copy

        # tokenize text
        text = self.tokenizer.encode(text, add_special_tokens=False)
        text = [0] + text + [2]
        text = torch.tensor(text)
        if self.transform:
            text = self.transform(text)

        return text, feat#, str(self.uuid[idx])

    def __len__(self):
        """
        :return length of dataset
        """
        return len(self.uuid)


class MILDataset(Dataset):
    """
    Multiple Instance Learning Dataset
    """
    def __init__(self, diag_path, feat_path, transform):
        """
        initiate with diag_path, feat_path, transform
        """
        self.transform = transform
        self.feat_path = feat_path
        self.max_len = 64000

        with h5py.File(feat_path, 'r') as feat_file:
            keys = list(feat_file.keys())

        label_data = torch.load(diag_path).tolist()
        label_map = {i[0].split('_')[0]: i[1] for i in label_data}

        error = [uuid for uuid in keys if uuid not in label_map]
        self.uuid = sorted(list(set(sorted(keys)) - set(sorted(error))))

        label_encoder = LabelEncoder()
        diag = [label_map.get(key) for key in self.uuid]
        self.diag = label_encoder.fit_transform(diag)

    def __getitem__(self, idx):
        """
        return feat, diag, length
        """
        with h5py.File(self.feat_path, 'r') as feat_file:
            feat = np.array(feat_file.get(self.uuid[idx]))
            feat = torch.tensor(feat)

        seq_len, _ = feat.size()
        diag = self.diag[idx]

        if seq_len > self.max_len:
            indices = random.sample(range(seq_len), self.max_len)
            feat = feat[indices, :]

        return feat, diag

    def __len__(self):
        """
        :return length of dataset
        """
        return len(self.uuid)
