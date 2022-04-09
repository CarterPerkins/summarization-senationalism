import torch

import os

import pandas as pd

class NewsHeadlineDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, article_max_len, headline_max_len, split):
        assert all(os.path.isfile(os.path.join(os.getcwd(), 'data', f'{t}.csv')) for t in ['train', 'val', 'test']), 'Dataset not found in `src/data`. Did you run `preprocessing.py`?'
        
        if split == 'train':
            self.dataframe = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train.csv'))
        elif split == 'val':
            self.dataframe = pd.read_csv(os.path.join(os.getcwd(), 'data', 'val.csv'))
        elif split == 'test':
            self.dataframe = pd.read_csv(os.path.join(os.getcwd(), 'data', 'test.csv'))
        else:
            raise ValueError(f'Unknow split `{split}`. Splits must be one of `train`, `val`, or `test`.')

        self.tokenizer = tokenizer
        self.article_max_len = article_max_len
        self.headline_max_len = headline_max_len
        self.article = self.dataframe.content
        self.headline = self.dataframe.title

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        article = self.article[index]
        headline = self.headline[index]

        source = self.tokenizer.batch_encode_plus([article], max_length=self.article_max_len, padding='max_length', return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([headline], max_length=self.headline_max_len, padding='max_length', return_tensors='pt')
        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }