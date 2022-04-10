import torch

import os

import pandas as pd

class NewsHeadlineDataset(torch.utils.data.Dataset):

    def __init__(self, split):
        assert all(os.path.isfile(os.path.join(os.getcwd(), 'data', f'{t}.csv')) for t in ['train', 'val', 'test']), 'Dataset not found in `src/data`. Did you run `preprocessing.py`?'
        
        if split == 'train':
            self.dataframe = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train.csv'))
        elif split == 'val':
            self.dataframe = pd.read_csv(os.path.join(os.getcwd(), 'data', 'val.csv'))
        elif split == 'test':
            self.dataframe = pd.read_csv(os.path.join(os.getcwd(), 'data', 'test.csv'))
        else:
            raise ValueError(f'Unknown split `{split}`. Must be one of `train`, `val`, or `test`.')

        self.article = self.dataframe.content
        self.headline = self.dataframe.title
        self.ids = self.dataframe.id # for analysis

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        article = self.article[index]
        headline = self.headline[index]
        identifier = self.ids[index]

        return (identifier, article, headline)

def transform_text(article, headline, tokenizer, article_max_len, headline_max_len):
    '''Transforms article and headline into numerical features.'''
    source = tokenizer.batch_encode_plus([article], max_length=article_max_len, padding='max_length', return_tensors='pt')
    target = tokenizer.batch_encode_plus([headline], max_length=headline_max_len, padding='max_length', return_tensors='pt')
    
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