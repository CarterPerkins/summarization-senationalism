import pandas as pd

import torch

from transformers import PegasusTokenizer, PegasusForConditionalGeneration#, PegasusConfig
from transformers import BartTokenizer, BartForConditionalGeneration#, BartConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration#, T5Config
from transformers import pipeline

import argparse
import os
import time

def get_model(name):
    # TODO: Add support for hyperparameters
    if name == 't5':
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
    elif name == 'bart':
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    elif name == 'pegasus':
        model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    else:
        raise ValueError(f'Unknown model {name}. Must be t5, bart, or pegasus.')

    return (model, tokenizer)

def run(args):
    params = pd.read_csv(args.params).loc[0]

    model, tokenizer = get_model(params['model'])
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.eval()

    # Setup data
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'test.csv'))
    df = df.rename(columns={'title': 'headline', 'content': 'article'})
    df = df.head(10)
    headline_max_len = int(params['headline_max_len'])
    article_max_len = int(params['article_max_len'])
    filename = params['results_name']

    # Truncate articles to fit max ar
    articles = df.article.apply(lambda x: x[:article_max_len]).to_list()
    start = time.time()
    headline_generator = pipeline('summarization', model=model, tokenizer=tokenizer)
    out = headline_generator(articles, max_length=headline_max_len)
    out = list(map(lambda x: x['summary_text'], out))

    df['generated_headline'] = out
    df.to_csv(filename + '.test', index=False)
    elapsed = time.time() - start
    print('Done in {:.3f}s.'.format(elapsed))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate headlines on test dataset for a model.')
    parser.add_argument('--params', type=str, help='training params file', required=True)
    parser.add_argument('--model_path', type=str, help='path to model weights', required=True)

    args = parser.parse_args()

    run(args)