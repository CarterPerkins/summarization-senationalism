import torch
from torch import nn, optim

import numpy as np
import pandas as pd

import argparse
import os
import random
import time

from models import get_model
from utils import NewsHeadlineDataset, transform_text

def train(epoch, model, tokenizer, loader, optimizer, device, args):
	'''Conduct an epoch of training.'''
	model.train()
	train_loss = 0
	start = time.time()
	for i, (_, article, headline) in enumerate(loader):
		batch = transform_text(article, headline, tokenizer, args.article_max_len, args.headline_max_len)
		y = batch['target_ids'].to(device, dtype=torch.long)
		y_ids = y[:, :-1].contiguous()
		lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = batch['source_ids'].to(device, dtype=torch.long)
        mask = batch['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        loss = outputs[0]

        elapsed_time = time.time() - start
        train_loss += loss.item()

        if i % 100 == 0:
        	print(f'Train - [EPOCH]: {epoch:<3}\t[LOSS]: {loss.item():.3f}\t[ELAPSED]: {elapsed_time:.2f}s')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

	train_duration = time.time() - start
	train_loss /= len(loader)
	return {
		'train_duration': train_duration,
		'train_loss': np.inf,
		'train_rogue': np.inf,
		'train_bleu': np.inf,
		'train_sensationalism': np.inf,
		'train_bias': np.inf
	}

def validate(epoch, model, tokenizer, loader, device, args):
	'''Conduct an epoch of validation.'''
	model.eval()
	with torch.no_grad():
		val_loss = 0
		start = time.time()
		for i, (_, article, headline) in enumerate(loader):
			batch = transform_text(article, headline, tokenizer, args.article_max_len, args.headline_max_len)
			y = batch['target_ids'].to(device, dtype=torch.long)
		    # --------------------------------------------------
			# Keep the following for calculating validation loss?
			y_ids = y[:, :-1].contiguous()
			lm_labels = y[:, 1:].clone().detach()
		    lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
		    # --------------------------------------------------
			ids = batch['source_ids'].to(device, dtype=torch.long)
	        mask = batch['source_mask'].to(device, dtype=torch.long)

			outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
		    loss = outputs[0]

		    elapsed_time = time.time() - start
		    train_loss += loss.item()

		    if i % 100 == 0:
		    	print(f'Validate - [EPOCH]: {epoch:<3}\t[LOSS]: {loss.item():.3f}\t[ELAPSED]: {elapsed_time:.2f}s')

	train_duration = time.time() - start
	train_loss /= len(loader)
	return {
		'train_duration': train_duration,
		'train_loss': np.inf,
		'train_rogue': np.inf,
		'train_bleu': np.inf,
		'train_sensationalism': np.inf,
		'train_bias': np.inf
	}

def test(model, tokenizer, loader):
	'''Test the model.''' 
	pass

def run(args):
	# Set seeds for reproducibility
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	# Get model and tokenizer
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model, tokenizer = get_model(args.model)
	model = model.to(device)

	# Setup datasets
	train_dataset = NewsHeadlineDataset(split='train')
	val_dataset = NewsHeadlineDataset(split='val')
	test_dataset = NewsHeadlineDataset(split='test')

	# Setup dataloaders
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

	optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

	# Setup result logging
	train_results = []
	val_results = []

	# Train
	for epoch in range(args.epochs):
		train_result = train(epoch, model, tokenizer, train_loader, optimizer, device, args)
		train_results.append(train_result)

	if args.tuning: # if tuning mode, only see validation set
		for epoch in range(args.epochs):
			val_result = validate(epoch, model, tokenizer, val_loader, device, args)
			val_results.append(val_result)

	else: # if not tuning mode, checking model performance against unseen test set
		test_result = test(model, tokenizer, test_loader)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Fine-tune a summarization model.')
	parser.add_argument('--epochs', type=int, help='number of training epochs')
	parser.add_argument('--model', type=str, help='base model', choices=['bart', 'pegasus', 't5'])
	parser.add_argument('--batch_size', type=int, help='batch size (per gpu)')
	parser.add_argument('--learning_rate', type=int, help='learning_rate')
	parser.add_argument('--article_max_len', type=int, help='article maximum length')
	parser.add_argument('--headline_max_len', type=int, help='headline maximum length')
	parser.add_argument('--tuning', action='store_true', help='set to hyperparameter tuning mode')
	parser.add_argument('--seed', type=int, help='random seed for reproducibility')
	# TODO: Add support for hyperparameters
	
	args = parser.parse_args()

	run(args)