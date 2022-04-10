from transformers import PegasusTokenizer, PegasusForConditionalGeneration#, PegasusConfig
from transformers import BartTokenizer, BartForConditionalGeneration#, BartConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration#, T5Config

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
		raise ValueError(f'Unknown model `{name}`. Must be one of `t5`, `bart`, or `pegasus`.')

	return (model, tokenizer)