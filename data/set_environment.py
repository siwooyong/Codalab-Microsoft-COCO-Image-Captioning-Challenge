  # set environment

from google.colab import drive
drive.mount('/content/gdrive/', force_remount = True)

!git clone https://github.com/poojahira/image-captioning-bottom-up-top-down
!wget https://raw.githubusercontent.com/lukemelas/EfficientNet-PyTorch/master/examples/simple/labels_map.txt

import sys
sys.path.insert(0, '/content/image-captioning-bottom-up-top-down/nlg-eval-master')

!pip install -U albumentations
!pip install pytorchcv
!pip install transformers

gpt2_tokenizer = 0.0
bert_tokenizer = 0.0

special_tokens = '[pad]', '[start]', '[end]', '[cls]'
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_config = GPT2Config.from_pretrained('gpt2')
gpt2_model = GPT2Model(gpt2_config).from_pretrained('gpt2', config = gpt2_config)
gpt2_tokenizer.add_tokens(special_tokens)
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))


#gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#gpt2_config = GPT2Config.from_pretrained('gpt2')
#gpt2_config.n_embd = 512
#gpt2_config.n_head = 8
#gpt2_model = GPT2Model(gpt2_config)
#gpt2_tokenizer.add_tokens(special_tokens)
#gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

#from transformers import BertTokenizer, BertConfig, BertModel
#bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#bert_config = BertConfig.from_pretrained('bert-base-uncased')
#bert_model = BertModel(bert_config).from_pretrained('bert-base-uncased', config = bert_config)
#bert_tokenizer.add_tokens(special_tokens)
#bert_model.resize_token_embeddings(len(bert_tokenizer)) 

#from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
#roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#roberta_config = RobertaConfig.from_pretrained('roberta-base')
#roberta_model = RobertaModel(roberta_config).from_pretrained('roberta-base', config = roberta_config)
#roberta_tokenizer.add_tokens(special_tokens)
#roberta_model.resize_token_embeddings(len(roberta_tokenizer)) 

path = '/content/gdrive/My Drive/coco_image_caption/'
