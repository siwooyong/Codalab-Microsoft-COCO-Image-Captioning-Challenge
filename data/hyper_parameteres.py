# hyper_parameters

hyper_parameters = {}

hyper_parameters['block_num'] = 10
hyper_parameters['img_size'] = 512
hyper_parameters['max_len'] = 20
hyper_parameters['shrink_factor'] = 0.7
hyper_parameters['topk'] = 5
hyper_parameters['patience'] = 1
hyper_parameters['tokenizer'] = gpt2_tokenizer
if hyper_parameters['tokenizer'] == bert_tokenizer:
  hyper_parameters['vocab_dim'] = 30526
elif hyper_parameters['tokenizer'] == gpt2_tokenizer:
  hyper_parameters['vocab_dim'] = 50261
else:
  hyper_parameters['vocab_dim'] = nltk_tokenizer.__len__()

hyper_parameters['grad_clip'] = 5.0
hyper_parameters['layer_num'] = 4
hyper_parameters['hypo_num'] = 20
hyper_parameters['bert_layers'] = 0
hyper_parameters['caption_num'] = 3#5
hyper_parameters['conv_num'] = 4
hyper_parameters['save_address'] = 'trained_model'
hyper_parameters['block_size'] = int(hyper_parameters['img_size'] / hyper_parameters['block_num'])
hyper_parameters['lr'] = 1e-3
hyper_parameters['batch_size'] = 32
