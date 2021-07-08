# model : ClipVinvlGpt

'''

origianl show_and_tell with improving abillity of each layer(ClipGpt) + meaningful features(Vinvl)

'''

import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.nn.utils.weight_norm import weight_norm
import clip

clip_model, preprocess = clip.load("ViT-B/32", device = device, jit = False)
device = 'cuda'
cnn = ptcv_get_model('resnet101', pretrained = True)   

class decoder(nn.Module):
  def __init__(self, features_dim, embed_dim, seq_len = hyper_parameters['max_len'], dropout = 0.5):
    super(decoder, self).__init__()
    self.features_dim = features_dim
    self.embed_dim = embed_dim

    self.seq_len = seq_len
    self.vocab_dim = hyper_parameters['vocab_dim']
    self.layer_num = hyper_parameters['layer_num']
    self.block_num = hyper_parameters['block_num']
    self.clip_dim = hyper_parameters['clip_dim']
    self.tags_num = 0#hyper_parameters['tags_num']
    self.tokenizer = hyper_parameters['tokenizer']
    self.start_token = hyper_parameters['start_token']
    self.end_token = hyper_parameters['end_token']
    self.feature_num = hyper_parameters['feature_num']
    self.clip_num = 1
    
    self.__clip__ = clip_model
    self.__cnn__ = cnn.features
    self.__img2embed__ = nn.Linear(self.features_dim, self.embed_dim)
    self.__clip_embed__ = nn.Linear(self.clip_dim, self.embed_dim)
    self.__content_embed__ = nn.Embedding.from_pretrained(pretrained_embedding)
    self.__position_embed__ = gpt2_model.wpe
    self.__hidden_layers__ = gpt2_model.h
    self.__layer_norm__ = gpt2_model.ln_f
    self.__fc_layer__ = nn.Linear(self.embed_dim, self.vocab_dim)
    self.__clip_drop__ = nn.Dropout(p = 0.1)
    self.__embed_drop__ = nn.Dropout(p = dropout)

    self.softmax = nn.Softmax(dim = 1)
    self.dropout = nn.Dropout(p = dropout)
    self.scst_dropout = nn.Dropout(p = 0.3)
    self.relu = nn.ReLU()

    self.__init_weights__()


  def __init_weights__(self):
    torch.nn.init.xavier_uniform_(self.__fc_layer__.weight)
    torch.nn.init.xavier_uniform_(self.__img2embed__.weight)  
    torch.nn.init.xavier_uniform_(self.__clip_embed__.weight)  

  def forward(self, images, features, input_ids = None, tag_ids = None):
    batch = features.shape[0] # (N)

    with torch.no_grad():
      clip_features = self.__clip__.encode_image(images)
    clip_embed = self.__clip_embed__(self.__clip_drop__(clip_features.float())).unsqueeze(1) # (N, clip_num = 1, embed_dim)
      
    imgs_embed = self.__img2embed__(self.__clip_drop__(features.float())) # (N, feature_num, embed_dim)

    words_embed = self.__content_embed__(input_ids) 
    indices  = torch.arange(self.seq_len + self.feature_num + self.clip_num).expand(batch, -1).to(device)
    position_embed = self.__position_embed__(indices)

    h = self.__embed_drop__(torch.cat([clip_embed, imgs_embed, words_embed], dim = 1) + position_embed).to(device) # (N, seq_len + feature_num + clip_num, embed_dim)
    for i in range(self.layer_num):
        h = self.__hidden_layers__[i](h)[0]        
        h[:, :int(self.clip_num + self.feature_num), :] = torch.cat([clip_embed, imgs_embed], dim = 1) + position_embed[:, :int(self.clip_num + self.feature_num), :]

    preds = self.__fc_layer__(self.dropout(self.__layer_norm__(h[:, int(self.clip_num + self.feature_num):, :]))) # (N, seg_len, vocab_dim)
    return preds

  def __beam_search__(self, images, features, tag_ids, beam_size, seq_len, penalty = None):
    k = beam_size
    batch = features.shape[0] 

    
    with torch.no_grad():
      clip_features = self.__clip__.encode_image(images)
    clip_embed = self.__clip_embed__(clip_features.float()).unsqueeze(1)
    
    imgs_embed = self.__img2embed__(features.float())

    tokens = torch.ones(batch, 1).to(device).long() * self.start_token
    start_embed = self.__content_embed__(tokens)

    indices = torch.arange(self.feature_num + self.clip_num + 1).expand(batch, -1).to(device).long()
    position_embed = self.__position_embed__(indices)
    
    h = (torch.cat([clip_embed, imgs_embed, start_embed], dim = 1) + position_embed).to(device)
    for j in range(self.layer_num):
      h = self.__hidden_layers__[j](h)[0]
      h[:, :int(self.clip_num + self.feature_num), :] = torch.cat([clip_embed, imgs_embed], dim = 1) + position_embed[:, :int(self.clip_num + self.feature_num), :]#h[:, :int(self

    dec_out = self.__fc_layer__(self.__layer_norm__(h[:, int(self.clip_num + self.feature_num):, :]))[:, -1, :].unsqueeze(1)  #(N, 1,vocab_dim)
    
    lprobs = F.log_softmax(dec_out, dim = 2)
    lprobs, tokens = torch.topk(lprobs, k = k, dim = 2)
    lprobs = lprobs.transpose(2, 1) # (N, k, 1)

    tokens = tokens.transpose(2, 1) # (N, k, 1)
    tokens = torch.cat([torch.ones(batch, k, 1).to(device).long() * self.start_token, tokens], dim = 2)

    mask = torch.zeros(batch, k, 3, dtype = torch.bool).to(device)  # (N, k, 3)
    mask[:, :, :] = True

    new_order = torch.tensor(np.repeat(range(batch), k)).to(device)
    imgs_embed = imgs_embed.index_select(0, new_order) # (N * k, feature_num, embed_dim)
    clip_embed = clip_embed.index_select(0, new_order)

    for step in range(seq_len):
      tokens_batch = tokens.flatten(0, 1) # (N * k, s)
      words_embed = self.__content_embed__(tokens_batch) # (N * k, s, embed_dim)

      indices = torch.arange(self.clip_num + self.feature_num + words_embed.shape[1]).expand(words_embed.shape[0], -1).to(device).long() 
      position_embed = self.__position_embed__(indices) # (N * k, s + block_num, embed_dim)

      h = (torch.cat([clip_embed, imgs_embed, words_embed], dim = 1) + position_embed).to(device)
      for j in range(self.layer_num):
        h = self.__hidden_layers__[j](h)[0]
        h[:, :int(self.clip_num + self.feature_num), :] = torch.cat([clip_embed, imgs_embed], dim = 1) + position_embed[:, :int(self.clip_num + self.feature_num), :]#h[:, :int(self.block_num + self.tags_num), :] = torch.cat([imgs_embed, tags_embed], dim = 1) + position_embed[:, :int(self.block_num + self.tags_num), :]

      dec_out = self.__fc_layer__(self.__layer_norm__(h[:, int(self.clip_num + self.feature_num):, :]))  #(N, s, vocab_dim)


      lprobs_batch = F.log_softmax(dec_out, dim = 2)
      lprobs_batch = lprobs_batch[:, -1, :] #(N * k, vocab_dim)
      lprobs_batch = lprobs_batch.reshape(tokens.shape[0], tokens.shape[1], -1)
      lprobs_k, tokens_k = torch.topk(lprobs_batch, k = k , dim = 2)  # (N, k, k)

      tokens_repeated = torch.repeat_interleave(tokens, k, dim = 1)
      tokens_k_flattened = tokens_k.flatten().view(batch, -1, 1)
      tokens_cat = torch.cat([tokens_repeated, tokens_k_flattened], dim = 2)

      mask_repeated = torch.repeat_interleave(mask, k, dim = 1)
      mask_k_flattened = (tokens_k_flattened != self.end_token) & mask_repeated[:, :, -1:]
      mask_cat = torch.cat([mask_repeated, mask_k_flattened], dim = 2)

      lprobs_repeated = torch.repeat_interleave(lprobs, k, dim = 1)
      lprobs_k_flattened = lprobs_k.flatten().view(batch, -1, 1)
      lprobs_cat = torch.cat([lprobs_repeated, lprobs_k_flattened], dim = 2)
      lprobs_cat_masked = lprobs_cat * mask_cat[:, :, 1:-1]

      num_tokens = torch.sum(mask_cat[:, :, 1:-1], dim = 2)
      scores_mask = torch.zeros(batch, k ** 2, dtype = torch.bool)

      if penalty is None:
        scores = torch.sum(lprobs_cat_masked, dim = 2) #/ num_tokens ** penalty
      else:
        scores = torch.sum(lprobs_cat_masked, dim = 2) / num_tokens ** penalty

      for i in range(batch):
        for j in range(k):
           first = j * k
           start = first + 1
           end = first + k
           scores_mask[i, start:end] = torch.sum(mask_cat[i, first:end, -1]) == 0

      for i in range(batch):
        scores[i][scores_mask[i]] = -1e8

      top_values, top_indices = torch.topk(scores, k = k)

      tokens_list = []
      lprobs_list = []
      mask_list = []

      for i in range(batch):
        tokens_selected = tokens_cat[i][top_indices[i]]
        tokens_list.append(tokens_selected)

        lprobs_selected = lprobs_cat[i][top_indices[i]]
        lprobs_list.append(lprobs_selected)

        mask_selected = mask_cat[i][top_indices[i]]
        mask_list.append(mask_selected)

      tokens = torch.stack(tokens_list, dim = 0)
      lprobs = torch.stack(lprobs_list, dim = 0)
      mask = torch.stack(mask_list, dim = 0)

      if torch.sum(mask[:, :, -1]) == 0:
        break

    result_mask = mask[:, :, 1:-1]
    result_tokens = tokens[:, :, 1:] * result_mask
    result_lprobs = lprobs * result_mask
    result_num_tokens = torch.sum(result_mask, dim = 2)

    if penalty is None:
      result_scores = torch.sum(result_lprobs, dim = 2) #/ result_num_tokens ** penalty
    else:
      result_scores = torch.sum(result_lprobs, dim = 2) / result_num_tokens ** penalty

    return result_scores, result_lprobs, result_tokens, result_mask

gpt_decoder = decoder(features_dim = 2054,
                      embed_dim = 768)















# cross_entropy_train

def convert_models_to_fp32(model): 
  for p in model.parameters(): 
    p.data = p.data.float() 

class CapBenchTrain(nn.Module):
  def __init__(self, model, fine_tune = False):
    super(CapBenchTrain, self).__init__()
    self.model = model
    self.fine_tune = fine_tune
    if self.fine_tune == False:
      self.model.__cnn__.requires_grad_ = False
    else:
      self.model.__cnn__.requires_grad_ = True

    self.loss_fn = nn.CrossEntropyLoss()

  def forward(self, images, features, input_ids, target, tag_ids, mask = None):
    preds_out = self.model(images, features, input_ids, tag_ids) # (N, seq_len, vocab_dim)
    loss = self.loss_fn(preds_out.reshape(-1, hyper_parameters['vocab_dim']), target.reshape(-1))
    
    return loss, preds_out

model = gpt_decoder
state = 'initial'

if state == 'initial':
  print('not trained')
  net = CapBenchTrain(model).to(device)
else:
  print('trained')
  net = CapBenchTrain(caption_net, False).to(device)

convert_models_to_fp32(net.model.__clip__)

















# scst train

from nlgeval.pycocoevalcap.cider.cider import Cider

class CapBenchTrain(nn.Module):
  def __init__(self, model, fine_tune = False):
    super(CapBenchTrain, self).__init__()
    self.model = model
    self.fine_tune = fine_tune
    if self.fine_tune == False:
      self.model.__cnn__.requires_grad_ = False
    else:
      self.model.__cnn__.requires_grad_ = True

    self.loss_fn = nn.CrossEntropyLoss()
  
  def __generate__(self, images, features, tag_ids, all_caps, beam_size):
    '''
     
    generate captions using batch beam search

    scores.shape : (N, beam_size)
    tokens.shape : (N, beam_size, max_len)

    '''
    batch = images.shape[0]

    references = list()
    hypothesis = list()

    scores, _, batch_tokens, _ = self.model.__beam_search__(images = images, 
                                                            features = features,
                                                            tag_ids = tag_ids,
                                                            beam_size = beam_size, 
                                                            seq_len = 25)
    
    for i in range(batch):
      reference = all_caps[i]
      tokens = batch_tokens[i]

      for k in range(beam_size):
         sample = tokens[k]
         if hyper_parameters['tokenizer'] == gpt2_tokenizer:
           indices = torch.where((sample < 50257) & (sample != 13) & (sample != 0))[0]
           decoded = hyper_parameters['tokenizer'].decode(sample[indices].long().tolist())
         else:
           indices = torch.where((sample > 3))[0]
           decoded = gpt2_tokenizer.decode(gpt2_tokenizer.encode(' '.join([hyper_parameters['tokenizer'].decode(token) for token in sample[indices].long().tolist()])))
          
         hypothesis.extend([decoded])   
         references.extend([reference])
    
    return references, hypothesis, scores

  def forward(self, images, features, tag_ids, input_ids, target, all_caps, beam_size, self_crit_seq_train = None):
    if self_crit_seq_train is None:
      preds_out = self.model(images, features, input_ids, tag_ids)
      loss = self.loss_fn(preds_out.reshape(-1, hyper_parameters['vocab_dim']), target.reshape(-1))
      return loss, preds_out

    else: # self_crit_seq_train
      ref_list, hyp_list, scores = self.__generate__(images, features, tag_ids, all_caps, beam_size)

      refs = {idx: lines for (idx, lines) in enumerate(ref_list)}
      hyps = {idx: [lines] for (idx, lines) in enumerate(hyp_list)}
      _, reward = Cider().compute_score(refs, hyps) # (N, beam_size)

      reward = torch.from_numpy(reward).to(device).view(scores.shape)
      reward_baseline = torch.mean(reward, dim = 1, keepdim = True)

      loss = - scores * (reward - reward_baseline)
      loss = loss.mean()

      return loss, hyp_list[::beam_size]


model = gpt_decoder
state = 'not initial'

if state == 'not initial':
  print('not trained')
  net = CapBenchTrain(model).to(device)
else:
  print('trained')
  net = CapBenchTrain(caption_net, False).to(device)
