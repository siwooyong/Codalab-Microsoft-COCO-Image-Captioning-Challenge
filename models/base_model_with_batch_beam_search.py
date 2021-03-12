# model ge1 : beam_search

'''

Put the whole sentence in gpt2 hidden layers : with multiple img_embed

'''

import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.nn.utils.weight_norm import weight_norm

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
    self.tokenizer = hyper_parameters['tokenizer']

    self.__cnn__ = cnn.features
    self.__img2embed_conv__ = nn.Conv2d(self.features_dim, int(self.embed_dim * 0.5), kernel_size = 1, stride = 1)
    self.__content_embed__ = gpt2_model.wte
    self.__position_embed__ = gpt2_model.wpe
    self.__hidden_layers__ = gpt2_model.h
    self.__layer_norm__ = gpt2_model.ln_f
    self.__fc_layer__ = nn.Linear(self.embed_dim, self.vocab_dim)
    self.__embed_drop__ = nn.Dropout(p = dropout)

    self.softmax = nn.Softmax(dim = 1)
    self.dropout = nn.Dropout(p = dropout)

    self.__init_weights__()

  def __init_weights__(self):
    torch.nn.init.xavier_uniform_(self.__fc_layer__.weight)
    if self.tokenizer == nltk_tokenizer:
      torch.nn.init.xavier_uniform_(self.__content_embed__.weight)

  def __random_topk__(self, pred, k): 
    prob_distribution = self.softmax(pred)
    top_indices = prob_distribution.topk(k = k).indices.permute(1, 0)

    return random.choice(top_indices)


  def forward(self, images, input_ids = None):
    batch = images.shape[0] # (N)
    with torch.no_grad():
      batch_features = self.__cnn__(images) # (N, features_dim, block_num, block_num)
    
    conv_features = self.__img2embed_conv__(batch_features).permute(0, 2, 3, 1) # (N, block_num, block_num, embed_dim * 0.5)
    apool = torch.mean(conv_features, dim = 1) # (N, block_num, embed_dim * 0.5)
    mpool, _ = torch.max(conv_features, dim = 1) # (N, block_num, embed_dim * 0.5)

    imgs_embed = torch.cat([apool, mpool], dim = 2) # (N, block_num, embed_dim)

    words_embed = self.__content_embed__(input_ids) # (N, seq_len, embed_dim)
    indices  = torch.arange(self.seq_len + self.block_num).expand(batch, -1).to(device)
    position_embed = self.__position_embed__(indices)

    h = self.__embed_drop__(torch.cat([imgs_embed, words_embed], dim = 1) + position_embed).to(device) # (N, seq_len + self.block_num, embed_dim)
    for i in range(self.layer_num):
        h = self.__hidden_layers__[i](h)[0]        
        h[:, :self.block_num, :] = imgs_embed + position_embed[:, :self.block_num, :]

    preds = self.__fc_layer__(self.dropout(self.__layer_norm__(h[:, self.block_num:, :]))) # (N, seg_len, vocab_dim)
    return preds
  

  def __sample__(self, images):
    batch = images.shape[0] # (N)
    with torch.no_grad():
      batch_features = self.__cnn__(images) # (N, features_dim, block_num, block_num)
    
    conv_features = self.__img2embed_conv__(batch_features).permute(0, 2, 3, 1) # (N, block_num, block_num, embed_dim * 0.5)
    apool = torch.mean(conv_features, dim = 1) # (N, block_num, embed_dim * 0.5)
    mpool, _ = torch.max(conv_features, dim = 1) # (N, block_num, embed_dim * 0.5)

    imgs_embed = torch.cat([apool, mpool], dim = 2) # (N, block_num, embed_dim)

    if self.tokenizer == gpt2_tokenizer:
      start_embed = self.__content_embed__(torch.Tensor(batch * [50258]).to(device).long())  
    elif self.tokenizer == nltk_tokenizer:
      start_embed = self.__content_embed__(torch.Tensor(batch * [1]).to(device).long())  

    indices = torch.arange(self.block_num + 1).expand(batch, -1).to(device).long()
    position_embed = self.__position_embed__(indices)

    h = (torch.cat([imgs_embed, start_embed.unsqueeze(1)], dim = 1) + position_embed).to(device) # (N, block_num + 1, embed_dim)

    preds = torch.zeros([batch, self.seq_len]).to(device)
    scores = torch.zeros([batch, self.seq_len, self.vocab_dim]).to(device)
    for i in range(self.seq_len):
      for j in range(self.layer_num):
        h = self.__hidden_layers__[j](h)[0]
        h[:, :self.block_num, :] = imgs_embed + position_embed[:, :self.block_num, :]

      pred = self.__fc_layer__(self.__layer_norm__(h[:, self.block_num:, :]))[:, -1, :] # (N, vocab_dim)
      preds[:, i] = self.__random_topk__(pred = pred, k = 1) 
      scores[:, i, :] = pred
      
      words_embed = self.__content_embed__(preds[:, :(i + 1)].long())
      indices = torch.arange(i + self.block_num + 2).expand(batch, -1).to(device)
      position_embed = self.__position_embed__(indices)

      h = torch.cat([imgs_embed, start_embed.unsqueeze(1), words_embed], dim = 1) + position_embed

    return scores
  
  def __beam_search__(self, images, beam_size, seq_len, penalty = None):
    k = beam_size

    batch = images.shape[0]
    with torch.no_grad():
      batch_features = self.__cnn__(images) # (N, features_dim, block_num, block_num)
    
    conv_features = self.__img2embed_conv__(batch_features).permute(0, 2, 3, 1) # (N, block_num, block_num, embed_dim * 0.5)
    apool = torch.mean(conv_features, dim = 1) # (N, block_num, embed_dim * 0.5)
    mpool, _ = torch.max(conv_features, dim = 1) # (N, block_num, embed_dim * 0.5)

    imgs_embed = torch.cat([apool, mpool], dim = 2) # (N, block_num, embed_dim)


    tokens = torch.ones(batch, 1).to(device).long() * 50258
    start_embed = self.__content_embed__(tokens)

    indices = torch.arange(self.block_num + 1).expand(batch, -1).to(device).long()
    position_embed = self.__position_embed__(indices)
    
    h = (torch.cat([imgs_embed, start_embed], dim = 1) + position_embed).to(device)
    for j in range(self.layer_num):
      h = self.__hidden_layers__[j](h)[0]
      h[:, :self.block_num, :] = imgs_embed + position_embed[:, :self.block_num, :]

    dec_out = self.__fc_layer__(self.__layer_norm__(h[:, self.block_num:, :]))[:, -1, :].unsqueeze(1)  #(N, 1,vocab_dim)
    
    lprobs = F.log_softmax(dec_out, dim = 2)
    lprobs, tokens = torch.topk(lprobs, k = k, dim = 2)
    lprobs = lprobs.transpose(2, 1) # (N, k, 1)

    tokens = tokens.transpose(2, 1) # (N, k, 1)
    tokens = torch.cat([torch.ones(batch, k, 1).to(device).long() * 50258, tokens], dim = 2) # (N, k, 2)

    mask = torch.zeros(batch, k, 3, dtype = torch.bool).to(device)  # (N, k, 3)
    mask[:, :, :] = True

    new_order = torch.tensor(np.repeat(range(batch), k)).to(device)
    imgs_embed = imgs_embed.index_select(0, new_order) # (N * k, block_num, embed_dim)

    for _ in range(seq_len):
      tokens_batch = tokens.flatten(0, 1) # (N * k, s)

      words_embed = self.__content_embed__(tokens_batch) # (N * k, s, embed_dim)

      indices = torch.arange(self.block_num + words_embed.shape[1]).expand(words_embed.shape[0], -1).to(device).long() 
      position_embed = self.__position_embed__(indices) # (N * k, s + block_num, embed_dim)


      h = (torch.cat([imgs_embed, words_embed], dim = 1) + position_embed).to(device)
      for j in range(self.layer_num):
        h = self.__hidden_layers__[j](h)[0]
        h[:, :self.block_num, :] = imgs_embed + position_embed[:, :self.block_num, :]    

      dec_out = self.__fc_layer__(self.__layer_norm__(h[:, self.block_num:, :]))  #(N, s, vocab_dim)

      lprobs_batch = F.log_softmax(dec_out, dim = 2)
      lprobs_batch = lprobs_batch[:, -1, :] #(N * k, vocab_dim)
      lprobs_batch = lprobs_batch.reshape(tokens.shape[0], tokens.shape[1], -1)
      lprobs_k, tokens_k = torch.topk(lprobs_batch, k = k , dim = 2)  # (N, k, k)

      tokens_repeated = torch.repeat_interleave(tokens, k, dim = 1)
      tokens_k_flattened = tokens_k.flatten().view(batch, -1, 1)
      tokens_cat = torch.cat([tokens_repeated, tokens_k_flattened], dim = 2)

      mask_repeated = torch.repeat_interleave(mask, k, dim = 1)
      mask_k_flattened = (tokens_k_flattened != 50259) & mask_repeated[:, :, -1:]
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

gpt_decoder = decoder(features_dim = 2048,
                      embed_dim = 768)
