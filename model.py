# model: decoder-transformer

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
    #with torch.no_grad():
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

gpt_decoder = decoder(features_dim = 2048,
                      embed_dim = 768)
