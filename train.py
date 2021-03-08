# caption_params 
caption_params = []
mode = 'gpt2'

if mode == 'gru':
  modules = [net.model.__img2embed_conv__,
             net.model.__content_embed__,
             net.model.__language_gru__,
             net.model.__fc_layer__] 

elif mode == 'gpt2':
  modules = [#net.model.__cnn__,
             net.model.__img2embed_conv__,
             net.model.__content_embed__,
             net.model.__position_embed__,
             net.model.__hidden_layers__[:hyper_parameters['layer_num']],
             net.model.__layer_norm__,
             net.model.__fc_layer__,
             #net.model.__cnn__.stage2,
             #net.model.__cnn__.stage3,
             #net.model.__cnn__.stage4,
             ]  

for module in modules:
  caption_params += list(module.parameters())

print(len(caption_params))








# caption_config

class caption_config:
  num_workers = 16
  batch_size = hyper_parameters['batch_size']
  n_epochs = 50
  lr = hyper_parameters['lr']
  folder = hyper_parameters['save_address']
  verbose = True
  verbose_step = 1
  step_scheduler = False
  validation_scheduler = True
  SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
  scheduler_params = dict(
      mode = 'min',
      factor = 0.5,
      patience = 2,
      verbose = False, 
      threshold = 0.0001,
      threshold_mode = 'abs',
      cooldown = 0, 
      min_lr = 1e-8,
      eps = 1e-08)
  
  
  
  
  
  
  
# caption_fitter : automatic_mixed_precision

import warnings
import os
import time
from datetime import datetime
from glob import glob
from nltk.translate.bleu_score import corpus_bleu

warnings.filterwarnings('ignore')
scaler = torch.cuda.amp.GradScaler() 

class caption_fitter:
  def __init__(self, model, config, params):
    self.model = model
    self.config = config
    
    self.epoch = 0
    self.base_dir = f'/content/gdrive/My Drive/coco_image_caption/train/{self.config.folder}'
    if not os.path.exists(self.base_dir):
      os.makedirs(self.base_dir)
        
    self.log_path = f'{self.base_dir}/log.txt'
    self.params = params

    self.best_summary_loss = 10 ** 5
    self.best_bleu4 = 0.0
    self.epochs_from_improvement = 0

    self.optimizer = torch.optim.AdamW(self.params, lr = config.lr)
    self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
    self.log(f'caption_fitter prepared. device is {device}')
  
  def fit(self, train_dataloader, valid_dataloader):
    for e in range(self.config.n_epochs):
      if self.epochs_from_improvement == 20:
        break
      if self.epochs_from_improvement > 0 and self.epochs_from_improvement % hyper_parameters['patience'] == 0:
        adjust_lr(self.optimizer, hyper_parameters['shrink_factor'])

      if self.config.verbose:
        lr = self.optimizer.param_groups[0]['lr']
        timestamp = datetime.utcnow().isoformat()
        self.log(f'\n{timestamp}\nLR: {lr}')

      t = time.time()
      summary_loss, topk_accuracy = self.train_function(train_dataloader)

      self.log(f'[RESULT]: train. epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, topk_accuracy: {topk_accuracy.avg: .5f}, time: {(time.time() - t):.5f}')
      self.save(f'{self.base_dir}/last-checkpoint.bin')

      t = time.time()
      summary_loss, topk_accuracy, bleu = self.valid_function(valid_dataloader)
      
      bleu1, bleu2, bleu3, bleu4 = bleu['bleu1'], bleu['bleu2'], bleu['bleu3'], bleu['bleu4']
      self.log(f'[RESULT]: valid. epoch: {self.epoch},' + 
               f'summary_loss: {summary_loss.avg:.5f},' +
               f'topk_accuracy: {topk_accuracy.avg: .5f},' +
               f'bleu1: {bleu1:.5f},' +
               f'bleu2: {bleu2:.5f},' +
               f'bleu3: {bleu3:.5f},' +
               f'bleu4: {bleu4:.5f},' +
               f'time: {(time.time() - t):.5f}')
      
      if summary_loss.avg < self.best_summary_loss:
        self.best_summary_loss = summary_loss.avg
      
      if bleu['bleu4'] > self.best_bleu4:
        self.best_bleu4 = bleu['bleu4']
        self.epochs_from_improvement = 0  

        ##self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
        ##for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
        ##  os.remove(path)

      else:
        self.epochs_from_improvement += 1

      if self.config.validation_scheduler:
        self.scheduler.step(metrics = summary_loss.avg)

      self.model.eval()
      self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
      for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
        os.remove(path)
        
      self.epoch += 1

  def valid_function(self, valid_dataloader):
    self.model.eval()

    summary_loss = averagemeter()
    topk_accuracy = averagemeter()
    
    t = time.time()
    valid_book = tqdm(valid_dataloader, total = len(valid_dataloader))

    references = list()
    hypothesis = list()
    for step, (images, input_ids, targets, all_caps) in enumerate(valid_book):
      with torch.no_grad():
        batch = torch.stack(images).shape[0]
        
        with torch.cuda.amp.autocast(): 
          loss, scores = self.model(torch.stack(images).to(device), 
                                    torch.stack(input_ids).to(device), 
                                    torch.stack(targets).to(device)) 

        summary_loss.update(loss.detach().item(), batch)

        topk = accuracy(scores = scores.reshape(-1, hyper_parameters['vocab_dim']), 
                        targets = torch.stack(targets).reshape(-1), 
                        k = hyper_parameters['topk'])
        topk_accuracy.update(topk, batch * hyper_parameters['max_len'])

        references.extend(all_caps)

        samples = scores.argmax(2) # (N, max_len)
        for i in range(batch):
          if hyper_parameters['tokenizer'] == nltk_tokenizer:
            indices = torch.where(samples[i] > 3)[0]
            hypothesis.extend([' '.join([hyper_parameters['tokenizer'].decode(token) for token in samples[i][indices].long().tolist()])])
          elif hyper_parameters['tokenizer'] == gpt2_tokenizer:
            indices = torch.where(samples[i] < 50257)[0]
            hypothesis.extend([hyper_parameters['tokenizer'].decode(samples[i][indices].long().tolist())])           
        
        assert len(references) == len(hypothesis)
    
    bleu = {
        'bleu1': round(corpus_bleu(__ref2word__(references), __hyp2word__(hypothesis), weights=(1, 0, 0, 0)), 4),
        'bleu2': round(corpus_bleu(__ref2word__(references), __hyp2word__(hypothesis), weights=(0.5, 0.5, 0, 0)), 4),
        'bleu3': round(corpus_bleu(__ref2word__(references), __hyp2word__(hypothesis), weights=(0.33, 0.33, 0.33, 0)), 4),
        'bleu4': round(corpus_bleu(__ref2word__(references), __hyp2word__(hypothesis), weights=(0.25, 0.25, 0.25, 0.25)), 4)
    } 

    return summary_loss, topk_accuracy, bleu

  def train_function(self, train_dataloader):
    self.model.train()

    summary_loss = averagemeter()
    topk_accuracy = averagemeter()

    t = time.time()
    train_book = tqdm(train_dataloader, total = len(train_dataloader))
    for step, (images, input_ids, targets) in enumerate(train_book):
      batch = torch.stack(images).shape[0]
     
      self.optimizer.zero_grad()

      torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.params), hyper_parameters['grad_clip'])   
      
      with torch.cuda.amp.autocast(): 
        loss, scores = self.model(torch.stack(images).to(device), 
                                  torch.stack(input_ids).to(device),
                                  torch.stack(targets).to(device))  

      scaler.scale(loss).backward()  
      #clip_gradient(self.optimizer, hyper_parameters['grad_clip'])      
      #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.params), hyper_parameters['grad_clip'])   
      scaler.step(self.optimizer) 
      scaler.update()

      summary_loss.update(loss.detach().item(), batch)

      topk = accuracy(scores = scores.reshape(-1, hyper_parameters['vocab_dim']), 
                      targets = torch.stack(targets).reshape(-1), 
                      k = hyper_parameters['topk'])
      topk_accuracy.update(topk, batch * hyper_parameters['max_len'])

      if self.config.step_scheduler:
          self.scheduler.step()

    return summary_loss, topk_accuracy
    
  def save(self, path):
    self.model.eval()
    torch.save({
        'model_state_dict': self.model.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
        'best_summary_loss': self.best_summary_loss,
        'epoch': self.epoch,
    }, path)

  def load(self, path):
    checkpoint = torch.load(path)
    self.model.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    self.best_summary_loss = checkpoint['best_summary_loss']
    self.epoch = checkpoint['epoch'] + 1
      
  def log(self, message):
    if self.config.verbose:
      print(message)
    with open(self.log_path, 'a+') as logger:
      logger.write(f'{message}\n')
      
      
      
      
      
      
 # caption_runner

from torch.utils.data.sampler import SequentialSampler, RandomSampler

def caption_runner():
  net.to(device)

  train_dataloader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size = caption_config.batch_size,
      sampler = RandomSampler(train_dataset),
      pin_memory = False,
      drop_last = True,
      num_workers = caption_config.num_workers,
      collate_fn = collate_fn)
  
  valid_dataloader = torch.utils.data.DataLoader(
      valid_dataset, 
      batch_size = caption_config.batch_size,
      num_workers = caption_config.num_workers,
      shuffle = False,
      sampler = SequentialSampler(valid_dataset),
      pin_memory = False,
      collate_fn = collate_fn)

  fitter = caption_fitter(model = net, config = caption_config, params = caption_params)
  fitter.fit(train_dataloader, valid_dataloader)
