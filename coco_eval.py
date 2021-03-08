# coco_eval

import nlgeval
from nlgeval import NLGEval
nlgeval = NLGEval()

def coco_eval(model, dataloader):
  #model.eval()

  references = list()
  hypothesis = list()

  data_tqdm = tqdm(dataloader, total = len(dataloader))
  for step, (images, input_ids, targets, all_caps) in enumerate(data_tqdm):
    with torch.no_grad():
      batch = torch.stack(images).shape[0]

      with torch.cuda.amp.autocast(): 
        scores = model(torch.stack(images).to(device))
      
      references.extend(all_caps)

      samples = scores.argmax(2) 
      for i in range(batch):
        if hyper_parameters['tokenizer'] == nltk_tokenizer:
          indices = torch.where((samples[i] > 3) & (samples[i] != 5))[0]
          hypothesis.extend([' '.join([hyper_parameters['tokenizer'].decode(token) for token in samples[i][indices].long().tolist()]) + '.'])
        elif hyper_parameters['tokenizer'] == gpt2_tokenizer:
          indices = torch.where((samples[i] < 50257) & (samples[i] != 13))[0]
          hypothesis.extend([hyper_parameters['tokenizer'].decode(samples[i][indices].long().tolist()) + '.'])          
      assert len(references) == len(hypothesis)
  
  return references, hypothesis


valid_dataloader = torch.utils.data.DataLoader(
      valid_dataset, 
      batch_size = 16,
      num_workers = caption_config.num_workers,
      shuffle = False,
      sampler = SequentialSampler(valid_dataset),
      pin_memory = False,
      collate_fn = collate_fn)

references, hypothesis = coco_eval(caption_net.__sample__, valid_dataloader)

for i in range(len(references)):
  for j in range(len(references[i])):
    if '\n' in references[i][j]:
      references[i][j] = references[i][j].replace('\n', '')

bleu = {
        'bleu1': round(corpus_bleu(__ref2word__(references), __hyp2word__(hypothesis), weights=(1, 0, 0, 0)), 4),
        'bleu2': round(corpus_bleu(__ref2word__(references), __hyp2word__(hypothesis), weights=(0.5, 0.5, 0, 0)), 4),
        'bleu3': round(corpus_bleu(__ref2word__(references), __hyp2word__(hypothesis), weights=(0.33, 0.33, 0.33, 0)), 4),
        'bleu4': round(corpus_bleu(__ref2word__(references), __hyp2word__(hypothesis), weights=(0.25, 0.25, 0.25, 0.25)), 4)
    } 


nlgeval = NLGEval()
nlgeval.compute_metrics(references, hypothesis)
