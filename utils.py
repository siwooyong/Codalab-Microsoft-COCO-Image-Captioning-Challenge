# utils: accuracy function, adjust_lr, averagemeter, collate_fn, __ref2word__, __hyp2word__

def accuracy(scores, targets, k):
  batch = targets.size(0)
  _, indices = scores.topk(k, 1, True, True)
  correct = indices.eq(targets.to(device).view(-1, 1).expand_as(indices))
  correct_total = correct.view(-1).float().sum() 
  return correct_total.item() * (100 / batch)

def adjust_lr(optimizer, shrink_factor):
  for param_group in optimizer.param_groups:
    param_group['lr'] = param_group['lr'] * shrink_factor
  print(' adjust_LR is completed')

class averagemeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n = 1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def collate_fn(batch):
  return tuple(zip(*batch))

def __ref2word__(all_refs):
  all_words = []
  for refs in all_refs:
    words = []
    for ref in refs:
      words.append(ref.split())
    all_words.append(words)
  return all_words

def __hyp2word__(hyps):
  words = []
  for hyp in hyps:
    words.append(hyp.split())
  return words

def restart_server():
  server_down_model = ptcv_get_model('efficientnet_b8c')
  server_down_model(torch.Tensor(1, 3, 4096, 4096))

def clip_gradient(optimizer, grad_clip):
  for group in optimizer.param_groups:
    for param in group['params']:
       param.grad.data.clamp_(-grad_clip, grad_clip)
