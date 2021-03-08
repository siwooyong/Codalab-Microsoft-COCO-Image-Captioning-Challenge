# prepare vocabs

import nltk
from collections import Counter
nltk.download('punkt')

class Vocabulary(object):
  def __init__(self):
    self.word2idx = {}
    self.idx2word = {}
    self.idx = 0

  def add_word(self, word):
    if not word in self.word2idx:
      self.word2idx[word] = self.idx
      self.idx2word[self.idx] = word
      self.idx += 1

  def __len__(self):
    return len(self.word2idx)

  def encode(self, word):
    if not word in self.word2idx:
      return self.word2idx['[unk]']
    return self.word2idx[word]

  def decode(self, idx):
    return self.idx2word[idx]

def build_vocab(json = '/content/gdrive/My Drive/captions_train2014.json', 
                threshold = 4, 
                max_words = 15000):
  
  coco = COCO(json)
  counter = Counter()
  ids = coco.anns.keys()
  for i, id in enumerate(ids):
    caption = str(coco.anns[id]['caption'])
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    counter.update(tokens)

    if i % 100000 == 0:
      print('[%d/%d] tokenized the captions.' %(i, len(ids)))

  words = counter.most_common(max_words - 5)
  words = [word for word, cnt in words if cnt >= threshold]

  vocab = Vocabulary()
  vocab.add_word('[pad]')
  vocab.add_word('[start]')
  vocab.add_word('[end]')
  vocab.add_word('[cls]')
  vocab.add_word('[unk]')

  for i, word in enumerate(words):
    vocab.add_word(word)
  print('total number of words in vocab:', vocab.__len__())
  return vocab, words

nltk_tokenizer, words = build_vocab()
