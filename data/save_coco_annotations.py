# save coco_caption annotations : train & valid

import torchvision
from pycocotools.coco import COCO
import pandas as pd
import pickle

!wget http://images.cocodataset.org/zips/train2014.zip
!unzip train2014.zip

!wget http://images.cocodataset.org/zips/val2014.zip
!unzip val2014.zip

train_captions = torchvision.datasets.CocoCaptions(root = 'train2014/',
                                                   annFile = '/content/gdrive/My Drive/captions_train2014.json',
                                                   transform = None,
                                                   target_transform = None,
                                                   transforms = None)

valid_captions = torchvision.datasets.CocoCaptions(root = 'val2014/',
                                                   annFile = '/content/gdrive/My Drive/captions_val2014.json',
                                                   transform = None,
                                                   target_transform = None,
                                                   transforms = None)


train_coco = COCO('/content/gdrive/My Drive/captions_train2014.json')
train_ids = list(sorted(train_coco.imgs.keys()))

valid_coco = COCO('/content/gdrive/My Drive/captions_val2014.json')
valid_ids = list(sorted(valid_coco.imgs.keys()))


train_data = pd.DataFrame()
train_direcs = []
train_imgids = []
train_annotations = []
for i in range(len(train_captions)):
  if i % 10000 == 0:
    print(i)
  direc = i // 2000
  
  train_direcs.append(direc)
  train_imgids.append(train_ids[i])
  train_annotations.append(train_captions[i][1])

train_data['image_id'] = train_imgids
train_data['annotation'] = train_annotations
train_data['directory'] = train_direcs

train_data = train_data.to_dict()
with open('/content/gdrive/My Drive/coco_image_caption/train/train_coco_captions.pickle','wb') as fw:
  pickle.dump(train_data, fw)

print('save train_coco_captions completed')


valid_data = pd.DataFrame()
valid_direcs = []
valid_imgids = []
valid_annotations = []
for j in range(len(valid_captions)):
  if j % 10000 == 0:
    print(j)
  direc = j // 2000

  valid_direcs.append(direc)
  valid_imgids.append(valid_ids[j])
  valid_annotations.append(valid_captions[j][1])

valid_data['image_id'] = valid_imgids
valid_data['annotation'] = valid_annotations
valid_data['directory'] = valid_direcs

valid_data = valid_data.to_dict()
with open('/content/gdrive/My Drive/coco_image_caption/valid/valid_coco_captions.pickle','wb') as fw:
  pickle.dump(valid_data, fw)

print('save valid_coco_captions completed')

with open('/content/gdrive/My Drive/coco_image_caption/train/train_coco_captions.pickle', 'rb') as fr:
  train_coco_captions = pickle.load(fr)
train_coco_captions = pd.DataFrame(train_coco_captions)

with open('/content/gdrive/My Drive/coco_image_caption/valid/valid_coco_captions.pickle', 'rb') as fr:
  valid_coco_captions = pickle.load(fr)
valid_coco_captions = pd.DataFrame(valid_coco_captions)
