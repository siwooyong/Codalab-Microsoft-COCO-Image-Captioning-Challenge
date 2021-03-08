# save coco_caption images: train & valid

import os
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def __mk_idform__(id):
  if len(id) == 1:
    id = '00000' + f'{id}'
  elif len(id) == 2:
    id = '0000' + f'{id}'
  elif len(id) == 3:
    id = '000' + f'{id}'
  elif len(id) == 4:
    id = '00' + f'{id}'
  elif len(id) == 5:
    id = '0' + f'{id}'
  else:
    id = id
  return id 

for i in range(42):
  os.mkdir('/content/gdrive/My Drive/coco_image_caption/train/images/coco_part' + f'{i+1}')

for i in range(21):
  os.mkdir('/content/gdrive/My Drive/coco_image_caption/valid/images/coco_part' + f'{i+1}')


class save_train_cocoimage_dataset(torch.utils.data.Dataset):
  def __init__(self, captions):
    self.captions = captions

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, index):
    caption = self.captions.loc[index]

    image_id = caption['image_id']
    image_id = __mk_idform__(f'{image_id}')

    image = cv2.imread('./train2014/COCO_train2014_000000' + f'{image_id}.jpg')

    direc = caption['directory'] + 1
    return direc, image_id, image

save_train_cocoimage_dataset = save_train_cocoimage_dataset(train_coco_captions)

save_train_cocoimage_dataloader = torch.utils.data.DataLoader(
    save_train_cocoimage_dataset,
    batch_size = 1,
    pin_memory = False,
    drop_last = False,
    shuffle = False,
    num_workers = 1)

save_train_cocoimage_book = tqdm(save_train_cocoimage_dataloader, 
                                 total = len(save_train_cocoimage_dataloader))

for step, data in enumerate(save_train_cocoimage_book):
  direc, image_id, image = data
  _ = cv2.imwrite('/content/gdrive/My Drive/coco_image_caption/train/images/coco_part' 
                  + f'{direc.tolist()[0]}/' + f'{image_id[0]}.jpg', np.array(image[0]))
  

class save_valid_cocoimage_dataset(torch.utils.data.Dataset):
  def __init__(self, captions):
    self.captions = captions

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, index):
    caption = self.captions.loc[index]

    image_id = caption['image_id']
    image_id = __mk_idform__(f'{image_id}')

    image = cv2.imread('./val2014/COCO_val2014_000000' + f'{image_id}.jpg')

    direc = caption['directory'] + 1
    return direc, image_id, image

save_valid_cocoimage_dataset = save_valid_cocoimage_dataset(valid_coco_captions)

save_valid_cocoimage_dataloader = torch.utils.data.DataLoader(
    save_valid_cocoimage_dataset,
    batch_size = 1,
    pin_memory = False,
    drop_last = False,
    shuffle = False,
    num_workers = 1)

save_valid_cocoimage_book = tqdm(save_valid_cocoimage_dataloader, 
                                 total = len(save_valid_cocoimage_dataloader))

for step, data in enumerate(save_valid_cocoimage_book):
  direc, image_id, image = data
  _ = cv2.imwrite('/content/gdrive/My Drive/coco_image_caption/valid/images/coco_part' 
                  + f'{direc.tolist()[0]}/' + f'{image_id[0]}.jpg', np.array(image[0]))
