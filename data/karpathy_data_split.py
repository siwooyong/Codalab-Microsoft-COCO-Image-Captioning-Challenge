# data split : train_coco_captions, valid_coco_captions

import glob2

root = ['train'] * len(train_coco_captions)
train_coco_captions['root'] = root

root = ['valid'] * len(valid_coco_captions)
valid_coco_captions['root'] = root

!wget http://images.cocodataset.org/zips/val2017.zip
!unzip val2017.zip

val_addresses = glob2.glob('./val2017/*.jpg')

val_list = []
for i in range(len(val_addresses)):
  val_address = val_addresses[i][16:-4]
  if len(val_address) != 6:
    print(i)
  val_list.append(val_address)

karpathy_val = []
karpathy_notval = []

for i in range(len(valid_coco_captions)):
  if i % 10000 == 0:
    print(i)
  if __mk_idform__(str(valid_coco_captions['image_id'][i])) in val_list:
    karpathy_val.append(i)
  else:
    karpathy_notval.append(i)

karpathy_train_captions = pd.concat([train_coco_captions, valid_coco_captions.iloc[karpathy_notval]], axis = 0)
karpathy_valid_captions = valid_coco_captions.iloc[karpathy_val]

karpathy_train_captions = karpathy_train_captions.reset_index(drop = True)
karpathy_valid_captions = karpathy_valid_captions.reset_index(drop = True)

karpathy_train_captions = karpathy_train_captions.to_dict()
with open('/content/gdrive/My Drive/coco_image_caption/data/karpathy_train_captions.pickle','wb') as fw:
  pickle.dump(karpathy_train_captions, fw)

karpathy_valid_captions = karpathy_valid_captions.to_dict()
with open('/content/gdrive/My Drive/coco_image_caption/data/karpathy_valid_captions.pickle','wb') as fw:
  pickle.dump(karpathy_valid_captions, fw)

with open('/content/gdrive/My Drive/coco_image_caption/data/karpathy_train_captions.pickle', 'rb') as fr:
  karpathy_train_captions = pickle.load(fr)
karpathy_train_captions = pd.DataFrame(karpathy_train_captions)

with open('/content/gdrive/My Drive/coco_image_caption/data/karpathy_valid_captions.pickle', 'rb') as fr:
  karpathy_valid_captions = pickle.load(fr)
karpathy_valid_captions = pd.DataFrame(karpathy_valid_captions)

train_captions = karpathy_train_captions
valid_captions = karpathy_valid_captions
