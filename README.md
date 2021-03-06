# Codalab-Microsoft-COCO-Image-Captioning-Challenge




## Getting started
This repository is based on many image captioning models like 
 * [Show-and-Tell-A-Neural-Image-Caption-Generator](https://arxiv.org/pdf/1411.4555.pdf), 
 * [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf), 
 * [Bottom-Up and Top-Down Attention for Image Captioning and Visual QuestionAnswering](https://arxiv.org/pdf/1707.07998.pdf),
 * [Meshed-Memory Transformer for Image Captioning](https://arxiv.org/pdf/1912.08226.pdf) 
 * [Self-critical Sequence Training for Image Captioning](https://arxiv.org/pdf/1612.00563.pdf)
and so on.

I develpoed the simplest and better performance model.









## Coco_dataset: data prepare(karpathy split)
* The coco data consists of 80k train images, 40k valid images, and 40k test images. Here, I did not use test data, but trained on 80k images, and only did validation on 40k images.


download images here : ['train_coco_images2014'](http://images.cocodataset.org/zips/train2014.zip), ['valid_coco_images2014'](http://images.cocodataset.org/zips/val2014.zip), ['test_coco_images2014'](http://images.cocodataset.org/zips/test2014.zip)

* download caption annotation here : (http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

* In order to get many training data, I followed the karpathy split.
I used 118287 training data and 5000 valid data. Karpathy split data is available on the coco dataset site.











## Vocab
As a vocabulary for embeddedding. I tried using gpt2 (50,257 tokens) and Bert (30,232 tokens), but this required a relatively large amount of computation and was slow at learning, so I created vocab_dict separately.(See vocab.py for this.)

I selected frequently used words from the coco annotation data and proceeded with encoding.(I selected 15,000 tokens.)



** After a number of later studies, I realized that pretrained gpt2 embedding layer performed better.(check model.py)











## Encoder : CLIP
I used CLIP as an encoder. At the beginning of training, we did not include encoders (resnet) in trainable params, but later re-training by including encoders parameters for the trained capture models showed improved performance.(fine-tuned)











## Decoder : gpt2 --> base_model.py
* The decoder structure is the simplest structure, but I used one trick. The image input was separated into several tokens and put into the gpt2 hidden layer. This means that 1 image tokens, along with 20 word tokens (N, 21, 768) are input to gpt2.

* Of course, there is no label for image token, so the loss function contains the latter 20 (N, 20, 768) of the (N, 21, 768).












## Research 
To achieve good performance, modern image captioning models use image detection by default. However, this makes it difficult for users with poor gpu environment to implement.
Therefore, I made various attempts to obtain a good model with less gpu.

1. tagging model
* The input of the image capture model: a word anchor as well as an image. 
I want to conduct another training on tag using various models such as cnn, lstm, etc.


* example )
model_input : '[dog] [bark]', 'INPUT_IDS', 'IMAGE'
(where <[dog] [bark]> corresponds to tag.)











## Ways to increase performance
* First, 'beam search'
* Second, 'CIDEr optimization'
* Third, 'Ensemble'
* Fourth, 'using random labels'
where random labels are selected as random from five captions. This not only prevents overfitting but also improves performance in evaluations such as bleu and cider.


* Here, I saw the performance improvement using only the fourth method. If all of the first, second, and third methods are used, performance improvement of 1-2 is expected based on bleu4.











## Evaluation for karpathy test: models/base_model.py
with beam_search(beam_search = 5) and self_critical_sequence_training and Ensemble(5 models)
|metric|score|
|---|---|
|BLEU1|0.8305|
|BLEU2|0.6816|
|BLEU3|0.5361|
|BLEU4|0.4158|
|CIDEr|1.3453|
|METEOR|0.2892|
|ROUGE_L|0.5935|











## Evaluation for karpathy test: models/base_model_with_detection.py

Originally, the goal of this project was to develop image captioning model with high performance at low cost. For additional research, I also used image detection features to produce better results. 


with beam_search(beam_search = 5) and self_critical_sequence_training and Ensemble(3 models)
|metric|score|
|---|---|
|BLEU1|0.8420|
|BLEU2|0.6986|
|BLEU3|0.5546|
|BLEU4|0.4336|
|CIDEr|1.4163|
|METEOR|0.2968|
|ROUGE_L|0.6047|

you can download the features from [VinVL: Revisiting Visual Representations in Vision-Language Models](https://github.com/pzzhang/VinVL) 

# ***3rdPlace at COCO Image Caption Challenge***
![result](https://user-images.githubusercontent.com/68500343/124973025-ff088800-e065-11eb-8e95-b1c5a08a1c5f.png)






## References
I got help from [sgrvinod-a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).


## Thank you for reading

