# Image_Captioning_Tutorial_using_Transformer_Pytorch
New model to achieve the better performance at a "low cost".  And I hope it will be a tutorial of image capture because I took really easy steps.



## getting started
This repo is based on many image capture models like 
 * [Show-and-Tell-A-Neural-Image-Caption-Generator](https://arxiv.org/pdf/1411.4555.pdf), 
 * [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf), 
 * [Bottom-Up and Top-Down Attention for Image Captioning and Visual QuestionAnswering](https://arxiv.org/pdf/1707.07998.pdf),
 * [Meshed-Memory Transformer for Image Captioning](https://arxiv.org/pdf/1912.08226.pdf) 
and so on.

I develpoed the simplest and better performance model.




/




## coco_dataset: data prepare(karpathy split)
* The coco data consists of 80k train images, 40k valid images, and 40k test images. Here, I did not use test data, but trained on 80k images, and only did validation on 40k images.


download images here : ['train_coco_images2014'](http://images.cocodataset.org/zips/train2014.zip), ['valid_coco_images2014'](http://images.cocodataset.org/zips/val2014.zip), ['test_coco_images2014'](http://images.cocodataset.org/zips/test2014.zip)

* download caption annotation here : (http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

* In order to get many training data, I followed the karpathy split.
I used 118287 training data and 5000 valid data. Karpathy split data is available on the coco dataset site.





/





## vocab
As a vocabulary for embeddedding. I tried using gpt2 (50,257 tokens) and Bert (30,232 tokens), but this required a relatively large amount of computation and was slow at learning, so I created vocab_dict separately.(See vocab.py for this.)

I selected frequently used words from the coco annotation data and proceeded with encoding.(I selected 15,000 tokens.)



** After a number of later studies, pretrained gpt2 embedding layer performed best.(check model.py)





/





## encoder : resnet101
I used resnet101 as an encoder. At the beginning of training, we did not include encoders (resnet) in trainable params, but later re-training by including encoders parameters for the trained capture models showed improved performance.(fine-tuned)






/





## decoder : gpt2
* The decoder structure is the simplest structure, but I used one trick. The image input was separated into several tokens and put into the gpt2 hidden layer. This means that 10 image tokens, along with 20 word tokens (N, 30, 768) are input to gpt2.

* Of course, there is no label for image token, so the loss function contains the latter 20 (N, 20, 768) of the (N, 30, 768).





/






## research 
To achieve good performance, modern image capture models use image detection by default. However, this makes it difficult for users with poor gpu environment to implement.
Therefore, I made various attempts to obtain a good model with less gpu.

1. tagging model
* The input of the image capture model: a word anchor as well as an image. 
I want to conduct another training on tag using various models such as cnn, lstm, etc.


* example )
model_input : '[dog] [bark]', 'INPUT_IDS', 'IMAGE'
(where <[dog] [bark]> corresponds to tag.)



2. another attempts
I wanted to see the image caption task as text -> text, not image -> text
I tried to do a training process that creates arbitrary text and uses image to refine it to the correct answer and is currently in progress.





/





## As a way to increase performance
* First, 'beam search'
* Second, 'CIDEr optimization'
* Third, 'Ensemble'
* Fourth, 'using random labels'
where random labels are selected as random from five captions. This not only prevents overfitting but also improves performance in evaluations such as bleu and cider.


* Here, I saw the performance improvement using only the fourth method. If all of the first, second, and third methods are used, performance improvement of 5-10 is expected based on bleu4.





/




## evaluation
teacher focing with out beam_search(beam_search = 1)
|metric|score|
|---|---|
|BLEU1|0.7016|
|BLEU2|0.5367|
|BLEU3|0.4013|
|BLEU4|0.3036|
|CIDEr|0.7783|
|METEOR|0.2613|
|ROUGE_L|0.5105|



/




## references
I got help from [sgrvinod-a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).


## Thank you for reading

