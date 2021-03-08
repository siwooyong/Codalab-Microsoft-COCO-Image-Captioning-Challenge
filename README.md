# Easy_Image_Caption_using_Transformer_Pytorch
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

* In order to get the many training data here, I followed the karpathy split.
I used 118287 training data and 5000 valid data. Karpathy split data is available on the coco dataset site.





/





## vocab
As a vocabulary for embeddedding. I tried using gpt2 (50,257 tokens) and Bert (30,232 tokens), but this required a relatively large amount of computation and was slow at learning, so I created vocab_dict separately.(See vocab.py for this.)

I selected frequently used words from the coco annotation data and proceeded with encoding.(I selected 15,000 tokens.)



** After a number of later studies, pretrained gpt2 embedding layer performed best.












