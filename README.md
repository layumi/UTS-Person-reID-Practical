# UTS-Person-reID-Practical
By Zhedong Zheng

This is a [University of Technology Sydney](https://www.uts.edu.au) computer vision practical, authored by Zhedong Zheng.

![](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/demo.png)

Person re-ID can be viewed as an image retrieval problem. Given one query image in Camera **A**, we need to find the images of the same person in other Cameras. The key of the person re-ID is to find a discriminative representation of the person. Many recent works apply deeply learned model to extract visual features, and achieve the state-of-the-art performance.

This practical explores the basic of learning pedestrian features. In this pratical, we will learn to build a simple person re-ID system step by step.

## Prerequisites
- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+ (http://pytorch.org/)
- Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```

## Getting started
Check the Prerequisites. The download links for this practical are:

- Code: [Practical-Baseline](https://github.com/layumi/Person_reID_baseline_pytorch/archive/master.zip)
- Data: [Market-1501](http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip)

## Part 1: Training
### Part 1.1: Prepare Data Folder
You may notice that the downloaded folder is organized as:
```
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* We do not use it 
│   ├── gt_query/                   /* Files for multiple query testing 
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
```
Open and edit the script `prepare.py` in the editor. Change the fifth line in `prepare.py` to your download path, such as `\home\zzd\Download\Market`. Run this script in the terminal.
```bash
python prepare.py
```
After runining, we create a subfolder called `pytorch` under the download folder. 
```
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* We do not use it 
│   ├── gt_query/                   /* Files for multiple query testing 
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
│   ├── pytorch/
│       ├── train/                   /* train 
│           ├── 0002
|           ├── 0007
|           ...
│       ├── val/                     /* val
│       ├── train_all/               /* train+val      
│       ├── query/                   /* query files  
│       ├── gallery/                 /* gallery files  
```

In every subdir, such as `pytorch/train/0002`, images with the same ID are arranged in the folder.
Now we have successfully prepared the data for `torchvision` to read the data. 

```diff
+ Quick Question. How to recognize the images of the same ID?
```

### Part 1.2: Build Neural Network
We can use the pretrained networks, such as `AlexNet`, `VGG16`, `ResNet` and `DenseNet`. Generally, the pretrained networks help to achieve a better performance, since it perserves some good visual patterns from ImageNet[1].

In pytorch, we can easily import them by two lines. For example,
```python
from torchvision import models
model = models.resnet50(pretrained=True)
```

But we need to modify the networks a little bit.

### Part 1.3: Training


## Part 2: Extracting feature

## Part 3: Evaluation

[1] Deng, Jia, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "Imagenet: A large-scale hierarchical image database." In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pp. 248-255. Ieee, 2009.


