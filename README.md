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
### Part 1.1: Prepare Data
You may notice that the downloaded folder is organized as:
```

```
Open and edit the script `prepare.py` in the editor. Change the fifth line in `prepare.py` to your download path. This script contains the code to prepare the data for the training.

## Part 2: Extracting feature

## Part 3: Evaluation


