# UTS-Person-reID-Practical
By [Zhedong Zheng](http://zdzheng.xyz/)

This is a [University of Technology Sydney](https://www.uts.edu.au) computer vision practical, authored by Zhedong Zheng.

![](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/show.png)

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
### Part 1.1: Prepare Data Folder (`prepare_data.py`)
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
We create a subfolder called `pytorch` under the download folder. 
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

### Part 1.2: Build Neural Network (`model.py`)
We can use the pretrained networks, such as `AlexNet`, `VGG16`, `ResNet` and `DenseNet`. Generally, the pretrained networks help to achieve a better performance, since it perserves some good visual patterns from ImageNet[1].

In pytorch, we can easily import them by two lines. For example,
```python
from torchvision import models
model = models.resnet50(pretrained=True)
```
You can simply check the structure of the model by:
```python
print(model)
```

But we need to modify the networks a little bit. There are 751 classes (different people) in Market-1501, which is different with 1,000 classes in ImageNet. So here we change the model to use our classifier.

```python
import torch
import torch.nn as nn
from torchvision import models

# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num = 751):
        super(ft_net, self).__init__()
        #load the model
        model_ft = models.resnet50(pretrained=True) 
        # change avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num) #define our classifier.

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x) #use our classifier.
        return x
```

```diff
+ Quick Question. Why we use AdaptiveAvgPool2d? What is the difference between the AvgPool2d and AdaptiveAvgPool2d?
+ Quick Question. Does the model have parameters now? How to intialize the parameter in the new layer?
```
More details are in `model.py`. You may check it later, after you have gone through this practical.

### Part 1.3: Training (`train.py`)
OK. Now we have prepared the training data and defined model structure.
Before we start training, the last thing is how to read data and their labels from the prepared folder.
Using `torch.utils.data.DataLoader`, we can obtain two iterators `dataloaders['train']` and `dataloaders['val']` to read data and label.
```python
image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8) # 8 workers may work faster
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
```

Let's train the model.
Yes. It's only about 20 lines. Make sure you can understand every line of the code.
```python
            # Iterate over data.
            for data in dataloaders[phase]:
                # get a batch of inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable, if gpu is used, we transform the data to cuda.
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                #-------- forward --------
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                #-------- backward + optimize -------- 
                # only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
```
```diff
+ Quick Question. Why we need optimizer.zero_grad()? What happens if we remove it?
+ Quick Question. The dimension of the outputs is batchsize*751. Why?
```
Every 10 training epoch, we save a snapshot and update the loss curve.
```python
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)
```

## Part 2: Extracting feature (`test.py`)
In this part, we load the network (we just trained) to extract the visual feature of every image.
```python

```


## Part 3: Evaluation

[1] Deng, Jia, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "Imagenet: A large-scale hierarchical image database." In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pp. 248-255. Ieee, 2009.


