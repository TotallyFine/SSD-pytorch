## SSD Pytorch

SSD300的代码，论文地址[Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)。代码主要是参考了[pytorch-ssd](https://github.com/kuangliu/pytorch-ssd)，原版的代码不能直接运行在pytorch 0.4.0上，以及它还有一些其他的问题，我都改过了，并且加上了我的所有注释和理解，可以直接运行。

环境安装好之后，下载VOC2012数据集，用utils/process\_voc.py处理VOC2012训练集和测试集的标注，生成voc12\_test.txt voc\_train.txt放在data/下。

然后下载预训练的[VGG模型](https://download.pytorch.org/models/vgg16-397923af.pth)，使用utils/process\_vgg.py来生成加载了部分VGG参数的SSD原始模型，并将模型放在model/下。

### 环境
 - cuda 8.0
 - cudnn 7.1.2
 - anaconda2
 - numpy 1.14.3
 - pytorch 0.4.0
 - torchvision 0.2.1
 - python 2.7
