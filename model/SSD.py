# coding:utf-8
from torch import nn
import torch.nn.functional as F
from .MultiBox import MultiBoxLayer
import math

class L2Norm2d(nn.Module):
    """
    在所有的channel上进行L2Norm
    当从vgg中提取出特征以后需要对其进行L2Norm
    不过是针对没有Batch Norm的情况
    """
    def __init__(self, scale):
        super(L2Norm2d, self).__init__()
        self.scale = scale
    
    def forward(self, x, dim=1):
        """
        out = scale * x / sqrt(sum x_i^2)
        Tensor.sum(dim)沿着这个维度进行求和，这个维度会缩减消失，假如原来是三维的w h z 沿着高z进行求和，则输出w h
        Tensor.clamp(max, min)将数据缩紧到[min, max]的范围内，这里是为了不除0
        Tensor.rsqrt() = 1/sqrt(Tensor)对每个维度每个元素都进行这样的操作
        Tensor.expand_as(another Tensor) 将Tensor进行扩展
        """

        return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().unsqueeze(1).expand_as(x)


def conv3x3(in_channels, out_channels):
    """
    in_channels: 输入的channal数
    out_channels: 输出的channel数
    s: int, stride 步长
    这个函数进行3X3卷积，不改变feature map的尺寸只改变channel数
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True)
    )

class VGG16(nn.Module):
    """
    基础的提取体征的网络vgg16，这里去掉了最后的classifier及倒数第一个maxpool
    输入的图片大小是300X300
    中间的Conv全都是不会使feature map大小改变的3X3Conv
    然后中间有五个MaxPool每次feature map的大小减半
    所以最终输出的feature map为(batch_size, 512, 38, 38)
    """

    def __init__(self):

        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            conv3x3(3, 64),
            conv3x3(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 64, 150, 150)
            
            conv3x3(64, 128),
            conv3x3(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 128, 75, 75)
            
            conv3x3(128, 256),
            conv3x3(256, 256),
            conv3x3(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 256, 38, 38)
            
            conv3x3(256, 512),
            conv3x3(512, 512),
            conv3x3(512, 512),# (batch_size, 512, 38, 38)
        )
        # if init_weights: 改为在SSD中进行初始化
        #    self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

class SSD(nn.Module):
    def __init__(self, opt, init_weights=True):
        """
        num_classes: int, 类别的个数
        backbone: nn.Module, 提取基本特征的模型
        init_weight: boolean, 是否进行初始化
        """
        super(SSD, self).__init__()
        self.multibox = MultiBoxLayer(opt)
        self.num_classes = opt.num_classes
        self.base = self.VGG16() # # (batch_size, 512, 38, 38)
        self.norm4 = L2Norm2d(20)
        # fc都属于feature extra layer用来得到不同scaling的feature
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1) # maxpool 在forward中
        
        # (H + 2*p - d(ks - 1) - 1) / 2 + 1
        # (38 + 12 - 6*(3 - 1) -1 ) / 2 + 1 = 19.5 向下取整 19
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)# (batch_size, 1024, 19, 19)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)# (batch_size, 1024, 19, 19)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)# (batch_size, 512, 10, 10)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)# (batch_size, 256, 5, 5)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)# (batch_size, 256, 3, 3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)# (batch_size, 256, 3, 3)
        
        
        if init_weights:
            self._initialize_weights()
            
    def forward(self, x):
        hs = []
        feature = self.base(x)
        hs.append(self.norm4(feature))
        
        h = F.max_pool2d(feature, kernel_size=2, stride=2, ceil_mode=True)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)  # conv7

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)  # conv8_2

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)  # conv9_2

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)  # conv10_2

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)  # conv11_2

        loc_preds, conf_preds = self.multibox(hs)
        return loc_preds, conf_preds
        
    def VGG16(self):
        '''VGG16 layers.'''
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        """
        对网络进行初始化
        """
        for m in self.modules():
            # 卷积的初始化方法
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # bias都初始化为0
                if m.bias is not None:
                    m.bias.data.zero_()
            # batchnorm使用全1初始化 bias全0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
