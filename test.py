# coding:utf-8
from __future__ import print_function
import torch
import torchvision.transforms as tfs
import torch.nn.functional as F

from model import SSD
from data import PriorBox
from config import opt

from PIL import Image, ImageDraw

net = SSD(opt)
net.load_state_dict(torch.load(opt.ckpt_path)['net'])
net.eval()

# 加载测试图片
img = Image.open('/home/j/MYSSD/pytorch-ssd-master/image/img1.jpg')
img1 = img.resize((300, 300))
transform = tfs.Compose([tfs.ToTensor(), tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

img1 = transform(img1)

# 前向传播
loc, conf = net(img1[None, :, :, :])

# 将数据转换格式
prior_box = PriorBox(opt)
# squeeze是把batch_size那一层去掉
boxes, labels, scores = prior_box.convert_result(loc.squeeze(0), F.softmax(conf.squeeze(0), dim=0))

draw = ImageDraw.Draw(img)
for box in boxes:
    box[::2] *= img.width
    box[1::2] *= img.height
    draw.rectangle(list(box), outline='red')
img.show()
