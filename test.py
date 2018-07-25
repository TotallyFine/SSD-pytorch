# coding:utf-8

import torch
import torchvision.transforms as tfs
import torch.nn.functional as F


from model import SSD
from data import PriorBox
from config import opt

from PIL import Image, ImageDraw

net = SSD(opt)
net.load_state_dict(torch.load('checkpoints/...'))
net.eval()

# 加载测试图片
img = Image.open('...')
img1 = img.resize((300, 300))
transform = tfs.Compose([tfs.ToTensor(), tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

img1 = transform(img1)

# 前向传播
loc, conf = net(img1[None, :, :, :])

# 将数据转换格式
prior_box = PriorBox(opt)
boxes, labels, scores = prior_box.convert_result(loc.data.sequeeze(0), F.softmax(conf.squeeze(0)).data)

draw = ImageDraw.Draw(img)
for box in boxes:
    box[::2] *= img.width
    box[1::2] *= img.height
    draw.rectangle(list(box), online='red')
img.show()
