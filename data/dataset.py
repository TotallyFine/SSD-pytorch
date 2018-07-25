# coding:utf-8
from __future__ import print_function
import os
import torch
from torch.utils.data import Dataset

from .Box import PriorBox

from PIL import Image
import random # for data argumentation

class ImageSet(Dataset):
    # 对于每个batch中的图片，产生这个图片中的ground truth对应的prior box，以及对应的laabel
    def __init__(self, opt, transform, is_train):
        self.img_size = opt.img_size # 300 300
        self.transform = transform

        # 如果是train阶段的话进行Crop等
        self.is_train = is_train
        # 存放文件的路径
        if self.is_train:
            self.data_path = opt.train_data_path
            label_file = opt.train_label_file
        else:
            self.data_path = opt.test_data_path
            label_file = opt.test_label_file
        self.fnames = []
        self.boxes = []
        self.labels = []

        # 构造default box，进行match nms等
        self.prior_box = PriorBox(opt)

        with open(label_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            # 一行（一张图片）可能会有多个物体
            num_objs = int(splited[1])
            box = []
            label = []
            for i in range(num_objs):
                xmin = splited[2+5*i]
                ymin = splited[3+5*i]
                xmax = splited[4+5*i]
                ymax = splited[5+5*i]
                c = splited[6+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box)) # (img_num, num_obj, 4)
            self.labels.append(torch.LongTensor(label)) # (img_num, num_obj, 1)
        
    def __getitem__(self, index):
        """
        加载一个图片到神经网络中去，同时产生default box
        以及每个default box最匹配的label的class
        """
        fname = self.fnames[index]
        img = Image.open(os.path.join(self.data_path, fname))
        # (num, 4)
        boxes = self.boxes[index].clone()
        labels = self.labels[index]

        # 在进行数据增强的时候，label box还没有被缩减到0-1之间
        if self.is_train:
            img,boxes = self.random_flip(img, boxes)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        # 把xmin ymin xmax ymax缩减到0-1之间
        # 这里得到的是w,h 但是在feature map中一般都是(batch_size, channel, h, w)
        # 因为内层是一行一行排起来的
        w, h = img.size
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)

        # rescale
        img = img.resize((self.img_size, self.img_size))
        img = self.transform(img)

        # 产生default box并进行对应
        loc_target, conf_target = self.prior_box.match(boxes, labels)
        return img, loc_target, conf_target
    
    def __len__(self):
        return len(self.fnames)

    def random_flip(self, img, boxes):
        """
        img: PIL.Image
        boxes: tensor (num, 4) 还未缩减到0-1之间
        随机进行左右翻转
        """
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
        return img, boxes

    def random_crop(self, img, boxes, labels):
        """
        img: PIL.Image
        boxes: tensor, (obj_num, 4) 还未缩减到0-1之间
        labels: tensor, (obj_num,)
        对图片进行随机的crop
        """
        imw, imh = img.size
        while True:
            # 随机一个crop后
            min_iou = random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
            # 如果随机到了None就不进行crop
            if min_iou is None:
                return img, boxes, labels

            # 如果循环了100次都没有好的结果就不再进行crop
            for _ in range(100):
                w = random.randrange(int(0.1*imw), imw)
                h = random.randrange(int(0.1*imh), imh)
                # 如果随机出来的h w不合理则再进行循环尝试
                if h > 2*w or w > 2*h:
                    continue

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)
                # region of interest 随机出来的新的图片在原图上的位置
                roi = torch.Tensor([[x, y, x+w, y+w]])

                # 得到每个object的中心位置
                center = (boxes[:, :2] + boxes[:, 2:]) / 2 # (num, 2)
                # 把roi进行重复，重复到len(center)个
                roi2 = roi.expand(len(center), 4) # (num, 4)
                # center > roi2[:, :2]得到一个bool矩阵，每个object的
                # center是否大于随机出来的框的xmax ymax
                # center < roi2[:, 2:]得到一个bool矩阵，每个object的
                # center是否小于随机出来的框的xmin ymin
                # 两个布尔矩阵再进行与操作，得到哪些object的x y不在随机出的框内
                mask = (center > roi2[:, :2]) & (center < roi2[:, 2:]) # (num, 2)
                mask = mask[:, 0] & mask[:, 1] # (num,)
                # 如果有所有object的中心落在了随机出来的框的外面则舍弃这次随机的结果再次循环
                if not mask.any():
                    continue

                # 从boxes中选出对应的center没有落在随机出来的框之外的box
                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))

                # 计算center落在框内的box和框的iou，如果iou太小则重新进行随机
                iou = self.prior_box.iou(selected_boxes, roi)
                if iou.min() < min_iou:
                    continue

                # 将图片进行crop
                img = img.crop((x, y, x+w, y+h))
                # 将center落在随机出来的框的box的边界缩减到框内
                selected_boxes[:,0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:,2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,3].add_(-y).clamp_(min=0, max=h)
                return img, selected_boxes, labels[mask]
