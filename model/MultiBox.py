# coding:utf-8
import torch
import torch.nn as nn


class MultiBoxLayer(nn.Module):
    """
    对多个不同scaling的特征进行提取位置和类别
    """
    
    def __init__(self, opt):
        super(MultiBoxLayer, self).__init__()
        
        self.num_classes = opt.num_classes
        # 每个in_plane都有一个num_anchor进行对应，不同的anocher表示的是用不同的aspect ratio
        self.num_anchors = opt.num_anchors
        self.in_planes = opt.in_planes
        
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        
        for i in range(len(self.in_planes)):
            # 第i个in_planes有num_anchors[i]个aspect ratio，每个ratio都有4个数字表示 centerX centerY w h
            # 这里输出每个位置的类别概率和box，不改变feature map的大小
            self.loc_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*self.num_classes, kernel_size=3, padding=1))
            
    def forward(self, xs):
        """
        xs: list，提取出来的特征构成的list，xs[i]的channel = self.in_planes[i]
        return: loc_preds, (batch_size, num_anchors*m*n, 4)预测的box
                conf_preds, (batch_size, num_anchors*m*n, num_classes)预测的概率
        """
        y_locs = []
        y_confs = []
        # 遍历每个in_plane进行提取位置和类别的概率
        for i, x in enumerate(xs):
            y_loc = self.loc_layers[i](x)
            batch_size = y_loc.size(0) # int
            # (batch_size, m, n, anchor*4)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            # 要先把4放到最后，然后再改变shape 变成(batch_size, anchor_all_number, 4)
            y_loc = y_loc.view(batch_size, -1, 4)
            y_locs.append(y_loc)
            
            y_conf = self.conf_layers[i](x)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(batch_size, -1, self.num_classes)
            y_confs.append(y_conf)
            
        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)
        return loc_preds, conf_preds
