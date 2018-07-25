# coding:utf-8
from __future__ import print_function
import os
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as tfs
# from torch.autograd import Variable

from model import SSD, MultiBoxLayer, MultiBoxLoss
from data import ImageSet
from config import opt

parser = argparse.ArgumentParser(description='SSD Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_loss = float('inf')
start_epoch = 0 # 从0开始或者从上一个epoch开始

print('## Preparing data ##')
# 对图片的变换
transform = tfs.Compose([tfs.ToTensor(),
    tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
# 构建trainset trainloader testset testloader
trainset = ImageSet(opt, transform, is_train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8)

testset = ImageSet(opt, transform, is_train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=8)
print('## Data preparation finish ##')

print('## Building net : SSD300 ##')
net = SSD(opt)
# 是否加载之前保存的模型
if args.resume:
    print(' # Resuming from checkpoint # ')
    checkpoint = torch.load(opt.ckpt_path)
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
# 加载预训练的模型
else:
    print(' # Loading pretrained model # ')
    net.load_state_dict(torch.load(opt.pretrained_model))

criterion = MultiBoxLoss()

if use_cuda:
    net.cuda()
    criterion.cuda()
    cudnn.benchmark = True
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
print('## SSD Build success ##')

#for param in net.parameters():
#    if param.requires_grad==True:
#        print('param autograd')
#        break
    

def train(epoch):
    print('\nTrain phase, Epoch: {}'.format(epoch))
    net.train()
    train_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(trainloader):
        images.requires_grad_()
        # loc_targets.requires_grad_()
        # conf_targets.requires_grad_()
        if use_cuda:
            images = images.cuda()
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()

        # images = Variable(images)
        # loc_targets = Variable(loc_targets)
        # conf_targets = Variable(conf_targets)

        optimizer.zero_grad()
        loc_preds, conf_preds = net(images)
        # print(loc_preds.requires_grad)
        # print(conf_preds.requires_grad)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        # print(loss.requires_grad)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() # 累加的loss
        print('  Train loss: %.3f, accumulated average loss: %.3f' % (loss.item(), train_loss/(batch_idx+1)))
        return

def test(epoch):
    print('\nTest phase, Epoch: {}'.format(epoch))
    net.eval()
    test_loss = 0 
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(testloader):
        images.requires_grad_()
        if use_cuda:
            images = images.cuda()
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()

        # images = Variable(images)
        # loc_targets = Variable(loc_targets)
        # conf_targets = Variable(conf_targets)

        loc_preds, conf_preds = net(images)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        test_loss += loss.item() # 累加的loss
        print('  Test loss : %.3f, accumulated average loss: %.3f' % (loss.item(), test_loss/(batch_idx+1)))

    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print(' # New best loss: {} #'.format(test_loss))
        print('## Saving ##')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, './checkpoints/SSD_'+str(epoch)+'epoch.pth')
        best_loss = test_loss

if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
