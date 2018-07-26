# coding:utf-8

class DefaultConfig(object):

    batch_size = 32
    # 保存的checkpoint路径
    ckpt_path = './checkpoints/SSD_160epoch.pth'
    # 有vgg预训练参数的模型，见utils/convert_fgg.py
    pretrained_model = 'checkpoints/ssd.pth'

    # 输入神经网络的图片的大小，所有原始图片都需要rescale到300*300
    img_size = 300
    # 先把所有的xml格式的标注转换成一行一行 保存在label_path文件里
    # img_file_name num_objs xmin ymin xmax ymax class_label ... xmin ymin xmax ymax class_label
    train_label_file = 'data/voc12_train.txt'
    test_label_file = 'data/voc12_test.txt'
    # 保存图片的文件夹的路径 结尾有/
    train_data_path = '/home/j/VOC2012TRAIN/JPEGImages/'
    test_data_path = '/home/j/VOC2012TEST/JPEGImages/'

    # 背景类的标记为0
    num_classes = 21

    # multibox 相关
    # num_anchor和aspect_ratio对应num_anchors[i] = 2+aspect_ratio[i]*2
    num_anchors = [4,6,6,6,4,4]
    # 每个feature map的channel in_planes[i]
    in_planes = [512,1024,512,256,256,256]

    # prior box相关   
    # prior box是从左上角开始产生的
    # cell size是每个feature map中每个cell对应原图的大小
    # 实际产生prior box的时候用的都是相对位置
    cell_size = (8, 16, 32, 64, 100, 300)
    # 预先设计好的prior box的大小，实际产生prior box的时候用的是相对位置
    prior_box_hw = (30, 60, 111, 162, 213, 264, 315)
    # 预先设计好的prior box的aspect ratio
    # 每个feature map都有几个ratio aspect_ratios[i]就是对第i个feature map的ratio
    aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2,), (2,))
    # feature map的大小
    feature_map_sizes = (38, 19, 10, 5, 3, 1)

opt = DefaultConfig()
