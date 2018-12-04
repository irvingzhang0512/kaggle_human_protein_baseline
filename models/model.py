from pretrainedmodels.models import bninception, inceptionresnetv2
from torch import nn
from config import config


def get_net():
    if config.model_name == 'bninception_bcelog':
        return get_bninception()
    elif config.model_name == 'inceptionresnetv2':
        return get_inception_resnet_v2()
    else:
        raise ValueError('Unknown Model Name %s' % config.model_name)


def get_bninception():
    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1_7x7_s2 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, config.num_classes),
    )
    return model


def get_inception_resnet_v2():
    class BasicConv2d(nn.Module):

        def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
            super(BasicConv2d, self).__init__()
            self.conv = nn.Conv2d(in_planes, out_planes,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, bias=False)  # verify bias false
            self.bn = nn.BatchNorm2d(out_planes,
                                     eps=0.001,  # value found in tensorflow
                                     momentum=0.1,  # default pytorch value
                                     affine=True)
            self.relu = nn.ReLU(inplace=False)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x

    model = inceptionresnetv2(pretrained="imagenet")
    model.conv2d_1a = BasicConv2d(config.channels, 32, kernel_size=3, stride=2)
    model.avgpool_1a = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(1536),
        nn.Dropout(0.5),
        nn.Linear(1536, config.num_classes),
    )
    return model
