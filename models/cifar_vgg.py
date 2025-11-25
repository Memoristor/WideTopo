#coding=utf-8

# Based on code taken from https://github.com/facebookresearch/open_lth

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CIFARVGG11',
    'CIFARVGG11BN',
    'CIFARVGG13',
    'CIFARVGG13BN',
    'CIFARVGG16',
    'CIFARVGG16BN',
    'CIFARVGG19',
    'CIFARVGG19BN',
]


class ConvModule(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))

class ConvBNModule(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super(ConvBNModule, self).__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class VGG(nn.Module):
    """A VGG-style neural network designed for CIFAR-10."""

    def __init__(self, plan, conv, num_classes=10):
        super(VGG, self).__init__()
        layer_list = []
        filters = 3

        for spec in plan:
            if spec == 'M':
                layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layer_list.append(conv(filters, spec))
                filters = spec

        self.nn = nn.Sequential(*layer_list)        

        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.nn(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def _plan(num):
    if num == 11:
        plan = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 13:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 16:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    elif num == 19:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    else:
        raise ValueError('Unknown VGG model: {}'.format(num))
    return plan

def _vgg(arch, plan, conv, num_classes, pretrained):
    model = VGG(plan, conv, num_classes)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-lottery.pt'.format(arch)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


class VGGImp(nn.Module):
    def __init__(self, plan, arch, conv, num_class, pretrained=False):
        super().__init__()
        self.vgg = _vgg(arch, plan, conv, num_class, pretrained)

    def forward(self, x):
        x = self.vgg(x)
        return x    
    

class CIFARVGG11(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(_plan(11), 'vgg11_bn', ConvModule, num_class=num_class, pretrained=pretrained)
    

class CIFARVGG11BN(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(_plan(11), 'vgg11_bn', ConvBNModule, num_class=num_class, pretrained=pretrained)
    
    
class CIFARVGG13(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(_plan(13), 'vgg13_bn', ConvModule, num_class=num_class, pretrained=pretrained)
    

class CIFARVGG13BN(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(_plan(13), 'vgg13_bn', ConvBNModule, num_class=num_class, pretrained=pretrained)
    
    
class CIFARVGG16(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(_plan(16), 'vgg16_bn', ConvModule, num_class=num_class, pretrained=pretrained)
    

class CIFARVGG16BN(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(_plan(16), 'vgg16_bn', ConvBNModule, num_class=num_class, pretrained=pretrained)
    
    
class CIFARVGG19(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(_plan(19), 'vgg19_bn', ConvModule, num_class=num_class, pretrained=pretrained)
    

class CIFARVGG19BN(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(_plan(19), 'vgg19_bn', ConvBNModule, num_class=num_class, pretrained=pretrained)
    