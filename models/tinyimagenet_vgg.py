#coding=utf-8

import torch
import torch.nn as nn

__all__ = [
    'TinyImagenetVGG11',
    'TinyImagenetVGG11BN',
    'TinyImagenetVGG13',
    'TinyImagenetVGG13BN',
    'TinyImagenetVGG16',
    'TinyImagenetVGG16BN',
    'TinyImagenetVGG19',
    'TinyImagenetVGG19BN',
]


cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=200):
        super().__init__()
        self.features = features
        
        self.Linear = nn.Linear

        dim = 512 * 4
        
        self.classifier = nn.Sequential(
            self.Linear(dim, dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.Linear(dim//2, dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.Linear(dim//2, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_nn(cfg, batch_norm=False):
    layer_list = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layer_list += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layer_list += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layer_list += [nn.BatchNorm2d(l)]
        
        layer_list += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layer_list)

    
def _vgg(arch, features, num_classes, pretrained):
    model = VGG(features, num_classes)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-cifar{}.pt'.format(arch, num_classes)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


class VGGImp(nn.Module):
    def __init__(self, cfg, arch, num_class, pretrained=False, batch_norm=False):
        super().__init__()
        features = make_nn(cfg, batch_norm=batch_norm)
        self.vgg = _vgg(arch, features, num_class, pretrained)

    def forward(self, x):
        x = self.vgg(x)
        return x    
    

class TinyImagenetVGG11(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(cfg['A'], 'vgg11', num_class=num_class, pretrained=pretrained, batch_norm=False)
    

class TinyImagenetVGG11BN(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(cfg['A'], 'vgg11_bn', num_class=num_class, pretrained=pretrained, batch_norm=True)
    

class TinyImagenetVGG13(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(cfg['B'], 'vgg13', num_class=num_class, pretrained=pretrained, batch_norm=False)
    

class TinyImagenetVGG13BN(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(cfg['B'], 'vgg13_bn', num_class=num_class, pretrained=pretrained, batch_norm=True)
    

class TinyImagenetVGG16(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(cfg['D'], 'vgg16', num_class=num_class, pretrained=pretrained, batch_norm=False)
    

class TinyImagenetVGG16BN(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(cfg['D'], 'vgg16_bn', num_class=num_class, pretrained=pretrained, batch_norm=True)
    

class TinyImagenetVGG19(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(cfg['E'], 'vgg19', num_class=num_class, pretrained=pretrained, batch_norm=False)
    

class TinyImagenetVGG19BN(VGGImp):
    def __init__(self, num_class, pretrained=False):
        super().__init__(cfg['E'], 'vgg19_bn', num_class=num_class, pretrained=pretrained, batch_norm=True)
    