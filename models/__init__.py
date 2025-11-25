# coding=utf-8

# CNN-based classification models
from .vgg import *
from .resnet import *
from .cifar_resnet import *
from .cifar_vgg import *
from .tinyimagenet_vgg import *
# from .dct_resnet import *
from .block_resnet import *
from .parcnet import *
from .convnetxt import *
from .mobilenet_v2 import *
from .freq_convnetxt import *

# ViT-based classification models
from .dino_vit import *
from .pretrained_vit import *

# from .mobile_former import *
from .edgevit import *
from .freq_edgevit import *

# CNN-based segmentation models
from .fcn import *
from .segnet import *
from .pspnet_vgg import *
from .pspnet_resnet import *
from .unet import *
from .deeplabv3 import *
from .deeplabv3_plus import *
from .hrnet_w48 import *
from .bisenet_v1 import *
from .bisenet_v2 import *

# ViT-based segmentation models
from .transunet import *
from .segformer import *
from .segmenter import *

# Hybrid models
from .sbcformer import *