
import torch
from torchvision import datasets, transforms, models
from torchvision.models import vgg16, VGG16_Weights, densenet121, DenseNet121_Weights
from torchvision.models import VGG16_Weights, VGG13_Weights, DenseNet121_Weights

from torch import nn, optim
import torch.nn.functional as F

model_functions = {
    'vgg16': models.vgg16,
    'vgg13': models.vgg13,
    'densenet121': models.densenet121
}

weights_mapping = {
    'vgg16': VGG16_Weights.DEFAULT,
    'vgg13': VGG13_Weights.DEFAULT,
    'densenet121': DenseNet121_Weights.DEFAULT
}