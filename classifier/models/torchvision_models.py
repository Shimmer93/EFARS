import torchvision.models as models
import torch.nn as nn

def ResNet18(num_classes, pretrained=False):
    model = models.resnet18(pretrained)
    model.fc = nn.Linear(512, num_classes)
    return model