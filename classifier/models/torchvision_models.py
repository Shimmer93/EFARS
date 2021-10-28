import torchvision.models as models
import torch.nn as nn

def ResNet18(num_classes, pretrained=False):
    model = models.resnet18(pretrained)
    model.fc = nn.Linear(512, num_classes)
    return model

def ResNet50(num_classes, pretrained=False):
    model = models.resnet50(pretrained)
    model.fc = nn.Linear(512, num_classes)
    return model

def MobileNetV3Small(num_classes, pretrained=False):
    model = models.mobilenet_v3_small(pretrained)
    model.classifier[3] = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    return model