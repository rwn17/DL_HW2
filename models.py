from torchvision import models
import torch.nn as nn
import DLResNet

def model_A(num_classes):
    model_resnet = models.resnet50(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

# my DenseNet
def model_B(num_classes):
    ## your code here
    model_resnet = DLResNet.dlresnet18(pretrained=False)
    # model_resnet = models.resnext50_32x4d(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet


def model_C(num_classes):
    ## your code here
    model_resnet = DLResNet.dlresnet18(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

