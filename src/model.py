import torch
import torch.nn as nn
from torchvision import models

def remove_layer(model, n):
    modules = list(model.children())[:-n]
    model = nn.Sequential(*modules)
    return model

def get_num_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

class RegNet(nn.Module):
    #21923016
    def __init__(self):
        super(RegNet, self).__init__()
        self.RGB_net = remove_layer(models.resnet18(pretrained=True), 2)
        self.depth_net = remove_layer(models.resnet18(pretrained=False), 2)
        modules = list(self.depth_net.children())
        modules[0] = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.depth_net = nn.Sequential(*modules)
        for param in self.RGB_net.parameters():
            param.requires_grad = False
        self.matching = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 8)

    def forward(self, rgb_img, depth_img):
        rgb_features = self.RGB_net(rgb_img)
        depth_features = self.depth_net(depth_img)
        concat_features = torch.cat((rgb_features, depth_features), 1)
        matching_features = self.matching(concat_features).squeeze()
        x = self.fc1(matching_features)
        x = self.fc2(x)

        return x