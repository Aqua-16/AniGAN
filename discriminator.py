import torch
import torch.nn as nn
from torchinfo import summary

class Discriminator(nn.Module):
    def __init__(self, feature_maps = 64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.02),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.02),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.02),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.02),

            nn.Conv2d(feature_maps * 8, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(-1, 1) # Reshaping for singular output value
    

if __name__ == '__main__':
    disc = Discriminator()
    img = torch.randn(64, 3, 64, 64)
    
    output = disc(img)
    
    summary(disc, (64, 3, 64, 64))