import torch
import torch.nn as nn

class SpectroNet(torch.nn.Module):
    # SpectroNet (architecture for spectrogram conversion piano to guitar)

    def __init__(self):
        super(SpectroNet, self).__init__()

        # Input spectrogram shape = (batch, 1, 1025, 130) = (batch, channel, freq bins (h), slices (w))

        # Encoder conv blocks (4)
        self.e1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.e3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.e4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )

        # Transformer blocks (4)
        self.t1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))
        )
        self.t2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))
        )
        self.t3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))
        )
        self.t4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))
        )

        # Decoder deconv blocks(4)
        self.d1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.d2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.d3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.d4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=(3,3), padding=1),
            #nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):

        # Convolutional encoder
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)

        # Convolutional transformer
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)

        # Convolutional decoder
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)

        return x