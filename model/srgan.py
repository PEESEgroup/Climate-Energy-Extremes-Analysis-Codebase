import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class SubPixelBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(SubPixelBlock, self).__init__()
        out_channels = in_channels * (upscale_factor ** 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x



class SRGAN_g(nn.Module):
    def __init__(self, in_channels=7, out_channels=9, num_res_blocks=32,
                 crop_size=(625, 1475)):
        super(SRGAN_g, self).__init__()
        self.crop_h, self.crop_w = crop_size
        channels = 128

        # First conv + ReLU
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)

        # Residual blocks
        self.residual_block = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])

        # Mid conv + BN
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)

        # Upsample layers: 3 × 2x -> 8x
        self.up1 = SubPixelBlock(channels, upscale_factor=2)
        self.up2 = SubPixelBlock(channels, upscale_factor=2)
        self.up3 = SubPixelBlock(channels, upscale_factor=2)

        # Final conv + tanh
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.conv_out.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.conv_out.bias)

    def center_crop(self, x):
        _, _, h, w = x.shape
        top = (h - self.crop_h) // 2
        left = (w - self.crop_w) // 2
        return x[:, :, top:top + self.crop_h, left:left + self.crop_w]

    def forward(self, x):
        x = self.relu(self.conv1(x))
        residual = x
        x = self.residual_block(x)
        x = self.bn1(self.conv2(x))
        x = x + residual
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.conv_out(x)
        x = self.center_crop(x)
        return x

 
class SRGAN_d(nn.Module):
    def __init__(self, in_channels=9, use_sigmoid=False):
        super(SRGAN_d, self).__init__()

        def conv_block(in_channels, out_channels, stride):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='leaky_relu')
            block = nn.Sequential(
                conv,
                nn.GroupNorm(num_groups=16, num_channels=out_channels),
                nn.LeakyReLU(0.2, inplace=False)
            )
            return block

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=False)
        )
        nn.init.kaiming_normal_(self.conv1[0].weight, mode='fan_in', nonlinearity='leaky_relu')

        self.conv2 = conv_block(64, 64, stride=2)
        self.conv3 = conv_block(64, 128, stride=1)
        self.conv4 = conv_block(128, 128, stride=2)
        self.conv5 = conv_block(128, 256, stride=1)
        self.conv6 = conv_block(256, 256, stride=2)
        self.conv7 = conv_block(256, 512, stride=1)
        self.conv8 = conv_block(512, 512, stride=2)

        # PatchGAN output: 1xHxW logits map
        self.conv_out = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_out.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.conv_out.bias)

        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        logits = self.conv_out(x)
        if self.use_sigmoid:
            prob = self.sigmoid(logits)
            return prob
        else:
            return logits  # for BCEWithLogitsLoss