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
    """8x SRGAN generator with optional STATIC PHYSICS CONDITIONING (Tier 2, ARCH_PHYSICS_FUSION.md §3B).
    static_lr (C,89,211)  : coarse terrain(z)/land-sea mask concatenated at the LR input  -> flows through the
                            32 residual blocks (terrain CONTEXT for the whole deep net).
    static_hr (C,625,1475): FINE terrain(z)/land-sea mask injected at HR (after upsample+crop) through a 3x3
                            refine conv -> places fine weather detail exactly on ridges/coastlines (land DETAIL).
    dyn_lr_ch/dyn_hr_ch   : reserve channels for a PER-SAMPLE dynamic conditioning field passed to forward()
                            (e.g. solar cos(SZA) for the radiation group) — concatenated the same dual way as
                            the static buffers (coarse@LR-input + fine@HR-refine). 0 -> no dynamic input.
    Static inputs default None -> identical to the original 7ch-in generator (backward compatible). Statics are
    registered as buffers so they move with .to(dev), are baked into the checkpoint, and need no call-site change."""
    def __init__(self, in_channels=7, out_channels=9, num_res_blocks=32,
                 crop_size=(625, 1475), static_lr=None, static_hr=None, dyn_lr_ch=0, dyn_hr_ch=0):
        super(SRGAN_g, self).__init__()
        self.crop_h, self.crop_w = crop_size
        channels = 128
        self.dyn_lr_ch = dyn_lr_ch; self.dyn_hr_ch = dyn_hr_ch

        s_lr_ch = 0
        s_hr_ch = 0
        if static_lr is not None:
            b = torch.as_tensor(static_lr, dtype=torch.float32)
            if b.dim() == 3: b = b.unsqueeze(0)
            self.register_buffer('static_lr', b); s_lr_ch = b.shape[1]
        else:
            self.static_lr = None
        if static_hr is not None:
            b = torch.as_tensor(static_hr, dtype=torch.float32)
            if b.dim() == 3: b = b.unsqueeze(0)
            self.register_buffer('static_hr', b); s_hr_ch = b.shape[1]
        else:
            self.static_hr = None

        # First conv + ReLU  (in = weather channels + coarse static + coarse dynamic conditioning)
        self.conv1 = nn.Conv2d(in_channels + s_lr_ch + dyn_lr_ch, channels, kernel_size=3, stride=1, padding=1)
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

        # HR fine conditioning refinement (when static and/or dynamic HR conditioning provided)
        if s_hr_ch + dyn_hr_ch > 0:
            self.refine = nn.Conv2d(channels + s_hr_ch + dyn_hr_ch, channels, kernel_size=3, stride=1, padding=1)
            nn.init.kaiming_normal_(self.refine.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.refine.bias)
        else:
            self.refine = None

        # Final conv + tanh
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.conv_out.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.conv_out.bias)

    def center_crop(self, x):
        _, _, h, w = x.shape
        top = (h - self.crop_h) // 2
        left = (w - self.crop_w) // 2
        return x[:, :, top:top + self.crop_h, left:left + self.crop_w]

    def forward(self, x, dyn_lr=None, dyn_hr=None):
        parts = [x]
        if self.static_lr is not None:
            parts.append(self.static_lr.expand(x.shape[0], -1, -1, -1))
        if dyn_lr is not None:
            parts.append(dyn_lr)                       # (B, dyn_lr_ch, 89, 211), per-sample
        x = torch.cat(parts, dim=1) if len(parts) > 1 else x
        x = self.relu(self.conv1(x))
        residual = x
        x = self.residual_block(x)
        x = self.bn1(self.conv2(x))
        x = x + residual
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        if self.refine is not None:
            x = self.center_crop(x)
            parts = [x]
            if self.static_hr is not None:
                parts.append(self.static_hr.expand(x.shape[0], -1, -1, -1))
            if dyn_hr is not None:
                parts.append(dyn_hr)                    # (B, dyn_hr_ch, 625, 1475), per-sample
            x = torch.cat(parts, dim=1) if len(parts) > 1 else x
            x = self.relu(self.refine(x))
            x = self.conv_out(x)
        else:
            x = self.conv_out(x)
            x = self.center_crop(x)
        return x


class SRGAN_d(nn.Module):
    def __init__(self, in_channels=9, use_sigmoid=False, use_sn=True):
        super(SRGAN_d, self).__init__()
        # Spectral normalization bounds the discriminator's Lipschitz constant -> keeps D outputs
        # bounded so the relativistic-MSE (RaGAN) loss can't blow up (fixes the epoch-5 divergence).
        SN = nn.utils.spectral_norm if use_sn else (lambda m: m)

        def conv_block(in_channels, out_channels, stride):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='leaky_relu')
            block = nn.Sequential(
                SN(conv),
                nn.GroupNorm(num_groups=16, num_channels=out_channels),
                nn.LeakyReLU(0.2, inplace=False)
            )
            return block

        _c1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(_c1.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.conv1 = nn.Sequential(
            SN(_c1),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.conv2 = conv_block(64, 64, stride=2)
        self.conv3 = conv_block(64, 128, stride=1)
        self.conv4 = conv_block(128, 128, stride=2)
        self.conv5 = conv_block(128, 256, stride=1)
        self.conv6 = conv_block(256, 256, stride=2)
        self.conv7 = conv_block(256, 512, stride=1)
        self.conv8 = conv_block(512, 512, stride=2)

        # PatchGAN 输出：1xHxW logits map
        _co = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(_co.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(_co.bias)
        self.conv_out = SN(_co)

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
