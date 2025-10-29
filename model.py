import torch
import torch.nn as nn
import torch.nn.functional as F


class PadIfNecessary(nn.Module):
    """Pad input to make it divisible by 2^depth. Has .pad() and .unpad() methods"""
    
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.two_to_depth = 2 ** depth
        self.pad_amt = None
        self.unpad_loc = None
    
    def get_pad_amt(self, x):
        b, c, h, w = x.shape
        pad_h = (self.two_to_depth - h % self.two_to_depth) % self.two_to_depth
        pad_w = (self.two_to_depth - w % self.two_to_depth) % self.two_to_depth
        # pad_amt = [pad_left, pad_right, pad_top, pad_bottom]
        pad_amt = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        return pad_amt
    
    def get_unpad_loc(self, x):
        b, c, h, w = x.shape
        # unpad will deal with padded inputs, so we need to account for the padding here
        h += self.pad_amt[2] + self.pad_amt[3]
        w += self.pad_amt[0] + self.pad_amt[1]
        
        # all elements in batch, all channels, top to bottom, left to right
        unpad_loc = [slice(None), slice(None),
                     slice(self.pad_amt[2], h - self.pad_amt[3]),
                     slice(self.pad_amt[0], w - self.pad_amt[1])]
        return unpad_loc
    
    def pad(self, x):
        if self.pad_amt is None:
            self.pad_amt = self.get_pad_amt(x)
            self.unpad_loc = self.get_unpad_loc(x)
        return F.pad(x, self.pad_amt)
    
    def unpad(self, x):
        if self.pad_amt is None:
            raise ValueError('Must call .pad() before .unpad()')
        return x[self.unpad_loc]


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, 
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, x_skip_channels=None,
                 up_type='conv_then_interpolate'):
        super().__init__()
        self.x_skip_channels = x_skip_channels or in_channels // 2
        
        # upsample will halve the number of channels
        conv1_in_channels = in_channels // 2 + self.x_skip_channels
        
        self.conv1 = ConvNormAct(conv1_in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size, padding=1)
        
        if up_type == 'conv_then_interpolate':
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
        elif up_type == 'convtranspose':
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            raise ValueError(f'Unknown up_type: {up_type}')

    def forward(self, x, x_skip):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=7, init_features=8, max_features=64,
                 up_type='conv_then_interpolate', extra_out_conv=True, dynamic_padding=True):
        super().__init__()
        
        # Generate channel progression like Choroidalyzer: doublemax-64
        channels = [min(init_features * (2 ** i), max_features) for i in range(depth + 1)]
        # channels = [8, 16, 32, 64, 64, 64, 64, 64] for depth=7
        
        self.dynamic_padding = dynamic_padding
        if dynamic_padding:
            self.pad_if_necessary = PadIfNecessary(depth)
        
        self.in_conv = ConvNormAct(in_channels, channels[0], kernel_size=3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        for d in range(depth):
            self.down_blocks.append(DownBlock(channels[d], channels[d + 1]))
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        for d in reversed(range(depth)):
            self.up_blocks.append(
                UpBlock(channels[d + 1], channels[d], 
                       x_skip_channels=channels[d],
                       up_type=up_type)
            )
        
        # Output
        if not extra_out_conv:
            self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        else:
            self.out_conv = nn.Sequential(
                ConvNormAct(channels[0], channels[0], kernel_size=3, padding=1),
                nn.Conv2d(channels[0], out_channels, kernel_size=1)
            )
        
    def forward(self, x):
        if self.dynamic_padding:
            x = self.pad_if_necessary.pad(x)
        
        x_skip = []
        x = self.in_conv(x)
        x_skip.append(x)
        
        # Encoder
        for down_block in self.down_blocks:
            x = down_block(x)
            x_skip.append(x)
        
        # Remove last skip connection (bottleneck output)
        x_skip.pop()
        
        # Decoder
        for up_block in self.up_blocks:
            x = up_block(x, x_skip.pop())
        
        x = self.out_conv(x)
        
        if self.dynamic_padding:
            x = self.pad_if_necessary.unpad(x)
        
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)