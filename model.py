import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
class ConvBlock(torch.nn.Module) :
    def __init__(self, input_size, output_size, kernel_size = 3, stride = 2, padding = 1,
                 activation = 'relu', batch_norm = True) :
        super(ConvBlock, self).__init__()

        self.activation = activation
        self.batch_norm = batch_norm

        self.bn = nn.InstanceNorm2d(output_size)

        ### build...
        # // conv -> bn -> relu // In this class deal with this part(block)
        # -> // conv -> bn -> relu //
        # -> (output + x)
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.batch_norm :  # batch_norm = True // conv -> batch
            out = self.bn(self.conv(x))
        else :
            out = self.conv(x)

        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        else :
            print('no activation')
            return out

class ResidualBlock(nn.Module) :
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()

        self.residual_block = nn.Sequential(nn.ReflectionPad2d(1),
                                          nn.Conv2d(in_channel, in_channel, kernel_size = 3),
                                          nn.InstanceNorm2d(in_channel),
                                          nn.ReLU(inplace=True),
                                          nn.ReflectionPad2d(1),
                                          nn.Conv2d(in_channel, in_channel, kernel_size = 3),
                                          nn.InstanceNorm2d(in_channel))

    def forward(self, x) :
        return self.residual_block(x) + x

class DeconvBlock(torch.nn.Module) :
    def __init__(self, input_channel, output_channel, kernel_size =3, stride = 2, padding = 1,
               output_padding = 1, activation = 'relu' ,batch_norm = True) :
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding, output_padding)
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn = nn.InstanceNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()


    def forward(self, x) :
        if self.batch_norm :
            out = self.bn(self.deconv(x))

        else :
            out = self.deconv(x)

        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        else :
            print('no activation')
            return out

class Generator(nn.Module):
    def __init__(self, input_channel, output_channel,num_residual, final_output_c=1): # 3, 64, 3, 9
        super(Generator, self).__init__()

        self.pad = nn.ReflectionPad2d(3)

        self.Encoder = nn.Sequential(ConvBlock(input_channel, output_channel, kernel_size = 7, stride = 1, padding = 0),
                                     ConvBlock(output_channel, output_channel * 2),
                                     ConvBlock(output_channel * 2, output_channel * 4))

        self.residual_blocks = []
        for _ in range(num_residual) :
            self.residual_blocks.append(ResidualBlock(output_channel * 4))

        self.transfer = nn.Sequential(*self.residual_blocks)

        self.Decoder = nn.Sequential(DeconvBlock(output_channel * 4, output_channel *2),
                                     DeconvBlock(output_channel * 2, output_channel),
                                     nn.ReflectionPad2d(3),
                                     ConvBlock(output_channel, final_output_c, kernel_size = 7, stride = 1, padding = 0,
                                               activation = 'tanh', batch_norm = False))


    def forward(self, x):
        x = self.pad(x)
        x = self.Encoder(x)
        x = self.transfer(x)
        x = self.Decoder(x)
        return x