import torch
import torch.nn as nn

# buat fungsi double conv
def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

# buat fungssi untuk pootng gambar
def crop_img(tensor, target_tensor):
    target_size = tensor.size()[2]
    tensor_size = tensor.size()[2]

    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


class UNet(nn.Module):
    """
    1. Contracting path (sisi kiri/encoder)
    2. Expansive path (sisi kanan/decoder)
    """
    def __init__(self):
        super(UNet, self).__init__()
        
        # encoder
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        # decoder
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, image):
        # image = [batch_size, channel, height, width]

        # encoder
        x1 = self.down_conv_1(image) # crop
        x2 = self.max_pool(x1)
        
        x3 = self.down_conv_2(x2) # crop
        x4 = self.max_pool(x3)
        
        x5 = self.down_conv_3(x4) # crop
        x6 = self.max_pool(x5)
        
        x7 = self.down_conv_4(x6) # crop
        x8 = self.max_pool(x7)
        
        x9 = self.down_conv_5(x8) #

        # decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], dim=1))

        x = self.up_trans_2(x)
        y = self.crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], dim=1))

        x = self.up_trans_3(x)
        y = self.crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], dim=1))

        x = self.up_trans_4(x)
        y = self.crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], dim=1))

        x = self.out(x)
        
        return x
