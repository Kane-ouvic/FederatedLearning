import torch
import torch.nn as nn
# from torchsummary import summary


def cropping(x, y):
    if x.shape == y.shape:
        return x
    else:
        return x[:, :, :y.shape[2], :y.shape[3]]


def Double_Conv2d(in_channels, out_channels, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=3, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=3, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def DeConv2D(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()

        self.num_filters = 64

        self.double1l = Double_Conv2d(in_channels, self.num_filters, padding=1)
        self.double2l = Double_Conv2d(
            self.num_filters, self.num_filters*2, padding=1)
        self.double3l = Double_Conv2d(
            self.num_filters*2, self.num_filters*4, padding=1)
        self.double4l = Double_Conv2d(
            self.num_filters*4, self.num_filters*8, padding=1)
        self.doubleb = Double_Conv2d(
            self.num_filters*8, self.num_filters*16, padding=1)

        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1 = DeConv2D(self.num_filters*16, self.num_filters*8)
        self.up2 = DeConv2D(self.num_filters*8, self.num_filters*4)
        self.up3 = DeConv2D(self.num_filters*4, self.num_filters*2)
        self.up4 = DeConv2D(self.num_filters*2, self.num_filters)

        self.double1r = Double_Conv2d(
            self.num_filters*16, self.num_filters*8, padding=1)
        self.double2r = Double_Conv2d(
            self.num_filters*8, self.num_filters*4, padding=1)
        self.double3r = Double_Conv2d(
            self.num_filters*4, self.num_filters*2, padding=1)
        self.double4r = Double_Conv2d(
            self.num_filters*2, self.num_filters, padding=1)

        self.final = nn.Conv2d(self.num_filters, out_channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        l1 = self.double1l(x)
        x = self.maxpooling(l1)

        l2 = self.double2l(x)
        x = self.maxpooling(l2)

        l3 = self.double3l(x)
        x = self.maxpooling(l3)

        l4 = self.double4l(x)
        x = self.maxpooling(l4)

        x = self.doubleb(x)

        x = self.up1(x)
        l4 = cropping(l4, x)
        x = torch.cat([l4, x], dim=1)
        x = self.double1r(x)

        x = self.up2(x)
        l3 = cropping(l3, x)
        x = torch.cat([l3, x], dim=1)
        x = self.double2r(x)

        x = self.up3(x)
        l2 = cropping(l2, x)
        x = torch.cat([l2, x], dim=1)
        x = self.double3r(x)

        x = self.up4(x)
        l1 = cropping(l1, x)
        x = torch.cat([l1, x], dim=1)
        x = self.double4r(x)

        x = self.final(x)
        x = self.act(x)
        return x


if __name__ == '__main__':
    model = Unet(1, 1)
    # print(model)
    # summary(model, [(1, 512, 512)])
