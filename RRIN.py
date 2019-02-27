"""
RIB Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model_utils import split, merge, injective_pad, psi
from torchvision.models.resnet import resnet50
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint



class SRResirevBlock(nn.Module):
    def __init__(self, layercount=18,inchannel = 3,outchannel=64,first_initpad=False,pooling=True):
        super(SRResirevBlock, self).__init__()

        self.inchannel = inchannel
        self.block_list = nn.ModuleList()
        self.psi = psi(2)
        self.pooling = pooling
        self.first = True
        block_list = nn.ModuleList()
        strides = []
        channels = []

        # for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
        #     strides = strides + ([stride] + [1] * (depth - 1))
        #     channels = channels + ([channel] * depth)
        # for channel, stride in zip(channels, strides):
        #     block_list.append(_block(in_ch, channel, stride,
        #                              first=self.first,
        #                              dropout_rate=dropout_rate,
        #                              affineBN=affineBN, mult=mult))

        for i in range(layercount):

            layer = irevnet_block(inchannel, outchannel, first=self.first);

            self.block_list.append(layer)

            inchannel = 2 * outchannel
            self.first = False

    def forward(self, x):

        stratx = x
        n = self.inchannel // 2

        out = (x[:, :n, :, :], x[:, n:, :, :])

        for i in range(len(self.block_list)):
            block = self.block_list[i]
            out = block.forward(out)


        out_bij = merge(out[0], out[1])

        # out = stratx + out_bij
        # if self.pooling:
        #     out =  self.psi.forward(out)
        #
        out_bij = out_bij+stratx

        return out_bij

    def inverse(self,x):
        out = split(x)
        for i in range(len(self.block_list)):
            out = self.stack[-1 - i].inverse(out)
        out = merge(out[0], out[1])
        return out


class SRRIN(nn.Module):
    def __init__(self, nblock=[6,8,10,8],nblock2=[6,6,12,4],INi=[3,16,64,256]):
        super(SRRIN, self).__init__()
        self.nblock = nblock;
        self.ini_psi = injective_pad(13)

        # self.primaryConv = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)),
        # ]))
        # self.primaryConv.add_module('norm0', nn.BatchNorm2d(16))
        # self.primaryConv.add_module('prelu0', nn.PReLU())
        self.qianx = None

        self.resblock1 = SRResirevBlock(layercount=nblock[0],inchannel=16,outchannel=8,first_initpad=True)
        self.resblock1_1 = SRResirevBlock(layercount=nblock[0], inchannel=16, outchannel=8, first_initpad=True)
        self.resblock2 = SRResirevBlock(layercount=nblock[1], inchannel=32,outchannel=16,first_initpad=True)
        self.resblock2_2 = SRResirevBlock(layercount=nblock2[1], inchannel=32, outchannel=16, first_initpad=True)

        self.resblock3 = SRResirevBlock(layercount=nblock[2], inchannel=64, outchannel=32,first_initpad=True,pooling=False)
        self.resblock3_3 = SRResirevBlock(layercount=nblock2[2], inchannel=64, outchannel=32, first_initpad=True,pooling=False)
        self.resblock4 = SRResirevBlock(layercount=nblock[2], inchannel=128, outchannel=64, first_initpad=True)
        self.resblock4_4 = SRResirevBlock(layercount=nblock2[1], inchannel=128, outchannel=64, first_initpad=True)

        self.eaualization = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )


        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.PReLU()
        )

        self.reconstruct = nn.Sequential(
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        )


    def forward_qian(self,x):
        x = self.ini_psi(x)

        x1 = self.resblock1(x)

        x = torch.cat((x1,x), 1)

        x1 = self.resblock2(x)
        x2 = self.resblock2_2(x)

        x = torch.cat((x2, x1), 1)
        return x

    def forward_zhong(self,x):

        x1 = self.resblock3(x)
        x2 = self.resblock3_3(x)
        x = torch.cat((x2, x1), 1)
        return x


    def forward(self, x):

        x = self.forward_qian(x)
        x = self.forward_zhong(x)

        x1 = self.resblock4(x)
        x2 = self.resblock4_4(x)
        x = torch.cat((x2,x1), 1)

        return x


    def reconstructure(self,x):

        x = self.eaualization(x)

        x = self.deconv(x)
        x = self.reconstruct(x)
        return x


    def inverse(self, x):

        x = self.resblock3.inverse(x)
        x = self.resblock2.inverse(x)
        x = self.resblock1.inverse(x)


class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=3,first_initpad=False):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        print out_ch
        print  in_ch
        self.stride = stride
        self.psi = psi(stride)
        self.inpx = None

        layers = []
        if not first:
            layers.append(nn.BatchNorm2d(in_ch//2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_ch//2, int(out_ch//mult), kernel_size=3,
                      stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch//mult), int(out_ch//mult),
                      kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch//mult), out_ch, kernel_size=3,
                      padding=1, bias=False))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """

        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        if self.pad != 0 and self.stride == 1:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = (x1, x2)
        return x


class iRevNet(nn.Module):
    def __init__(self, nBlocks, nStrides, nClasses, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=4):
        super(iRevNet, self).__init__()
        self.ds = in_shape[2]//2**(nStrides.count(2)+init_ds//2)
        self.init_ds = init_ds
        self.in_ch = in_shape[0] * 2**self.init_ds
        self.nBlocks = nBlocks
        self.first = True

        print('')
        print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3 + 1))
        if not nChannels:
            nChannels = [self.in_ch//2, self.in_ch//2 * 4,
                         self.in_ch//2 * 4**2, self.in_ch//2 * 4**3]

        self.init_psi = psi(self.init_ds)
        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)
        self.bn1 = nn.BatchNorm2d(nChannels[-1]*2, momentum=0.9)
        self.linear = nn.Linear(nChannels[-1]*2, nClasses)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)
        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, x):
        """ irevnet forward """
        n = self.in_ch//2
        if self.init_ds != 0:
            x = self.init_psi.forward(x)
        out = (x[:, :n, :, :], x[:, n:, :, :])
        for block in self.stack:
            out = block.forward(out)
        out_bij = merge(out[0], out[1])
        out = F.relu(self.bn1(out_bij))
        out = F.avg_pool2d(out, self.ds)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, out_bij

    def inverse(self, out_bij):
        """ irevnet inverse """
        out = split(out_bij)
        for i in range(len(self.stack)):
            out = self.stack[-1-i].inverse(out)
        out = merge(out[0],out[1])
        x = self.init_psi.inverse(out)
        return x


if __name__ == '__main__':
    model = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2, 2, 2, 2],
                    nChannels=None, nClasses=1000, init_ds=2,
                    dropout_rate=0., affineBN=True, in_shape=[3, 224, 224],
                    mult=4)
    y = model(Variable(torch.randn(1, 3, 224, 224)))
    print(y.size())
