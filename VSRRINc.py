from RRIN import SRRIN
from RDBCLSTM import CLSTM
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as  np
from torch.utils.checkpoint import checkpoint


class VSRRINc(nn.Module):
    def __init__(self):
        super(VSRRINc, self).__init__()
        self.realtionlen = 5

        netG = SRRIN()
        #netG = torch.nn.DataParallel(netG)
        #netG.load_state_dict(torch.load('./RISTNsp.pth'))
        #netG = netG.cuda()
        self.sptioCNN = netG
        #self.sptioCNN.load_state_dict(torch.load('./netG_epoch_4_39.pth'))
        self.sptioCNN = torch.nn.DataParallel(self.sptioCNN, device_ids=[0])
        self.temporalRNN = CLSTM(320, 3, 16, 10)
        # self.recon = Reconsturcture(256)
        self.trainMode = True
        w = torch.empty(512, 256)
        self.FW = nn.Parameter(nn.init.normal_(w, mean=0, std=0.01)).cuda()

        self.convertsTot = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=320, kernel_size=1, stride=1, padding=0, bias=False),
        )


        self.eaualization = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )


        self.convertTtos = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def calc_sp(self, x):
        x = x.transpose(0, 1)  # now is seq_len,B,C,H,W
        t = None
        t2 = None
        for i in range(self.realtionlen):
            ax = x[i]
            ax_s = self.sptioCNN(ax)
            ax = self.convertsTot(ax_s)
            ax = torch.unsqueeze(ax, 0)
            ax_s = torch.unsqueeze(ax_s, 0)

            if i == 0:
                t = ax
                t2 = ax_s
            else:
                t = torch.cat((t, ax), 0)
                t2 = torch.cat((t2, ax_s), 0)
        return t, t2

    def forward(self, x,longrange):
        if self.trainMode:
            t, t2 = self.calc_sp(x)
            x = self.temporalRNN(t)
        else:
            t, t2 = checkpoint(self.calc_sp, x)

            self.temporalRNN.setlongrange(longrange)
            x,newlongrange = self.temporalRNN(t)

        out = []
        for i in range(len(x)):
            ax = x[i]
            ax = self.convertTtos(ax)

            at = t2[i]
            features = torch.cat((ax, at), 1)

            B, C, W, H = features.shape
            features = features.permute(0, 2, 3, 1)
            selfeature = torch.matmul(features, self.FW)
            totalfeature = selfeature.permute(0, 3, 1, 2)

            if self.trainMode:
                totalfeature = self.eaualization(totalfeature)
                rec = self.sptioCNN.module.reconstructure(totalfeature)
            else:
                totalfeature = checkpoint(self.eaualization, totalfeature)
                rec = checkpoint(self.sptioCNN.module.reconstructure, totalfeature)
            out.append(rec)

        return out,newlongrange


if __name__ == '__main__':

    a = torch.randn((5,5,3,20,20))
    a =Variable(a).cuda()
    vsr = VSRRINc()
    b = vsr(a)

    print('# generator parameters:', sum(param.numel() for param in vsr.parameters()))

