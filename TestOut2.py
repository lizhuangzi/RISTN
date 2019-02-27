import argparse
import os
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import skimage.color
import skimage.io
import skimage.measure
from datautils_argument import ValDatasetFromFolder2
import VSRRINc

val_set = ValDatasetFromFolder2('./TestHR', upscale_factor=4)

netG = torch.load('netGallaverage26.344714.pkl')

print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

netG.eval()

if torch.cuda.is_available():
    netG.cuda()


netG.trainMode = False
allcount = 0
for j in range(4):
    val_lr, val_bilr, val_hr = val_set.generatebatch(j)
    batch_size = val_lr.size(0)

    lr = Variable(val_lr, volatile=True)
    val_bilr = Variable(val_bilr, volatile=True)
    hr = Variable(val_hr, volatile=True)
    if torch.cuda.is_available():
        lr = lr.cuda()
        val_bilr = val_bilr.cuda()

    result = 0.0
    result1 = 0.0

    totalnumber = 0

    for k in range(batch_size):
        lrseq = lr[k]
        hrseq = hr.cpu()[k]
        biseq = val_bilr[k]

        lrseq = torch.unsqueeze(lrseq, dim=0)

        longrange = None

        if k %6 ==0:
            longrange = None

        srs,longrange = netG(lrseq,longrange)

        for s in range(3):
            srimg = srs[s]
            hrimg = hrseq[s]
            biimg = biseq[s][0]

            # SSIM
            srdata = srimg.data[0].permute(1, 2, 0).cpu().numpy()
            hrdata = hrimg.permute(1, 2, 0).cpu().numpy()

            srdata[srdata < 0.0] = 0.0
            srdata[srdata > 1.0] = 1.0
            skimage.io.imsave("./Vid4Result/%d_%d.jpg" % (j, totalnumber), srdata)

