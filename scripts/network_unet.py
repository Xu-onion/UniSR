import torch
import torch.nn as nn
import networks.basicblock as B
import math
import numpy as np

class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[32, 64, 128, 256], nb=8, act_mode='L', downsample_mode='strideconv', upsample_mode='pixelshuffle', bias=False):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='CL')

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode) for _ in range(nb)],
                                    B.downsample_maxpool(nc[0], nc[1], bias=bias))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode) for _ in range(nb)],
                                    B.downsample_maxpool(nc[1], nc[2], bias=bias))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode) for _ in range(nb)],
                                    B.downsample_maxpool(nc[2], nc[3], bias=bias))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=bias, mode='C'+act_mode) for _ in range(nb)])

        self.m_up3 = B.sequential(B.upsample_pixelshuffle(nc[3], nc[2]),
                                  *[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode) for _ in range(nb)])
        self.m_up2 = B.sequential(B.upsample_pixelshuffle(nc[2], nc[1]),
                                  *[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(B.upsample_pixelshuffle(nc[1], nc[0]),
                                  *[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode) for _ in range(nb)])

        self.m_tail = B.sequential(B.conv(nc[0], out_nc, bias=bias, mode='C'))


    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)

        x = self.m_tail(x+x1)

        return x


if __name__ == '__main__':
    x = torch.rand(4, 1, 256, 256)
    net = UNetRes(nc=[32, 64, 128, 256, 256])

    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.size())

# run models/network_unet.py
