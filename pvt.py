import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x
class CEM(nn.Module):
    def __init__(self, nIn, kernel_size=3,reduction=4,bias=False,act=nn.PReLU()):
        super(CEM, self).__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.bn1 = nn.BatchNorm2d(nIn)
        self.psa1 = CAB(nIn, kernel_size, reduction, bias=bias, act=act)
        self.conv2_1 = conv2d(nIn, nIn // 2, (3, 3), padding=1, dilation=(1, 1), act=False)
        self.conv2_2 = conv2d(nIn // 2, nIn // 2, (3, 3), padding=1, dilation=(1, 1), act=False)

        self.conv3_1 = conv2d(nIn, nIn // 2, (3, 3), padding=1, dilation=(1, 1), act=False)
        self.conv3_3 = conv2d(nIn // 2, nIn // 2, (3, 3), padding=3, dilation=(3, 3), act=False)
        self.conv4_1 = conv2d(nIn, nIn // 2, (3, 3), padding=1, dilation=(1, 1), act=False)
        self.conv4_3 = conv2d(nIn // 2, nIn // 2, (3, 3), padding=5, dilation=(5, 5), act=False)
        self.conv_out = conv2d(nIn, nIn)

    def forward(self, x):
        #print(f"BAqian-------: {x.shape}")
        o1_2 = self.psa1(x)
        #print(f"BAhou-------: {o1_2.shape}")
        o2_1 = self.conv2_1(x)
        o2_2 = self.conv2_2(o2_1)

        o3_1 = self.conv3_1(x)
        o3_3 = self.conv3_3(o3_1)

        o4_1 = self.conv4_1(x)
        o4_3 = self.conv4_3(o4_1)

        o4 = torch.cat([o4_1, o4_3], 1)
        o3_4 = torch.cat([o3_1, o3_3], 1)
        o2_3 = torch.cat([o2_1, o2_2], 1)

        x_out = self.bn1(o4 + o3_4 + o2_3)
        x_out = x_out + o1_2
        x_out = self.conv_out(x_out)
        return x_out
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class MFAM(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(MFAM, self).__init__()
        self.cat1 = BasicConv2d(hidden_channels * 2, out_channels, kernel_size=3, padding=1)
        self.cat2 = BasicConv2d(hidden_channels * 2, out_channels, kernel_size=3, padding=1)
        self.cat3 = BasicConv2d(hidden_channels * 2, out_channels, kernel_size=3, padding=1)
    def forward(self, x, y, edge):
        edge = torch.sigmoid(edge)
        ones_matrix = torch.ones_like(edge)
        inverse_edge = ones_matrix - edge
        x_1 = x * inverse_edge
        x_2 = x * edge
        y_1 = y * inverse_edge
        y_2 = y * edge
        y_12 = torch.cat((y_1, y_2), dim=1)
        x_12 = torch.cat((x_1, x_2), dim=1)
        y_12 = self.cat1(y_12)
        x_12 = self.cat2(x_12)
        out = self.cat3(torch.cat((y_12,x_12), dim=1)) + y + x
        return out

class Network(nn.Module):
    def __init__(self, channel=64):
        super(Network, self).__init__()
        self.backbone = pvt_v2_b2()
        path = '/data1/lx/lx_3090_4_home_ubuntu/HitNet/pretrained_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.cem0 = CEM(64)
        self.cem1 = CEM(64)
        self.cem2 = CEM(64)
        self.cem3 = CEM(64)

        self.linearr1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.FF1 = MFAM(64, 64)
        self.FF2 = MFAM(64, 64)
        self.FF3 = MFAM(64, 64)
        self.FF4 = MFAM(64, 64)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        image_shape = x.size()[2:]
        pvt = self.backbone(x)

        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        #print(f"x1-------: {x1.shape}")
        #print(f"x2-------: {x2.shape}")
        #print(f"x3-------: {x3.shape}")
        #print(f"x4-------: {x4.shape}")

        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        x1_t = x1
        
        #print(f"x2_tCEMqian-------: {x2_t.shape}")
        #print(f"x3_tCEMqian-------: {x3_t.shape}")
        #print(f"x4_tCEMqian-------: {x4_t.shape}")
        #print(f"x1_tCEMqian-------: {x1_t.shape}")

        x1 = self.ca(x1) * x1
        #print(f"x1-------: {x1.shape}")
        F2 = self.sa(x1) * x1
        #print(f"F2-------: {F2.shape}")
        # F2 = x1
        x3_t_feed = torch.cat((x3_t, self.upsample(x4_t)), 1)
        x2_t_feed = torch.cat((x2_t, self.upsample(x3_t_feed)), 1)
        #print(f"x3_t_feed-------: {x3_t_feed.shape}")
        #print(f"x2_t_feed-------: {x2_t_feed.shape}")

        F1 = self.conv4(x2_t_feed)
        #print(f"F1-------: {F1.shape}")
        F1 = self.up2(F1)


        F2 = self.Translayer2_0(F2)
        #print(f"F1-------: {F1.shape}")
        #print(f"F2-------: {F2.shape}")

        cmd = F1 + F2
        #print(f"cmd-------: {F1.shape}")
        cmd = self.linearr5(cmd)
        #print(f"cmd-------: {cmd.shape}")
        

        C_1 = self.cem0(x1_t)
        C_2 = self.cem1(x2_t)
        C_3 = self.cem2(x3_t)
        C_4 = self.cem3(x4_t)
        #print(f"C_1CEMhou------: {C_1.shape}")
        #print(f"C_2CEMhou------: {C_2.shape}")
        #print(f"C_3CEMhou------: {C_3.shape}")
        #print(f"C_4CEMhou------: {C_4.shape}")
        G5 = F.interpolate(cmd, size=image_shape, mode='bilinear')
        E22 = F.interpolate(C_2, scale_factor=2, mode='bilinear')
        E23 = F.interpolate(C_3, scale_factor=4, mode='bilinear')
        E24 = F.interpolate(C_4, scale_factor=8, mode='bilinear')
        
        #print(f"E22------C_2resizeMFAMqian: {E22.shape}")
        #print(f"E23------C_3resizeMFAMqian: {E23.shape}")
        #print(f"E24------C_4resizeMFAMqian: {E24.shape}")

        M_4 = E24
        M_3 = self.FF3(E23, M_4, cmd)
        M_2 = self.FF2(E22, M_3, cmd)
        M_1 = self.FF1(C_1, M_2, cmd)
        
        #print(f"M_4------M_4resizeMFAMhou: {M_4.shape}")
        #print(f"M_3------M_3resizeMFAMhou: {M_3.shape}")
        #print(f"M_2------M_2resizeMFAMhou: {M_2.shape}")
        #print(f"M_1------M_1resizeMFAMhou: {M_1.shape}")

        # map_4 = self.linearr4(M_4)
        map_3 = self.linearr3(M_3)
        map_2 = self.linearr2(M_2)
        map_1 = self.linearr1(M_1)

        G1 = F.interpolate(map_1, size=image_shape, mode='bilinear')
        G2 = F.interpolate(map_2, size=image_shape, mode='bilinear')
        G3 = F.interpolate(map_3, size=image_shape, mode='bilinear')
        # G4 = F.interpolate(map_4, size=image_shape, mode='bilinear')

        return G1, G2, G3, G5


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network().cuda()
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352).cuda()
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        torch.cuda.synchronize()
        start = time()
        y = net(dump_x)
        torch.cuda.synchronize()
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        frame_rate[i] = running_frame_rate