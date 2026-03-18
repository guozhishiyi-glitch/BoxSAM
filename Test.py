import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from lib.pvt import Network
from Src.utils.Dataloader import test_dataset1
import imageio
from skimage import img_as_ubyte

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=480, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='/data/New/biye/pth/shad/Net_epoch_best.pth')
opt = parser.parse_args()

model = Network().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()
for dataset in ['SBU']:
    save_path = '/data/New/biye/result/shadow/{}/'.format(dataset)
    os.makedirs(save_path, exist_ok=True)
    test_loader = test_dataset1(image_root='/data/lxr/data/Shadow/Test/{}/Imgs/'.format(dataset),
                               gt_root='/data/lxr/data/Shadow/Test/{}/GT/'.format(dataset),
                               edge_root='/data/lxr/data/Shadow/Test/{}/GT/'.format(dataset),
                               testsize=opt.testsize)
    img_count = 1 
    for iteration in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        G, _1, _2, _3 = model(image)
        G = F.upsample(G, size=gt.shape, mode='bilinear', align_corners=True)
        G = G.sigmoid().data.cpu().numpy().squeeze()
        G = (G - G.min()) / (G.max() - G.min() + 1e-8)
        imageio.imsave(save_path+name, img_as_ubyte(G))
        img_count += 1

