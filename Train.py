import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from torch import optim
from torchvision.utils import make_grid
from utils.utils import clip_gradient
import torch
from Src.utils.Dataloader import get_loader, test_dataset
import torch.nn.functional as F
import os

from torch import nn
import logging
import torch.backends.cudnn as cudnn
from lib.pvt import Network



def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function

    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, edges) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            map_4, map_3, map_2, clm = model(images)
            loss1 = structure_loss(map_4, gts)
            loss2 = structure_loss(map_3, gts)
            loss3 = structure_loss(map_2, gts)
            loss_clm = structure_loss(clm, gts)
            # loss_clm = nn.BCEWithLogitsLoss()(clm, gts)
            loss_out = loss1 + loss2 + loss3
            loss = loss_out + loss_clm
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data



            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}   loss_out: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_out.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}, loss_out: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_out.data))

                writer.add_scalars('Loss_Statistics',
                                   {'loss_out': loss_out.data,
                                    'Loss_total': loss.data},
                                   global_step=step)

                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)


        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise
def val(test_loader, model, epoch, save_path, writer):

    global best_metric_dict, best_score, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, _, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res = model(image)

            res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        if epoch == 1:
            best_score = mae
            print('[Cur Epoch: {}] Metrics (mae={})'.format(
                epoch, mae))
            logging.info('[Cur Epoch: {}] Metrics (mae={})'.format(
                epoch, mae))
        else:
            if mae <= best_score:

                best_score = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            else:
                print('>>> not find the best epoch -> continue training ...')
            print('[Cur Epoch: {}] Metrics (mae={})\n[Best Epoch: {}] Metrics (mae={})'.format(
                epoch, mae,
                best_epoch, best_score))
            logging.info('[Cur Epoch: {}] Metrics (mae={})\n[Best Epoch:{}] Metrics (mae={})'.format(
                epoch, mae,
                best_epoch, best_score))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')  #pvtv2_B2 16  pvtv2_B4 12
    parser.add_argument('--trainsize', type=int, default=480, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default='/data',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='/data/lxr/data/Shadow/Test/ISTD/',
                        help='the test rgb images root')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='train use gpu')
    parser.add_argument('--epoch_save', type=int, default=1,
                        help='every N epochs save your trained snapshot')
    parser.add_argument('--save_epoch', type=int, default=50,
                        help='every N epochs save your trained snapshot')

    parser.add_argument('--save_path', type=str, default='/data/New/biye/pth/shad/',
                        help='the path to save model and log')

    opt = parser.parse_args()

    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')
    cudnn.benchmark = True

    model = Network().cuda()
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + '/lxr/data/Shadow/Train/Imgs_jpg/',
                              gt_root=opt.train_root + '/lxr/data/Shadow/Train/GT/',
                              edge_root=opt.train_root + '/lxr/data/Shadow/Train/GT/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=4)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs_jpg/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)


    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> current mode: network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    step = 0
    writer = SummaryWriter(save_path + 'summary')

    best_score = 0
    best_epoch = 0

    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    print(">>> start train...")
    for epoch in range(1, opt.epoch):
        save_path = opt.save_path
        os.makedirs(save_path, exist_ok=True)
        if epoch % opt.epoch_save == 0:
            val(val_loader, model, epoch, opt.save_path, writer)
        if (epoch + 1) % opt.save_epoch == 0:
            torch.save(model.state_dict(), save_path + 'LNet_%d.pth' % (epoch + 1))
        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        train(train_loader, model, optimizer, epoch, save_path, writer)




