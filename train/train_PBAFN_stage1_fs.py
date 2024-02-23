import time
from options.train_options import TrainOptions
from models.networks import VGGLoss, save_checkpoint
from models.afwm import TVLoss, AFWM
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import cv2
import datetime

opt = TrainOptions().parse()
path = 'runs/' + opt.name
os.makedirs(path, exist_ok=True)
os.makedirs(opt.checkpoints_dir, exist_ok=True)


def CreateDataset(opt):
    # training with augmentation
    # from data.aligned_dataset import AlignedDataset_aug
    # dataset = AlignedDataset_aug()
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


os.makedirs('sample_fs', exist_ok=True)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

device = torch.device('cpu')

start_epoch, epoch_iter = 1, 0

train_data = CreateDataset(opt)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                          num_workers=4, pin_memory=True)
dataset_size = len(train_loader)
print('#training images = %d' % dataset_size)

warp_model = AFWM(opt, 45)
print(warp_model)
warp_model.train()
warp_model.to(device)

criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()

params_warp = [p for p in warp_model.parameters()]
optimizer_warp = torch.optim.Adam(params_warp, lr=opt.lr, betas=(opt.beta1, 0.999))

total_steps = (start_epoch - 1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size

writer = SummaryWriter(path)

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(train_loader):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1

        t_mask = torch.FloatTensor((data['label'].numpy() == 7).astype(np.float))
        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        edge = data['edge']
        pre_clothes_edge = torch.FloatTensor((edge.numpy() > 0.5).astype(np.int))
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        person_clothes_edge = torch.FloatTensor((data['label'].numpy() == 4).astype(np.int))
        real_image = data['image']
        person_clothes = real_image * person_clothes_edge
        pose = data['pose']
        size = data['label'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1, data['densepose'].data.long(), 1.0)
        densepose_fore = data['densepose'] / 24.0
        face_mask = torch.FloatTensor((data['label'].numpy() == 1).astype(np.int)) + torch.FloatTensor(
            (data['label'].numpy() == 12).astype(np.int))
        other_clothes_mask = torch.FloatTensor((data['label'].numpy() == 5).astype(np.int)) + torch.FloatTensor(
            (data['label'].numpy() == 6).astype(np.int)) + \
                             torch.FloatTensor((data['label'].numpy() == 8).astype(np.int)) + torch.FloatTensor(
            (data['label'].numpy() == 9).astype(np.int)) + \
                             torch.FloatTensor((data['label'].numpy() == 10).astype(np.int))
        preserve_mask = torch.cat([face_mask, other_clothes_mask], 1)
        concat = torch.cat([preserve_mask, densepose, pose], 1)

        flow_out = warp_model(concat, clothes, pre_clothes_edge)
        warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]

        epsilon = 0.001
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        loss_all = 0

        for num in range(5):
            cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5 ** (4 - num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5 ** (4 - num),
                                                     mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes)
            loss_vgg = criterionVGG(x_all[num], cur_person_clothes)
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge)
            b, c, h, w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x) / (b * c * h * w)
            loss_flow_y = (delta_y_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y) / (b * c * h * w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            loss_all = loss_all + (num + 1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num + 1) * 2 * loss_edge + (
                    num + 1) * 6 * loss_second_smooth

        loss_all = 0.01 * loss_smooth + loss_all

        writer.add_scalar('loss_all', loss_all, step)

        optimizer_warp.zero_grad()
        loss_all.backward()
        optimizer_warp.step()

        step += 1
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch - step % step_per_batch) + step_per_batch * (
                    opt.niter + opt.niter_decay - epoch)
        eta = iter_delta_time * step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
        if step % 100 == 0:
            print('{}:{}:[step-{}]--[loss-{:.6f}]--[ETA-{}]'.format(now, epoch_iter, step, loss_all, eta))

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        save_checkpoint(warp_model, os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_warp_epoch_%03d.pth' % (epoch + 1)))

    if epoch > opt.niter:
        # Update learning rate
        pass

