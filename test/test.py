import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from torchvision import utils
from util import flow_util

opt = TestOptions().parse()

print(opt)
def de_offset(s_grid):
    [b, _, h, w] = s_grid.size()

    x = torch.arange(w).view(1, -1).expand(h, -1).float()
    y = torch.arange(h).view(-1, 1).expand(-1, w).float()
    x = 2 * x / (w - 1) - 1
    y = 2 * y / (h - 1) - 1
    grid = torch.stack([x, y], dim=0).float().cuda()
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)

    offset = grid - s_grid

    offset_x = offset[:, 0, :, :] * (w - 1) / 2
    offset_y = offset[:, 1, :, :] * (h - 1) / 2

    offset = torch.cat((offset_y, offset_x), 0)

    return offset


start_epoch, epoch_iter = 1, 0

f2c = flow_util.flow2color()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print(dataset_size)

warp_model = AFWM(opt, 3)
print(warp_model)
warp_model.eval()
#warp_model.cuda()
load_checkpoint(warp_model, opt.warp_checkpoint)

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
gen_model.eval()
#gen_model.cuda()
load_checkpoint(gen_model, opt.gen_checkpoint)

total_steps = (start_epoch - 1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size / opt.batchSize

if not os.path.exists('our_t_results'):
    os.mkdir('our_t_results')

for epoch in range(1, 2):

    for i, data in enumerate(dataset, start=epoch_iter):
        print(data)
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        real_image = data['image']
        clothes = data['clothes']
        edge = data['edge']
        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
        clothes = clothes * edge

        #flow_out = warp_model(real_image.cuda(), clothes.cuda())
        flow_out = warp_model(real_image, clothes)
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge, last_flow.permute(0, 2, 3, 1),
                                    mode='bilinear', padding_mode='zeros')

        gen_inputs = torch.cat([real_image, warped_cloth, warped_edge], 1)
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        path = 'results/' + opt.name
        os.makedirs(path, exist_ok=True)
        print(data['p_name'])

        if step % 1 == 0:

            utils.save_image(
                p_tryon,
                os.path.join('./our_t_results', data['p_name'][0]),
                nrow=int(1),
                normalize=True,
                value_range=(-1, 1),
            )

        step += 1
        if epoch_iter >= dataset_size:
            break

