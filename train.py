import argparse
import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import yaml

from PIL import Image

from datasets import *
from nets import *
from functions import *
from trainer import *

# paddle.enable_static()
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
parser.add_argument('--dataset_path', type=str, default='./data/ffhq/', help='dataset path')
parser.add_argument('--label_file_path', type=str, default='./data/ffhq.npy', help='label file path')
parser.add_argument('--vgg_model_path', type=str, default='./models/dex_imdb_wiki.caffemodel.pt', help='pretrained age classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--multigpu', type=bool, default=False, help='use multiple gpus')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
opts = parser.parse_args()

log_dir = os.path.join(opts.log_path, opts.config) + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# logger = Logger(log_dir)

config = yaml.safe_load(open('./configs/' + opts.config + '.yaml', 'r'))
epochs = config['epochs']
age_min = config['age_min']
age_max = config['age_max']

batch_size = 4
img_size = (512, 512)

# Load dataset
dataset_A = MyDataSet(age_min, age_max, opts.dataset_path, opts.label_file_path, output_size=img_size, training_set=True)
dataset_B = MyDataSet(age_min, age_max, opts.dataset_path, opts.label_file_path, output_size=img_size, training_set=True)
loader_A = paddle.io.DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=0)
loader_B = paddle.io.DataLoader(dataset_B, batch_size=batch_size, shuffle=True, num_workers=0)


trainer = Trainer(config)
epoch_0 = 0

# Start Training
n_iter = 0
print("Start!")
print('Reconstruction weight: ', config['w']['recon'])
print('Classification weight: ', config['w']['class'])
print('Adversarial loss weight: ', config['w']['adver'])
print('TV weight: ', config['w']['tv'])

for n_epoch in range(epoch_0, epoch_0+epochs):
    print(n_epoch)

    if n_epoch == 10:
        trainer.config['w']['recon'] = 0.1*trainer.config['w']['recon']
        # Load dataset at 1024 x 1024 resolution for the next 10 epochs
        batch_size = config['batch_size']
        img_size = (config['input_h'], config['input_w'])
        dataset_A = MyDataSet(age_min, age_max, opts.dataset_path, opts.label_file_path, output_size=img_size,
                              training_set=True)
        dataset_B = MyDataSet(age_min, age_max, opts.dataset_path, opts.label_file_path, output_size=img_size,
                              training_set=True)
        loader_A = paddle.io.DataLoader(dataset_A, batch_size=batch_size, shuffle=True, num_workers=0)
        loader_B = paddle.io.DataLoader(dataset_B, batch_size=batch_size, shuffle=True, num_workers=0)

    iter_B = iter(loader_B)
    for i, list_A in enumerate(loader_A):
        image_A, age_A = list_A
        image_B, age_B = next(iter_B)
        if age_A.size != batch_size: break
        if age_B.size != batch_size:
            iter_B = iter(loader_B)
            image_B, age_B = next(iter_B)

        # image_A, age_A = image_A.to(device), age_A.to(device)
        # image_B, age_B = image_B.to(device), age_B.to(device)
        trainer.update(image_A, image_B, age_A, age_B, n_iter)

        # if (n_iter + 1) % config['log_iter'] == 0:
        #     trainer.log_loss(logger, n_iter)
        # if (n_iter + 1) % config['image_log_iter'] == 0:
        #     trainer.log_image(image_A, age_A, logger, n_epoch, n_iter)
        if (n_iter + 1) % config['image_save_iter'] == 0:
            trainer.save_image(image_A, age_A, log_dir, n_epoch, n_iter)

        n_iter += 1

    trainer.gen_scheduler.step()
    trainer.dis_scheduler.step()
