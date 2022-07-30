# Copyright (c) 2020, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from PIL import Image
# from torchvision import transforms, utils
from paddle.io import Dataset
from paddle.vision import transforms
import cv2


class MyDataSet(Dataset):
    def __init__(self, age_min, age_max, image_dir, label_dir, output_size=(256, 256), training_set=True,
                 obscure_age=True):
        self.image_dir = image_dir
        self.transform = transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392], std=[1, 1, 1])
        self.resize = transforms.Compose([
            transforms.Resize(size=output_size[0]),
            transforms.ToTensor()
        ])

        # load label file
        label = np.load(label_dir)
        label = label[:9]
        train_len = int(0.95 * len(label))
        self.training_set = training_set
        self.obscure_age = obscure_age
        if training_set:
            label = label[:train_len]
        else:
            label = label[train_len:]
        a_mask = np.zeros(len(label), dtype=bool)
        for i in range(len(label)):
            if int(label[i, 1]) in range(age_min, age_max): a_mask[i] = True
        self.label = label[a_mask]
        self.length = len(self.label)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 1000:
            self.image_dir = 'data/ffhq'
        elif index < 2000:
            self.image_dir = 'data/01000'
        elif index < 3000:
            self.image_dir = 'data/02000'
        elif index < 4000:
            self.image_dir = 'data/03000'
        elif index < 5000:
            self.image_dir = 'data/04000'
        elif index < 6000:
            self.image_dir = 'data/05000'
        elif index < 7000:
            self.image_dir = 'data/06000'
        elif index < 8000:
            self.image_dir = 'data/07000'
        elif index < 9000:
            self.image_dir = 'data/08000'
        img_name = os.path.join(self.image_dir, self.label[index][0])
        if self.training_set and self.obscure_age:
            age_val = int(self.label[index][1]) + np.random.randint(-1, 1)
        else:
            age_val = int(self.label[index][1])
        age = paddle.to_tensor(age_val)

        image = cv2.imread(img_name)
        image = Image.fromarray(image)
        img = self.resize(image)
        if img.size == 1:
            img = paddle.concat((img, img, img), axis=0)
        img = self.transform(img)

        return img, age
