import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from PIL import Image
from paddle import grad


def clip_img(x):
    img_tmp = x.clone()[0]
    img_tmp[0] += 0.48501961
    img_tmp[1] += 0.45795686
    img_tmp[2] += 0.40760392
    img_tmp = paddle.clip(img_tmp, 0, 1)
    return [img_tmp.detach().cpu()]


def hist_transform(source_tensor, target_tensor):
    c, h, w = source_tensor.size()
    s_t = source_tensor.view(c, -1)
    t_t = target_tensor.view(c, -1)
    s_t_sorted, s_t_indices = paddle.sort(s_t)
    t_t_sorted, t_t_indices = paddle.sort(t_t)
    for i in range(c):
        s_t[i, s_t_indices[i]] = t_t_sorted[i]
    return s_t.view(c, h, w)


def reg_loss(img):
    reg_loss_ = paddle.mean(paddle.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) \
                + paddle.mean(paddle.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return reg_loss_


def vgg_transform(x):
    r, g, b = paddle.split(x, 1, 1)
    out = paddle.concat((b, g, r), axis=1)
    out = F.interpolate(out, size=(224, 224), mode='bilinear')
    out = out*255
    return out


def get_predict_age(age_pb):
    predict_age_pb = F.softmax(age_pb)
    predict_age = paddle.zeros([age_pb.size(0)]).type_as(predict_age_pb)
    for i in range(age_pb.size(0)):
        for j in range(age_pb.size(1)):
            predict_age[i] += j*predict_age_pb[i][j]

    return  predict_age