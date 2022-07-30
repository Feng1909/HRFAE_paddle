import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from PIL import Image
from paddle import grad
from paddle import utils
import yaml
import random

from nets import *
from functions import *
from yep import *


class Trainer(nn.Layer):
    def __init__(self, config):
        super(Trainer, self).__init__()
        # Load Params
        self.n_iter = None
        self.loss_dis_gp = None
        self.loss_dis = None
        self.loss_gp = None
        self.realism_a_modif = None
        self.realism_b = None
        self.loss_class = None
        self.loss_recon = None
        self.loss_adver = None
        self.loss_tv = None
        self.loss_gen = None
        self.age_diff = None
        self.content_code_a = None
        self.target_age = None
        self.config = config
        # Networks
        self.enc = Encoder()
        self.dec = Decoder()
        self.mlp_style = Mod_Net()
        self.dis = Dis_PatchGAN()
        self.classifier = VGG()
        # Optimizers
        self.gen_params = list(self.enc.parameters()) + list(self.dec.parameters()) + list(self.mlp_style.parameters())
        self.dis_params = list(self.dis.parameters())
        self.scheduler = paddle.optimizer.lr.StepDecay(learning_rate=self.config['lr'], step_size=self.config['step_size'], gamma=self.config['gamma'], verbose=True)
        self.gen_opt = paddle.optimizer.Adam(learning_rate=self.scheduler, parameters=self.gen_params)
        self.scheduler = paddle.optimizer.lr.StepDecay(learning_rate=config['lr'], step_size=self.config['step_size'], gamma=self.config['gamma'], verbose=True)
        self.dis_opt = paddle.optimizer.Adam(learning_rate=self.scheduler, parameters=self.dis_params)
        self.mse_loss = paddle.nn.MSELoss(reduction='none')
        self.ce_loss = paddle.nn.CrossEntropyLoss()

    def L1loss(self, input, target):
        return paddle.mean(paddle.abs(input - target))

    def L2loss(self, input, target):
        return paddle.mean((input - target) ** 2)

    def CEloss(self, x, target_age):
        return self.ce_loss(x, target_age)

    def GAN_loss(self, x, real=True):
        if real:
            target = paddle.ones([x.size()]).type_as(x)
        else:
            target = paddle.zeros([x.size()]).type_as(x)
        return self.mse_loss(x, target)

    def grad_penalty_r1(self, net, x, coeff=10):
        # x.requires_grad = True
        real_predict = net(x)
        gradients = grad(outputs=real_predict.mean(), inputs=x, create_graph=True)
        gradients = gradients.view(gradients.size(0), -1)
        gradients_penality = (coeff / 2) * ((gradients.norm(2, dim=1) ** 2).mean())
        return gradients_penality

    def random_age(self, age_input, diff_val=20):
        age_output = age_input.clone()
        if diff_val > (self.config['age_max'] - self.config['age_min']) / 2:
            diff_val = (self.config['age_max'] - self.config['age_min']) // 2
        for i, age_ele in enumerate(age_output):
            if age_ele < self.config['age_min'] + diff_val:
                age_target = age_ele.clone()
                age_target = random.randint(age_ele + diff_val, self.config['age_max'])
            elif (self.config['age_min'] + diff_val) <= age_ele <= (self.config['age_max'] - diff_val):
                age_target = age_ele.clone()
                age_target = random.randint(self.config['age_min'] + 2 * diff_val, self.config['age_max'] + 1)
                if age_target <= age_ele + diff_val:
                    age_target = age_target - 2 * diff_val
            elif age_ele > self.config['age_max'] - diff_val:
                age_target = age_ele.clone()
                age_target = random.randint(self.config['age_min'], age_ele - diff_val)
            age_output[i] = age_target
        return age_output

    def gen_encode(self, x_a, age_a, training=False, target_age=0):
        if target_age:
            self.target_age = target_age
            age_modif = self.target_age * paddle.ones([age_a.size()]).type_as(age_a)
        else:
            age_modif = self.random_age(age_a, diff_val=25)

        # Generate modified image
        self.content_code_a, skip_1, skip_2 = self.enc(x_a)
        style_params_a = self.mlp_style(age_a)
        style_params_b = self.mlp_style(age_modif)

        x_a_recon = self.dec(self.content_code_a, style_params_a, skip_1, skip_2)
        # x_a_modif = self.dec(self.content_code_a, style_params_b, skip_1, skip_2)
        x_a_modif = x_a_recon

        return x_a_recon, x_a_modif, age_modif

    def compute_gen_loss(self, x_a, age_a, age_b):
        # Generate modified image
        x_a_recon, x_a_modif, age_a_modif = self.gen_encode(x_a, age_a, age_b, training=True)

        # Feed into discriminator
        realism_a_modif = self.dis(x_a_modif)
        predict_age_pb = self.classifier(vgg_transform(x_a_modif))['fc8']

        # Get predicted age
        predict_age = get_predict_age(predict_age_pb)
        self.age_diff = paddle.mean(paddle.abs(predict_age - age_a_modif.float()))

        # Classification loss
        self.loss_class = self.CEloss(predict_age_pb, age_a_modif)

        # Reconstruction loss
        self.loss_recon = self.L1loss(x_a_recon, x_a)

        # Adversarial loss
        self.loss_adver = self.GAN_loss(realism_a_modif, True)

        # Total Variation
        self.loss_tv = reg_loss(x_a_modif)

        self.loss_gen = self.config['w']['recon'] * self.loss_recon + \
                        self.config['w']['class'] * self.loss_class + \
                        self.config['w']['adver'] * self.loss_adver + \
                        self.config['w']['tv'] * self.loss_tv

        return self.loss_gen

    def compute_dis_loss(self, x_a, x_b, age_a, age_b):
        # Generate modified image
        x_a_recon, x_a_modif, age_a_modif = self.gen_encode(x_a, age_a, training=True)

        self.realism_b = self.dis(x_b)
        self.realism_a_modif = self.dis(x_a_modif.detach())

        self.loss_gp = self.grad_penalty_r1(self.dis, x_b)
        self.loss_dis = self.GAN_loss(self.realism_b, True).mean() + self.GAN_loss(self.realism_a_modif, True).mean()

        self.loss_dis_gp = self.config['w']['dis'] * self.loss_dis + self.config['w']['gp'] * self.loss_gp

        return self.loss_dis_gp

    def log_image(self, x_a, age_a, logger, n_epoch, n_iter):
        x_a_recon, x_a_modif, age_a_modif = self.gen_encode(x_a, age_a)
        logger.log_images('epoch' + str(n_epoch + 1) + '/iter' + str(n_iter + 1) + '/content', clip_img(x_a),
                          n_iter + 1)
        logger.log_images('epoch' + str(n_epoch + 1) + '/iter' + str(n_iter + 1) + '/content_recon' +
                          str(age_a.cpu().numpy()[0]), clip_img(x_a_recon), n_iter + 1)
        logger.log_images('epoch' + str(n_epoch + 1) + '/iter' + str(n_iter + 1) + '/content_modif_' +
                          str(age_a_modif.cpu().numpy()[0]), clip_img(x_a_modif), n_iter + 1)

    def log_loss(self, logger, n_iter):
        logger.log_value('loss/total', self.loss_gen.item() + self.loss_dis_gp.item(), n_iter + 1)
        logger.log_value('loss/recon', self.loss_recon.item(), n_iter + 1)
        logger.log_value('loss/class', self.loss_class.item(), n_iter + 1)
        logger.log_value('loss/adv', self.loss_adver.item(), n_iter + 1)
        logger.log_value('loss/dis', self.loss_dis_gp.item(), n_iter + 1)
        logger.log_value('age_diff', self.age_diff.item(), n_iter + 1)
        logger.log_value('dis/realism_A_modif', self.realism_a_modif.mean().item(), n_iter + 1)
        logger.log_value('dis/realism_B', self.realism_b.mean().item(), n_iter + 1)

    def save_image(self, x_a, age_a, log_dir, n_epoch, n_iter):
        x_a_recon, x_a_modif, age_a_modif = self.gen_encode(x_a, age_a)
        save_image(clip_img(x_a), log_dir + 'epoch' +str(n_epoch+1)+ 'iter' +str(n_iter+1)+ '_content.png')
        save_image(clip_img(x_a_recon), log_dir + 'epoch' +str(n_epoch+1)+ 'iter' +
                         str(n_iter+1)+ '_content_recon_'+str(age_a.cpu().numpy()[0])+'.png')
        save_image(clip_img(x_a_modif), log_dir + 'epoch' +str(n_epoch+1)+ 'iter' +
                         str(n_iter+1)+ '_content_modif_'+str(age_a_modif.cpu().numpy()[0])+'.png')

    def test_eval(self, x_a, age_a, target_age=0, hist_trans=True):
        _, x_a_modif, _= self.gen_encode(x_a, age_a, target_age=target_age)
        if hist_trans:
            for j in range(x_a_modif.size(0)):
                x_a_modif[j] = hist_transform(x_a_modif[j], x_a[j])
        return x_a_modif

    def save_model(self, log_dir):
        paddle.save(self.enc.state_dict(),'{:s}/enc.pdparams'.format(log_dir))
        paddle.save(self.mlp_style.state_dict(),'{:s}/mlp_style.pdparams'.format(log_dir))
        paddle.save(self.dec.state_dict(),'{:s}/dec.pdparams'.format(log_dir))
        paddle.save(self.dis.state_dict(),'{:s}/dis.pdparams'.format(log_dir))

    def update(self, x_a, x_b, age_a, age_b, n_iter):
        print("1")
        self.n_iter = n_iter
        print("2")
        self.compute_dis_loss(x_a, x_b, age_a, age_b).backward()
        print("3")
        self.dis_opt.step()
        print("4")
        self.dis_opt.clear_gradients()
        print("5")
        self.compute_gen_loss(x_a, x_b, age_a, age_b).backward()
        print("6")
        self.gen_opt.step()
        print("7")
        self.gen_opt.clear_gradients()
        print("8")




if __name__ == '__main__':
    config = yaml.safe_load(open('./configs/001' + '.yaml', 'r'))
    trainer = Trainer(config=config)
