import logging
import torch
from torch import nn

from .perception_loss import vgg16_feat, perceptual_loss
from .base_model import BaseModel
from . import networks
# from parse_config import ConfigParser
# import parse_config
import numpy as np


class DadganModelG(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
    By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    def __init__(self, opt, device):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        opt.gpu_ids = [device]
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN', 'G_L1', 'G_perceptual']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>

        self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionSeg = nn.BCEWithLogitsLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if opt.client_optimizer == "sgd":
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay,
                                                   nesterov=opt.nesterov)
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay,
                                                    amsgrad=True)

            self.optimizers.append(self.optimizer_G)

            self.vgg_model = vgg16_feat().to(self.gpu_ids[0])  #.cuda()
            self.criterion_perceptual = perceptual_loss()

    def set_input(self, input):
        self.real_A = input['A'].to(self.gpu_ids[0])

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        # logging.info(f"forward:{self.real_A.shape}, return:{self.fake_B.shape}")
        return self.fake_B.cpu().detach().numpy()

    def backward_G(self, grad_fake_B):
        """Calculate gradients of the generator from gradients of output"""
        grad_fake_B = grad_fake_B.to(self.gpu_ids[0])
        self.optimizer_G.zero_grad()
        self.fake_B.backward(grad_fake_B)
        self.optimizer_G.step()
        return

    def optimize_parameters(self):
        pass

    def evaluate(self, input):
        real_A = input['A'].to(self.gpu_ids[0])
        # real_B = input['B'].to(self.gpu_ids[0])

        with torch.no_grad():
            # logging.info(f"evaluate, real_A:{real_A.shape}")
            fake_B = self.netG(real_A)  # G(A)

        ## TODO: compare real_B and fake_B?

        return fake_B.cpu().detach().numpy()


class DadganModelD(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
    By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    def __init__(self, opt, device):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        opt.gpu_ids = [device]
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>

        self.model_names = ['D']

        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.gpu_ids[0])
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionSeg = nn.BCEWithLogitsLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if opt.client_optimizer == "sgd":
                self.optimizer_D = torch.optim.SGD(self.netD.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay,
                                                   nesterov=opt.nesterov)

            else:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay,
                                                    amsgrad=True)

            self.optimizers.append(self.optimizer_D)

            self.vgg_model = vgg16_feat().to(self.gpu_ids[0])  #.cuda()
            self.criterion_perceptual = perceptual_loss()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        # self.image_paths = []
        self.real_A = input['A'].to(self.gpu_ids[0])
        self.real_B = input['B'].to(self.gpu_ids[0])
        # self.image_paths.append(input['A_paths_' + str(i)])

    def backward_D(self, fake_B):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, fake_B), 1)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake_AB.detach())
        pred_real = self.netD(real_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * self.opt.lambda_D  # 0.5
        loss_val = self.loss_D.item()
        self.loss_D.backward()

        return loss_val

    def backward_G(self, fake_B):
        """Calculate GAN and L1 loss for the generator"""

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(fake_B, self.real_B) * self.opt.lambda_L1
        # Third, perceptual loss
        pred_feat = self.vgg_model(networks.gray2color(fake_B))
        target_feat = self.vgg_model(networks.gray2color(self.real_B))
        self.loss_G_perceptual = self.criterion_perceptual(pred_feat, target_feat) * self.opt.lambda_perceptual

        self.loss_G = (self.loss_G_GAN + self.loss_G_L1 + self.loss_G_perceptual)*self.opt.lambda_G  #0.1

        loss_val = self.loss_G.item()
        self.loss_G.backward()

        return loss_val

    def forward(self):
        pass

    def optimize_parameters(self):
        pass

    def optimize(self, fake_B):
        # fake_B = fake_B.new_tensor(fake_B, device=self.gpu_ids[0], requires_grad=True)
        fake_B = fake_B.to(self.gpu_ids[0])
        losses_dict = {}
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        loss_D = self.backward_D(fake_B)                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        losses_dict['loss_D'] = loss_D
        losses_dict['loss_D_fake'] = self.loss_D_fake.item()
        losses_dict['loss_D_real'] = self.loss_D_real.item()
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        fake_B.retain_grad()
        fake_B.requires_grad_(True)
        if fake_B.grad:
            fake_B.grad.zero_()         # set fake_B's gradients to zero
        loss_G = self.backward_G(fake_B)                   # calculate graidents for G
        losses_dict['loss_G'] = loss_G
        losses_dict['loss_G_GAN'] = self.loss_G_GAN.item()
        losses_dict['loss_G_L1'] = self.loss_G_L1.item()
        losses_dict['loss_G_perceptual'] = self.loss_G_perceptual.item()
        # get gradients of fake_B
        grad_fake_B = fake_B.grad.cpu().detach().numpy()
        # print('grad_fake_B: ', grad_fake_B.max(), grad_fake_B.min())
        return losses_dict, grad_fake_B

    def evaluate(self, input, fake_B):
        real_A = input['A'].to(self.gpu_ids[0])
        real_B = input['B'].to(self.gpu_ids[0])

        losses_dict = {}
        with torch.no_grad():
            fake_AB = torch.cat((real_A, fake_B), 1)
            real_AB = torch.cat((real_A, real_B), 1)
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake = self.netD(fake_AB.detach())
            pred_real = self.netD(real_AB)
            loss_D_fake = self.criterionGAN(pred_fake, False).item()
            loss_D_real = self.criterionGAN(pred_real, True).item()

            # D loss
            losses_dict['loss_D'] = (loss_D_fake + loss_D_real) * self.opt.lambda_D
            losses_dict['loss_D_fake'] = loss_D_fake
            losses_dict['loss_D_real'] = loss_D_real

            loss_G_GAN = self.criterionGAN(pred_fake, True).item()
            loss_G_L1 = self.criterionL1(fake_B, real_B).item()
            pred_feat = self.vgg_model(networks.gray2color(fake_B))
            target_feat = self.vgg_model(networks.gray2color(real_B))
            loss_G_perceptual = self.criterion_perceptual(pred_feat, target_feat).item()

            losses_dict['loss_G'] = (loss_G_GAN + loss_G_L1 * self.opt.lambda_L1 + loss_G_perceptual * self.opt.lambda_perceptual) * self.opt.lambda_G
            losses_dict['loss_G_GAN'] = loss_G_GAN
            losses_dict['loss_G_L1'] = loss_G_L1
            losses_dict['loss_G_perceptual'] = loss_G_perceptual

        return losses_dict
