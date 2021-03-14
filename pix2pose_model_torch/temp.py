import sys, os
import numpy as np
import scipy.ndimage
import tensorflow as tf

import keras
from keras import backend as K
from keras.initializers import glorot_normal, glorot_normal
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout, Activation, RepeatVector, Lambda, Reshape, Subtract
from keras.layers import Concatenate
from keras.layers import Layer
from keras.layers import merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras import losses
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
import pix2pose_model_tf.resnet50_mod as resnet
import math


###########
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import itertools

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only. If you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")








class transformer_loss(Layer):
    def __init__(self, sym=0, **kwargs):
        self.sym = sym

        super(transformer_loss, self).__init__(**kwargs)

    def build(self, input_shape):
        super(transformer_loss, self).build(input_shape)

    def call(self, x):
        y_pred = x[0]
        y_recont_gt = x[1]
        y_prob_pred = tf.squeeze(x[2], axis=3)
        y_prob_gt = x[3]
        visible = tf.cast(y_prob_gt > 0.5, y_pred.dtype)
        visible = tf.squeeze(visible, axis=3)
        # generate transformed values using sym
        if (len(self.sym) > 1):
            # if(True):
            for sym_id, transform in enumerate(self.sym):  # 3x3 matrix
                tf_mat = tf.convert_to_tensor(transform, y_recont_gt.dtype)
                y_gt_transformed = tf.transpose(tf.matmul(tf_mat, tf.transpose(tf.reshape(y_recont_gt, [-1, 3]))))
                y_gt_transformed = tf.reshape(y_gt_transformed, [-1, 128, 128, 3])
                loss_xyz_temp = K.sum(K.abs(y_gt_transformed - y_pred), axis=3) / 3
                loss_sum = K.sum(loss_xyz_temp, axis=[1, 2])
                if (sym_id > 0):
                    loss_sums = tf.concat([loss_sums, tf.expand_dims(loss_sum, axis=0)], axis=0)
                    loss_xyzs = tf.concat([loss_xyzs, tf.expand_dims(loss_xyz_temp, axis=0)], axis=0)
                else:
                    loss_sums = tf.expand_dims(loss_sum, axis=0)
                    loss_xyzs = tf.expand_dims(loss_xyz_temp, axis=0)

            min_values = tf.reduce_min(loss_sums, axis=0, keepdims=True)
            loss_switch = tf.cast(tf.equal(loss_sums, min_values), y_pred.dtype)
            loss_xyz = tf.expand_dims(tf.expand_dims(loss_switch, axis=2), axis=3) * loss_xyzs
            loss_xyz = K.sum(loss_xyz, axis=0)
        else:
            loss_xyz = K.sum(K.abs(y_recont_gt - y_pred), axis=3) / 3
        prob_loss = K.square(y_prob_pred - K.minimum(loss_xyz, 1))
        loss_invisible = (1 - visible) * loss_xyz
        loss_visible = visible * loss_xyz
        loss = loss_visible * 3 + loss_invisible + 0.5 * prob_loss
        loss = K.mean(loss, axis=[1, 2])
        return loss

    def compute_output_shape(self, input_shape):
        return (tuple([input_shape[0][0], 1]))


class aemodel_unet_prob(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        # ============== YOUR CODE HERE ============== #

        # construct unet structure
        input_nc, output_nc, ngf = 3, 3, 64

        ##encoder
        self.down1 = [nn.Conv2d(input_nc, ngf, kernel_size=(5, 5), stride=(2, 2), padding_mode=(2, 2)),
                      nn.BatchNorm2d(ngf, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        self.down2 = [nn.Conv2d(ngf, ngf, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.BatchNorm2d(ngf * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        self.down3 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.BatchNorm2d(ngf * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        self.down4 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.BatchNorm2d(ngf * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        self.down5 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.BatchNorm2d(ngf * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        self.down6 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.BatchNorm2d(ngf * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        self.down7 = [nn.Conv2d(ngf * 4, ngf * 4, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.BatchNorm2d(ngf * 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        self.down8 = [nn.Conv2d(ngf * 4, ngf * 4, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                      nn.BatchNorm2d(ngf * 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        self.medium = [nn.Linear(out_features=64 * 4),
                       nn.Linear(out_features=64 * 4 * 8 * 8)]

        ##decoder
        self.up1 = [nn.ConvTranspose2d(ngf * 4, ngf * 4, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                    nn.BatchNorm2d(ngf * 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)]
                    #nn.Dropout2d(p=0.5, inplace=True)

        self.up2 = [nn.Conv2d(ngf * 4, ngf * 4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                    nn.BatchNorm2d(ngf * 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        self.up3 = [nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                    nn.BatchNorm2d(ngf * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        self.up4 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                    nn.BatchNorm2d(ngf * 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        self.up5 = [nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                    nn.BatchNorm2d(ngf, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        self.up6 = [nn.Conv2d(ngf, ngf * 2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                    nn.BatchNorm2d(ngf * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        self.up7 = [nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                    nn.Tanh()]
        self.up8 = [nn.ConvTranspose2d(ngf * 2, 1, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                    nn.Sigmoid()]

        self.down1 = nn.Sequential(*self.down1)
        self.down2 = nn.Sequential(*self.down2)
        self.down3 = nn.Sequential(*self.down3)
        self.down4 = nn.Sequential(*self.down4)
        self.down5 = nn.Sequential(*self.down5)
        self.down6 = nn.Sequential(*self.down6)
        self.down7 = nn.Sequential(*self.down7)
        self.down8 = nn.Sequential(*self.down8)

        self.medium = nn.Sequential(*self.medium)

        self.up1 = nn.Sequential(*self.up1)
        self.up2 = nn.Sequential(*self.up2)
        self.up3 = nn.Sequential(*self.up3)
        self.up4 = nn.Sequential(*self.up4)
        self.up5 = nn.Sequential(*self.up5)
        self.up6 = nn.Sequential(*self.up6)
        self.up7 = nn.Sequential(*self.up7)
        self.up8 = nn.Sequential(*self.up8)

        #self.last = nn.Sequential(*self.last)

        # ============== END OF CODE ================= #
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        # ++++++++++++++++++++++++++++++++++++++++++++++#

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        # ============== YOUR CODE HERE ============== #
        d1 = self.down1(input)
        d2 = self.down2(input)
        d12_cat = torch.cat([d1, d2], 3)


        d3 = self.down3(d12_cat)
        d4 = self.down4(d12_cat)
        d34_cat = torch.cat([d3, d4], 3)

        d5 = self.down5(d34_cat)
        d6 = self.down6(d34_cat)
        d56_cat = torch.cat([d5, d6], 3)

        d7 = self.down7(d56_cat)
        d8 = self.down8(d56_cat)
        d78_cat = torch.cat([d7, d8], 3)

        d_medium = torch.flatten(d78_cat)
        d_medium = self.medium(d_medium)
        d_medium = d_medium.Reshape(-1, 8, 8, 256)

        u1 = self.up1(d_medium)
        u1 = torch.cat([u1, d6], 3)

        u2 = self.up2(u1)
        #u2 = torch.cat([u2, d6], 3)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d4], 3)


        u4 = self.up4(u3)
        #u4 = torch.cat([u4, d4], 3)
        u5 = self.up5(u4)
        u5 = torch.cat([u5, d2], 3)


        u6 = self.up6(u5)
        #u6 = torch.cat([u6, d2], 3)


        u7 = self.up7(u6)
        #u7 = torch.cat([u7, d1], 3)
        u8 = self.up8(u6)
        #o = self.last(u8)

        # ============== END OF CODE ================= #
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        # ++++++++++++++++++++++++++++++++++++++++++++++#

        return u7, u8


class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        # ============== YOUR CODE HERE ============== #
        input_nc, output_nc, ngf = 6, 1, 64

        self.net = [nn.Conv2d(input_nc, ngf, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),

                    nn.Conv2d(ngf, ngf * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                    nn.BatchNorm2d(ngf * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),

                    nn.Conv2d(ngf * 2, ngf * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                    nn.BatchNorm2d(ngf * 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),

                    nn.Conv2d(ngf * 4, ngf * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                    nn.BatchNorm2d(ngf * 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),

                    nn.Conv2d(ngf * 8, output_nc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                    nn.Sigmoid()]

        self.net = nn.Sequential(*self.net)

        # ============== END OF CODE ================= #
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        # ++++++++++++++++++++++++++++++++++++++++++++++#

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        # ============== YOUR CODE HERE ============== #
        x = torch.cat((label, input), 1)
        x = self.net(x)

        # ============== END OF CODE ================= #
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        # ++++++++++++++++++++++++++++++++++++++++++++++#
        return x


def aemodel_unet_prob(p=0.5):
    input_img = Input(shape=(128, 128, 3))
    bn_axis = 3

    f1_1 = Conv2D(64, (5, 5), strides=(2, 2), name='conv1_1', padding='same')(input_img)
    f1_1 = BatchNormalization(axis=bn_axis)(f1_1)

    # f1_1 = Activation('relu')(f1_1) #64x64x64
    f1_1 = LeakyReLU()(f1_1)
    f1_2 = Conv2D(64, (5, 5), strides=(2, 2), name='conv1_2', padding='same')(input_img)
    f1_2 = BatchNormalization(axis=bn_axis)(f1_2)
    f1_2 = LeakyReLU()(f1_2)
    f1 = Concatenate()([f1_1, f1_2])  # 64x64x128

    f2_1 = Conv2D(128, (5, 5), strides=(2, 2), name='conv2_1', padding='same')(f1)
    f2_1 = BatchNormalization(axis=bn_axis)(f2_1)
    f2_1 = LeakyReLU()(f2_1)
    f2_2 = Conv2D(128, (5, 5), strides=(2, 2), name='conv2_2', padding='same')(f1)
    f2_2 = BatchNormalization(axis=bn_axis)(f2_2)
    f2_2 = LeakyReLU()(f2_2)
    f2 = Concatenate()([f2_1, f2_2])  # 32x32x256

    f3_1 = Conv2D(128, (5, 5), strides=(2, 2), name='conv3_1', padding='same')(f2)
    f3_1 = BatchNormalization(axis=bn_axis)(f3_1)
    f3_1 = LeakyReLU()(f3_1)
    f3_2 = Conv2D(128, (5, 5), strides=(2, 2), name='conv3_2', padding='same')(f2)
    f3_2 = BatchNormalization(axis=bn_axis)(f3_2)
    f3_2 = LeakyReLU()(f3_2)
    f3 = Concatenate()([f3_1, f3_2])  # 16x16x256

    f4_1 = Conv2D(256, (5, 5), strides=(2, 2), name='conv4_1', padding='same')(f3)
    f4_1 = BatchNormalization(axis=bn_axis)(f4_1)
    f4_1 = LeakyReLU()(f4_1)
    f4_2 = Conv2D(256, (5, 5), strides=(2, 2), name='conv4_2', padding='same')(f3)
    f4_2 = BatchNormalization(axis=bn_axis)(f4_2)
    f4_2 = LeakyReLU()(f4_2)
    f4 = Concatenate()([f4_1, f4_2])  # 8x8x512

    x = Flatten()(f4)
    encoded = Dense(256)(x)  # 128:default, 256:large #bottle
    d1 = Dense(8 * 8 * 256)(encoded)
    d1 = Reshape((8, 8, -1))(d1)  # 8x8x256
    d1 = Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same')(d1)  # 16x16x256
    d1 = BatchNormalization(axis=bn_axis)(d1)
    # d1 = Dropout(p)(d1)
    d1 = LeakyReLU()(d1)

    d1_uni = Concatenate()([d1, f3_2])  # 16x16x256
    d1_uni = Conv2D(256, (5, 5), strides=(1, 1), name='deconv1', padding='same')(d1_uni)  # 16x16x256
    d1_uni = BatchNormalization(axis=bn_axis)(d1_uni)  #
    # d1_uni = Dropout(p)(d1_uni)
    d1_uni = LeakyReLU()(d1_uni)  #

    d2 = Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(d1_uni)  # 32x32x128
    d2 = BatchNormalization(axis=bn_axis)(d2)
    # d2 = Dropout(p)(d2)
    d2 = LeakyReLU()(d2)
    d2_uni = Concatenate()([d2, f2_2])  # 32x32x256
    d2_uni = Conv2D(256, (5, 5), strides=(1, 1), name='deconv2', padding='same')(d2_uni)  # 32x32x256
    d2_uni = BatchNormalization(axis=bn_axis)(d2_uni)  #
    d2_uni = LeakyReLU()(d2_uni)  #
    # to 32x32x256

    d3 = Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same')(d2_uni)  # 64x64x64
    d3 = BatchNormalization(axis=bn_axis)(d3)  #
    # d3 = Dropout(p)(d3)
    d3 = LeakyReLU()(d3)
    d3_uni = Concatenate()([d3, f1_2])  # 64x64x128
    d3_uni = Conv2D(128, (5, 5), strides=(1, 1), name='deconv3', padding='same')(d3_uni)  # 64x64x128
    d3_uni = BatchNormalization(axis=bn_axis)(d3_uni)  #
    d3_uni = LeakyReLU()(d3_uni)
    # to 64x64x128

    decoded = Conv2DTranspose(3, kernel_size=(5, 5), strides=(2, 2), padding='same')(d3_uni)  # 128x128x3
    decoded = Activation('tanh')(decoded)  # 8x8x256
    pixel_prob = Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), padding='same')(d3_uni)  # 128x128x3
    pixel_prob = Activation('sigmoid')(pixel_prob)  # 8x8x256
    # has to be sigmoid..
    generator_train = Model(inputs=[input_img], outputs=[decoded, pixel_prob])

    return generator_train


def DCGAN_discriminator():
    nb_filters = 64
    nb_conv = int(np.floor(np.log(128) / np.log(2)))
    list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    input_img = Input(shape=(128, 128, 3))
    x = Conv2D(list_filters[0], (3, 3), strides=(2, 2), name="disc_conv2d_1", padding="same")(input_img)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(0.2)(x)
    # Next convs
    for i, f in enumerate(list_filters[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name, padding="same")(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU(0.2)(x)

    x_flat = Flatten()(x)
    x_out = Dense(1, activation="sigmoid", name="disc_dense")(x_flat)
    discriminator_model = Model(inputs=input_img, outputs=[x_out])
    return discriminator_model


def aemodel_unet_resnet50(p=0.5):
    bn_axis = 3
    input_img = Input(shape=(128, 128, 3))
    resnet_model = resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    resnet_part = Model(inputs=resnet_model.input,
                        outputs=[resnet_model.get_layer('act_conv1').output,
                                 resnet_model.get_layer('act2c_branch').output,
                                 resnet_model.get_layer('act3d_branch').output])

    f1, f2, f3 = resnet_part(input_img)

    f1_2 = Lambda(lambda x: x[:, :, :, :32])(f1)
    f2_2 = Lambda(lambda x: x[:, :, :, :128])(f2)
    f3_2 = Lambda(lambda x: x[:, :, :, :128])(f3)

    f4_1 = Conv2D(256, (5, 5), strides=(2, 2), name='conv4_1', padding='same')(f3)
    f4_1 = BatchNormalization(axis=bn_axis)(f4_1)
    f4_1 = LeakyReLU()(f4_1)
    f4_2 = Conv2D(256, (5, 5), strides=(2, 2), name='conv4_2', padding='same')(f3)
    f4_2 = BatchNormalization(axis=bn_axis)(f4_2)
    f4_2 = LeakyReLU()(f4_2)
    f4 = Concatenate()([f4_1, f4_2])  # 8x8x512

    x = Flatten()(f4)
    encoded = Dense(256)(x)  # 128:default, 256:large #bottle
    d1 = Dense(8 * 8 * 256)(encoded)
    d1 = Reshape((8, 8, -1))(d1)  # 8x8x256
    d1 = Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same')(d1)  # 16x16x256
    d1 = BatchNormalization(axis=bn_axis)(d1)
    # d1 = Dropout(p)(d1)
    d1 = LeakyReLU()(d1)

    d1_uni = Concatenate()([d1, f3_2])  # 16x16x256
    d1_uni = Conv2D(256, (5, 5), strides=(1, 1), name='deconv1', padding='same')(d1_uni)  # 16x16x256
    d1_uni = BatchNormalization(axis=bn_axis)(d1_uni)  #
    # d1_uni = Dropout(p)(d1_uni)
    d1_uni = LeakyReLU()(d1_uni)  #

    d2 = Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(d1_uni)  # 32x32x128
    d2 = BatchNormalization(axis=bn_axis)(d2)
    # d2 = Dropout(p)(d2)
    d2 = LeakyReLU()(d2)
    d2_uni = Concatenate()([d2, f2_2])  # 32x32x256
    d2_uni = Conv2D(256, (5, 5), strides=(1, 1), name='deconv2', padding='same')(d2_uni)  # 32x32x256
    d2_uni = BatchNormalization(axis=bn_axis)(d2_uni)  #
    d2_uni = LeakyReLU()(d2_uni)  #
    # to 32x32x256

    d3 = Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same')(d2_uni)  # 64x64x64
    d3 = BatchNormalization(axis=bn_axis)(d3)  #
    # d3 = Dropout(p)(d3)
    d3 = LeakyReLU()(d3)
    d3_uni = Concatenate()([d3, f1_2])  # 64x64x128
    d3_uni = Conv2D(128, (5, 5), strides=(1, 1), name='deconv3', padding='same')(d3_uni)  # 64x64x128
    d3_uni = BatchNormalization(axis=bn_axis)(d3_uni)  #
    d3_uni = LeakyReLU()(d3_uni)
    # to 64x64x128

    decoded = Conv2DTranspose(3, kernel_size=(5, 5), strides=(2, 2), padding='same')(d3_uni)  # 128x128x3
    decoded = Activation('tanh')(decoded)  # 8x8x256
    pixel_prob = Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), padding='same')(d3_uni)  # 128x128x3
    pixel_prob = Activation('sigmoid')(pixel_prob)  # 8x8x256
    # has to be sigmoid..
    generator_train = Model(inputs=[input_img], outputs=[decoded, pixel_prob])

    return generator_train