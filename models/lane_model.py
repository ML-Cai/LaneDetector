import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import sys
import time
import datetime
from models.resnet_identity_block import ResnetIdentityBlock
from models.mobilenet_v2_block import MobilenetV2IdentityBlock
from losses.lane_loss import LaneLoss
from PIL import Image
import cv2

# ---------------------------------------------------
class SqueezeAndExcitiationBlock(tf.keras.Model):
    def __init__(self, dim,  name=''):
        super(SqueezeAndExcitiationBlock, self).__init__(name=name)
        H, W, C = dim

        # self.global_pool = keras.layers.MaxPool2D(pool_size=(H, W))
        self.global_pool = keras.layers.AveragePooling2D(pool_size=(H, W)) 
        self.dense_A     = keras.layers.Dense(units=C / 2)
        self.relu_A      = keras.layers.ReLU()
        self.dense_B     = keras.layers.Dense(units=C)
        self.reshape     = keras.layers.Reshape((1, 1, C))
    
    def call(self, x, training=False):
        x_in = x
        x = self.global_pool(x)
        x = self.dense_A(x)
        x = self.relu_A(x)
        x = self.dense_B(x)
        x = tf.nn.sigmoid(x)
        x = self.reshape(x)
        x = x_in * x
        
        return x

# ---------------------------------------------------
class FeatureDownsamplingBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), pool_size=(1, 1), padding='same', name=''):
        super(FeatureDownsamplingBlock, self).__init__(name=name)

        self.conv = keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=pool_size)
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        # self.pool = keras.layers.MaxPool2D(pool_size=pool_size)
        self.se = None
    
    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training= training)
        x = self.relu(x)
        # x = self.pool(x)

        # ---------------------------------
        # if self.se is None:
        #     H, W, C = x.shape.as_list()[1:]
        #     self.se = SqueezeAndExcitiationBlock(dim=(H, W, C))
        
        # x = self.se(x)
        # ---------------------------------

        return x

# ---------------------------------------------------
class FeatureUpsamplingBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name='', activation=True):
        super(FeatureUpsamplingBlock, self).__init__(name=name)

        self.conv = keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding=padding)
        self.bn = keras.layers.BatchNormalization()
        if activation:
            self.activeation = keras.layers.ReLU()
        else:
            self.activeation = None
        self.se = None
    
    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training= training)

        if not (self.activeation is None):
            x = self.activeation(x)

        # ---------------------------------
        # if self.se is None:
        #     H, W, C = x.shape.as_list()[1:]
        #     self.se = SqueezeAndExcitiationBlock(dim=(H, W, C))
        
        # x = self.se(x)
        # ---------------------------------

        return x

# ---------------------------------------------------
class DensenetBlock(tf.keras.Model):
    def __init__(self, filters, connect_count, name=''):
        super(DensenetBlock, self).__init__(name=name)

        self.se_list = []
        self.conv_list = []
        self.bn_list = []
        self.relu_list = []
        self.concate_list = []
        for ci in range(connect_count):
            self.conv_list.append(keras.layers.Conv2D(filters, (3, 3), padding='same'))
            self.bn_list.append(keras.layers.BatchNormalization())
            self.relu_list.append(keras.layers.ReLU())
            self.concate_list.append(keras.layers.Concatenate())
            self.se_list.append(None)
        
        self.connect_count = connect_count

    def call(self, x, training=False):
        
        for ci in range(self.connect_count):
            inputs = x
            x = self.conv_list[ci](x)
            x = self.bn_list[ci](x)
            x = self.relu_list[ci](x)

            # -------------------------------------------
            # se
            
            # if self.se_list[ci] is None:
            #     H, W, C = x.shape.as_list()[1:]
            #     self.se_list[ci] = SqueezeAndExcitiationBlock(dim=(H, W, C))
            
            # x = self.se_list[ci](x)
        
            # -------------------------------------------
            x = self.concate_list[ci]([x, inputs])
        
        return x



def AlphaLaneModel(net_input_img_size, x_anchors, y_anchors, max_lane_count, name=''):
    # input = keras.Input(shape=(net_input_img_size[1], net_input_img_size[0], 3))
    # x = input
    # x = FeatureDownsamplingBlock(10, kernel_size=(3, 3), pool_size=(2, 2), padding='same', name='FeatureDownsamplingBlock_1')(x)
    # x = FeatureDownsamplingBlock(20, kernel_size=(3, 3), pool_size=(2, 2), padding='same', name='FeatureDownsamplingBlock_2')(x)
    # x = FeatureDownsamplingBlock(30, kernel_size=(3, 3), pool_size=(2, 2), padding='same', name='FeatureDownsamplingBlock_3')(x)

    # # dense
    # x = MobilenetV2IdentityBlock(kernel_size=(3, 3), filters=(30*6, 30))(x)
    # x = MobilenetV2IdentityBlock(kernel_size=(3, 3), filters=(30*6, 30))(x)
    # x_denseA = x

    # ## down
    # x = FeatureDownsamplingBlock(64, kernel_size=(3, 3), pool_size=(2, 2), padding='same', name='FeatureDownsamplingBlock_5')(x)
        
    # # dense
    # x = MobilenetV2IdentityBlock(kernel_size=(3, 3), filters=(64*6, 64))(x)
    # x = MobilenetV2IdentityBlock(kernel_size=(3, 3), filters=(64*6, 64))(x)
    # x = MobilenetV2IdentityBlock(kernel_size=(3, 3), filters=(64*6, 64))(x)
        
    # # upsampling A
    # x = FeatureUpsamplingBlock(64, kernel_size=(3, 3), padding='same', strides=(2, 2), name='FeatureDownsamplingBlock_7')(x)
    # x = MobilenetV2IdentityBlock(kernel_size=(3, 3), filters=(64*6, 64))(x)
    # x = MobilenetV2IdentityBlock(kernel_size=(3, 3), filters=(64*6, 64))(x)
    
    # # concate
    # x = keras.layers.Concatenate()([x, x_denseA])

    # # upsampling B
    # x = FeatureUpsamplingBlock(max_lane_count, kernel_size=(3, 3), padding='same', strides=(2, 2), activation=False, name='FeatureDownsamplingBlock_9')(x)
    # x = keras.layers.Softmax(axis=2)(x)

    # # cvt 72 128 4 --> 4 72 128
    # output = keras.backend.permute_dimensions(x, (0, 3, 1, 2))  # transpose

    # return keras.Model(input, output, name="test")




    input = keras.Input(shape=(net_input_img_size[1], net_input_img_size[0], 3))
    x = input
    x = FeatureDownsamplingBlock(10, kernel_size=(3, 3), pool_size=(2, 2), padding='same', name='FeatureDownsamplingBlock_1')(x)
    x = FeatureDownsamplingBlock(20, kernel_size=(3, 3), pool_size=(2, 2), padding='same', name='FeatureDownsamplingBlock_2')(x)
    x_feature2 = x

    x = FeatureDownsamplingBlock(30, kernel_size=(3, 3), pool_size=(2, 2), padding='same', name='FeatureDownsamplingBlock_3')(x)

    # dense
    x = DensenetBlock(filters=20, connect_count=4, name='DensenetBlock_4')(x)

    # se
    H, W, C = x.shape.as_list()[1:]
    x = SqueezeAndExcitiationBlock(dim=(H, W, C))(x)
    x_denseA = x

    ## down
    x = FeatureDownsamplingBlock(128, kernel_size=(3, 3), pool_size=(2, 2), padding='same', name='FeatureDownsamplingBlock_5')(x)

    # se
    H, W, C = x.shape.as_list()[1:]
    x = SqueezeAndExcitiationBlock(dim=(H, W, C))(x)

    # dense
    x = DensenetBlock(filters=20, connect_count=8, name='DensenetBlock_6')(x)

    # se
    H, W, C = x.shape.as_list()[1:]
    x = SqueezeAndExcitiationBlock(dim=(H, W, C))(x)

    # upsampling A
    x = FeatureUpsamplingBlock(128, kernel_size=(3, 3), padding='same', strides=(2, 2), name='FeatureDownsamplingBlock_7')(x)

    # concate
    x = keras.layers.Concatenate()([x, x_denseA])

    # se
    H, W, C = x.shape.as_list()[1:]
    x = SqueezeAndExcitiationBlock(dim=(H, W, C))(x)

    # upsampling B
    x = FeatureUpsamplingBlock(max_lane_count, kernel_size=(5, 5), padding='same', strides=(2, 2), activation=False, name='FeatureDownsamplingBlock_9')(x)
    x = keras.layers.Softmax(axis=2)(x)
    x = keras.backend.permute_dimensions(x, (0, 3, 1, 2))  # cvt 72 128 4 --> 4 72 128
    output = x

    # x = FeatureUpsamplingBlock(32, kernel_size=(3, 3), padding='same', strides=(2, 2), name='FeatureDownsamplingBlock_9')(x)
    # x = keras.layers.Concatenate()([x, x_feature2])
    # x = keras.layers.Conv2D(max_lane_count, kernel_size=(1, 1), padding='same')(x)
    # x = keras.layers.Softmax(axis=2)(x)
    # x = keras.backend.permute_dimensions(x, (0, 3, 1, 2))  # cvt 72 128 4 --> 4 72 128
    # output = x

    # H, W, C = x.shape.as_list()[1:]
    # x = keras.layers.AveragePooling2D(pool_size=(H, W))(x)
    # x = keras.layers.Dense(128)(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Dense(x_anchors* y_anchors* max_lane_count)(x)
    # x = keras.layers.Reshape((max_lane_count, y_anchors, x_anchors))(x)
    # x = keras.layers.Softmax()(x)
    # output = x

    return keras.Model(input, output, name="test")









