import tensorflow as tf
import tensorflow_model_optimization as tfmot
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
class Default8BitOutputQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig which only quantizes the output from a layer."""

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
        num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]

  def get_config(self):
    return {}

# ---------------------------------------------------
class Conv2DTranspose_QuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig which only quantizes the output from a layer."""

  def get_weights_and_quantizers(self, layer):
    return []

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
        num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]

  def get_config(self):
    return {}



# ---------------------------------------------------
class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """QuantizeConfig which does not quantize any part of the layer."""
    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []
        # return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
        # num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]


    def get_config(self):
        return {}


# ---------------------------------------------------
# class SqueezeAndExcitiationBlock(tf.keras.Model):
#     def __init__(self, dim,  name=''):
#         super(SqueezeAndExcitiationBlock, self).__init__(name=name)
#         H, W, C = dim

#         # self.global_pool = keras.layers.MaxPool2D(pool_size=(H, W))
#         self.global_pool = tf.keras.layers.AveragePooling2D(pool_size=(H, W)) 
#         self.dense_A     = tf.keras.layers.Dense(units=C / 2)
#         self.relu_A      = tf.keras.layers.ReLU()
#         self.dense_B     = tf.keras.layers.Dense(units=C)
#         self.reshape     = tf.keras.layers.Reshape((1, 1, C))
    
#     def call(self, x, training=False):
#         x_in = x
#         x = self.global_pool(x)
#         x = self.dense_A(x)
#         x = self.relu_A(x)
#         x = self.dense_B(x)
#         x = tf.nn.sigmoid(x)
#         x = self.reshape(x)
#         x = x_in * x
        
#         return x

# ---------------------------------------------------
# class FeatureDownsamplingBlock(tf.keras.Model):
#     def __init__(self, filters, kernel_size=(3, 3), pool_size=(1, 1), padding='same', name=''):
#         super(FeatureDownsamplingBlock, self).__init__(name=name)

#         self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=pool_size)
#         self.bn = tf.keras.layers.BatchNormalization()
#         self.relu = tf.keras.layers.ReLU()
#         # self.pool = tf.keras.layers.MaxPool2D(pool_size=pool_size)
#         self.se = None
    
#     def call(self, x, training=False):
#         x = self.conv(x)
#         x = self.bn(x, training= training)
#         x = self.relu(x)
#         # x = self.pool(x)

#         # ---------------------------------
#         # if self.se is None:
#         #     H, W, C = x.shape.as_list()[1:]
#         #     self.se = SqueezeAndExcitiationBlock(dim=(H, W, C))
        
#         # x = self.se(x)
#         # ---------------------------------

#         return x

# ---------------------------------------------------
# class FeatureUpsamplingBlock(tf.keras.Model):
#     def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name='', activation=True):
#         super(FeatureUpsamplingBlock, self).__init__(name=name)

#         self.conv = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding=padding)
#         self.bn = tf.keras.layers.BatchNormalization()
#         if activation:
#             self.activeation = tf.keras.layers.ReLU()
#         else:
#             self.activeation = None
#         self.se = None
    
#     def call(self, x, training=False):
#         x = self.conv(x)
#         x = self.bn(x, training= training)

#         if not (self.activeation is None):
#             x = self.activeation(x)

#         # ---------------------------------
#         # if self.se is None:
#         #     H, W, C = x.shape.as_list()[1:]
#         #     self.se = SqueezeAndExcitiationBlock(dim=(H, W, C))
        
#         # x = self.se(x)
#         # ---------------------------------

#         return x

# ---------------------------------------------------
# class DensenetBlock(tf.keras.Model):
#     def __init__(self, filters, connect_count, name=''):
#         super(DensenetBlock, self).__init__(name=name)

#         self.se_list = []
#         self.conv_list = []
#         self.bn_list = []
#         self.relu_list = []
#         self.concate_list = []
#         for ci in range(connect_count):
#             self.conv_list.append(tf.keras.layers.Conv2D(filters, (3, 3), padding='same'))
#             self.bn_list.append(tf.keras.layers.BatchNormalization())
#             self.relu_list.append(tf.keras.layers.ReLU())
#             self.concate_list.append(tf.keras.layers.Concatenate())
#             self.se_list.append(None)
        
#         self.connect_count = connect_count

#     def call(self, x, training=False):
        
#         for ci in range(self.connect_count):
#             inputs = x
#             x = self.conv_list[ci](x)
#             x = self.bn_list[ci](x)
#             x = self.relu_list[ci](x)

#             # -------------------------------------------
#             # se
            
#             # if self.se_list[ci] is None:
#             #     H, W, C = x.shape.as_list()[1:]
#             #     self.se_list[ci] = SqueezeAndExcitiationBlock(dim=(H, W, C))
            
#             # x = self.se_list[ci](x)
        
#             # -------------------------------------------
#             x = self.concate_list[ci]([x, inputs])
        
#         return x


# ---------------------------------------------------
def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2

    x = tf.keras.layers.Reshape((height, width, 2, channels_per_split))(x)
    x = tf.keras.layers.Permute((1, 2, 4, 3))(x)
    x = tf.keras.layers.Reshape((height, width, channels))(x)

    return x

# ---------------------------------------------------
def _FeatureDownsamplingBlock(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name='', quantization_aware_training=False):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides, use_bias=False, name=name+'_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'_bn')(x)
    x = tf.keras.layers.ReLU(name=name+'_relu')(x)
    
    # input = x
    # x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    # x_out = x

    # x = input
    # x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)

    # x = tfmot.quantization.keras.quantize_annotate_layer(
    #                 tf.keras.layers.Concatenate(name=name+'_concat'), quantize_config=Default8BitOutputQuantizeConfig()) ([x, x_out])
    # x = channel_shuffle(x)
    
    return x

# ---------------------------------------------------
def _FeatureUpsamplingBlock(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name='', quantization_aware_training=False):
    if not quantization_aware_training:
        # upsampling A
        x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, name=name+'_convTranspose')(x)
        # x = tf.keras.layers.BatchNormalization(name=name+'_bn')(x)
        
    else:
        # upsampling A
        # x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = tfmot.quantization.keras.quantize_annotate_layer(
            tf.keras.layers.Conv2DTranspose(filters,
                                            kernel_size=kernel_size,
                                            strides=strides,
                                            padding=padding,
                                            use_bias=False,
                                            name=name+'_convTranspose'),
                                            quantize_config=Conv2DTranspose_QuantizeConfig()) (x)
        
        # x = tf.keras.layers.Dropout()(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tfmot.quantization.keras.quantize_annotate_layer(
        #     tf.keras.layers.BatchNormalization(name=name+'_bn'), quantize_config=NoOpQuantizeConfig()) (x)

    return x

# ---------------------------------------------------
def _DensenetBlock(x, filters, connect_count, name='', quantization_aware_training=False):
    if not quantization_aware_training:
        for ci in range(connect_count):
            inputs = x
            x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same', name=name+'_conv2'+str(ci), use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(name=name+'_bn2'+str(ci))(x)
            x = tf.keras.layers.ReLU(name=name+'_relu2'+str(ci))(x)
            x = tf.keras.layers.Concatenate()([x, inputs])
    else:
        for ci in range(connect_count):
            inputs = x
            x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding='same', name=name+'_conv2'+str(ci), use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization(name=name+'_bn2'+str(ci))(x)
            x = tf.keras.layers.ReLU(name=name+'_relu2'+str(ci))(x)

            # x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x)
            # x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
            # x = tf.keras.layers.BatchNormalization()(x)
            # x = tf.keras.layers.ReLU()(x)


            x = tfmot.quantization.keras.quantize_annotate_layer(
                    tf.keras.layers.Concatenate(), quantize_config=NoOpQuantizeConfig()) ([x, inputs])

    return x



    # if not quantization_aware_training:
    #     for ci in range(connect_count):
    #         inputs = x
    #         x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), dilation_rate=(1, 1), padding='same', use_bias=False)(x)
    #         x = tf.keras.layers.BatchNormalization()(x)
    #         x = tf.keras.layers.ReLU()(x)
    #         x = tf.keras.layers.Concatenate()([x, inputs])
    # else:
    #     for ci in range(connect_count):
    #         inputs = x
    #         x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), dilation_rate=(1, 1), padding='same', use_bias=False)(x)
    #         x = tf.keras.layers.BatchNormalization()(x)
    #         x = tf.keras.layers.ReLU()(x)
    #         x = tfmot.quantization.keras.quantize_annotate_layer(
    #                 tf.keras.layers.Concatenate(), quantize_config=NoOpQuantizeConfig()) ([x, inputs])
    # return x



# ---------------------------------------------------
class FeatureDownsamplingBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 name=None,
                 is_training=True):
        super(FeatureDownsamplingBlock, self).__init__(name=name)

        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU(max_value=6.0)

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training= training)
        x = self.relu(x)

        return x

# ---------------------------------------------------
class MarkingSegmentationFeatureDecoder(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(MarkingSegmentationFeatureDecoder, self).__init__()
        self.softmax_x_anchors = tf.keras.layers.Softmax()
        self.pooling = tf.keras.layers.MaxPool2D(pool_size=(10, 1))

    def call(self, inputs):
        batch, lane_count, y_anchors, x_anchors = inputs.get_shape().as_list()
        
        x = inputs
        x = self.softmax_x_anchors(inputs)
        
        # x = self.pooling(x)

        # max_list = []
        # for i in range(lane_count):
        #     m = tf.slice(x, [0, i, 0, 0], [-1, 1, -1, x_anchors -1])
        #     max_list.append(m)
        
        # x = tf.keras.layers.Maximum()(max_list)

        return x

# ---------------------------------------------------
def AlphaLaneModel(net_input_img_size,
                   x_anchors,
                   y_anchors,
                   max_lane_count,
                   name='',
                   training=True,
                   input_batch_size=None,
                   quantization_aware_training=False):

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  97% accuracy
    input = tf.keras.Input(name='input', shape=(net_input_img_size[1], net_input_img_size[0], 3), batch_size=input_batch_size)
    x = input

    QAT = tfmot.quantization.keras.quantize_annotate_layer

    # FeatureDownsamplingBlock A
    # x = FeatureDownsamplingBlock(filters=16, kernel_size=(3, 3), padding='same', strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', strides=(2, 2), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)

    # FeatureDownsamplingBlock B
    x_in = x
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x_in = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), padding='same', use_bias=False, strides=(2, 2))(x_in)
    x_in = tf.keras.layers.BatchNormalization()(x_in)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)


    # FeatureDownsamplingBlock C
    x_in = x
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x_in = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), padding='same', use_bias=False, strides=(2, 2))(x_in)
    x_in = tf.keras.layers.BatchNormalization()(x_in)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)

    x_in = x
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)

    x_feature_C = x

    # FeatureDownsamplingBlock D
    x_in = x
    x = tf.keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x_in = tf.keras.layers.Conv2D(96, kernel_size=(1, 1), padding='same', use_bias=False, strides=(2, 2))(x_in)
    x_in = tf.keras.layers.BatchNormalization()(x_in)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)

    x_in = x
    x = tf.keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)

    x_in = x
    x = tf.keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)

    # FeatureUpsamplingBlock
    if not quantization_aware_training:
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    else:
        x = QAT(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'),
                quantize_config=NoOpQuantizeConfig()) (x)

    # concat
    if not quantization_aware_training:
        x = tf.keras.layers.Concatenate()([x, x_feature_C])
    else:
        x = QAT(tf.keras.layers.Concatenate(),
                quantize_config=NoOpQuantizeConfig()) ([x, x_feature_C])
    
    x_in = x
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x_in = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), padding='same', use_bias=False)(x_in)
    x_in = tf.keras.layers.BatchNormalization()(x_in)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)
    
    x_in = x
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)

    if not quantization_aware_training:
        x = tf.keras.layers.UpSampling2D(size=(1, 2), interpolation='nearest')(x)
    else:
        x = QAT(tf.keras.layers.UpSampling2D(size=(1, 2), interpolation='nearest'),
                quantize_config=NoOpQuantizeConfig()) (x)

    x_in = x
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x_in = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), padding='same', use_bias=False)(x_in)
    x_in = tf.keras.layers.BatchNormalization()(x_in)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)
        
    x_in = x
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)

    x = tf.keras.layers.Conv2D(max_lane_count, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    confidence = x


    # output
    if not quantization_aware_training:
        x = tf.keras.layers.Permute((3, 1, 2), name='output')(x)
    else:
        x = tfmot.quantization.keras.quantize_annotate_layer(
            tf.keras.layers.Permute((3, 1, 2), name='output') , quantize_config=NoOpQuantizeConfig()) (x)
    
    # for (index, net) in enumerate(x):
    #     tf.print("index ", index)
    #     # tf.print(" , net", tf.shape(net))
    # tf.print("----------------")

    if training:
        output = x
    else:
        # x = tfmot.quantization.keras.quantize_annotate_layer(
            # tf.keras.layers.Softmax(axis=3),  quantize_config=NoOpQuantizeConfig()) (x)
        x = MarkingSegmentationFeatureDecoder()(x)
        
        output = x


    if quantization_aware_training:
        with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig,
                                                 'Default8BitOutputQuantizeConfig':Default8BitOutputQuantizeConfig,
                                                 'Conv2DTranspose_QuantizeConfig':Conv2DTranspose_QuantizeConfig}):
            model = tf.keras.Model(input, output, name="AlphaLaneNet")
            q_aware_model = tfmot.quantization.keras.quantize_model(model)
            
            return q_aware_model
    else:
        with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig}):
            model = tf.keras.Model(input, output, name="AlphaLaneNet")

            return model
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  97% accuracy

    
    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  97% accuracy
    # input = tf.keras.Input(name='input', shape=(net_input_img_size[1], net_input_img_size[0], 3), batch_size=input_batch_size)
    # x = input

    # QAT = tfmot.quantization.keras.quantize_annotate_layer

    # # FeatureDownsamplingBlock A
    # x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same', strides=(2, 2), use_bias=False, name='conv_A')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU(6)(x)

    # # FeatureDownsamplingBlock B
    # x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', strides=(2, 2), use_bias=False, name='conv_B')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU(6)(x)

    # # FeatureDownsamplingBlock C
    # x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', strides=(2, 2), use_bias=False, name='conv_C')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU(6)(x)
    
    # # DenseBlock A
    # # x = _DensenetBlock(x, filters=32, connect_count=4, name='DensenetBlock_4', quantization_aware_training=quantization_aware_training)
    # x_in = x
    # x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU(6)(x)
    # x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Add()([x, x_in])
    # x = tf.keras.layers.ReLU(6)(x)


    # x_feature_C = x

    # # FeatureDownsamplingBlock D
    # # x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    # # x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', strides=(2, 2), use_bias=False)(x)
    # # x = tf.keras.layers.BatchNormalization()(x)
    # # x = tf.keras.layers.ReLU(6)(x)

    # # dense 
    # # x = _DensenetBlock(x, filters=32, connect_count=6, name='DensenetBlock_6', quantization_aware_training=quantization_aware_training)
    # x_in = x
    # x = tf.keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2), name='conv_F')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU(6)(x)
    # x = tf.keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x_in = tf.keras.layers.Conv2D(96, kernel_size=(1, 1), padding='same', use_bias=False, strides=(2, 2))(x_in)
    # x_in = tf.keras.layers.BatchNormalization()(x_in)
    # x = tf.keras.layers.Add()([x, x_in])
    # x = tf.keras.layers.ReLU(6)(x)

    # x_in = x
    # x = tf.keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', use_bias=False, name='conv_G')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU(6)(x)
    # x = tf.keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Add()([x, x_in])
    # x = tf.keras.layers.ReLU(6)(x)

    # x_in = x
    # x = tf.keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', use_bias=False, name='conv_H')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU(6)(x)
    # x = tf.keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Add()([x, x_in])
    # x = tf.keras.layers.ReLU(6)(x)


    # # FeatureUpsamplingBlock
    # # x = _FeatureUpsamplingBlock(x, 128, kernel_size=(3, 3), strides=(2, 2), padding='same', name ='up_7', quantization_aware_training=quantization_aware_training)
    # # x = tf.keras.layers.ReLU(6)(x)
     
    # x_in = x
    # x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU(6)(x)
    # x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x_in = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), padding='same', use_bias=False)(x_in)
    # x_in = tf.keras.layers.BatchNormalization()(x_in)
    # x = tf.keras.layers.Add()([x, x_in])
    # x = tf.keras.layers.ReLU(6)(x)

    # if not quantization_aware_training:
    #     x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    # else:
    #     x = QAT(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'),
    #             quantize_config=NoOpQuantizeConfig()) (x)


    # # concat
    # if not quantization_aware_training:
    #     x = tf.keras.layers.Concatenate()([x, x_feature_C])
    # else:
    #     x = QAT(tf.keras.layers.Concatenate(),
    #             quantize_config=NoOpQuantizeConfig()) ([x, x_feature_C])

    # if not quantization_aware_training:
    #     x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    # else:
    #     x = QAT(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'),
    #             quantize_config=NoOpQuantizeConfig()) (x)

    # x_in = x
    # x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU(6)(x)
    # x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x_in = tf.keras.layers.Conv2D(8, kernel_size=(1, 1), padding='same', use_bias=False)(x_in)
    # x_in = tf.keras.layers.BatchNormalization()(x_in)
    # x = tf.keras.layers.Add()([x, x_in])
    # x = tf.keras.layers.ReLU(6)(x)
   

    # x = tf.keras.layers.Conv2D(max_lane_count, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)




    # # # FeatureUpsamplingBlock
    # # # x = _FeatureUpsamplingBlock(x, max_lane_count, kernel_size=(3, 3), strides=(2, 2), padding='same', name ='up_8', quantization_aware_training=quantization_aware_training)
    # # x = tfmot.quantization.keras.quantize_annotate_layer(
    # #         tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'),
    # #         quantize_config=NoOpQuantizeConfig()) (x)
    # # x = tf.keras.layers.Conv2D(max_lane_count, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # # x = tf.keras.layers.BatchNormalization()(x)
    # # x = tf.keras.layers.ReLU(6)(x)

    # # output
    # x = tfmot.quantization.keras.quantize_annotate_layer(
    #     tf.keras.layers.Permute((3, 1, 2), name='output') , quantize_config=NoOpQuantizeConfig()) (x)
    
    # output = x


    # if quantization_aware_training:
    #     with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig,
    #                                              'Default8BitOutputQuantizeConfig':Default8BitOutputQuantizeConfig,
    #                                              'Conv2DTranspose_QuantizeConfig':Conv2DTranspose_QuantizeConfig}):
    #         model = tf.keras.Model(input, output, name="AlphaLaneNet")
    #         q_aware_model = tfmot.quantization.keras.quantize_model(model)
            
    #         return q_aware_model
    # else:
    #     with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig}):
    #         model = tf.keras.Model(input, output, name="AlphaLaneNet")

    #         return model
    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  97% accuracy
    




    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  95% accuracy
    # input = tf.keras.Input(name='input', shape=(net_input_img_size[1], net_input_img_size[0], 3), batch_size=input_batch_size)
    # x = input

    # # FeatureDownsamplingBlock A
    # x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same', strides=(2, 2), use_bias=False, name='conv_A')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)

    # # FeatureDownsamplingBlock B
    # x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', strides=(2, 2), use_bias=False, name='conv_B')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)

    # # FeatureDownsamplingBlock C
    # x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', strides=(2, 2), use_bias=False, name='conv_C')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    
    # # DenseBlock A
    # # x = _DensenetBlock(x, filters=32, connect_count=4, name='DensenetBlock_4', quantization_aware_training=quantization_aware_training)
    # x_in = x
    # x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False, name='conv_D')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x_in = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), padding='same', use_bias=False)(x_in)
    # x_in = tf.keras.layers.BatchNormalization()(x_in)
    # x = tf.keras.layers.Add()([x, x_in])
    # x = tf.keras.layers.ReLU()(x)

    # x_in = x
    # x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False, name='conv_E')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Add()([x, x_in])
    # x = tf.keras.layers.ReLU()(x)


    # x_feature_C = x

    # # FeatureDownsamplingBlock D
    # x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    # # dense 
    # # x = _DensenetBlock(x, filters=32, connect_count=6, name='DensenetBlock_6', quantization_aware_training=quantization_aware_training)
    # x_in = x
    # x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False, name='conv_F')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x_in = tf.keras.layers.Conv2D(128, kernel_size=(1, 1), padding='same', use_bias=False)(x_in)
    # x_in = tf.keras.layers.BatchNormalization()(x_in)
    # x = tf.keras.layers.Add()([x, x_in])
    # x = tf.keras.layers.ReLU()(x)

    # x_in = x
    # x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False, name='conv_G')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Add()([x, x_in])
    # x = tf.keras.layers.ReLU()(x)

    # x_in = x
    # x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False, name='conv_H')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Add()([x, x_in])
    # x = tf.keras.layers.ReLU()(x)


    # # FeatureUpsamplingBlock
    # x = _FeatureUpsamplingBlock(x, 128, kernel_size=(3, 3), strides=(2, 2), padding='same', name ='up_7', quantization_aware_training=quantization_aware_training)
    # # x = tf.keras.layers.ReLU()(x)

    # # concat
    # if not quantization_aware_training:
    #     x = tf.keras.layers.Concatenate()([x, x_feature_C])
    # else:
    #     x = tfmot.quantization.keras.quantize_annotate_layer(
    #             tf.keras.layers.Concatenate(),
    #             quantize_config=NoOpQuantizeConfig()) ([x, x_feature_C])

    # # FeatureUpsamplingBlock
    # x = _FeatureUpsamplingBlock(x, max_lane_count, kernel_size=(3, 3), strides=(2, 2), padding='same', name ='up_8', quantization_aware_training=quantization_aware_training)


    # # output
    # x = tfmot.quantization.keras.quantize_annotate_layer(
    #     tf.keras.layers.Permute((3, 1, 2), name='output') , quantize_config=NoOpQuantizeConfig()) (x)
    
    # output = x


    # if quantization_aware_training:
    #     with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig,
    #                                              'Default8BitOutputQuantizeConfig':Default8BitOutputQuantizeConfig}):
    #         model = tf.keras.Model(input, output, name="AlphaLaneNet")
    #         q_aware_model = tfmot.quantization.keras.quantize_model(model)
            
    #         return q_aware_model
    # else:
    #     with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig}):
    #         model = tf.keras.Model(input, output, name="AlphaLaneNet")

    #         return model
    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  95% accuracy
    




    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  90% accuracy
    # input = tf.keras.Input(name='input', shape=(net_input_img_size[1], net_input_img_size[0], 3), batch_size=input_batch_size)
    # x = input

    # # FeatureDownsamplingBlock A
    # x = _FeatureDownsamplingBlock(x, 8, kernel_size=(3, 3), padding='same', strides=(2, 2), name ='down_1', quantization_aware_training = quantization_aware_training)
    
    # # FeatureDownsamplingBlock B
    # x = _FeatureDownsamplingBlock(x, 16, kernel_size=(3, 3), padding='same', strides=(2, 2), name ='down_2', quantization_aware_training = quantization_aware_training)
    # # x = channel_shuffle(x)

    # # FeatureDownsamplingBlock C
    # x = _FeatureDownsamplingBlock(x, 32, kernel_size=(3, 3), padding='same', strides=(2, 2), name ='down_3', quantization_aware_training = quantization_aware_training)
    # x_feature_C = x

    # # FeatureDownsamplingBlock C
    # x = _FeatureDownsamplingBlock(x, 64, kernel_size=(3, 3), padding='same', strides=(2, 2), name ='down_5', quantization_aware_training = quantization_aware_training)
    
    # # dense V
    # x = _DensenetBlock(x, filters=16, connect_count=12, name='DensenetBlock_6', quantization_aware_training=quantization_aware_training)

    # # FeatureUpsamplingBlock B
    # x = _FeatureUpsamplingBlock(x, 32, kernel_size=(3, 3), strides=(2, 2), padding='same', name ='up_7', quantization_aware_training=quantization_aware_training)

    # # concat
    # if not quantization_aware_training:
    #     x = tf.keras.layers.Concatenate()([x, x_feature_C])
    # else:
    #     x = tfmot.quantization.keras.quantize_annotate_layer(
    #             tf.keras.layers.Concatenate(), quantize_config=NoOpQuantizeConfig()) ([x, x_feature_C])

    # x = _DensenetBlock(x, filters=8, connect_count=4, name='DensenetBlock_7', quantization_aware_training=quantization_aware_training)


    # # FeatureUpsamplingBlock B
    # x = _FeatureUpsamplingBlock(x, max_lane_count, kernel_size=(5, 5), strides=(2, 2), padding='same', name ='up_8', quantization_aware_training=quantization_aware_training)
    
    # x = tf.keras.layers.Softmax(axis=2)(x)
    
    
    # # output
    # if not quantization_aware_training:
    #     x = tf.keras.layers.Permute((3, 1, 2), name='output')(x)  # cvt 72 128 4 --> 4 72 128
    # else:
    #     x = tfmot.quantization.keras.quantize_annotate_layer(
    #         tf.keras.layers.Permute((3, 1, 2), name='output') , quantize_config=NoOpQuantizeConfig()) (x)
    
    # output = x

    

    # if quantization_aware_training:
    #     with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig,
    #                                              'Default8BitOutputQuantizeConfig':Default8BitOutputQuantizeConfig}):
    #         model = tf.keras.Model(input, output, name="AlphaLaneNet")
    #         q_aware_model = tfmot.quantization.keras.quantize_model(model)
            
    #         return q_aware_model
    # else:
    #     with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig}):
    #         model = tf.keras.Model(input, output, name="AlphaLaneNet")

    #         return model
    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  90% accuracy
    

    



    # x = FeatureDownsamplingBlock(10, kernel_size=(3, 3), pool_size=(2, 2), padding='same', name='FeatureDownsamplingBlock_1')(x)
    # x = FeatureDownsamplingBlock(20, kernel_size=(3, 3), pool_size=(2, 2), padding='same', name='FeatureDownsamplingBlock_2')(x)
    # x = FeatureDownsamplingBlock(30, kernel_size=(3, 3), pool_size=(2, 2), padding='same', name='FeatureDownsamplingBlock_3')(x)
    # x_feature_3 = x

    # # dense
    # x = DensenetBlock(filters=20, connect_count=4, name='DensenetBlock_4')(x)

    # # se
    # H, W, C = x.shape.as_list()[1:]
    # x = SqueezeAndExcitiationBlock(dim=(H, W, C))(x)
    # x_denseA = x

    # ## down
    # x = FeatureDownsamplingBlock(128, kernel_size=(3, 3), pool_size=(2, 2), padding='same', name='FeatureDownsamplingBlock_5')(x)

    # # se
    # H, W, C = x.shape.as_list()[1:]
    # x = SqueezeAndExcitiationBlock(dim=(H, W, C))(x)

    # # dense
    # x = DensenetBlock(filters=20, connect_count=6, name='DensenetBlock_6')(x)

    # # se
    # H, W, C = x.shape.as_list()[1:]
    # x = SqueezeAndExcitiationBlock(dim=(H, W, C))(x)

    # # upsampling A
    # x = FeatureUpsamplingBlock(128, kernel_size=(3, 3), padding='same', strides=(2, 2), name='FeatureDownsamplingBlock_7')(x)

    # # concate (x_feature will increase 2% accuracy, but not perform help at different angle of camera)
    # x = tf.keras.layers.Concatenate()([x, x_denseA, x_feature_3])

    # # se
    # H, W, C = x.shape.as_list()[1:]
    # x = SqueezeAndExcitiationBlock(dim=(H, W, C))(x)

    # # upsampling B
    # x = FeatureUpsamplingBlock(max_lane_count, kernel_size=(5, 5), padding='same', strides=(2, 2), activation=False, name='FeatureDownsamplingBlock_9')(x)
    # x = tf.keras.layers.Softmax(axis=2)(x)
    # x = tf.keras.backend.permute_dimensions(x, (0, 3, 1, 2))  # cvt 72 128 4 --> 4 72 128
    # output = x

    # return tf.keras.Model(input, output, name="test")









