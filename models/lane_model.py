import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import sys
import time
import datetime
from losses.lane_loss import LaneLoss
from PIL import Image
import cv2

# ---------------------------------------------------
class FeatureDownsamplingBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 name=None,
                 trainable=True):
        super(FeatureDownsamplingBlock, self).__init__(name=name)

        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           padding=padding,
                                           strides=strides,
                                           use_bias=False,
                                           trainable=trainable)
        self.bn   = tf.keras.layers.BatchNormalization(trainable=trainable)
        self.relu = tf.keras.layers.ReLU(6.0, trainable=trainable)

    def call(self, x, training=True):
        
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)

        return x

# ---------------------------------------------------
class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 name=None,
                 trainable=True):
        super(ResnetBlock, self).__init__(name=name)

        self.filters = filters

        self.conv_1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same', use_bias=False, strides=strides, trainable=trainable)
        self.bn_1 = tf.keras.layers.BatchNormalization(trainable=trainable)
        self.relu_1 = tf.keras.layers.ReLU(6, trainable=trainable)

        self.conv_2 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same', use_bias=False, trainable=trainable)
        self.bn_2 = tf.keras.layers.BatchNormalization(trainable=trainable)

        self.conv_sc = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False, strides=strides, trainable=trainable)
        self.bn_sc = tf.keras.layers.BatchNormalization(trainable=trainable)

        self.add = tf.keras.layers.Add(trainable=trainable)
        self.relu_out = tf.keras.layers.ReLU(6, trainable=trainable)

    def call(self, x, training=True):
        x_in = x
        
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)

        _, _, inC = x_in.shape.as_list()[1:]
        if self.filters != inC:
            sc = self.conv_sc(x_in)
            sc = self.bn_sc(sc)
            x = self.add([x, sc])
            x = self.relu_out(x)
        else:
            x = self.add([x, x_in])
            x = self.relu_out(x)

        return x

# ---------------------------------------------------
class Backbone(tf.keras.layers.Layer):
    def __init__(self,
                 trainable=True):
        super(Backbone, self).__init__()

        # FeatureDownsamplingBlock A
        self.feature_A1 = FeatureDownsamplingBlock(filters=16, kernel_size=(3, 3), strides=(2, 2), trainable=trainable)

        # FeatureDownsamplingBlock B
        self.feature_B1 = ResnetBlock(filters=32, kernel_size=(3, 3), strides=(2, 2), trainable=trainable)
        self.feature_B2 = ResnetBlock(filters=32, kernel_size=(3, 3), strides=(1, 1), trainable=trainable)

        # FeatureDownsamplingBlock C
        self.feature_C1 = ResnetBlock(filters=64, kernel_size=(3, 3), strides=(2, 2), trainable=trainable)
        self.feature_C2 = ResnetBlock(filters=64, kernel_size=(3, 3), strides=(1, 1), trainable=trainable)

        # FeatureDownsamplingBlock D
        self.feature_D1 = ResnetBlock(filters=128, kernel_size=(3, 3), strides=(2, 2), trainable=trainable)
        self.feature_D2 = ResnetBlock(filters=128, kernel_size=(3, 3), strides=(1, 1), trainable=trainable)
        self.feature_D3 = ResnetBlock(filters=128, kernel_size=(3, 3), strides=(1, 1), trainable=trainable)

        # FeatureUpsamplingBlock
        self.upSampling_E1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')

        # concat
        self.concate_C2_E1 = tf.keras.layers.Concatenate()
        
    def call(self, x, training=False):
        x = self.feature_A1(x, training=False)

        x = self.feature_B1(x, training=False)
        x = self.feature_B2(x, training=False)

        x = self.feature_C1(x, training=False)
        x = self.feature_C2(x, training=False)
        x_feature_C = x

        x = self.feature_D1(x, training=False)
        x = self.feature_D2(x, training=False)
        x = self.feature_D3(x, training=False)

        x = self.upSampling_E1(x)
        x = self.concate_C2_E1([x, x_feature_C])

        return x

# ---------------------------------------------------
class _CoordinateChannel(tf.keras.layers.Layer):
    def __init__(self, rank,
                 use_radius=False,
                 data_format=None,
                 **kwargs):
        super(_CoordinateChannel, self).__init__(**kwargs)

        if data_format not in [None, 'channels_first', 'channels_last']:
            raise ValueError('`data_format` must be either "channels_last", "channels_first" '
                             'or None.')

        self.rank = rank
        self.use_radius = use_radius
        self.data_format = tf.keras.backend.image_data_format() if data_format is None else data_format
        self.axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[self.axis]
        print("_CoordinateChannel input_dim ", input_shape)
        
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=self.rank + 2, axes={self.axis: input_dim})
        self.built = True

    def call(self, inputs, training=None, mask=None):
        input_shape = inputs.get_shape()
        # input_shape = tf.shape(inputs)  # training

        if self.rank == 2:
            input_shape = [input_shape[i] for i in range(4)]
            batch_size, H, W, C = input_shape

            xx_channels = tf.range(0.0, float(W), dtype=tf.float32)
            xx_channels = tf.expand_dims(xx_channels, axis=0)
            xx_channels = tf.tile(xx_channels, [H, 1])
            xx_channels = xx_channels / (float(W) - 1.0)
            xx_channels = (xx_channels * 2.0) - 1.0
            xx_channels = tf.expand_dims(xx_channels, axis=0)
            xx_channels = tf.expand_dims(xx_channels, axis=-1)
            xx_channels = tf.tile(xx_channels, [batch_size, 1, 1, 1])
                        
            yy_channels = tf.range(0.0, float(W), dtype=tf.float32)
            yy_channels = tf.expand_dims(yy_channels, axis=1)
            yy_channels = tf.tile(yy_channels, [1, W])
            yy_channels = yy_channels / (float(H) - 1.0)
            yy_channels = yy_channels * 2.0 -1.0
            yy_channels = tf.expand_dims(yy_channels, axis=0)
            yy_channels = tf.expand_dims(yy_channels, axis=-1)
            yy_channels = tf.tile(yy_channels, [batch_size, 1, 1, 1])

            outputs = tf.concat([inputs, xx_channels, yy_channels], axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[self.axis]

        if self.use_radius and self.rank == 2:
            channel_count = 3
        else:
            channel_count = self.rank

        output_shape = list(input_shape)
        output_shape[self.axis] = input_shape[self.axis] + channel_count
        return tuple(output_shape)

    def get_config(self):
        config = {
            'rank': self.rank,
            'use_radius': self.use_radius,
            'data_format': self.data_format
        }
        base_config = super(_CoordinateChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CoordinateChannel2D(_CoordinateChannel):
    def __init__(self, use_radius=False,
                 data_format=None,
                 **kwargs):
        super(CoordinateChannel2D, self).__init__(
            rank=2,
            use_radius=use_radius,
            data_format=data_format,
            **kwargs
        )

    def get_config(self):
        config = super(CoordinateChannel2D, self).get_config()
        config.pop('rank')
        return config


# ---------------------------------------------------
class OutputMuxer(tf.keras.layers.Layer):
    def __init__(self):
        super(OutputMuxer, self).__init__()

    def build(self, input_shape):
        super(OutputMuxer, self).build(input_shape)

    def call(self, prediction):
        # batch, lane_count, y_anchors, x_anchors = inputs.get_shape().as_list()
        x_cls, x_offset, x_embedding = prediction

        
        x_anchors = 32
        y_anchors = 32
        embedding_count = 6
        groundSize = (256, 256)
        inv_anchor_scale_x = (float)(groundSize[1]) / (float)(x_anchors)
        inv_anchor_scale_y = (float)(groundSize[0]) / (float)(y_anchors) 
        
        # ==================================================================
        # post processing
        pred_offset = x_offset
        pred_embeddings = x_embedding

        # do threshold to crop class prob and create a bool mask 
        lane_prob = x_cls[:,:,:,0]
        lane_prob = tf.expand_dims(lane_prob, axis=-1)
        lane_prob = tf.tile(lane_prob, multiples=[1, 1, 1, embedding_count])

        # remove the embeddings which value less than 0.5
        inter_data = pred_embeddings - 0.5
        inter_data = tf.nn.relu(inter_data)
        inter_data *= 2
        embeddings = tf.multiply(inter_data, lane_prob)
        
        # create anchor coordinate maps
        anchor_x_axis = tf.range(0, x_anchors, dtype=tf.float32)
        anchor_x_axis = tf.multiply(anchor_x_axis, tf.constant([inv_anchor_scale_x], dtype=tf.float32))
        anchor_x_axis = tf.expand_dims(anchor_x_axis, axis=0)
        anchor_x_axis = tf.tile(anchor_x_axis, [y_anchors, 1])
        anchor_x_axis = tf.expand_dims(anchor_x_axis, axis=-1)  # expand as same dim as embeddings
        anchor_x_axis = tf.expand_dims(anchor_x_axis, axis=0)   # expand batch
      
        # generate y axis data
        anchor_y_axis = tf.range(0, y_anchors, dtype=tf.float32)
        anchor_y_axis = tf.multiply(anchor_y_axis, tf.constant([inv_anchor_scale_y], dtype=tf.float32 ))
        anchor_y_axis = tf.expand_dims(anchor_y_axis, axis=-1)
        anchor_y_axis = tf.tile(anchor_y_axis, [1, x_anchors])
        anchor_y_axis = tf.expand_dims(anchor_y_axis, axis=-1)  # expand as same dim as embeddings
        anchor_y_axis = tf.expand_dims(anchor_y_axis, axis=0)   # expand batch
    
        
        # filter data by embeddings and decode offsets by exp
        offsets = tf.exp(pred_offset)
        # anchor_x_axis = tf.add(anchor_x_axis, offsets) 
     
        # generate confidence data
        anchor_axis = tf.concat([anchor_x_axis, anchor_y_axis], axis=-1)
        result = [embeddings, offsets, anchor_axis]

        return result

# ---------------------------------------------------
def AlphaLaneModel(net_input_img_size,
                   x_anchors,
                   y_anchors,
                   name='',
                   training=True,
                   input_batch_size=None,
                   output_as_raw_data=False):

    input = tf.keras.Input(name='input', shape=(net_input_img_size[1], net_input_img_size[0], 3), batch_size=input_batch_size)
    x = input


    # FeatureDownsamplingBlock A
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
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x_in = tf.keras.layers.Conv2D(128, kernel_size=(1, 1), padding='same', use_bias=False, strides=(2, 2))(x_in)
    x_in = tf.keras.layers.BatchNormalization()(x_in)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)
    

    x_in = x
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)
    

    x_in = x
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)
    x_bone_non_ipsampling = x
    
    # FeatureUpsamplingBlock
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)

    # concat
    x = tf.keras.layers.Concatenate()([x, x_feature_C])
    x_out = x

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
        
    x_in = x
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.ReLU(6)(x)

    x_bone = x

    # class
    class_count = 2
    x_cls = tf.keras.layers.Conv2D(class_count, kernel_size=(3, 3), padding='same', use_bias=False)(x_bone)
    x_cls = tf.keras.layers.BatchNormalization()(x_cls)
    x_cls = tf.keras.layers.Softmax()(x_cls)

    # offset
    offset_count = 1
    x_offset = tf.keras.layers.Conv2D(offset_count, kernel_size=(3, 3), padding='same', use_bias=False)(x_bone)
    x_offset = tf.keras.layers.BatchNormalization()(x_offset)
    

    # embedding (instance segmentation)
    instance_feature_dim = 6
    x_embedding = CoordinateChannel2D()(x_out)
    x_embedding = ResnetBlock(filters=32, kernel_size=(3, 3), strides=(1, 1))(x_embedding)
    x_embedding = CoordinateChannel2D()(x_embedding)
    x_embedding = ResnetBlock(filters=32, kernel_size=(3, 3), strides=(1, 1),)(x_embedding)
    x_embedding = CoordinateChannel2D()(x_embedding)
    x_embedding = ResnetBlock(filters=32, kernel_size=(3, 3), strides=(1, 1),)(x_embedding)
    x_embedding = CoordinateChannel2D()(x_embedding)
    x_embedding = tf.keras.layers.Conv2D(instance_feature_dim, kernel_size=(3, 3), padding='same', use_bias=False)(x_embedding)
    x_embedding = tf.keras.layers.BatchNormalization()(x_embedding)
    x_embedding = tf.keras.layers.Softmax()(x_embedding)
   

    if training:
        # concatenate offset and class as final output data
        x = tf.keras.layers.Concatenate()([x_cls, x_offset, x_embedding])
        output = x
    else:
        if output_as_raw_data:
            # use following output for raw data visualization
            x = [x_cls, x_offset, x_embedding]
        else:
            # do OutputMuxer to re-arrange output
            x = OutputMuxer()([x_cls, x_offset, x_embedding])

        output = x

    return  tf.keras.Model(input, output, name=name)





