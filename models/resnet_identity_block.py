import tensorflow as tf
# import tensorflow.keras as keras


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.shortcut = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2d_shortcut = tf.keras.layers.BatchNormalization()

        self.conv2d_a = tf.keras.layers.Conv2D(filters1, kernel_size=(1,1), padding='same')
        self.bn2d_a = tf.keras.layers.BatchNormalization()
        
        self.depthwise_b = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same')
        self.depthwise_bn2d_b = tf.keras.layers.BatchNormalization()
       
        self.conv2d_c = tf.keras.layers.Conv2D(filters3, kernel_size=(1,1), padding='same')
        self.bn2d_c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = input_tensor

        # expand
        x = self.bn2d_a(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2d_a(x)
        
        
        # depthwise
        x = self.depthwise_bn2d_b(x, training=training)
        x = tf.nn.relu(x)
        x = self.depthwise_b(x)
        
        # project
        x = self.bn2d_c(x, training=training)
        x = self.conv2d_c(x)
        
        # shortcut
        sc = self.shortcut(input_tensor)
        sc = self.bn2d_shortcut(sc)

        output = x + sc
        # output = x
        return output






# class ResnetIdentityBlock(tf.keras.Model):
#     def __init__(self, kernel_size, filters):
#         super(ResnetIdentityBlock, self).__init__(name='')
#         filters1, filters2, filters3 = filters

#         self.shortcut = tf.keras.layers.Conv2D(filters3, (1, 1))
#         self.bnshortcut = tf.keras.layers.BatchNormalization()

#         self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
#         self.bn2a = tf.keras.layers.BatchNormalization()

#         self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
#         self.bn2b = tf.keras.layers.BatchNormalization()

#         self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
#         self.bn2c = tf.keras.layers.BatchNormalization()

#     def call(self, input_tensor, training=False):
#         x = input_tensor
#         x = self.bn2a(x, training=training)
#         x = tf.nn.relu(x)
#         x = self.conv2a(x)
        

#         x = self.bn2b(x, training=training)
#         x = tf.nn.relu(x)
#         x = self.conv2b(x)
        
#         x = self.bn2c(x, training=training)
#         x = self.conv2c(x)
        
        
#         sc = self.shortcut(input_tensor)
#         sc = self.bnshortcut(sc)

#         output = x + sc
#         return output


