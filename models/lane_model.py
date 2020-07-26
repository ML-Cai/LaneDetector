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


class LaneModel():
    def __init__(self, net_input_img_size, x_anchors, y_anchors, max_lane_count):
        self.net_input_img_size = net_input_img_size
        self.x_anchors = x_anchors
        self.y_anchors = y_anchors
        self.max_lane_count = max_lane_count


    def create(self):
        print("-------------------------------------------------------------------")
        print("create_model")

        input = keras.Input(shape=(self.net_input_img_size[1], self.net_input_img_size[0], 3))
        x = input

        x = MobilenetV2IdentityBlock(3, [16*2, 16])(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_1')(x)

        x = MobilenetV2IdentityBlock(3, [24*2, 24])(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_2')(x)
        
        x = MobilenetV2IdentityBlock(3, [32*2, 32])(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_3')(x)

        x = MobilenetV2IdentityBlock(3, [64*2, 64])(x)
        x = MobilenetV2IdentityBlock(3, [64*2, 64])(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_4')(x)

        x = MobilenetV2IdentityBlock(3, [96*2, 96])(x)
        x = MobilenetV2IdentityBlock(3, [96*2, 96])(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_5')(x)

        x = MobilenetV2IdentityBlock(3, [128*2, 128])(x)
        x = keras.layers.MaxPool2D(pool_size=(3, 4), name='bone_pool_6')(x)


        # --------------------------------------------------------
        

        # --------------------------------------------------------
        # feature extend
        x = keras.layers.Flatten()(x)
        lane_1 = keras.layers.Dense(80, name='dense_ext_1')(x)
        lane_2 = keras.layers.Dense(80, name='dense_ext_2')(x)
        lane_3 = keras.layers.Dense(80, name='dense_ext_3')(x)
        lane_4 = keras.layers.Dense(80, name='dense_ext_4')(x)
        lane_1 = keras.activations.relu(lane_1)
        lane_2 = keras.activations.relu(lane_2)
        lane_3 = keras.activations.relu(lane_3)
        lane_4 = keras.activations.relu(lane_4)
        lane_1 = keras.layers.Dense(self.x_anchors * self.y_anchors)(lane_1)
        lane_2 = keras.layers.Dense(self.x_anchors * self.y_anchors)(lane_2)
        lane_3 = keras.layers.Dense(self.x_anchors * self.y_anchors)(lane_3)
        lane_4 = keras.layers.Dense(self.x_anchors * self.y_anchors)(lane_4)

        lane_group =[lane_1, lane_2, lane_3, lane_4]
        x = keras.layers.concatenate(lane_group, name='concat')
        
        # --------------------------------------------------------


        # # --------------------------------------------------------
        # # feature extend
        # lane_1 = keras.layers.Conv2D(self.x_anchors * 2, 1, padding='same')(x)
        # lane_2 = keras.layers.Conv2D(self.x_anchors * 2, 1, padding='same')(x)
        # lane_3 = keras.layers.Conv2D(self.x_anchors * 2, 1, padding='same')(x)
        # lane_4 = keras.layers.Conv2D(self.x_anchors * 2, 1, padding='same')(x)

        # lane_1 = keras.layers.BatchNormalization()(lane_1)
        # lane_2 = keras.layers.BatchNormalization()(lane_2)
        # lane_3 = keras.layers.BatchNormalization()(lane_3)
        # lane_4 = keras.layers.BatchNormalization()(lane_4)
        # lane_1 = keras.activations.relu(lane_1)
        # lane_2 = keras.activations.relu(lane_2)
        # lane_3 = keras.activations.relu(lane_3)
        # lane_4 = keras.activations.relu(lane_4)

        # lane_1 = keras.layers.Conv2D(self.x_anchors, 1, padding='same')(lane_1)
        # lane_2 = keras.layers.Conv2D(self.x_anchors, 1, padding='same')(lane_2)
        # lane_3 = keras.layers.Conv2D(self.x_anchors, 1, padding='same')(lane_3)
        # lane_4 = keras.layers.Conv2D(self.x_anchors, 1, padding='same')(lane_4)

        # lane_1 = keras.layers.Flatten()(lane_1)
        # lane_2 = keras.layers.Flatten()(lane_2)
        # lane_3 = keras.layers.Flatten()(lane_3)
        # lane_4 = keras.layers.Flatten()(lane_4)
         
        # # merge
        # lane_group =[lane_1, lane_2, lane_3, lane_4]
        # x = keras.layers.concatenate(lane_group, name='concat')
        # # --------------------------------------------------------


        # # --------------------------------------------------------
        # # feature extend
        # # ex_1 = keras.layers.Flatten()(x)
        # # ex_1 = keras.layers.Dense(36, name='dense_ext_1')(ex_1)
        # # ex_1 = keras.activations.relu(ex_1)
        # # ex_1 = keras.layers.Dense(self.max_lane_count * self.x_anchors * self.y_anchors)(ex_1)
        # # x = ex_1
        # # --------------------------------------------------------

        # # reshape output
        x = keras.layers.Flatten()(x)
        x = tf.keras.layers.Reshape((self.max_lane_count, self.y_anchors, self.x_anchors))(x)
        x = tf.keras.activations.softmax(x)
        output = x

        model = keras.Model(input, output, name="test")

        optimizer = keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        model.summary()
        # keras.utils.plot_model(model, "my_first_model.png")
       
        self.model = model

    # -------------------------------------------------------------------
    def train(self, dataset, train_epochs = 100):
        print("-------------------------------------------------------------------")
        print("train_model")

        checkpoint_path = "/home/dana/tmp/ccp-{epoch:04d}.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                        verbose=1, 
                                                        save_weights_only=True,
                                                        period=2)
        # self.model.save_weights(checkpoint_path.format(epoch=0))
        log_dir = "/home/dana/tmp/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1,
                                                            #   profile_batch = '100,200'
                                                              )

        history = self.model.fit(dataset,
                                 callbacks=[cp_callback, tensorboard_callback],
                                 epochs = train_epochs)

        return history

    # -------------------------------------------------------------------
    def evaluate(self, dataset):
        print("-------------------------------------------------------------------")
        print("evaluate_model")

        test_loss, test_acc = self.model.evaluate(dataset, verbose=2)

        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

        # #######################################3
        # label = np.zeros((1, 360, 640, 3), dtype=np.int8)
        # time1 = datetime.datetime.now()
        # prediction = self.model.predict(x=label)
        # time2 = datetime.datetime.now()
        # diff = time2 - time1
        # print("function took ",  (diff.total_seconds() * 1000.0), "ms")
        # time1 = datetime.datetime.now()
        # prediction = self.model.predict(x=label)
        # time2 = datetime.datetime.now()
        # diff = time2 - time1
        # print("function took ",  (diff.total_seconds() * 1000.0), "ms")
        # time1 = datetime.datetime.now()
        # prediction = self.model.predict(x=label)
        # time2 = datetime.datetime.now()
        # diff = time2 - time1
        # print("function took ",  (diff.total_seconds() * 1000.0), "ms")
        # time1 = datetime.datetime.now()
        # prediction = self.model.predict(x=label)
        # time2 = datetime.datetime.now()
        # diff = time2 - time1
        # print("function took ",  (diff.total_seconds() * 1000.0), "ms")
        # #######################################3
        
        prediction = self.model.predict(dataset)

        for idx in range(min(5, len(prediction))):
            print("save ", idx)
            prefix = str(idx)
            img0 = Image.fromarray(np.uint8(prediction[idx][0] * 255) , 'L')
            img1 = Image.fromarray(np.uint8(prediction[idx][1] * 255) , 'L')
            img2 = Image.fromarray(np.uint8(prediction[idx][2] * 255) , 'L')
            img3 = Image.fromarray(np.uint8(prediction[idx][3] * 255) , 'L')
            img01 = Image.blend(img0, img1, 0.5)
            img23 = Image.blend(img2, img3, 0.5)
            img = Image.blend(img01, img23, 0.5)
            img.save(prefix + "_aa.png")
            
            # img = Image.fromarray(np.uint8(prediction[0][0] * 255) , 'L')
            # img.save(prefix + "_aa0.png")
            # img = Image.fromarray(np.uint8(prediction[0][1] * 255) , 'L')
            # img.save(prefix + "_aa1.png")
            # img = Image.fromarray(np.uint8(prediction[0][2] * 255) , 'L')
            # img.save(prefix + "_aa2.png")
            # img = Image.fromarray(np.uint8(prediction[0][3] * 255) , 'L')
            # img.save(prefix + "_aa3.png")
        

        x_in = np.zeros((1, 288, 512, 3), dtype=np.int8)
        result = self.model.predict(x_in)
        start = time.time()
        for i in range(1000):
            result = self.model.predict(x_in)
        end = time.time()
        print("round ", i , " , cost : ", (end - start) / 1000.0, "s")


    # -------------------------------------------------------------------
    def load_weight(self):
        print("-------------------------------------------------------------------")
        print("load_model")

        checkpoint_path = "/home/dana/tmp/"
        self.model.load_weights(tf.train.latest_checkpoint(checkpoint_path))

    # -------------------------------------------------------------------
    def save(self):
        print("-------------------------------------------------------------------")
        print("save_model")
        self.model.save("/home/dana/tmp/my_model")
