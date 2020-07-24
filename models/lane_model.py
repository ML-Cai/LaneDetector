import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import sys
import time
import datetime
from models.resnet_identity_block import ResnetIdentityBlock
from losses.lane_loss import LaneLoss
from PIL import Image

class LaneModel():
    def create(self):
        print("-------------------------------------------------------------------")
        print("create_model")

        input = keras.Input(shape=(360, 640, 3))
        x = input

        x = ResnetIdentityBlock(3, [16, 16, 16])(x)
        x = ResnetIdentityBlock(3, [16, 16, 16])(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        x = ResnetIdentityBlock(3, [32, 32, 32])(x)
        x = ResnetIdentityBlock(3, [32, 32, 32])(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        x = ResnetIdentityBlock(3, [32, 32, 32])(x)
        x = ResnetIdentityBlock(3, [32, 32, 32])(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        x = ResnetIdentityBlock(3, [64, 64, 64])(x)
        x = ResnetIdentityBlock(3, [64, 64, 64])(x)
        x = keras.layers.MaxPool2D(pool_size=(1, 2))(x)

        x = ResnetIdentityBlock(3, [64, 64, 96])(x)
        x = ResnetIdentityBlock(3, [64, 64, 96])(x)

        x = ResnetIdentityBlock(3, [64, 64, 128])(x)
        x = ResnetIdentityBlock(3, [64, 64, 128])(x)
        x = keras.layers.MaxPool2D(pool_size=(3, 2))(x)

        x_1 = keras.layers.SeparableConv2D(64, (3, 3), name="decode_1")(x)
        x_1 = keras.layers.BatchNormalization()(x_1)
        x_1 = keras.activations.relu(x_1)

        x_2 = keras.layers.SeparableConv2D(64, (3, 3), name="decode_2")(x_1)
        x_2 = keras.layers.BatchNormalization()(x_2)
        x_2 = keras.activations.relu(x_2)

        x_3 = keras.layers.SeparableConv2D(50, (3, 3), name="decode_3")(x_2)
        x_3 = keras.layers.BatchNormalization()(x_3)
        x_3 = keras.activations.relu(x_3)

        # slice_1 = keras.layers.Flatten()(x_1)
        # slice_1 = keras.layers.Dense(128, name='dense_slice_1')(slice_1)
        # slice_1 = keras.activations.relu(slice_1)
        
        # slice_2 = keras.layers.Flatten()(x_2)
        # slice_2 = keras.layers.Dense(32, name='dense_slice_2')(slice_2)
        # slice_2 = keras.activations.relu(slice_2)

        slice_3 = keras.layers.Flatten()(x_3)
        slice_3 = keras.layers.Dense(25, name='dense_slice_3')(slice_3)
        slice_3 = keras.activations.relu(slice_3)


        lane_1 = keras.layers.Dense(50*100)(slice_3)
        lane_2 = keras.layers.Dense(50*100)(slice_3)
        lane_3 = keras.layers.Dense(50*100)(slice_3)
        lane_4 = keras.layers.Dense(50*100)(slice_3)
        lane_1 = keras.layers.Flatten()(lane_1)
        lane_2 = keras.layers.Flatten()(lane_2)
        lane_3 = keras.layers.Flatten()(lane_3)
        lane_4 = keras.layers.Flatten()(lane_4)

        # # build lane_1
        # lane_1_s1 = keras.layers.Dense(10*100)(slice_1)
        # lane_1_s2 = keras.layers.Dense(10*100)(slice_2)
        # lane_1_s3 = keras.layers.Dense(10*100)(slice_3)
        # lane_1 = keras.layers.concatenate([lane_1_s1, lane_1_s2, lane_1_s3], name='lane_1')
        # lane_1 = keras.layers.Flatten()(lane_1)

        # # build lane_2
        # lane_2_s1 = keras.layers.Dense(10*100)(slice_1)
        # lane_2_s2 = keras.layers.Dense(10*100)(slice_2)
        # lane_2_s3 = keras.layers.Dense(10*100)(slice_3)
        # lane_2 = keras.layers.concatenate([lane_2_s1, lane_2_s2, lane_2_s3], name='lane_2')
        # lane_2 = keras.layers.Flatten()(lane_2)

        # # build lane_3
        # lane_3_s1 = keras.layers.Dense(10*100)(slice_1)
        # lane_3_s2 = keras.layers.Dense(10*100)(slice_2)
        # lane_3_s3 = keras.layers.Dense(10*100)(slice_3)
        # lane_3 = keras.layers.concatenate([lane_3_s1, lane_3_s2, lane_3_s3], name='lane_3')
        # lane_3 = keras.layers.Flatten()(lane_3)

        # # build lane_4
        # lane_4_s1 = keras.layers.Dense(10*100)(slice_1)
        # lane_4_s2 = keras.layers.Dense(10*100)(slice_2)
        # lane_4_s3 = keras.layers.Dense(10*100)(slice_3)
        # lane_4 = keras.layers.concatenate([lane_4_s1, lane_4_s2, lane_4_s3], name='lane_4')
        # lane_4 = keras.layers.Flatten()(lane_4)



        # 4 lane
        lane_group =[lane_1, lane_2, lane_3, lane_4]
        # lane_group =[lane_1, lane_2]
        # x = keras.layers.concatenate([lane_1, lane_2, lane_3, lane_4], name='concat')

        if (len(lane_group) == 1):
            x = lane_group[0]
        else:
            x = keras.layers.concatenate(lane_group, name='concat')
        x = keras.layers.Flatten()(x)
        x = tf.keras.layers.Reshape((len(lane_group), 50, 100))(x)
        x = tf.keras.activations.softmax(x)
        output = x

        model = keras.Model(input, output, name="test")
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                    #   loss=LaneLoss(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        model.summary()
        # keras.utils.plot_model(model, "my_first_model.png")
       
        self.model = model

    # -------------------------------------------------------------------
    def train(self, dataset, train_epochs = 100):
        print("-------------------------------------------------------------------")
        print("train_model")

        checkpoint_path = "/tmp/cp-{epoch:04d}.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                        verbose=1, 
                                                        save_weights_only=True,
                                                        period=5)
        # self.model.save_weights(checkpoint_path.format(epoch=0))

        history = self.model.fit(dataset,
                                 callbacks=cp_callback,
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
        img = Image.fromarray(np.uint8(prediction[0][0] * 255) , 'L')
        img.save("aa0.png")
        img = Image.fromarray(np.uint8(prediction[0][1] * 255) , 'L')
        img.save("aa1.png")
        img = Image.fromarray(np.uint8(prediction[0][2] * 255) , 'L')
        img.save("aa2.png")
        img = Image.fromarray(np.uint8(prediction[0][3] * 255) , 'L')
        img.save("aa3.png")
        

    # -------------------------------------------------------------------
    def load_weight(self):
        print("-------------------------------------------------------------------")
        print("load_model")

        checkpoint_path = "/tmp/"
        self.model.load_weights(tf.train.latest_checkpoint(checkpoint_path))

    # -------------------------------------------------------------------
    def save(self):
        print("-------------------------------------------------------------------")
        print("save_model")
        self.model.save("/tmp/my_model")
