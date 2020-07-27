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


class LaneModel():
    def __init__(self, net_input_img_size, x_anchors, y_anchors, max_lane_count):
        self.net_input_img_size = net_input_img_size
        self.x_anchors = x_anchors
        self.y_anchors = y_anchors
        self.max_lane_count = max_lane_count


    def _bottleneck(self, inputs, nb_filters, t):
        x = tf.keras.layers.Conv2D(filters=nb_filters * t, kernel_size=(1,1), padding='same')(inputs)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same')(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Conv2D(filters=nb_filters, kernel_size=(1,1), padding='same')(x)

        # do not use activation function
        if not inputs.get_shape()[3] == nb_filters:
            inputs = tf.keras.layers.Conv2D(filters=nb_filters, kernel_size=(1,1), padding='same')(inputs)

        outputs = tf.keras.layers.add([x, inputs])
        return outputs

    def create(self):
        print("-------------------------------------------------------------------")
        print("create_model")

        input = keras.Input(shape=(self.net_input_img_size[1], self.net_input_img_size[0], 3))
        x = input

        x = keras.layers.Conv2D(16, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_1')(x)

        x = keras.layers.Conv2D(24, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_2')(x)
        
        x = keras.layers.Conv2D(32, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_3')(x)

        # mobilenet
        x = MobilenetV2IdentityBlock(3, [60*6, 60])(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_4')(x)

        x = MobilenetV2IdentityBlock(3, [92*6, 92])(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_5')(x)

        x = MobilenetV2IdentityBlock(3, [128*6, 128])(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_6')(x)

        # down sample
        # x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        # x = keras.layers.BatchNormalization()(x)
        # x = keras.activations.relu(x)
        # --------------------------------------------------------
        # x = MobilenetV2IdentityBlock(3, [16*2, 16])(x)
        # x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_1')(x)
        # x = MobilenetV2IdentityBlock(3, [24*2, 24])(x)
        # x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_2')(x)
        # x = MobilenetV2IdentityBlock(3, [32*2, 32])(x)
        # x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_3')(x)
        # x = MobilenetV2IdentityBlock(3, [64*2, 64])(x)
        # x = MobilenetV2IdentityBlock(3, [64*2, 64])(x)
        # x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_4')(x)
        # x = MobilenetV2IdentityBlock(3, [96*2, 96])(x)
        # x = MobilenetV2IdentityBlock(3, [96*2, 96])(x)
        # x = keras.layers.MaxPool2D(pool_size=(2, 2), name='bone_pool_5')(x)
        # x = MobilenetV2IdentityBlock(3, [128*2, 128])(x)
        # x = keras.layers.MaxPool2D(pool_size=(3, 2), name='bone_pool_6')(x)
        # --------------------------------------------------------

        # --------------------------------------------------------
        # feature extend
        x = keras.layers.Flatten()(x)
        lane_1 = keras.layers.Dense(90, name='dense_ext_11')(x)
        lane_2 = keras.layers.Dense(90, name='dense_ext_21')(x)
        lane_3 = keras.layers.Dense(90, name='dense_ext_31')(x)
        lane_4 = keras.layers.Dense(90, name='dense_ext_41')(x)
        lane_1 = keras.activations.relu(lane_1) 
        lane_2 = keras.activations.relu(lane_2)
        lane_3 = keras.activations.relu(lane_3)
        lane_4 = keras.activations.relu(lane_4)
        lane_1 = keras.layers.Dense(int(self.x_anchors /4 * self.y_anchors / 4))(lane_1)
        lane_2 = keras.layers.Dense(int(self.x_anchors /4 * self.y_anchors / 4))(lane_2)
        lane_3 = keras.layers.Dense(int(self.x_anchors /4 * self.y_anchors / 4))(lane_3)
        lane_4 = keras.layers.Dense(int(self.x_anchors /4 * self.y_anchors / 4))(lane_4)
        lane_1 = keras.activations.relu(lane_1) ####
        lane_2 = keras.activations.relu(lane_2)
        lane_3 = keras.activations.relu(lane_3)
        lane_4 = keras.activations.relu(lane_4)
        lane_1 = keras.layers.Reshape((1, int(self.y_anchors /4), int(self.x_anchors / 4)))(lane_1)
        lane_2 = keras.layers.Reshape((1, int(self.y_anchors /4), int(self.x_anchors / 4)))(lane_2)
        lane_3 = keras.layers.Reshape((1, int(self.y_anchors /4), int(self.x_anchors / 4)))(lane_3)
        lane_4 = keras.layers.Reshape((1, int(self.y_anchors /4), int(self.x_anchors / 4)))(lane_4)

        lane_1 = keras.layers.Conv2DTranspose (self.x_anchors, kernel_size=(3, 3), strides=(1, 2), padding='same')(lane_1)
        lane_2 = keras.layers.Conv2DTranspose (self.x_anchors, kernel_size=(3, 3), strides=(1, 2), padding='same')(lane_2)
        lane_3 = keras.layers.Conv2DTranspose (self.x_anchors, kernel_size=(3, 3), strides=(1, 2), padding='same')(lane_3)
        lane_4 = keras.layers.Conv2DTranspose (self.x_anchors, kernel_size=(3, 3), strides=(1, 2), padding='same')(lane_4)
        lane_1 = keras.activations.relu(lane_1)
        lane_2 = keras.activations.relu(lane_2)
        lane_3 = keras.activations.relu(lane_3)
        lane_4 = keras.activations.relu(lane_4)

        lane_1 = keras.layers.Conv2DTranspose (self.x_anchors, kernel_size=(3, 3), strides=(1, 2), padding='same')(lane_1)
        lane_2 = keras.layers.Conv2DTranspose (self.x_anchors, kernel_size=(3, 3), strides=(1, 2), padding='same')(lane_2)
        lane_3 = keras.layers.Conv2DTranspose (self.x_anchors, kernel_size=(3, 3), strides=(1, 2), padding='same')(lane_3)
        lane_4 = keras.layers.Conv2DTranspose (self.x_anchors, kernel_size=(3, 3), strides=(1, 2), padding='same')(lane_4)



        lane_1 = keras.layers.Flatten()(lane_1)
        lane_2 = keras.layers.Flatten()(lane_2)
        lane_3 = keras.layers.Flatten()(lane_3)
        lane_4 = keras.layers.Flatten()(lane_4)
        lane_group =[lane_1, lane_2, lane_3, lane_4]
        x = keras.layers.concatenate(lane_group, name='concat')
        # --------------------------------------------------------
        
        # # --------------------------------------------------------
        # # feature extend
        # x = keras.layers.Flatten()(x)
        # lane_1 = keras.layers.Dense(60, name='dense_ext_11')(x)
        # lane_2 = keras.layers.Dense(60, name='dense_ext_21')(x)
        # lane_3 = keras.layers.Dense(60, name='dense_ext_31')(x)
        # lane_4 = keras.layers.Dense(60, name='dense_ext_41')(x)
        # # lane_1 = keras.activations.relu(lane_1)
        # # lane_2 = keras.activations.relu(lane_2)
        # # lane_3 = keras.activations.relu(lane_3)
        # # lane_4 = keras.activations.relu(lane_4)
        # lane_1 = keras.layers.Dense(self.x_anchors * self.y_anchors)(lane_1)
        # lane_2 = keras.layers.Dense(self.x_anchors * self.y_anchors)(lane_2)
        # lane_3 = keras.layers.Dense(self.x_anchors * self.y_anchors)(lane_3)
        # lane_4 = keras.layers.Dense(self.x_anchors * self.y_anchors)(lane_4)
        # lane_group =[lane_1, lane_2, lane_3, lane_4]
        # x = keras.layers.concatenate(lane_group, name='concat')
        # # --------------------------------------------------------

        # # reshape output
        x = keras.layers.Flatten()(x)
        x = tf.keras.layers.Reshape((self.max_lane_count, self.y_anchors, self.x_anchors))(x)
        x = tf.keras.activations.softmax(x)
        output = x

        model = keras.Model(input, output, name="test")

        optimizer = keras.optimizers.Adam(lr=0.0001)
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

        
        # for elem in dataset:
        #     test_loss, test_acc =  self.model.test_on_batch (x=elem[0], y=elem[1])
        #     print('loss: %s, accuracy %s', test_loss, test_acc)
        #     # asdasd
        

        test_loss, test_acc = self.model.evaluate(dataset, verbose=2)

        print('evaluate loss:', test_loss)
        print('evaluate accuracy:', test_acc)


        idx = 0
        for elem in dataset:
            test_loss, test_acc =  self.model.test_on_batch (x=elem[0], y=elem[1])
            print('loss: %s, accuracy %s', test_loss, test_acc)

            prediction = self.model.predict(x=elem[0])
            main_img = np.uint8(elem[0] * 255)
            main_img = cv2.cvtColor(main_img[0], cv2.COLOR_BGR2GRAY)
            main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
            
            zeros = np.zeros((self.y_anchors,  self.x_anchors), dtype=np.uint8)
            raw_output= []
            for si in range(self.max_lane_count):
                output_int8 = np.uint8(prediction[0][si] * 255)
                img = cv2.merge([zeros, zeros, output_int8])
                img = cv2.resize(img, (512, 288))
                main_img = main_img + img
            
            prefix = str(idx)
            
            cv2.imwrite(prefix + "_aaw.png", main_img)
            idx += 1
            if (idx >=5):
                break;

        for idx in range(min(5, len(prediction))):
            print("save ", idx)
            prefix = str(idx)

            

            # self.max_lane_count, self.y_anchors,
            
            # img0 = Image.fromarray(raw_output[0] , 'RGB')
            # img1 = Image.fromarray(raw_output[1] , 'RGB')
            # img2 = Image.fromarray(raw_output[2] , 'RGB')
            # img3 = Image.fromarray(raw_output[3] , 'RGB')
            # img01 = Image.blend(img0, img1, 0.5)
            # img23 = Image.blend(img2, img3, 0.5)
            # img = Image.blend(img01, img23, 0.5)
            # img.save(prefix + "_aa.png")
            
            # img = Image.fromarray(np.uint8(prediction[0][0] * 255) , 'L')
            # img.save(prefix + "_aa0.png")
            # img = Image.fromarray(np.uint8(prediction[0][1] * 255) , 'L')
            # img.save(prefix + "_aa1.png")
            # img = Image.fromarray(np.uint8(prediction[0][2] * 255) , 'L')
            # img.save(prefix + "_aa2.png")
            # img = Image.fromarray(np.uint8(prediction[0][3] * 255) , 'L')
            # img.save(prefix + "_aa3.png")
        
        # for idx in range(min(5, len(prediction))):
        #     print("save ", idx)
        #     prefix = str(idx)
        #     img0 = Image.fromarray(np.uint8(prediction[idx][0] * 255) , 'L')
        #     img1 = Image.fromarray(np.uint8(prediction[idx][1] * 255) , 'L')
        #     img2 = Image.fromarray(np.uint8(prediction[idx][2] * 255) , 'L')
        #     img3 = Image.fromarray(np.uint8(prediction[idx][3] * 255) , 'L')
        #     img01 = Image.blend(img0, img1, 0.5)
        #     img23 = Image.blend(img2, img3, 0.5)
        #     img = Image.blend(img01, img23, 0.5)
        #     img.save(prefix + "_aa.png")
        

        # -------------------------------------
        # time test 
        mm = keras.applications.MobileNetV2()
        # mm.summary()
        x_in = np.zeros((1, 224, 224, 3), dtype=np.int8)
        result =mm.predict(x_in)
        start = time.time()
        for i in range(300):
            result = mm.predict(x_in)
        end = time.time()
        print("predict mobilenet ", i , " , cost : ", (end - start) / 300.0, "s")


        x_in = np.zeros((1, 288, 512, 3), dtype=np.int8)
        result = self.model.predict(x_in)
        start = time.time()
        for i in range(300):
            result = self.model.predict(x_in)
        end = time.time()
        print("predict lanenet ", i , " , cost : ", (end - start) / 300.0, "s")


        x_in = np.zeros((1, 224, 224, 3), dtype=np.int8)
        result =mm.predict_on_batch(x_in)
        start = time.time()
        for i in range(300):
            result = mm.predict_on_batch(x_in)
        end = time.time()
        print("predict_on_batch mobilenet ", i , " , cost : ", (end - start) / 300.0, "s")


        x_in = np.zeros((1, 288, 512, 3), dtype=np.int8)
        result = self.model.predict_on_batch(x_in)
        start = time.time()
        for i in range(300):
            result = self.model.predict_on_batch(x_in)
        end = time.time()
        print("predict_on_batch lanenet ", i , " , cost : ", (end - start) / 300.0, "s")


        x_in = np.zeros((1, 224, 224, 3), dtype=np.int8)
        result =  mm(x_in)
        start = time.time()
        for i in range(300):
            result = mm(x_in)
        end = time.time()
        print("mobilenet ", i , " , cost : ", (end - start) / 300.0, "s")


        x_in = np.zeros((1, 288, 512, 3), dtype=np.int8)
        result =  self.model(x_in)
        start = time.time()
        for i in range(300):
            result = self.model(x_in)
        end = time.time()
        print("lanenet ", i , " , cost : ", (end - start) / 300.0, "s")


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
