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


class LaneModel_densenet_segmentation():
    def __init__(self, net_input_img_size, x_anchors, y_anchors, max_lane_count):
        self.net_input_img_size = net_input_img_size
        self.x_anchors = x_anchors
        self.y_anchors = y_anchors
        self.max_lane_count = max_lane_count


    def _downsample_block(self, inputs, nb_filters, name=''):
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv')(inputs)
        x = keras.layers.BatchNormalization(name=name+'_bn')(x)
        x = keras.activations.relu(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), name=name+'_maxpool')(x)

        return x


    def _dense_block5(self, inputs, nb_filters, name=''):
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv1')(inputs)
        x = keras.layers.BatchNormalization(name=name+'_bn1')(x)
        x = keras.activations.relu(x)
        con_1 = keras.layers.concatenate([x, inputs], name=name+'con_1')
         
        x = con_1
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv2')(x)
        x = keras.layers.BatchNormalization(name=name+'_bn2')(x)
        x = keras.activations.relu(x)
        con_2 = keras.layers.concatenate([x, con_1], name=name+'con_2')
     
        x = con_2
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv3')(x)
        x = keras.layers.BatchNormalization(name=name+'_bn3')(x)
        x = keras.activations.relu(x)
        con_3 = keras.layers.concatenate([x, con_2], name=name+'con_3')
     
        x = con_3
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv4')(x)
        x = keras.layers.BatchNormalization(name=name+'_bn4')(x)
        x = keras.activations.relu(x)
        con_4 = keras.layers.concatenate([x, con_3], name=name+'con_4')
    
        x = con_4
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv5')(x)
        x = keras.layers.BatchNormalization(name=name+'_bn5')(x)
        x = keras.activations.relu(x)
        con_5 = keras.layers.concatenate([x, con_4], name=name+'con_5')
    
        x = con_5
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv6')(x)
        x = keras.layers.BatchNormalization(name=name+'_bn6')(x)
        x = keras.activations.relu(x)
        con_6 = keras.layers.concatenate([x, con_5], name=name+'con_6')

        out = con_6

        return out

    def _denseInput_block5(self, inputs, nb_filters, name=''):
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv1')(inputs)
        x = keras.layers.BatchNormalization(name=name+'_bn1')(x)
        x = keras.activations.relu(x)
        con_1 = keras.layers.concatenate([x, inputs], name=name+'con_1')
         
        x = con_1
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv2')(x)
        x = keras.layers.BatchNormalization(name=name+'_bn2')(x)
        x = keras.activations.relu(x)
        con_2 = keras.layers.concatenate([x, inputs], name=name+'con_2')
     
        x = con_2
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv3')(x)
        x = keras.layers.BatchNormalization(name=name+'_bn3')(x)
        x = keras.activations.relu(x)
        con_3 = keras.layers.concatenate([x, inputs], name=name+'con_3')
     
        x = con_3
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv4')(x)
        x = keras.layers.BatchNormalization(name=name+'_bn4')(x)
        x = keras.activations.relu(x)
        con_4 = keras.layers.concatenate([x, inputs], name=name+'con_4')
    
        x = con_4
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv5')(x)
        x = keras.layers.BatchNormalization(name=name+'_bn5')(x)
        x = keras.activations.relu(x)
        con_5 = keras.layers.concatenate([x, inputs], name=name+'con_5')
    
        x = con_5
        x = keras.layers.Conv2D(nb_filters, (3, 3), padding='same', name=name+'_conv6')(x)
        x = keras.layers.BatchNormalization(name=name+'_bn6')(x)
     
        out = x

        return out


    def create(self):
        print("-------------------------------------------------------------------")
        print("create_model")

        input = keras.Input(shape=(self.net_input_img_size[1], self.net_input_img_size[0], 3))
        x = input

        # res block 1
        x = self._downsample_block(x, 10, name='feature_block_A')
        x = self._downsample_block(x, 20, name='feature_block_B')
        x = self._downsample_block(x, 32, name='feature_block_C')
        
        # dense
        x = self._dense_block5(x, 16, name='dense_block_A')

        # transition
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        # dense
        x = self._dense_block5(x, 16, name='dense_block_B')

        # transition
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
        

        # transition
        groups = 16
        height, width, in_channels = x.shape.as_list()[1:]
        channels_per_group = in_channels // groups
        shape_size = height * width * in_channels

        x = keras.backend.reshape(x, [-1, height, width, groups, channels_per_group])
        x = keras.backend.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
        x = keras.backend.reshape(x, [-1, self.max_lane_count, self.y_anchors, int((shape_size / self.max_lane_count) / self.y_anchors)])

        
        # dense
        x = self._denseInput_block5(x, 128, name='dense_block_C')


        x = tf.keras.activations.softmax(x)
        output = x

        model = keras.Model(input, output, name="test")
        loss = LaneLoss()
        optimizer = keras.optimizers.Adam(lr=0.0001)
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        model.summary()
        # keras.utils.plot_model(model, "my_first_model.png")
       
        self.model = model

    # -------------------------------------------------------------------
    def train(self, dataset, train_epochs = 200):
        print("-------------------------------------------------------------------")
        print("train_model")

        checkpoint_path = "/home/dana/tmp/ccp-{epoch:04d}.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                        verbose=1, 
                                                        save_weights_only=True,
                                                        period=2)
        log_dir = "/home/dana/tmp/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1,
                                                            #   profile_batch = '100,200'
                                                              )
        # ---------------------------
        # recover point
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
            print('[%d] loss: %s, accuracy %s', idx, test_loss, test_acc)

            prediction = self.model.predict(x=elem[0])
            main_img = np.uint8(elem[0] * 255)
            main_img = cv2.cvtColor(main_img[0], cv2.COLOR_BGR2GRAY)
            main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
            
            zeros = np.zeros((self.y_anchors,  self.x_anchors), dtype=np.uint8)
            raw_output= []
            for si in range(self.max_lane_count):
                output_int8 = np.uint8(prediction[0][si] * 255)
                if si == 0:
                    img = cv2.merge([zeros, zeros, output_int8])
                elif si ==1:
                    img = cv2.merge([zeros, output_int8, zeros])
                elif si ==2:
                    img = cv2.merge([output_int8, zeros, zeros])
                elif si ==3:
                    img = cv2.merge([zeros, output_int8, output_int8])
                else:
                    img = cv2.merge([output_int8, zeros, output_int8])

                img = cv2.resize(img, (512, 288))
                main_img = main_img + img
                # main_img = img
            
            prefix = str(idx)
            
            cv2.imwrite(prefix + "_aaw.png", main_img)
            idx += 1
            if (idx >=20):
                break;
            

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
        self.model.save('model.h5', save_format='h5')
        
