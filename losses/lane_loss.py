import tensorflow as tf
import tensorflow.keras as keras

# custom loss test
def LaneLoss(coeff = 1.0):
    categoricalCrossentropy = tf.keras.losses.CategoricalCrossentropy()
    
    def _loss(y_true, y_pred):
        # batch, lane_count, y_anchors, x_anchors = y_pred.shape
        # print("y_true ", y_true.shape)
        # print("y_pred ", y_pred.shape)
        # tf.print("y_true ", y_true)
        # tf.print("y_pred ", y_pred)

        class_loss = categoricalCrossentropy(y_true, y_pred)
        # tf.print("ret ", ret)

        loss = class_loss
        return loss

    return _loss
