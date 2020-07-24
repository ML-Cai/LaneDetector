import tensorflow as tf
import tensorflow.keras as keras

# custom loss test
def LaneLoss(coeff = 1.0):
    scce = tf.keras.losses.CategoricalCrossentropy()
    
    def _loss(y_true, y_pred):
        # print("y_true ", y_true.shape)
        # print("y_pred ", y_pred.shape)
        # tf.print("y_true ", y_true)
        # tf.print("y_pred ", y_pred)

        ret = scce(y_true, y_pred)
        # tf.print("ret ", ret)
        return ret

    return _loss
