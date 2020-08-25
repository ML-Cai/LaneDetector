import tensorflow as tf
# import tensorflow.keras as keras


# custom loss test
def LaneAccuracy(coeff = 1.0):
    categoricalAccuracy = tf.keras.metrics.CategoricalAccuracy()
     
    def _accuracy(y_true, y_pred):
        # batch, lane_count, y_anchors, x_anchors = y_pred.get_shape().as_list()
        
        y_pred = tf.keras.activations.softmax(y_pred, axis=3)

        return categoricalAccuracy(y_true, y_pred)

    return _accuracy

# custom loss test
def LaneLoss(coeff = 1.0):
    categoricalCrossentropy = tf.keras.losses.CategoricalCrossentropy()

    def _loss(y_true, y_pred):
        # batch, lane_count, y_anchors, x_anchors = y_pred.get_shape().as_list()
        # m = tf.slice(x, [0, i, 0, 0], [-1, 1, -1, x_anchors -1])

        y_pred = tf.keras.activations.softmax(y_pred, axis=3)

        loss = categoricalCrossentropy(y_true, y_pred) 
        return loss

    return _loss
