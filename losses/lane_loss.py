import tensorflow as tf
import tensorflow.keras as keras

# custom loss test
def LaneLoss(coeff = 1.0):
    categoricalCrossentropy = tf.keras.losses.CategoricalCrossentropy()
    # categoricalCrossentropy = tf.keras.losses.CategoricalCrossentropy(axis=2)
    
    def _loss(y_true, y_pred):
        batch, lane_count, y_anchors, x_anchors = y_pred.get_shape().as_list()


        # tf.print(y_pred.get_shape())
        # tf.print(y_pred.get_shape().as_list())
        # tf.print("y_true ", tf.shape(y_true))
        # tf.print("y_pred ", tf.shape(y_pred))
        # tf.print("y_pred ---->", tf.shape(y_pred[:,1]))
        
        true_cls = tf.slice(y_true, [0, 0, 0, 0], [-1, -1, -1, x_anchors -1])
        pred_cls = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, x_anchors -1])
    #     tf.print("pred_cls ", tf.shape(pred_cls))
        
        # pred_score = tf.slice(y_pred, [0, 0, 0, x_anchors -1], [-1, -1, -1, 1])
        # pred_score = 1.0 - pred_score
        # tf.print("pred_score ", tf.shape(pred_score), summarize=-1)
        # tf.print("pred_score ", pred_score, summarize=-1)


        wIdxList = tf.range(1, x_anchors, 1, dtype=tf.float32)
        # tf.print("wIdxList ", wIdxList, summarize=-1)

        # tf.print("ori pred_cls ", pred_cls)

        continue_scores = wIdxList * pred_cls
        continue_scores = keras.backend.sum(continue_scores, axis=3)
        s0 = tf.slice(continue_scores, [0, 0, 0], [-1, -1, y_anchors -2])
        s1 = tf.slice(continue_scores, [0, 0, 1], [-1, -1, y_anchors -2])
        s2 = tf.slice(continue_scores, [0, 0, 2], [-1, -1, y_anchors -2])
        # tf.print("continue_scores ", continue_scores)
        # tf.print("continue_scores shape ", tf.shape(continue_scores))

        s = keras.backend.abs(s0-s1) - keras.backend.abs(s1-s2)
        s = keras.backend.sum(s, axis=-1)
        continue_loss = keras.backend.abs(keras.backend.sum(s, axis=-1))

        # tf.print("s0 ", s0)
        # tf.print("s0 shape ", tf.shape(s0))
        # tf.print("s1 ", s1)
        # tf.print("s1 shape ", tf.shape(s1))
        # tf.print("s2 ", s2)
        # tf.print("s2 shape ", tf.shape(s2))
        # tf.print("s ", s)
        # tf.print("s shape ", tf.shape(s))

        class_loss = categoricalCrossentropy(y_true, y_pred)
        # tf.print("ret ", ret)

        # loss = class_loss + continue_loss * 0.01
        loss = class_loss + continue_loss * 0.1
        # loss = class_loss 
        return loss

    return _loss
