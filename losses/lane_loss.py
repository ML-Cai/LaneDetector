import tensorflow as tf


def DiscriminativeLoss_single(y_true,
                              y_pred,
                              delta_v,
                              delta_d,
                              param_var,
                              param_dist,
                              param_reg):
    # reference from https://github.com/hq-jiang/instance-segmentation-with-discriminative-loss-tensorflow/blob/master/loss.py
    label_shape = tf.shape(y_true)      # [height, width]
    pred_shape = tf.shape(y_pred)       # [height, width, feature_dim]
    feature_dim = pred_shape[2]

    correct_label = tf.reshape(y_true, [label_shape[1] * label_shape[0]])
    reshaped_pred = tf.reshape(y_pred, [pred_shape[1] * pred_shape[0], feature_dim])

    # calculate instance nums
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)
    C = tf.cast(num_instances, tf.float32)
    # tf.print("C ", C, " , unique_labels ", unique_labels)

    # calculate instance pixel embedding mean vec
    segmented_sum = tf.math.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)
    mu = tf.math.divide(segmented_sum, tf.reshape(counts, (-1, 1)))
    mu_expand = tf.gather(mu, unique_id)

    # Calculate l_var
    distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1, ord=1)
    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    l_var = tf.math.unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = tf.math.divide(l_var, counts)
    l_var = tf.reduce_sum(l_var)
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))

    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
    mu_band_rep = tf.tile(mu, [1, num_instances])
    mu_band_rep = tf.reshape(mu_band_rep, (num_instances * num_instances, feature_dim))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)

    intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff), axis=1)
    zero_vector = tf.zeros(1, dtype=tf.float32)
    bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

    mu_norm = tf.norm(mu_diff_bool, axis=1, ord=1)
    mu_norm = tf.subtract(2.0 * delta_d, mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm = tf.square(mu_norm)

    l_dist = tf.reduce_mean(mu_norm)
    # l_dist = tf.math.divide(l_dist, C * (C - 1))

    l_reg = tf.reduce_mean(tf.norm(mu, axis=1, ord=1))

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    return loss

def smooth_L1_loss(y_true, y_pred):
    # reference from https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_loss_function/keras_ssd_loss.py
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred)**2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)


def log_loss(y_true, y_pred):
    # reference from https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_loss_function/keras_ssd_loss.py
    # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
    y_pred = tf.maximum(y_pred, 1e-15)
    
    # Compute the log loss
    log_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    return log_loss

# custom loss test
def LaneLoss(coeff = 1.0):
    def _LaneLoss(y_true, y_pred):

        true_cls = tf.slice(y_true, [0, 0, 0, 0], [-1, -1, -1, 2])
        pred_cls = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, 2])

        true_offset = tf.slice(y_true, [0, 0, 0, 2], [-1, -1, -1, 1])
        pred_offset = tf.slice(y_pred, [0, 0, 0, 2], [-1, -1, -1, 1])

        loss_cls = log_loss(true_cls, pred_cls)
        loss_offset = smooth_L1_loss(true_offset, pred_offset)


        # calcuate discriminative loss from embeddings
        instance_feature_dim =6
        delta_v=0.3
        delta_d=1.0
        param_var=1.0
        param_dist=10.0
        param_reg=0.001

        instance_label_dim = 1
        y_true_instance = tf.slice(y_true, [0, 0, 0, 3], [-1, -1, -1, instance_label_dim])
        y_pred_instance = tf.slice(y_pred, [0, 0, 0, 3], [-1, -1, -1, instance_feature_dim])

        batch_size = tf.shape(y_pred_instance)[0]

        def loop_cond(y_true_instance, y_pred_instance, loss, loopIdx):
            return tf.less(loopIdx, batch_size)
            
        def loop_body(y_true_instance, y_pred_instance, loss, loopIdx):
            loss += DiscriminativeLoss_single(y_true_instance[loopIdx],
                                              y_pred_instance[loopIdx],
                                              delta_v=delta_v,
                                              delta_d=delta_d,
                                              param_var=param_var,
                                              param_dist=param_dist,
                                              param_reg=param_reg)
            return y_true_instance, y_pred_instance, loss, loopIdx +1
        

        loss = 0.0
        _, _, loss, _ = tf.while_loop(cond=loop_cond,
                                      body=loop_body,
                                      loop_vars=[y_true_instance, y_pred_instance, loss, 0])
        return (loss_cls + loss_offset) + 0.01 * loss

    return _LaneLoss
