import tensorflow as tf

def masked_crossentropy(y_true, y_pred):

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    mask = tf.math.logical_not(tf.math.equal(y_true,2))
    mask = tf.cast(mask, dtype = y_pred.dtype)

    print(tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)*mask)
    # Tensor("masked_crossentropy/Mean:0", shape=(None,), dtype=float32)
    return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1) * mask
