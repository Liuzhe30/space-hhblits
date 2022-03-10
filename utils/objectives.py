import tensorflow as tf

def weighted_masked_crossentropy(weight):


    def masked_crossentropy(y_true, y_pred):

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true * weight, y_pred.dtype)

        mask = tf.math.logical_not(tf.math.equal(y_true,2))
        mask = tf.cast(mask, dtype = y_pred.dtype)

        return tf.reduce_mean(tf.math.square(y_pred - y_true) * mask, axis=-1)
    
    return masked_crossentropy


def binary_cross_entropy(y_true, y_pred):

    class_weights = tf.cast(tf.constant([[[10., 1.]]]),y_pred.dtype)

    y_pred = tf.convert_to_tensor(y_pred)

    y_true = tf.cast(y_true, y_pred.dtype)

    weights = tf.reduce_sum(class_weights * y_true, axis=-1)



    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    loss = bce(y_true, y_pred)
    loss = weights * bce(y_true, y_pred)
    
    return tf.reduce_sum(loss)


