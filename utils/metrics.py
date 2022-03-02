import keras.backend as K
import tensorflow as tf

def six_metrics(y_true, y_pred):
    # y_pred_tmp = K.cast(tf.equal(K.argmax(y_pred, axis=-1), channel), "float32")
    threshold = 0
    y_pred_false = K.cast(tf.math.less_equal(y_pred, 0), "float32")
    y_pred_true = tf.logical_not(y_pred_false)
    y_true_true = K.cast(tf.equal(y_true, 1),"float32")
    y_true_false = tf.logical_not(y_pred_false)
    
    true_positives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_true))
    true_negatives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_false))
    false_positives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_true))
    false_negatives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_false))

    sensitivities = tf.math.divide_no_nan(
        true_positives,
        tf.math.add(true_positives, false_negatives))
    specificities = tf.math.divide_no_nan(
        true_negatives,
        tf.math.add(true_negatives, false_positives))
    precision = tf.math.divide_no_nan(
        true_positives,
        tf.math.add(true_positives, false_positives))
    accuracy = tf.math.divide_no_nan(
        tf.math.add(true_positives, true_negatives),
        tf.math.add_n([true_positives, false_negatives, true_negatives, false_positives]))
    f1_score = tf.multiply(
        2,
        tf.math.divide_no_nan(
            tf.math.multiply(sensitivities, precision),
            tf.math.add(sensitivities, precision)
        ))
    mcc = tf.math.divide_no_nan(
        tf.math.subtract(
            tf.math.multiply(true_positives,true_negatives),
            tf.math.multiply(false_negatives, false_positives)
        ),
        tf.math.sqrt((true_positives + false_positives)*
        (true_positives + false_negatives)*
        (true_negatives + false_positives)*
        (true_negatives + false_negatives))
    )
    return sensitivities,specificities,precision,accuracy,f1_score,mcc