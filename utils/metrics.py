import keras.backend as K
import tensorflow as tf

def sensitivities_metric(func_name = 'sensitivities'):
    def sensitivities_func(y_true, y_pred):
        y_pred_false = tf.math.less_equal(y_pred, 0)
        y_pred_true = K.cast(tf.logical_not(y_pred_false),'float32')
        y_pred_false = K.cast(y_pred_false,'float32')
        y_true_true = tf.equal(y_true, 1)
        y_true_false = K.cast(tf.logical_not(y_true_true),'float32')
        y_true_true = K.cast(y_true_true,'float32')
        
        true_positives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_true))
        true_negatives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_false))
        false_positives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_true))
        false_negatives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_false))
        sensitivities = tf.math.divide_no_nan(
            true_positives,
            tf.math.add(true_positives, false_negatives))
        return sensitivities
    sensitivities_func.__name__ = func_name
    return sensitivities_func

def specificities_metric(func_name = 'specificities'):
    def specificities_func(y_true, y_pred):
        y_pred_false = tf.math.less_equal(y_pred, 0)
        y_pred_true = K.cast(tf.logical_not(y_pred_false),'float32')
        y_pred_false = K.cast(y_pred_false,'float32')
        y_true_true = tf.equal(y_true, 1)
        y_true_false = K.cast(tf.logical_not(y_true_true),'float32')
        y_true_true = K.cast(y_true_true,'float32')
        
        true_positives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_true))
        true_negatives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_false))
        false_positives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_true))
        false_negatives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_false))
        specificities = tf.math.divide_no_nan(
            true_negatives,
            tf.math.add(true_negatives, false_positives))
        return specificities
    specificities_func.__name__ = func_name
    return specificities_func

def precision_metric(func_name = 'specificities'):
    def precision_func(y_true, y_pred):
        y_pred_false = tf.math.less_equal(y_pred, 0)
        y_pred_true = K.cast(tf.logical_not(y_pred_false),'float32')
        y_pred_false = K.cast(y_pred_false,'float32')
        y_true_true = tf.equal(y_true, 1)
        y_true_false = K.cast(tf.logical_not(y_true_true),'float32')
        y_true_true = K.cast(y_true_true,'float32')
        
        true_positives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_true))
        true_negatives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_false))
        false_positives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_true))
        false_negatives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_false))
        precision = tf.math.divide_no_nan(
            true_positives,
            tf.math.add(true_positives, false_positives))
        return precision
    precision_func.__name__ = func_name
    return precision_func

def accuracy_metric(func_name = 'accuracy'):
    def accuracy_func(y_true, y_pred):
        y_pred_false = tf.math.less_equal(y_pred, 0)
        y_pred_true = K.cast(tf.logical_not(y_pred_false),'float32')
        y_pred_false = K.cast(y_pred_false,'float32')
        y_true_true = tf.equal(y_true, 1)
        y_true_false = K.cast(tf.logical_not(y_true_true),'float32')
        y_true_true = K.cast(y_true_true,'float32')
        
        true_positives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_true))
        true_negatives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_false))
        false_positives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_true))
        false_negatives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_false))
        accuracy = tf.math.divide_no_nan(
            tf.math.add(true_positives, true_negatives),
            tf.math.add_n([true_positives, false_negatives, true_negatives, false_positives]))
        return accuracy
    accuracy_func.__name__ = func_name
    return accuracy_func

def f1_score_metric(func_name = 'f1_score'):
    def f1_score_func(y_true, y_pred):
        y_pred_false = tf.math.less_equal(y_pred, 0)
        y_pred_true = K.cast(tf.logical_not(y_pred_false),'float32')
        y_pred_false = K.cast(y_pred_false,'float32')
        y_true_true = tf.equal(y_true, 1)
        y_true_false = K.cast(tf.logical_not(y_true_true),'float32')
        y_true_true = K.cast(y_true_true,'float32')
        
        true_positives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_true))
        true_negatives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_false))
        false_positives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_true))
        false_negatives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_false))
        sensitivities = tf.math.divide_no_nan(
            true_positives,
            tf.math.add(true_positives, false_negatives))
        precision = tf.math.divide_no_nan(
            true_positives,
            tf.math.add(true_positives, false_positives))
        f1_score = tf.multiply(
            2.0,
            tf.math.divide_no_nan(
                tf.math.multiply(sensitivities, precision),
                tf.math.add(sensitivities, precision)
            ))
        return f1_score
    f1_score_func.__name__ = func_name
    return f1_score_func

def mcc_metric(func_name = 'mcc'):
    def mcc_func(y_true, y_pred):
        threshold = 100
        y_pred_false = tf.math.less_equal(y_pred, 0)
        y_pred_true = K.cast(tf.logical_not(y_pred_false),'float32')
        y_pred_false = K.cast(y_pred_false,'float32')
        y_true_true = tf.equal(y_true, 1)
        y_true_false = K.cast(tf.logical_not(y_true_true),'float32')
        y_true_true = K.cast(y_true_true,'float32')
        
        true_positives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_true))
        true_negatives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_false))
        false_positives = tf.reduce_sum(tf.math.multiply(y_true_false,y_pred_true))
        false_negatives = tf.reduce_sum(tf.math.multiply(y_true_true,y_pred_false))
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
        return mcc
    mcc_func.__name__ = func_name
    return mcc_func