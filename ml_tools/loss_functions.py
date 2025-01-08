import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from sklearn.metrics import roc_auc_score, precision_score
from scipy.special import softmax


def weighted_categorical_crossentropy(class_weights):

    def wce_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
        weight_log = tf.math.multiply(tf.math.log1p(1 - y_pred), class_weights)
        loss = y_true * weight_log
        return loss

    return wce_loss


@tf.function
def gaussian_pdf(y_true, mu, sigma):
    y_true = tf.expand_dims(y_true, -1)  # Add Gaussian component dimension
    prob = tfp.distributions.Normal(loc=mu, scale=sigma).prob(y_true)

    return prob


@tf.function
def mdn_loss(y_true, preds):
    pi, mu, sigma = tf.split(preds, 3, axis=-1)
    prob = gaussian_pdf(y_true, mu, sigma)  # Likelihood for each Gaussian
    weighted_prob = tf.reduce_sum(pi * prob, axis=-1)  # Mixture likelihood
    nll = -tf.math.log(weighted_prob + 1e-8)  # Avoid log(0)
    return tf.reduce_mean(nll)


@tf.function
def focal_loss(gamma=2.0, alpha=0.25):
    def fc_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)  # Avoid log(0)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1 - pt, gamma) * tf.math.log(pt))
    return fc_loss


def weighted_fc_loss(class_weights):
    """
    Weighted Focal Loss function with class weights.

    Parameters:
    - class_weights: A list or tensor of weights for each class.

    Returns:
    - A loss function that can be used in model compilation.
    """

    @tf.function
    def fc_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        foc_loss = 0.25 * tf.pow(1 - pt, 2.0) * tf.math.log1p(pt)

        class_weights_tensor = tf.gather(class_weights, tf.argmax(y_true, axis=-1))
        class_weights_tensor = tf.expand_dims(class_weights_tensor, -1)

        weighted_loss = tf.multiply(foc_loss, class_weights_tensor)
        return weighted_loss
    return fc_loss


def weighted_f1_loss(class_weights):
    @tf.function
    def f1_loss(y_true, y_pred):
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        loss = 3.25 * precision * recall / (2.25 * precision + recall + 1e-7)
        loss = tf.multiply(tf.math.log1p(1 - loss), class_weights)
        
        return loss
    return f1_loss


def weighted_precision_loss(class_weights):
    @tf.function
    def f1_loss(y_true, y_pred):
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)

        precision = tp / (tp + fp + 1e-7)

        loss = tf.multiply(tf.math.log1p(1 - precision), class_weights)

        return loss

    return f1_loss


def weighted_tnr_loss(class_weights):
    @tf.function
    def tnr_loss(y_true, y_pred):
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=-1)
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=-1)
        
        loss = tn / (tn + fn + 1e-7)
        loss = tf.clip_by_value(loss, 1e-7, 1 - 1e-7)
        loss = tf.multiply(tf.math.log1p(1 - loss), class_weights)
        
        return loss
    return tnr_loss


def xgb_weighted_auc(preds, dtrain):
    labels = dtrain.get_label()
    weights = dtrain.get_weight()

    num_classes = len(np.unique(labels))
    preds = preds.reshape(-1, num_classes)
    preds = softmax(preds)

    weighted_auc_score = roc_auc_score(
        labels, preds,
        sample_weight=weights,
        multi_class='ovr',
        average='weighted'
    )
    return 'weighted_auc', weighted_auc_score


def xgb_weighted_precision(preds, dtrain):
    labels = dtrain.get_label()
    weights = dtrain.get_weight()

    num_classes = len(np.unique(labels))

    preds = preds.reshape(-1, num_classes)
    preds = softmax(preds, axis=1)

    preds_classes = np.argmax(preds, axis=1)

    weighted_precision_score = precision_score(
        labels, preds_classes,
        sample_weight=weights,
        average='weighted'
    )

    return 'weighted_precision', weighted_precision_score


"""-------------------------------------------Combined Functions-----------------------------------------------------"""


def weighted_prec_recall_loss(class_weights):
    class_w_tensor = tf.convert_to_tensor(class_weights, dtype=tf.float32)
    @tf.function
    def prec_recall_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        f1_loss = weighted_f1_loss(class_weights)
        prec_loss = weighted_precision_loss(class_weights)
        # fc_loss = weighted_fc_loss(class_w_tensor)
        # cat_loss = weighted_categorical_crossentropy(class_w_tensor)

        w_f1_loss = tf.reduce_sum(f1_loss(y_true, y_pred), axis=-1)
        w_prec_loss = tf.reduce_sum(prec_loss(y_true, y_pred), axis=-1)
        # w_cat = tf.reduce_sum(cat_loss(y_true, y_pred), axis=-1)
        # w_fc = tf.reduce_mean(fc_loss(y_true, y_pred), axis=-1)

        loss = (tf.math.log1p(w_f1_loss) + (tf.math.log1p(w_prec_loss) * 0.25)) # + (w_cat * 0.25) + (w_fc * 0.25))
        return loss

    return prec_recall_loss


"""-------------------------------------------Other-----------------------------------------------------"""

