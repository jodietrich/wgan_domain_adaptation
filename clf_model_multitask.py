# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

# used for field strength classifier

import tensorflow as tf
from tfwrapper import losses


def inference(images, inference_handle, nlabels, training=True):
    '''
    Wrapper function to provide an interface to a model from the model_zoo inside of the model module. 
    '''

    diag_logits, age_logits = inference_handle(images, training, nlabels)

    return diag_logits, age_logits


def loss(diag_logits,
         age_logits,
         diag_labels,
         age_labels,
         nlabels,
         diag_weight=1.0,
         age_weight=1.0,
         weight_decay=0.0,
         use_ordinal_reg=False,
         **kwargs):
    '''
    Loss to be minimised by the neural network
    :param diag_logits: The output of the neural network before the softmax
    :param labels: The ground truth labels in standard (i.e. not one-hot) format
    :param nlabels: The number of GT labels
    :param loss_type: Can be 'weighted_crossentropy'/'crossentropy'/'dice'/'crossentropy_and_dice'
    :param weight_decay: The weight for the L2 regularisation of the network paramters
    :return: The total loss including weight decay, the loss without weight decay, only the weight decay 
    '''

    diag_labels = tf.one_hot(diag_labels, depth=nlabels)

    with tf.variable_scope('weights_norm') as scope:

        weights_norm = tf.reduce_sum(
            input_tensor = weight_decay*tf.stack(
                [tf.nn.l2_loss(ii) for ii in tf.get_collection('weight_variables')]
            ),
            name='weights_norm'
        )

    classification_loss = diag_weight*losses.cross_entropy_loss(diag_logits, diag_labels)

    ordinal_reg_weights = kwargs['ordinal_reg_weights'] if 'ordinal_reg_weights' in kwargs else None

    if use_ordinal_reg:
        age_loss = losses.ordinal_prediction_loss(age_logits, age_labels, weights=ordinal_reg_weights)
    else:
        age_labels_one_hot = tf.one_hot(age_labels, depth=6)
        age_loss = 0.30*losses.cross_entropy_loss(age_logits, age_labels_one_hot)

    age_loss = age_weight*age_loss

    total_loss = classification_loss + age_loss + weights_norm

    return total_loss, classification_loss, age_loss, weights_norm


def predict(images, exp_config):
    '''
    Returns the prediction for an image given a network from the model zoo
    :param images: An input image tensor
    :param inference_handle: A model function from the model zoo
    :return: A prediction mask, and the corresponding softmax output
    '''

    diag_logits, ages_logits = exp_config.clf_model_handle(images,
                               training=tf.constant(False, dtype=tf.bool),
                               nlabels=exp_config.nlabels,
                               n_age_thresholds=len(exp_config.age_bins))

    diag_softmax = tf.nn.softmax(diag_logits)
    diag_labels = tf.arg_max(diag_softmax, dimension=-1)

    age_softmaxs = []
    for logit in ages_logits:
        age_softmaxs.append(tf.nn.softmax(logit))

    return diag_labels, diag_softmax, age_softmaxs



def evaluation(diag_logits, age_logits, diag_labels, age_labels, images, nlabels, age_weight, diag_weight, use_ordinal_reg):
    '''
    A function for evaluating the performance of the netwrok on a minibatch. This function returns the loss and the
    current foreground Dice score, and also writes example segmentations and imges to to tensorboard.
    :param logits: Output of network before softmax
    :param labels: Ground-truth label mask
    :param images: Input image mini batch
    :param nlabels: Number of labels in the dataset
    :param loss_type: Which loss should be evaluated
    :return: The loss without weight decay, the foreground dice of a minibatch
    '''

    total_loss, diag_loss, age_loss, weight_norm = loss(diag_logits,
                                                        age_logits,
                                                        diag_labels,
                                                        age_labels,
                                                        nlabels,
                                                        age_weight=age_weight,
                                                        diag_weight=diag_weight,
                                                        use_ordinal_reg=use_ordinal_reg)

    diag_softmax = tf.nn.softmax(diag_logits)
    diag_pred_labels = tf.arg_max(diag_softmax, dimension=-1)

    if use_ordinal_reg:
        age_softmaxs = []
        for logit in age_logits:
            age_softmaxs.append(tf.nn.softmax(logit))
    else:
        age_softmaxs = tf.nn.softmax(age_logits)

    return diag_loss, age_loss, diag_pred_labels, age_softmaxs


