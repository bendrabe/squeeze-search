import argparse
import inputs
import os
import tensorflow as tf
from tensorflow.layers import conv2d, average_pooling2d, max_pooling2d, dropout
#from tensorflow.python import debug as tf_debug
from time import sleep

_NUM_TRAIN_FILES=1024
_NUM_TRAIN_IMAGES=1281167
_SHUFFLE_BUFFER=10000
_DEFAULT_IMAGE_SIZE = 227
_NUM_CHANNELS = 3
_NUM_CLASSES = 1000
DATA_FORMAT='channels_first'
BASE_LR=0.04
WEIGHT_DECAY=0.0002
MOMENTUM=0.9

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpus", type=int, help="GPUs per node", default=4, choices=[1,2,4])
parser.add_argument("-b", "--batch_size", type=int, help="global batch size", default=512)
parser.add_argument("-ne", "--num_epochs", type=int, help="number of epochs to train", default=70)
parser.add_argument("-dd", "--data_dir", type=str, help="path to ImageNet data", default=None)
parser.add_argument("-md", "--model_dir", type=str, help="path to store summaries / checkpoints", default='summary')
args = parser.parse_args()

def fire_module(inputs, squeeze_depth, expand_depth, data_format):
    net = _squeeze(inputs, squeeze_depth, data_format)
    net = _expand(net, expand_depth, data_format)
    return net

def _squeeze(inputs, num_outputs, data_format):
    return conv2d(inputs=inputs,
                  filters=num_outputs,
                  kernel_size=[1, 1],
                  strides=1,
                  padding='valid',
                  data_format=data_format,
                  activation=tf.nn.relu,
                  use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  bias_initializer=tf.zeros_initializer())

def _expand(inputs, num_outputs, data_format):
    e1x1 = conv2d(inputs=inputs,
                  filters=num_outputs,
                  kernel_size=[1, 1],
                  strides=1,
                  padding='valid',
                  data_format=data_format,
                  activation=tf.nn.relu,
                  use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  bias_initializer=tf.zeros_initializer())

    e3x3 = conv2d(inputs=inputs,
                  filters=num_outputs,
                  kernel_size=[3, 3],
                  strides=1,
                  padding='same',
                  data_format=data_format,
                  activation=tf.nn.relu,
                  use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  bias_initializer=tf.zeros_initializer())

    return tf.concat([e1x1, e3x3], 1)

def model_fn(features, labels, mode, params):
    tf.summary.image("inputs", tf.transpose(features, [0,2,3,1]), max_outputs=6)
    lr0 = params['init_learning_rate']
    ds = params['decay_steps']
    wd = params['weight_decay']
    df = params['data_format']
    net = conv2d(inputs=features,
                 filters=96,
                 kernel_size=[7, 7],
                 strides=2,
                 padding='valid',
                 data_format=df,
                 activation=tf.nn.relu,
                 use_bias=True,
                 kernel_initializer=tf.variance_scaling_initializer(),
                 bias_initializer=tf.zeros_initializer())
    net = max_pooling2d(inputs=net,
                        pool_size=[3, 3],
                        strides=2,
                        data_format=df)
    net = fire_module(net, 16, 64, df)
    net = fire_module(net, 16, 64, df)
    net = fire_module(net, 32, 128, df)
    net = max_pooling2d(inputs=net,
                     pool_size=[3, 3],
                     strides=2,
                     data_format=df)
    net = fire_module(net, 32, 128, df)
    net = fire_module(net, 48, 192, df)
    net = fire_module(net, 48, 192, df)
    net = fire_module(net, 64, 256, df)
    net = max_pooling2d(inputs=net,
                     pool_size=[3, 3],
                     strides=2,
                     data_format=df)
    net = fire_module(net, 64, 256, df)
    net = dropout(inputs=net,
                  rate=0.5,
                  training=mode == tf.estimator.ModeKeys.TRAIN)
    net = conv2d(inputs=net,
                 filters=_NUM_CLASSES,
                 kernel_size=[1, 1],
                 strides=1, 
                 padding='valid', # no padding eqv. to pad=1 for 1x1 conv?
                 data_format=df,
                 activation=tf.nn.relu,
                 use_bias=True,
                 kernel_initializer=tf.initializers.random_normal(mean=0.0, stddev=0.01),
                 bias_initializer=tf.zeros_initializer())
    net = average_pooling2d(inputs=net,
                            pool_size=[13, 13],
                            strides=1,
                            data_format=df)

    # TODO fix for data_format later
    logits = tf.squeeze(net, [2,3])

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, axis=1)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels
    )
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    l2_loss = wd * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
    )
    tf.identity(l2_loss, name='l2_loss')
    tf.summary.scalar('l2_loss', l2_loss)

    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.polynomial_decay(
            learning_rate=lr0,
            global_step=global_step,
            decay_steps=ds,
            end_learning_rate=0.0,
            power=1.0
        )
        tf.identity(learning_rate, 'learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=MOMENTUM
        )
        grad_vars = optimizer.compute_gradients(loss)
        minimize_op = optimizer.apply_gradients(grad_vars, global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(
        tf.nn.in_top_k(
            predictions=logits, targets=labels, k=5, name='top_5_op'
        )
    )

    metrics = {
        'accuracy': accuracy,
        'accuracy_top_5': accuracy_top_5
    }

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

local_batch_size = args.batch_size // args.gpus
steps_per_epoch = ((_NUM_TRAIN_IMAGES - 1 ) // args.batch_size) + 1
decay_steps = args.num_epochs*steps_per_epoch
init_learning_rate = BASE_LR * args.batch_size / 512
input_shape = (_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS)

if args.data_dir is not None:
    input_fn = inputs.get_real_input_fn
else:
    input_fn = inputs.get_synth_input_fn(list(input_shape), _NUM_CLASSES)
    init_learning_rate = 0.0 # protect from NaN

def train_input_fn():
    return input_fn(
        is_training=True,
        data_dir=args.data_dir,
        batch_size=local_batch_size
    )

def eval_input_fn():
    return input_fn(
        is_training=False,
        data_dir=args.data_dir,
        batch_size=local_batch_size
    )

tf.logging.set_verbosity( tf.logging.INFO )

session_config = tf.ConfigProto(allow_soft_placement=True)

config = tf.estimator.RunConfig(
    session_config=session_config,
    save_checkpoints_steps=steps_per_epoch,
    train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=args.gpus)
)

classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=args.model_dir,
    config=config, params={
        'init_learning_rate': init_learning_rate,
        'decay_steps': decay_steps,
        'weight_decay': WEIGHT_DECAY,
        'data_format': DATA_FORMAT
    })

_TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',
                                        'cross_entropy',
                                        'train_accuracy'])

hooks = [tf.train.LoggingTensorHook(_TENSORS_TO_LOG, every_n_iter=100)]
#hooks = [tf_debug.LocalCLIDebugHook()]
#hooks = [tf.train.ProfilerHook(save_steps=1000)]

for _ in range(args.num_epochs):
    classifier.train(input_fn=train_input_fn, steps=steps_per_epoch, hooks=hooks)
    classifier.evaluate(input_fn=eval_input_fn)
