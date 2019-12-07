import argparse
import json
import numpy as np
import tensorflow as tf
#from tensorflow.python import debug as tf_debug

import inputs
import squeezenet

_NUM_TRAIN_IMAGES=1281167
_DEFAULT_IMAGE_SIZE = 227
_NUM_CHANNELS = 3
_NUM_CLASSES = 1000
DATA_FORMAT='channels_first'
MOMENTUM=0.9

def model_fn(features, labels, mode, params):
    tf.summary.image("inputs", tf.transpose(features, [0,2,3,1]), max_outputs=6)
    lr0 = params['lr0']
    lr_decay_rate = params['lr_decay_rate']
    warmup_steps = params['warmup_steps']
    weight_decay = params['weight_decay']
    data_format = params['data_format']
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    logits = squeezenet.model(features, is_training, data_format, _NUM_CLASSES)

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

    l2_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
    )
    tf.identity(l2_loss, name='l2_loss')
    tf.summary.scalar('l2_loss', l2_loss)

    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        lr = tf.train.natural_exp_decay(
            learning_rate=lr0,
            global_step=global_step,
            decay_steps=1,
            decay_rate=lr_decay_rate
        )
        post_warmup_lr = lr0 * np.exp(-1 * lr_decay_rate * warmup_steps)
        warmup_lr = (post_warmup_lr * 
            tf.cast(global_step, tf.float64) / tf.cast(warmup_steps, tf.float64))
        learning_rate = tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

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

class Experiment:
    def __init__(self,
                 num_gpus=4,
                 num_epochs=70,
                 data_dir='/data/imagenet-tfrecord/',
                 model_dir='summary',
                 global_batch_size=512,
                 lr0=0.04,
                 lr_decay_rate=3.1e-5,
                 warmup_epochs=0,
                 weight_decay=0.0002):
        self.num_gpus = num_gpus
        self.num_epochs = num_epochs
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.global_batch_size = global_batch_size
        self.lr0 = lr0
        self.lr_decay_rate = lr_decay_rate
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay

        # TODO: error handling, make sure gbs is multiple of num_gpus
        self.local_batch_size = global_batch_size // num_gpus
        self.steps_per_epoch = ((_NUM_TRAIN_IMAGES - 1 ) // global_batch_size) + 1
        self.input_shape = (_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS)
        self.warmup_steps = int(self.steps_per_epoch * self.warmup_epochs)

        self.hyperparams = {
            'global_batch_size': self.global_batch_size,
            'lr0': self.lr0,
            'lr_decay_rate': self.lr_decay_rate,
            'warmup_epochs': self.warmup_epochs,
            'weight_decay': self.weight_decay
        }

    def log_hyperparams(self):
        with open(self.model_dir + 'hyperparams.txt', 'w') as f:
            json.dump(self.hyperparams, f)

    def execute(self):
        if self.data_dir is not None:
            input_fn = inputs.get_real_input_fn
        else:
            input_fn = inputs.get_synth_input_fn(list(self.input_shape), _NUM_CLASSES)
            self.lr0 = 0.0 # protect from NaN

        def train_input_fn():
            return input_fn(
                is_training=True,
                data_dir=self.data_dir,
                batch_size=self.local_batch_size
            )

        def eval_input_fn():
            return input_fn(
                is_training=False,
                data_dir=self.data_dir,
                batch_size=self.local_batch_size
            )

        tf.logging.set_verbosity( tf.logging.INFO )

        session_config = tf.ConfigProto(allow_soft_placement=True)

        config = tf.estimator.RunConfig(
            session_config=session_config,
            save_checkpoints_steps=self.steps_per_epoch,
            keep_checkpoint_max=1,
            save_summary_steps=500,
            train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=self.num_gpus)
        )

        classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=self.model_dir,
            config=config, params={
                'lr0': self.lr0,
                'lr_decay_rate': self.lr_decay_rate,
                'weight_decay': self.weight_decay,
                'warmup_steps': self.warmup_steps,
                'data_format': DATA_FORMAT
            })

        _TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',
                                                'cross_entropy',
                                                'train_accuracy'])

        hooks = [tf.train.LoggingTensorHook(_TENSORS_TO_LOG, every_n_iter=500)]
        #hooks = [tf_debug.LocalCLIDebugHook()]
        #hooks = [tf.train.ProfilerHook(save_steps=1000)]

        max_val = 0.0
        max_epoch = 0

        for epoch in range(self.num_epochs):
            classifier.train(input_fn=train_input_fn, steps=self.steps_per_epoch, hooks=hooks)
            eval_results = classifier.evaluate(input_fn=eval_input_fn)
            val = eval_results['accuracy']
            if val > max_val:
                max_val = val
                max_epoch = epoch
            # if at least 2 epochs have trained and acc still not responding, quit
            if epoch >= 1 and val < 0.01:
                break
            # if peak performance was more than 5 epochs ago, quit
            if epoch - max_epoch >= 5:
                break
