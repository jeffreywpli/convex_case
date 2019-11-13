"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf

import os
import sys

from privacy.optimizers import dp_optimizer
DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)
# pylint: disable=R0903
class OmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes=5, max_grad_norm=0.5, noise_multiplier=0.51, microbatches=10, dp_sgd_lr=0.01, refine_optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.microbatches = microbatches
        self.dp_sgd_lr = dp_sgd_lr
        self.refine_optimizer = refine_optimizer
        print("Noise: ", self.noise_multiplier)
        print("Clipping Threshold", self.max_grad_norm)

        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = self._build_train_op(self.dp_sgd_lr, self.loss)

        self.refine_op = self.refine_optimizer(**optim_kwargs).minimize(self.loss)

    def _build_train_op(self, lr, vector_loss):
        #optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
        optimizer = dp_optimizer.DPAdamGaussianOptimizer(
            l2_norm_clip=self.max_grad_norm,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=None,
            ledger=None,
            learning_rate=lr,
            unroll_microbatches=False,
            beta1=0)
        train_op = optimizer.minimize(loss=vector_loss, global_step=tf.train.get_or_create_global_step())
        return train_op


class MiniImageNetModel:
    """
    A model for MiniImageNet classification.
    """
    def __init__(self, num_classes=5, max_grad_norm=0.5, noise_multiplier=0.51, microbatches=10, dp_sgd_lr=0.01, refine_optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.microbatches = microbatches
        self.dp_sgd_lr = dp_sgd_lr
        self.refine_optimizer = refine_optimizer
        print("Noise: ", self.noise_multiplier)
        print("Clipping Threshold", self.max_grad_norm)

        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3))
        out = self.input_ph
        for _ in range(4):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))

        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = self._build_train_op(self.dp_sgd_lr, self.loss)

        self.refine_op = self.refine_optimizer(**optim_kwargs).minimize(self.loss)

    def _build_train_op(self, lr, vector_loss):
        #optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
        optimizer = dp_optimizer.DPAdamGaussianOptimizer(
            l2_norm_clip=self.max_grad_norm,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=None,
            ledger=None,
            learning_rate=lr,
            unroll_microbatches=False,
            beta1=0)
        train_op = optimizer.minimize(loss=vector_loss, global_step=tf.train.get_or_create_global_step())
        return train_op


# pylint: disable=R0903
class WikiModel:
    """
    A model for the wiki dataset.
    """
    def __init__(self, num_classes=5, input_dim=50, max_grad_norm=0.5, noise_multiplier=0.51, microbatches=10, dp_sgd_lr=0.01, refine_optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.microbatches = microbatches
        self.dp_sgd_lr = dp_sgd_lr
        self.refine_optimizer = refine_optimizer
        print("Noise: ", self.noise_multiplier)
        print("Clipping Threshold", self.max_grad_norm)

        self.input_ph = tf.placeholder(tf.float32, shape=(None, input_dim), name='input_ph')
        self.label_ph = tf.placeholder(tf.int32, shape=(None,), name='label_ph')

        self.logits = tf.layers.dense(self.input_ph, num_classes)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)

        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = self._build_train_op(self.dp_sgd_lr, self.loss)

        self.refine_op = self.refine_optimizer(**optim_kwargs).minimize(self.loss)

    def _build_train_op(self, lr, vector_loss):
        optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
        #optimizer = dp_optimizer.DPAdamGaussianOptimizer(
            l2_norm_clip=self.max_grad_norm,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=None,
            ledger=None,
            learning_rate=lr,
            unroll_microbatches=False,
            beta1=0)
        train_op = optimizer.minimize(loss=vector_loss, global_step=tf.train.get_or_create_global_step())
        return train_op
