"""
Training helpers for supervised meta-learning.
"""

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import tensorflow as tf

from .reptile import Reptile
from .variables import weight_decay
from .eval import evaluate
from .writer import print_metrics
from .args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs

try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except AttributeError:
    tf.logging.set_verbosity(tf.logging.ERROR)

# pylint: disable=R0913,R0914
def train(sess,
          model,
          train_set,
          test_set,
          save_dir,
          result_dir,
          result_file,
          num_classes=5,
          num_shots=5,
          inner_batch_size=5,
          inner_iters=20,
          replacement=False,
          meta_step_size=0.1,
          meta_step_size_final=0.1,
          meta_batch_size=1,
          meta_iters=400000,
          eval_inner_batch_size=5,
          eval_inner_iters=50,
          eval_interval=10,
          weight_decay_rate=1,
          time_deadline=None,
          train_shots=None,
          transductive=False,
          reptile_fn=Reptile):
    """
    Train a model on a dataset.
    """
    args = argument_parser().parse_args()
    eval_kwargs = evaluate_kwargs(args)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    saver = tf.train.Saver()
    reptile = reptile_fn(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    accuracy_ph = tf.placeholder(tf.float32, shape=())
    tf.summary.scalar('accuracy', accuracy_ph)
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'test'), sess.graph)

    tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())

    # Actual training begins
    for i in range(meta_iters):
        if i % 10 == 0:
            print("Reached Iteration: ", i)

        # Linearly Decaying meta step size
        frac_done = i / meta_iters
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size

        reptile.train_step(train_set, model.input_ph, model.label_ph, model.minimize_op,
                           num_classes=num_classes, num_shots=(train_shots or num_shots),
                           inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                           replacement=replacement,
                           meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size)

        if not args.hyperparameter_tuning and (i % eval_interval) == 0:
            test_acc = evaluate(sess, model, test_set, **eval_kwargs)
            #accuracies = []
            #for dataset, writer in [(test_set, test_writer)]:
            #    correct = reptile.evaluate(dataset, model.input_ph, model.label_ph,
            #                               model.refine_op, model.predictions,
            #                               num_classes=num_classes, num_shots=num_shots,
            #                               inner_batch_size=eval_inner_batch_size,
            #                               inner_iters=eval_inner_iters, replacement=replacement)
            #    summary = sess.run(merged, feed_dict={accuracy_ph: correct/num_classes})
            #    writer.add_summary(summary, i)
            #    writer.flush()
            #    accuracies.append(correct / num_classes)
            print('batch %d: test=%f' % (i, test_acc))
            print_metrics(i, test_acc, result_dir, result_file)

            if i > 2 and test_acc < 0.50:
                break

        if i % 100 == 0 or i == meta_iters-1:
            saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=i)
        if time_deadline is not None and time.time() > time_deadline:
            break
