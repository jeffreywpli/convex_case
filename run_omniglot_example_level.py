"""
Train a model on Omniglot.
"""

import random

import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models_dp_example_level import OmniglotModel
from supervised_reptile.omniglot import read_dataset, split_dataset, augment_dataset
from supervised_reptile.train_example_level import train
from supervised_reptile.writer import print_metrics

DATA_DIR = 'data/omniglot'

def main():
    """
    Load data and train a model on it.
    """
    print("In Example Level")
    args = argument_parser().parse_args()
    print(args.dp_notion)
    random.seed(args.seed)

    train_set, test_set = split_dataset(read_dataset(DATA_DIR))
    train_set = list(augment_dataset(train_set))
    test_set = list(test_set)

    model = OmniglotModel(args.classes,
        max_grad_norm=args.max_grad_norm, noise_multiplier=args.noise_multiplier, microbatches=None, dp_sgd_lr=args.dp_sgd_lr,
        **model_kwargs(args))

    with tf.Session() as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model, train_set, test_set, args.checkpoint, args.result_dir, args.result_file, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        #final_train_acc = evaluate(sess, model, train_set, **eval_kwargs)
        #print_metrics(0, final_train_acc, args.result_dir, args.result_file)
        #print('Train accuracy: ' + str(initial_test_acc))
        final_test_acc = evaluate(sess, model, test_set, **eval_kwargs)
        print_metrics(args.meta_iters, final_test_acc, args.result_dir, args.result_file)
        print('Test accuracy: ' + str(final_test_acc))

if __name__ == '__main__':
    main()
