"""
Train a model on Omniglot.
"""

import random

import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models_dp_example_level import WikiModel
from supervised_reptile.wiki3029 import Wiki3029DatasetsBuilder
from supervised_reptile.train import train
from supervised_reptile.writer import print_metrics

DATA_DIR = 'data/wiki3029/wiki3029_pck/'

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    wiki_dataset_builder = Wiki3029DatasetsBuilder(DATA_DIR)
    train_set, val_set, test_set = wiki_dataset_builder._read_dataset()

    model = WikiModel(args.classes,
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
        #print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
        final_test_acc = evaluate(sess, model, test_set, **eval_kwargs)
        print_metrics(args.meta_iters, final_test_acc, args.result_dir, args.result_file)
        print('Test accuracy: ' + str(final_test_acc))

if __name__ == '__main__':
    main()
