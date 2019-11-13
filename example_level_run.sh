python -u run_omniglot_example_level.py --dp-notion 'example_level' --result-dir 'results/example_level' --result-file 'test' --eval-interval 10 --eval-samples 200 --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 10 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint example_Level_ckpt_o15t --transductive --noise-multiplier 0.51 --max-grad-norm 0.1 --dp-sgd-lr 0.001


