python -u run_omniglot_user_level.py --dp-notion 'user_level' --result-dir 'results/user_level' --result-file 'test' --eval-interval 5000 --eval-samples 1000 --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint user_Level_ckpt_o15t --transductive --noise-multiplier 0.51 --max-grad-norm 0.1


