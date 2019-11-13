# Omniglot
#python -u run_omniglot.py --result-dir 'results/noiseless' --result-file 'nichol_params' --eval-interval 4000 --eval-samples 10000 --shots 1 --inner-batch 10 --inner-iters 5 --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 10 --checkpoint ckpt_o15t --transductive

# Mini-Imagenet
python -u run_miniimagenet.py --result-dir 'results/mini_imagenet' --result-file 'nichol_params' --eval-interval 4000  --eval-samples 10000 --shots 5 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 20000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m55 --transductive

