CUDA_VISIBLE_DEVICES=0 python train_monot53b.py --model "t5-base"
CUDA_VISIBLE_DEVICES=1 python train_monot53b.py --model "t5-large"
wait
# CUDA_VISIBLE_DEVICES=2 python train_monot53b.py --model "t5-3b" &
# wait