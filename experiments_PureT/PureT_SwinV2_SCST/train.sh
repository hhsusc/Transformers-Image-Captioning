# CUDA_VISIBLE_DEVICES=3,2,1,0 python3 -m torch.distributed.launch --nproc_per_node=4 main.py --folder ./experiments/xtransformer

CUDA_VISIBLE_DEVICES=0 python main.py --folder ./experiments_PureT/PureT_SwinV2_SCST/ --resume 10 --load_epoch

# CUDA_VISIBLE_DEVICES=0, 1 python -m torch.distributed.launch --master_port=3142 --nproc_per_node=2 main_multi_gpu.py --folder ./experiments_PureT/PureT_SCST/ --resume 15