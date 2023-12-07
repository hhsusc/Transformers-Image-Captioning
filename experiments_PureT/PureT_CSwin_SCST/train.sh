CUDA_VISIBLE_DEVICES=0 python main.py --folder ./experiments_PureT/PureT_CSwin_SCST/ --resume 14
# CUDA_VISIBLE_DEVICES=0 python main.py --folder ./experiments_PureT/PureT_CSwin_SCST/ --resume 3
# CUDA_VISIBLE_DEVICES=0 python main.py --folder ./experiments_PureT/PureT_CSwin_SCST/ --resume 2 --load_epoch

# CUDA_VISIBLE_DEVICES=0, 1 python -m torch.distributed.launch --master_port=3142 --nproc_per_node=2 main_multi_gpu.py --folder ./experiments_PureT/PureT_XE/