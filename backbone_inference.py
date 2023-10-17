import torch
import numpy as np
import datasets.data_loader as data_loader
from models.backbone.swin_transformer_backbone import SwinTransformer

# constants
batch_size = 100

# Define model
model = SwinTransformer(
    img_size=384, 
    embed_dim=128, 
    depths=[ 2, 2, 18, 2 ], 
    num_heads=[ 4, 8, 16, 32 ], 
    window_size=12, 
    drop_path_rate=0.2
)

# Load model weights
model.load_weights("/home/arlamb/school/ds565/Swin-Transformer/swin_base_patch4_window12_384_22kto1k_no_head.pth")

# Freeze model weights
for _name, _weight in model.named_parameters():
            _weight.requires_grad = False

# Make sure model is on gpu and in eval mode
model = model.to('cuda').eval()

# Get test dataset
dataset = data_loader.load_val('./mscoco/misc/ids2path_json/coco_test_ids2path.json', '', './mscoco/feature/coco2014').dataset

# Get actual images and delete dataset dataloader
dataset_size = len(dataset)
data = torch.zeros((dataset_size, 3, 384, 384), dtype=torch.float32)
for i, it in enumerate(dataset):
    data[i] = it[2]
del dataset

# Send data to gpu
data = data.to('cuda')

# Define outputs
output = torch.zeros((dataset_size, 144, 1024), dtype=torch.float32, device='cpu')

# Inference
for idx in range(batch_size, dataset_size, batch_size):
        with torch.no_grad():
         output[idx-batch_size:idx] = model.forward(data[idx-batch_size:idx])

# Save outputs
with open('swin_transfomer_output.npy', 'wb') as f:
    np.save(f, output)