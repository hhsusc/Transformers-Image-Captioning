import torch
from collections import OrderedDict

model_path2 = './swin_base_patch4_window12_384_22kto1k.pth'
weights2 = torch.load(model_path2, 'cpu')

swin_weights = OrderedDict()
for key in weights2['model'].keys():
    if 'head' in key:
        continue
    new_layer_name = key
    swin_weights[new_layer_name] = weights2['model'][key]

torch.save(swin_weights, './swin_base_patch4_window12_384_22kto1k_no_head.pth')