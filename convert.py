import torch
from collections import OrderedDict

# model_path2 = './swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth'
# model_path2 = './swinv2_large_patch4_window12_192_22k.pth'
model_path2 = './cswin_large_384.pth'
weights2 = torch.load(model_path2, 'cpu')

swin_weights = OrderedDict()
# for key in weights2['model'].keys():
for key in weights2['state_dict_ema'].keys():
    if 'head' in key:
        continue
    new_layer_name = key
    # swin_weights[new_layer_name] = weights2['model'][key]
    swin_weights[new_layer_name] = weights2['state_dict_ema'][key]

# torch.save(swin_weights, './swinv2_large_patch4_window12to24_192to384_22kto1k_no_head.pth')
# torch.save(swin_weights, './swinv2_large_patch4_window12_192_22k_no_head.pth')
torch.save(swin_weights, './cswin_large_384_no_head.pth')