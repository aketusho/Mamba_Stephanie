# -- coding:utf-8 --
#pth_file_path = '../../../pretrainedmodels/PointMamba/pretrain.pth'


import torch


model_path = '../../../pretrainedmodels/PointMamba/pretrain.pth'


model = torch.load(model_path, map_location=torch.device('cpu'))

# 打印模型内容
print(model)
