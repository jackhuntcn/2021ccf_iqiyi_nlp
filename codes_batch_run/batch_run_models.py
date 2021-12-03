#-*- coding: utf8 -*-

import os
import glob

all_files = glob.glob('./data/train_text02*.csv')
for file_input in all_files:
    file_input = file_input.split('/')[2].replace('train_', '').replace('.csv', '')
    for model_type in [0, 1, 2]:
        for loss_type in [0]:                               # 暂时先跑 RMSELoss
            print('file_input', file_input)
            print('model_type', model_type)
            print('loss_type', loss_type)
            os.system(f'python run_model.py {file_input} {model_type} {loss_type}')
