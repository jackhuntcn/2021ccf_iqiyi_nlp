import warnings
warnings.simplefilter('ignore')

import os
import gc
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

PREV_TEXT_ORDER = int(sys.argv[1])
CUR_PREV_ORDER = int(sys.argv[2])
ADD_SCENE_TEXT = int(sys.argv[3])
ADD_CHARACTER_TEXT = int(sys.argv[4])
MAX_SEQ_LEN = int(sys.argv[5])

config = {
    # 前文排序方式
    # 0: 倒序
    # 1: 正序
    'PREV_TEXT_ORDER': PREV_TEXT_ORDER, 
    
    # 前文和当前次序
    # 0: 前文在前
    # 1: 前文在后
    'CUR_PREV_ORDER': CUR_PREV_ORDER,
    
    # 是否添加场景辅助文本
    # 0: 不添加
    # 1: 添加
    'ADD_SCENE_TEXT': ADD_SCENE_TEXT,
    
    # 角色名是否添加剧本前缀
    # 0: 不添加
    # 1: 添加
    'ADD_CHARACTER_TEXT': ADD_CHARACTER_TEXT,
    
    # 文本最长长度
    'MAX_SEQ_LEN': MAX_SEQ_LEN
}

train = pd.read_csv('data/train_with_names.csv')
test = pd.read_csv('data/test_with_names.csv')

train = train.sort_values(by=['movie', 'scene', 'movie_id']).reset_index(drop=True)
train.fillna('', inplace=True)
test = test.sort_values(by=['movie', 'scene', 'movie_id']).reset_index(drop=True)
test.fillna('', inplace=True)

df_train = train.groupby(['movie', 'scene'])[['id', 'content', 'character', 'emotions', 'character_name']].agg(list).reset_index()
df_test = test.groupby(['movie', 'scene'])[['id', 'content', 'character', 'character_name']].agg(list).reset_index()

train_data = list()

for _, row in tqdm(df_train.iterrows()):
    
    movie = row['movie']
    scene = row['scene']
    contents = row['content']
    characters = row['character']
    character_names = row['character_name']
    emotions = row['emotions']
    ids = row['id']
    
    # 以 content 开始历遍, 方便 test 处理
    for idx, content in enumerate(contents):
        id_ = ids[idx]                                                      # 当前 id
        emotion = emotions[idx]                                             # 当前情绪标签
        character = characters[idx].split('_')[-1]                          # 当前角色
        character_name = character_names[idx]                               # 当前角色名
        
        res = dict()
        if emotion == '':                                                   # 无标签跳过
            continue
        else:
            text_list = contents[:idx+1]                                    # 包括当前文本在内的所有句子
            
            # 排除重复文本
            seqs = list()
            for item in text_list:
                if item not in seqs:
                    seqs.append(item)  
                    
            # 添加辅助文本
            if config['ADD_SCENE_TEXT'] == 1:
                text_begin = f'剧本: {movie} 场景: {scene} 角色: {character_name}' 
            else:
                text_begin = f'剧本: {movie} 角色: {character_name}'
                
            if config['ADD_CHARACTER_TEXT'] == 1 and character != 'nan':    # 存在 character 为空的情况
                content = content.replace(character_name, 
                                      f' 剧本{movie} {character_name}')
            text = ' 当前: ' + content
            text_prev = ''
            prev_str = ' 前文: '
            cur_len = len(text) + len(text_begin) + len(text_prev) + len(prev_str)         # 用来做长度计算
            for s in seqs[:-1][::-1]:                                       # 不包括当前句子, 倒序取出
                if cur_len + len(s) <= config['MAX_SEQ_LEN']:               # 没有超出最大长度
                    if not re.search(character_name, s):                    # 没有当前角色不拼接
                        continue
                    if config['PREV_TEXT_ORDER'] == 0:
                        text_prev = text_prev + s
                    else:
                        text_prev = s + text_prev
                    cur_len = cur_len + len(s)
                else:
                    break
            # 组装起来
            if config['CUR_PREV_ORDER'] == 1:
                text = text_begin + text + prev_str + text_prev
            else:
                text = text_begin + prev_str + text_prev + text
            text = text.strip()
            
            # 组成数据集
            res['id'] = id_
            res['movie'] = movie
            res['scene'] = scene
            res['character'] = character
            res['character_name'] = character_name
            res['text'] = text
            res['emotions'] = emotion
            train_data.append(res)
            
train_data = pd.DataFrame(train_data)

test_data = list()

for _, row in tqdm(df_test.iterrows()):
    
    movie = row['movie']
    scene = row['scene']
    contents = row['content']
    characters = row['character']
    character_names = row['character_name']
    ids = row['id']
    
    # 以 content 开始历遍, 方便 test 处理
    for idx, content in enumerate(contents):
        id_ = ids[idx]                                                      # 当前 id
        character = characters[idx].split('_')[-1]                          # 当前角色
        character_name = character_names[idx]                               # 当前角色名
        
        res = dict()

        text_list = contents[:idx+1]                                    # 包括当前文本在内的所有句子

        # 排除重复文本
        seqs = list()
        for item in text_list:
            if item not in seqs:
                seqs.append(item)  

        # 添加辅助文本
        if config['ADD_SCENE_TEXT'] == 1:
            text_begin = f'剧本: {movie} 场景: {scene} 角色: {character_name}' 
        else:
            text_begin = f'剧本: {movie} 角色: {character_name}'

        if config['ADD_CHARACTER_TEXT'] == 1 and character != 'nan':    # 存在 character 为空的情况
            content = content.replace(character_name, 
                                  f' 剧本{movie} {character_name}')
        text = ' 当前: ' + content
        text_prev = ''
        prev_str = ' 前文: '
        cur_len = len(text) + len(text_begin) + len(text_prev) + len(prev_str)         # 用来做长度计算
        for s in seqs[:-1][::-1]:                                       # 不包括当前句子, 倒序取出
            if cur_len + len(s) <= config['MAX_SEQ_LEN']:               # 没有超出最大长度
                if not re.search(character_name, s):                    # 没有当前角色不拼接
                    continue
                if config['PREV_TEXT_ORDER'] == 0:
                    text_prev = text_prev + s
                else:
                    text_prev = s + text_prev
                cur_len = cur_len + len(s)
            else:
                break
        # 组装起来
        if config['CUR_PREV_ORDER'] == 1:
            text = text_begin + text + prev_str + text_prev
        else:
            text = text_begin + prev_str + text_prev + text
        text = text.strip()

        # 组成数据集
        res['id'] = id_
        res['movie'] = movie
        res['scene'] = scene
        res['character'] = character
        res['character_name'] = character_name
        res['text'] = text
        test_data.append(res)
            
            
test_data = pd.DataFrame(test_data)

assert len(test_data) == len(test)

train_data.to_csv(
    f'./data2/train_text02_{PREV_TEXT_ORDER}_{CUR_PREV_ORDER}_{ADD_SCENE_TEXT}_{ADD_CHARACTER_TEXT}_{MAX_SEQ_LEN}.csv')
test_data.to_csv(
    f'./data2/test_text02_{PREV_TEXT_ORDER}_{CUR_PREV_ORDER}_{ADD_SCENE_TEXT}_{ADD_CHARACTER_TEXT}_{MAX_SEQ_LEN}.csv')
