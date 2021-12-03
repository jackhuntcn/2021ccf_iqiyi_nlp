# 2021 CCF 爱奇艺剧本角色情感识别 方案总结

### 赛题信息

赛道链接：https://www.datafountain.cn/competitions/518

### 最后排名

排名结果：A 榜 第 11 (0.71061951)；B 榜 第 13 (0.70903144)

在赛期前半段尝试了很多，我们混了两个<del>充电器</del>周冠军，最后一个月因为<del>肝</del>赶不上前排所以弃赛了。

### 代码结构

```
.
├── batch_run
│   ├── batch_get_data.py
│   ├── batch_run_models.py
│   ├── get_data_v1.py
│   ├── get_data_v2.py
│   └── run_model.py
├── notebooks
│   ├── 01_process_character_names.ipynb
│   ├── 02_prepare_data.ipynb
│   ├── 03_roberta_regression.ipynb
│   └── extra
│       ├── fix_data.ipynb
│       └── pretrain.ipynb
└── README.md
```

