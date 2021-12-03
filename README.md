# 2021 CCF 爱奇艺剧本角色情感识别 方案总结

### 赛题信息

赛道链接：https://www.datafountain.cn/competitions/518

### 最后排名

排名结果：A 榜 第 11 (0.71061951)；B 榜 第 13 (0.70903144)

在赛期前半段尝试了很多，我们混了两个<del>充电器</del>周冠军，最后一个月因为<del>肝</del>赶不上前排所以弃赛了。

### 代码结构

```
.
├── codes
│   ├── 01_process_character_names.ipynb
│   ├── 02_prepare_data.ipynb
│   ├── 03_roberta_regression.ipynb
│   ├── emb_svr
│   │   ├── emb_svr_rapids.py
│   │   ├── get_folds.py
│   │   └── run_folds.py
│   └── extra
│       ├── fix_data.ipynb
│       └── pretrain.ipynb
├── codes_batch_run
│   ├── batch_get_data.py
│   ├── batch_run_models.py
│   ├── get_data_v1.py
│   ├── get_data_v2.py
│   └── run_model.py
└── README.md
```

部分代码可能路径问题没改对，请酌情修改。

### 代码说明

`codes` 目录：

- `01_process_character_names` 替换原始数据集里的 a1 等脱敏角色名为随机的中文名字；
- `02_prepare_data` 添加前置提示语、拼接前文等预处理操作，形成可用的训练集和测试集；
- `03_roberta_regression` RoBERTa 回归模型的训练和预测，9:1 划分训练集和验证集，**单模单折单 epoch，线上 A 榜 0.707+**；
- `emb_svr` 目录内是分 fold 生成 bert 768 embedding OOF 后再使用 RAPIDS SVR 进行回归预测结果，线上 A 榜 0.704 左右，用于融合；
- `extra` 目录里是一些其他的<del>值得总结但</del>没上分的操作。

`codes_batch_run` 目录里是批量生成多种预测结果的脚本：

- `batch_get_data` 根据拼接前文的方式(所有前文/前文里只包含当前角色的)生成两种数据集；
- `batch_run_models` 根据前置提示语、MAX_LEN、前文在文本的前面还是后面、不同的预训练模型等批量跑出上百个不同结果用来测试和融合。

### 尝试有效的 trick

- 文本里增加前置提示语，如在文本开头添加 `剧本: {movie} 场景: {scene} 角色: {character_name}`、文本角色前面添加 `剧本{movie} `等：+0.003
- 拼接前文：+0.006
- 替换脱敏角色名，因为大量的**不同**剧本都使用 a1,b2 这种相同的名字，从直觉上就觉得模型会有点受干扰，所以使用了随机生成且比较常见的人名来替代原先的脱敏角色名：+0.003
- 融合方面，与 emb_svr 融合上分比较明显，建议以后的比赛可以多多尝试: +0.001

```
# 随机生成大众名字示例:

FIRST_NAMES = '羿祥惠盛捷霞阳豪誉涵颖梅湘丹勇苗悦朝君杰毓乐曦瑶全恒裕帅馨秋山诗东雯紫木水骏昊艳宗国源莲子锦尔蕾兵天钰财桥轩桐海运坤信卿诚欣茂明晓月韬泳绮侦熙龙舟雨晴元峻程金宇启浩莉彤槐巧艺伟伊扬洋琪正森文鹏辉泽婷美超玉娴智敬奎强玄心高嵘思朗萱昆宸甜凌俊治云仕亭苹喜寅书华瑜晨益仁璇满贵利沁淳林伯晞嘉辰'
SECOND_NAMES = '李王张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于'

def gen_names():
    f1 = FIRST_NAMES[random.randint(0, len(FIRST_NAMES)-1)]
    f2 = FIRST_NAMES[random.randint(0, len(FIRST_NAMES)-1)]
    s1 = SECOND_NAMES[random.randint(0, len(SECOND_NAMES)-1)]
    return f'{s1}{f1}{f2}'
```

预训练模型方面：

测试了大量模型后，比较稳定的：

- `hfl/chinese-roberta-wwm-ext`
- `nghuyong/ernie-1.0`
- `uer/chinese_roberta_L-12_H-768`

其他尝试过的模型有：

- bert-wwm
- roberta-large
- macbert
- roformer
- ernie-gram
- mengzi
- 等等

### 尝试无效的 trick 

同一个 SEED 划分数据的情况下

- 五折掉分挺多，十折没有明显上分
- FGM 没有明显上分
- SmoothL1Loss 轻微掉分
- 训练集+测试集 预训练，掉分严重
- 未脱敏的剧本 (写了个小爬虫抓取了原本的剧本，基本可以确认数据收集于 https://www.1bianju.com) 预训练，掉分严重，参考 extra 里面的代码
- 对照抓取下来的剧本进行还原未脱敏的角色名，掉分严重，参考 extra 里面的代码 (这确实出乎我的意料，因为赛题数据的脱敏非常容易产生歧义，比如`刘亦菲`和`亦菲`其实是同一个角色，但是被脱敏为了`a1`和`b2`两个不同的角色，还原后应该能改善；但结果才 0.702 左右，不知道是不是有哪个地方没做对)

### 融合

- 使用了四个单模单折的结果与两个 emb_svr 结果融合，应该能到 A 榜 0.7096+，
- 榜上的 0.710+ 是面向榜上分数进行拟合出来的，忘了怎么复现了

### 代码参考

- https://www.kaggle.com/prithvijaunjale/scibert-multi-label-classification
- https://www.kaggle.com/cdeotte/rapids-svr-boost-17-8

