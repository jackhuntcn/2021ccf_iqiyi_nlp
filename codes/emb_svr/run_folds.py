import warnings
warnings.simplefilter('ignore')

import os
import sys
import time
import copy
import json
import random

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_colwidth', 400)
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, AutoConfig

# 参数设置

FOLD = int(sys.argv[1])

class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.SEED = 71
        self.MODEL_PATH = 'hfl/chinese-roberta-wwm-ext'
        self.NUM_LABELS = 6

        # data
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        self.MAX_LENGTH = 400
        self.BATCH_SIZE = 8

        # model
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = True
        self.LR = 3e-5
        self.OPTIMIZER = 'AdamW'
        self.N_VALIDATE_DUR_TRAIN = 3
        self.N_WARMUP = 0
        self.SAVE_BEST_ONLY = True
        self.EPOCHS = 1
        self.USE_FGM = False


config = Config()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


np.random.seed(config.SEED)
seed_torch(seed=config.SEED)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train = pd.read_csv('../data/train_scenes.csv')
test = pd.read_csv('../data/test_scenes.csv')

submit = pd.read_csv('../raw_data/submit_example.tsv', sep='\t')

train['labels'] = train['emotions'].apply(lambda x: [int(i) for i in x.split(',')])
for i in range(6):
    train[f'label_{i}'] = train['labels'].apply(lambda x: x[i])

train_df = train[['id', 'movie', 'text'] + [f'label_{i}' for i in range(6)]].copy().reset_index(drop=True)
test_df = test[['id', 'movie', 'text']].copy().reset_index(drop=True)

class TransformerDataset(Dataset):
    def __init__(self, df, indices, set_type=None):
        super(TransformerDataset, self).__init__()

        df = df.iloc[indices]
        self.texts = df['text'].values.tolist()
        self.set_type = set_type
        if self.set_type != 'test':
            self.labels = df.iloc[:, 3:].values

        self.tokenizer = config.TOKENIZER
        self.max_length = config.MAX_LENGTH

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        tokenized = self.tokenizer.encode_plus(
            self.texts[index],
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'].squeeze()
        attention_mask = tokenized['attention_mask'].squeeze()

        if self.set_type != 'test':
            return {
                'input_ids': input_ids.long(),
                'attention_mask': attention_mask.long(),
                'labels': torch.Tensor(self.labels[index]).float(),
            }

        return {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.long(),
        }

with open('folds.json') as f:
    kfolds = json.load(f)

train_indices = kfolds[f'fold_{FOLD}']['train']
valid_indices = kfolds[f'fold_{FOLD}']['valid']

train_data = TransformerDataset(train_df, train_indices)
valid_data = TransformerDataset(train_df, valid_indices)

train_dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=config.BATCH_SIZE, shuffle=False)

class FGM(object):
    def __init__(self, model, emb_name, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return


class Model(nn.Module):
    def __init__(self, ):
        super().__init__()

        cfg = AutoConfig.from_pretrained(config.MODEL_PATH)
        cfg.update({"output_hidden_states": True,
                    "hidden_dropout_prob": 0.0,
                    "layer_norm_eps": 1e-7})

        self.roberta = AutoModel.from_pretrained(config.MODEL_PATH, config=cfg)

        dim = self.roberta.pooler.dense.bias.shape[0]

        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = 12
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.attention = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Linear(768, 1),
            nn.Softmax(dim=1)
        )
        self.cls = nn.Sequential(
            nn.Linear(dim, 6)
        )
        init_params([self.cls, self.attention])

    def get_bert_emb(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)
        last_hidden_state = roberta_output.last_hidden_state
        CLS_token_state = last_hidden_state[:, 0, :]
        return CLS_token_state

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)

        cls_outputs = torch.stack(
            [self.dropout(layer) for layer in roberta_output[2][-12:]], dim=0
        )
        cls_output = (
                torch.softmax(self.layer_weights, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(1) * cls_outputs).sum(
            0)

        logits = torch.mean(
            torch.stack(
                [torch.sum(self.attention(self.high_dropout(cls_output)) * cls_output, dim=1) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )
        return self.cls(logits)
    
device = config.DEVICE

def val(model, valid_dataloader, criterion):
    val_loss = 0
    true, pred = [], []

    # set model.eval() every time during evaluation
    model.eval()

    for step, batch in enumerate(valid_dataloader):
        b_input_ids = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        with torch.no_grad():
            # forward pass
            logits = model(input_ids=b_input_ids, attention_mask=b_attention_mask)

            # calculate loss
            loss = criterion(logits, b_labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_dataloader)
    print('Val loss:', avg_val_loss)
    return avg_val_loss


def train(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, epoch):
    # we validate config.N_VALIDATE_DUR_TRAIN times during the training loop
    nv = config.N_VALIDATE_DUR_TRAIN
    temp = len(train_dataloader) // nv
    temp = temp - (temp % 100)
    validate_at_steps = [temp * x for x in range(1, nv + 1)]
    
    if config.USE_FGM:
        fgm = FGM(model, epsilon=1, emb_name='word_embeddings.')

    train_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader,
                                      desc='Epoch ' + str(epoch))):
        # set model.eval() every time during training
        model.train()

        # unpack the batch contents and push them to the device (cuda or cpu).
        b_input_ids = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        # clear accumulated gradients
        optimizer.zero_grad()

        # forward pass
        logits = model(input_ids=b_input_ids, attention_mask=b_attention_mask)

        # calculate loss
        loss = criterion(logits, b_labels)
        train_loss += loss.item()

        # backward pass
        loss.backward()
        
        # fgm attack
        if config.USE_FGM:
            fgm.attack()
            logits_adv = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
            loss_adv = criterion(logits_adv, b_labels)
            loss_adv.backward()
            fgm.restore()

        # update weights
        optimizer.step()

        # update scheduler
        scheduler.step()

        if step in validate_at_steps:
            print(f'-- Step: {step}')
            _ = val(model, valid_dataloader, criterion)

    avg_train_loss = train_loss / len(train_dataloader)
    print('Training loss:', avg_train_loss)

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss
    
def run():
    # setting a seed ensures reproducible results.
    # seed may affect the performance too.
    torch.manual_seed(config.SEED)

    # criterion = nn.BCEWithLogitsLoss()
    criterion = RMSELoss()

    # define the parameters to be optmized -
    # - and add regularization
    if config.FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_parameters, lr=config.LR)

    num_training_steps = len(train_dataloader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    min_avg_val_loss = float('inf')
    for epoch in range(config.EPOCHS):
        train(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, epoch)
        avg_val_loss = val(model, valid_dataloader, criterion)

        if config.SAVE_BEST_ONLY:
            if avg_val_loss < min_avg_val_loss:
                best_model = copy.deepcopy(model)
                best_val_mse_score = avg_val_loss

                model_name = f'model_{FOLD}'
                torch.save(best_model.state_dict(), 'models/' + model_name + '.pt')

                print(f'--- Best Model. Val loss: {min_avg_val_loss} -> {avg_val_loss}')
                min_avg_val_loss = avg_val_loss

    return best_model, best_val_mse_score


model = Model()
model.to(device)

best_model, best_val_mse_score = run()

def predict(model, loader):
    preds = []
    model.eval()
    for step, batch in enumerate(loader):
        b_input_ids = batch['input_ids'].to(device)
        b_attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            embs = model.get_bert_emb(input_ids=b_input_ids, attention_mask=b_attention_mask)
            embs = embs.cpu().numpy()
            preds.extend(embs)

    preds = np.array(preds)
    return preds

dataset_size = len(test_df)
test_indices = list(range(dataset_size))

test_data = TransformerDataset(test_df, test_indices, set_type='test')
test_dataloader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

oof_preds = predict(best_model, valid_dataloader)
oof_df = pd.DataFrame({'idx': valid_indices, 'embs': list(oof_preds)})

test_preds = predict(best_model, test_dataloader)
test['embs'] = list(test_preds)
sub = submit.copy()
del sub['emotion']
sub = sub.merge(test[['id', 'embs']], how='left', on='id')

oof_df.to_pickle(f'oof/oof_{FOLD}.pickle')
sub.to_pickle(f'sub/test_{FOLD}.pickle')
