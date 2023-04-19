import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Union, List
from pathlib import Path
import copy
from tqdm import tqdm
import random
import datasets
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModel


from common_utils.common_data_processing_utils import *
from common_utils.common_ML_utils import *


def tokenize_flat_tables(config, flat_tables: pd.DataFrame, cols_to_tokenize: List[str], pooling_mtd:str=None):
    """
    pooling_mtd: pooler_output, last_hidden_state
    """

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained)
    tokenizer.add_special_tokens(additional_special_tokens)
    
    for col in cols_to_tokenize:
        # features = datasets.Features({col: datasets.Value('string')})
        # flat_tables[col] = flat_tables[col].astype(str)
        # ds = datasets.Dataset.from_pandas(flat_tables, features=features)
        tmp = {col: flat_tables[col].astype(str).tolist()}
        ds = datasets.Dataset.from_dict(tmp)
        
        def tokenize_function(example):
            return tokenizer(example[col], truncation=True, max_length=512)

        tokenized_ds = ds.map(tokenize_function, batched=True, load_from_cache_file=False)
        tokenized_ds = tokenized_ds.remove_columns([col])

        flat_tables[f'tokenized_{col}'] = tokenized_ds['input_ids']
    
    return flat_tables


def load_single_DataWithTOkenType(config, df, drop_duplicates):
    return DataWithTOkenType(config, df, config.input_cols, config.input_num_cols, drop_duplicates=drop_duplicates)


def load_DataWithTOkenType(config, df, input_cols, input_num_cols, valid_fold, test_fold, augment=False, load_train=True):
    is_test = df.fold == test_fold
    is_valid = df.fold == valid_fold
    test_df = df[is_test].copy()
    valid_df = df[is_valid].copy()
    train_df = df[(~is_test) & (~is_valid)].copy()

    if augment is True:
        print("<<< Augmenting data for CTC!")
        for label in (1, 2, 3, 4):
            train_df = augment_data(train_df, label, int(len(df[df.labels==0]) / len(df[df.labels==label])))
        # train_df = augment_data(train_df, 4, 10)
        # train_df = augment_data(train_df, 3, 5)
        # train_df = augment_data(train_df, 1, 3)

    if load_train and 'labels' in train_df.columns:
        print(train_df.labels.value_counts())

    # don't drop duplicates for validation and test folds.
    if load_train:
        return DataWithTOkenType(config, train_df, input_cols, input_num_cols), DataWithTOkenType(config, valid_df, input_cols, input_num_cols, drop_duplicates=False), DataWithTOkenType(config, test_df, input_cols, input_num_cols, drop_duplicates=False)
    else:
        return DataWithTOkenType(config, valid_df, input_cols, input_num_cols, drop_duplicates=False), DataWithTOkenType(config, test_df, input_cols, input_num_cols, drop_duplicates=False)


def split_and_pick(x, times):
    sep_token = '[SEP]'
    if sep_token not in x: return x
    x = x.split(sep_token)
    rst = []
    for _ in range(times):
        tmp = copy.copy(x)
        random.shuffle(tmp)
        rst.append(sep_token.join(tmp))
    return rst


def augment_data(df: pd.DataFrame, classes_to_aug: int, times):
    is_class = df.labels == classes_to_aug
    to_aug_df = df[is_class].copy()
    remaining_df = df[~is_class].copy()
    to_aug_df['text_sentence_no_mask'] = to_aug_df['text_sentence_no_mask'].apply(lambda x: split_and_pick(x, times))
    to_aug_df = to_aug_df.explode('text_sentence_no_mask', ignore_index=True)

    return pd.concat([to_aug_df, remaining_df], ignore_index=True)


class DataWithTOkenType(Dataset):
    def __init__(self, config, input_df, input_cols: List[str], input_num_cols: List[str], drop_duplicates=None):
        super().__init__()

        self.config = config
        self.input_num_cols = input_num_cols
        self.input_cols = input_cols

        # Only keeping the columns that are going to be used
        cols_to_keep = input_cols + input_num_cols
        if getattr(self.config, 'use_labels', False) is True:
            cols_to_keep.append('labels')
        self.input_df = input_df[cols_to_keep].copy()
        # print(f'cols to keep: {cols_to_keep}')

        # Drop duplicates
        if drop_duplicates is None:
            drop_duplicates = getattr(self.config, 'drop_duplicates', False)
        if drop_duplicates is True:
            print('Dropping duplicates!')
            print(len(self.input_df))
            self.input_df = self.input_df.drop_duplicates(ignore_index=True)
            print(len(self.input_df))

        self.input_df = tokenize_flat_tables(config, self.input_df, input_cols).reset_index(drop=True)

        # analyze how many are longer than 512
        # count = 0
        # for i in tqdm(range(len(self))):
        #     if len(self.__getitem__(i)['input_ids']) >= 512:
        #         count += 1
        # print(f'Number of samples longer than 512: {count}, out of {len(self)}')
    
    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, idx):
        inputs = [self.input_df.loc[idx, f"tokenized_{col}"][1:-1] for col in self.input_cols]

        input_type_ids = [0]
        input_ids = [102]
        for type_idx, l in enumerate(inputs):
            input_type_ids += [type_idx] * (len(l) + 1)
            input_ids += l + [103]
        input_type_ids = torch.LongTensor(input_type_ids[0:512])
        input_ids = torch.LongTensor(input_ids[0:512])
        if len(input_ids) == 512:
            input_ids[-1] = 103
        
        item = {col: int(self.input_df.loc[idx, col]) for col in self.input_num_cols}
        if getattr(self.config, 'use_labels', False) is True:
            item['labels'] = self.input_df.iloc[idx]['labels'].item()
        item['input_type_ids'] = input_type_ids
        item['input_ids'] = input_ids

        return item
