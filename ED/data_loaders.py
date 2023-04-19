from torch.utils.data import Dataset
from typing import List
import torch


from CTC.TE_loaders import tokenize_flat_tables
from common_utils.common_data_processing_utils import *


def load_bi_encoder_input(config, *dfs):
    """
    Assumet the first df is the training data. Do not drop duplicates on others
    """
    if len(dfs) == 1:
        return DatasetWithType(dfs[0], config)
    else:
        return [DatasetWithType(df, config) if idx == 0 else DatasetWithType(df, config, drop_duplicates=False) for idx, df in enumerate(dfs)]


def load_triplet_encoder_input(config, *dfs):
    if len(dfs) == 1:
        return TripletInput(dfs[0], config)
    else:
        return [TripletInput(df, config) if idx == 0 else TripletInput(df, config, drop_duplicates=False) for idx, df in enumerate(dfs)]


def padding_bi_encoder_data(features):
    first = features[0]
    cell_features = ent_features = pos_ent_features = neg_ent_features = label_features = None

    if first.get('cell', None):
        cell_features = [f['cell'] for f in features]
        cell_features = padding_by_batch(cell_features)
    if first.get('ent', None):
        ent_features = [f['ent'] for f in features]
        ent_features = padding_by_batch(ent_features)
    if first.get('pos_ent', None):
        pos_ent_features = [f['pos_ent'] for f in features]
        pos_ent_features = padding_by_batch(pos_ent_features)
    if first.get('neg_ent', None):
        neg_ent_features = [f['neg_ent'] for f in features]
        neg_ent_features = padding_by_batch(neg_ent_features)
    if first.get('labels', None):
        label_features = torch.tensor([f['labels'] for f in features])

    batch = {
        'cell': cell_features,
        'ent': ent_features,
        'labels': label_features,
        'pos_ent': pos_ent_features,
        'neg_ent': neg_ent_features
    }

    return batch


def process_input(inputs):
    input_type_ids = [0]
    input_ids = [102]
    for type_idx, l in enumerate(inputs):
        input_type_ids += [type_idx] * (len(l) + 1)
        input_ids += l + [103]
    input_type_ids = torch.LongTensor(input_type_ids[0:512])
    input_ids = torch.LongTensor(input_ids[0:512])
    if len(input_ids) == 512:
        input_ids[-1] = 103
    return input_ids, input_type_ids


class TripletInput(Dataset):
    def __init__(self, input_df, config, drop_duplicates=None):
        super().__init__()
        self.config = config
        
        # Only keeping the columns that are going to be used
        cols_to_keep = (config.cell_embedding.input_cols if getattr(config, 'cell_embedding') is not None else []) + (config.ent_embedding.input_cols if getattr(config, 'ent_embedding') is not None else [])\
            + (config.pos_ent_embedding.input_cols if getattr(config, 'pos_ent_embedding') is not None else []) + (config.neg_ent_embedding.input_cols if getattr(config, 'neg_ent_embedding') is not None else [])
        
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
        
        if getattr(config, 'cell_embedding', None):
            self.input_df = tokenize_flat_tables(config.cell_embedding, self.input_df, config.cell_embedding.input_cols).reset_index(drop=True)
        if getattr(config, 'pos_ent_embedding', None):
            self.input_df = tokenize_flat_tables(config.pos_ent_embedding, self.input_df, config.pos_ent_embedding.input_cols).reset_index(drop=True)
        if getattr(config, 'neg_ent_embedding', None):
            self.input_df = tokenize_flat_tables(config.neg_ent_embedding, self.input_df, config.neg_ent_embedding.input_cols).reset_index(drop=True)
        if getattr(config, 'ent_embedding', None):
            self.input_df = tokenize_flat_tables(config.ent_embedding, self.input_df, config.ent_embedding.input_cols).reset_index(drop=True)

    def __len__(self):
        return len(self.input_df)
    
    def __getitem__(self, idx):
        cell = ent = pos_ent = neg_ent = None

        if self.config.cell_embedding is not None:
            cell_inputs = [self.input_df.loc[idx, f"tokenized_{col}"][1:-1] for col in self.config.cell_embedding.input_cols]
            input_ids, input_type_ids = process_input(cell_inputs)
            cell = {
                "input_ids": input_ids,
                "input_type_ids": input_type_ids,
            }

        if self.config.ent_embedding is not None:
            ent_inputs = [self.input_df.loc[idx, f"tokenized_{col}"][1:-1] for col in self.config.ent_embedding.input_cols]
            input_ids, input_type_ids = process_input(ent_inputs)
            ent = {
                "input_ids": input_ids,
                "input_type_ids": input_type_ids,
            }

        if self.config.pos_ent_embedding is not None:
            ent_inputs = [self.input_df.loc[idx, f"tokenized_{col}"][1:-1] for col in self.config.pos_ent_embedding.input_cols]
            input_ids, input_type_ids = process_input(ent_inputs)
            pos_ent = {
                "input_ids": input_ids,
                "input_type_ids": input_type_ids,
            }

        if self.config.neg_ent_embedding is not None:
            ent_inputs = [self.input_df.loc[idx, f"tokenized_{col}"][1:-1] for col in self.config.neg_ent_embedding.input_cols]
            input_ids, input_type_ids = process_input(ent_inputs)
            neg_ent = {
                "input_ids": input_ids,
                "input_type_ids": input_type_ids,
            }

        return {"cell": cell, "ent": ent, "pos_ent": pos_ent, "neg_ent": neg_ent}


class DatasetWithType(Dataset):
    def __init__(self, input_df, config, drop_duplicates=None):
        super().__init__()

        self.config = config
        
        # Only keeping the columns that are going to be used
        cols_to_keep = (config.cell_embedding.input_cols if getattr(config, 'cell_embedding') is not None else []) + (config.ent_embedding.input_cols if getattr(config, 'ent_embedding') is not None else [])
        if getattr(self.config, 'use_labels', False) is True:
            cols_to_keep.append('labels')
        self.input_df = input_df[cols_to_keep].copy()
        print(f'cols to keep: {cols_to_keep}')

        # Drop duplicates
        if drop_duplicates is None:
            drop_duplicates = getattr(self.config, 'drop_duplicates', False)
        if drop_duplicates is True:
            print('Dropping duplicates!')
            print(len(self.input_df))
            self.input_df = self.input_df.drop_duplicates(ignore_index=True)
            print(len(self.input_df))

        if config.cell_embedding is not None:
            self.input_df = tokenize_flat_tables(config.cell_embedding, self.input_df, config.cell_embedding.input_cols).reset_index(drop=True)
        if config.ent_embedding is not None:
            self.input_df = tokenize_flat_tables(config.ent_embedding, self.input_df, config.ent_embedding.input_cols).reset_index(drop=True)

    def __len__(self):
        return len(self.input_df)
    
    def __getitem__(self, idx):
        cell = ent = None

        if self.config.cell_embedding is not None:
            cell_inputs = [self.input_df.loc[idx, f"tokenized_{col}"][1:-1] for col in self.config.cell_embedding.input_cols]
            input_ids, input_type_ids = process_input(cell_inputs)
            cell = {
                "input_ids": input_ids,
                "input_type_ids": input_type_ids,
            }
        if self.config.ent_embedding is not None:
            ent_inputs = [self.input_df.loc[idx, f"tokenized_{col}"][1:-1] for col in self.config.ent_embedding.input_cols]
            input_ids, input_type_ids = process_input(ent_inputs)
            ent = {
                "input_ids": input_ids,
                "input_type_ids": input_type_ids,
            }

        labels = None
        if getattr(self.config, 'use_labels', False) is True:
            labels = self.input_df.iloc[idx]['labels'].item()

        return {"cell": cell, "ent": ent, "labels": labels}