import numpy as np
import random
import os
import torch
from datasets import load_metric
from collections import namedtuple
from sklearn.metrics import pairwise_distances
from typing import Dict, List, Tuple, Union, Optional
from enum import Enum


class Config(object):
    def __init__(self, **kwargs):
        self.dict = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.dict[k] = v
    
    def __repr__(self) -> str:
        return repr(self.dict)


def set_seed(seed, all_gpus=True):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    if all_gpus:
        torch.cuda.manual_seed_all(seed)
    return g


def seed_worker(worker_id):
    worker_seed = torch.initial_seed()
    np.random.seed(worker_seed)
    random.seed(worker_seed)


add_token_mapping = {
    '[FIELD]': '[unused1]',
    '[REF]': '[unused2]',
    # '[BIB_REF]': '[unused3]',
    # '[OTHER_REF]': '[unused4]',
    '[TABLE_TITLE]': '[unused5]',
    '[BIB_ITEM]': '[unused6]',
    '[SEC_OR_FIG_TITLE]': '[unused7]',
    '[ROW_BORDER]': '[unused8]',
    '[COL_BORDER]': '[unused9]',
    '[BORDER]': '[unused10]',
    '[EMPTY]': '[unused11]',
    '[FLOAT]': '[unused12]',
    '[INT]': '[unused13]'
}

additional_special_tokens = { 'additional_special_tokens': list(add_token_mapping.values())}


Metric = namedtuple('Metric', ['name', 'instance', 'keywargs', 'prefix'])

metrics = [
    Metric(name='accuracy', instance=load_metric('accuracy'), keywargs={}, prefix=''),
    Metric(name='f1', instance=load_metric('f1'), keywargs={'average': None}, prefix='')
    # Metric(name='recall', instance=load_metric('recall'), keywargs={'average': None}, prefix='per_class_'),
    # Metric(name='precision', instance=load_metric('precision'), keywargs={'average': None}, prefix='per_class_')
]

val_metrics = [
    Metric(name='f1', instance=load_metric('f1'), keywargs={'average': 'micro'}, prefix='val_micro_'),
    Metric(name='f1', instance=load_metric('f1'), keywargs={'average': None}, prefix='val_')
]


def sim_func(model, cell_embeds, ent_embeds):
    """
    Computes the  similarity cell_embeds[i], ent_embeds[j]) for all i and j.
    :return: Matrix with res[i][j]  = model(cell_embeds[i], ent_embeds[j]).relevance_score
    """
    if len(cell_embeds.shape) == 1:
        cell_embeds = cell_embeds.unsqueeze(0)
    if len(ent_embeds.shape) == 1:
        ent_embeds = ent_embeds.unsqueeze(0)

    model.eval()
    metric = lambda cell_embed, ent_embed: model(cell_emb=cell_embed, ent_emb=ent_embed).relevance_score.squeeze()
    
    return torch.from_numpy(pairwise_distances(cell_embeds, ent_embeds, metric=metric))


def pooler(input, mtd, attn_mask: Optional[torch.LongTensor] = None):
    last_hidden_state = input.last_hidden_state  # (BS, seq_len, hidden_size)
    if mtd == 'max':
        raise ValueError("Shouldn'd be used!")
    elif mtd == 'avg':
        mask = torch.unsqueeze(attn_mask, dim=-1).expand(last_hidden_state.size()).float()
        embed = torch.sum(last_hidden_state * mask, dim=1) / torch.sum(mask, dim=1)
        return torch.nan_to_num(embed)
    elif mtd == 'cls':
        return last_hidden_state[:, 0]
        # if source == 'ent':
        #     return self.ent_pooler(last_hidden_state[:, 0])
        # elif source == 'cell':
        #     return self.cell_pooler(last_hidden_state[:, 0])
        # else:
        #     raise ValueError("Invalid source type")
    else:
        raise ValueError(f"{mtd}")


class LabelsExt(Enum):
    OTHER=0
    DATASET=1
    METHOD=2
    METRIC=3
    DATASET_AND_METRIC=4


class Labels(Enum):
    OTHER=0
    DATASET=1
    METHOD=2
    METRIC=3


m = {
    'Method': "method",
    'Dataset': "dataset",
    'Metric': "metric",
    'Other': "other",
    'DatasetAndMetric': "dataset&metric"
}

m_reverse = {v:k for k, v in m.items()}