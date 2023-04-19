import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import numpy as np

from CTC.models import SciBertWithAdditionalFeatures
from common_utils.common_ML_utils import pooler, Config


@dataclass
class LinkingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    relevance_score: Optional[torch.FloatTensor] = None
    cell_emb: Optional[torch.FloatTensor] = None
    ent_emb: Optional[torch.FloatTensor] = None


class CrossEocoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = SciBertWithAdditionalFeatures(config)

        self.transform = None

        if config.loss == 'MSE':
            self.loss_func = nn.MSELoss()
            self.transform = nn.Linear(768, 1, bias=True)
        elif config.loss == 'BCE':
            self.loss_func = nn.BCEWithLogitsLoss()
            self.transform = nn.Linear(768, 1, bias=True)
        else:
            raise ValueError(f"Unknown loss function {config.loss}")
        
        self._init_weights(self.transform)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def pooler(self, input, mtd, attn_mask: Optional[torch.LongTensor] = None):
        last_hidden_state = input.last_hidden_state  # (BS, seq_len, hidden_size)
        if mtd == 'max':
            raise ValueError("Shouldn'd be used!")
        elif mtd == 'avg':
            mask = torch.unsqueeze(attn_mask, dim=-1).expand(last_hidden_state.size()).float()
            embed = torch.sum(last_hidden_state * mask, dim=1) / torch.sum(mask, dim=1)
            return torch.nan_to_num(embed)
        elif mtd == 'cls':
            return last_hidden_state[:, 0]
        else:
            raise ValueError(f"{mtd}")
    
    def forward(self, 
        input_ids: torch.LongTensor,  # (BS, seq_length)
        attention_mask: torch.LongTensor,
        input_type_ids: Optional[torch.LongTensor] = None,
        labels = None
    ):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        input_ids = input_ids.to(device) if input_ids is not None else input_ids
        attention_mask = attention_mask.to(device) if attention_mask is not None else attention_mask
        input_type_ids = input_type_ids.to(device) if input_type_ids is not None else input_type_ids
        labels = labels.to(device).to(torch.float32) if labels is not None else labels
        
        last_hidden_state = self.encoder(input_ids, attention_mask, input_type_ids)  # BaseModelOutput. (BS, seq_length, 768)
        embedding = self.pooler(last_hidden_state, self.config.pool_mtd, attention_mask)  # (BS, 768)

        # logits basically are distances
        if self.config.loss in ('MSE', 'BCE'):
            logits = self.transform(embedding)

        loss = None
        if labels is not None:
            if self.config.loss in ('MSE', 'BCE'):
                loss = self.loss_func(logits.squeeze(), labels.squeeze())
        
        if self.config.loss == 'BCE':
            logits = torch.sigmoid(logits.detach().squeeze())
        
        return LinkingOutput(loss=loss, relevance_score=-logits.squeeze())


class BiEncoderTriplet(nn.Module):
    """
    Bi-encoder with triplet objective function.
    """
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder = PairEmbedding(Config(cell_embedding=config.cell_embedding, ent_embedding=config.pos_ent_embedding if getattr(config, 'ent_embedding') is None else config.ent_embedding))
        self.loss_func = nn.TripletMarginLoss(margin=1, p=2)
    
    def forward(self, cell: Optional[Dict] = None, pos_ent: Optional[Dict] = None, neg_ent: Optional[Dict] = None, output_emb=False,
                cell_emb: Optional[torch.FloatTensor]=None, pos_ent_emb: Optional[torch.FloatTensor]=None, neg_ent_emb: Optional[torch.FloatTensor]=None,
                ent: Optional[Dict] = None, ent_emb: Optional[torch.FloatTensor]=None, **kwargs):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if cell is not None:
            cell = {k: v.to(device) for k, v in cell.items()}
        if pos_ent is not None:
            pos_ent = {k: v.to(device) for k, v in pos_ent.items()}
        if neg_ent is not None:
            neg_ent = {k: v.to(device) for k, v in neg_ent.items()}
        if ent is not None:
            ent = {k: v.to(device) for k, v in ent.items()}

        if ent_emb is not None and isinstance(ent_emb, (np.ndarray, np.generic)):
            ent_emb = torch.from_numpy(ent_emb).to(device)
            if ent_emb.dim() == 1:
                ent_emb = ent_emb.unsqueeze(0)
        if cell_emb is not None and isinstance(cell_emb, (np.ndarray, np.generic)):
            cell_emb = torch.from_numpy(cell_emb).to(device)
            if cell_emb.dim() == 1:
                cell_emb = cell_emb.unsqueeze(0)

        cell_emb = self.encoder(cell=cell)[0] if cell_emb is None else cell_emb
        pos_ent_emb = self.encoder(ent=pos_ent)[1] if pos_ent_emb is None else pos_ent_emb
        neg_ent_emb = self.encoder(ent=neg_ent)[1] if neg_ent_emb is None else neg_ent_emb
        ent_emb = self.encoder(ent=ent)[1] if ent_emb is None else ent_emb

        loss = relevance_score = None
        if cell_emb is not None and pos_ent_emb is not None and neg_ent_emb is not None:  # training
            loss = self.loss_func(cell_emb, pos_ent_emb, neg_ent_emb)
        
        if cell_emb is not None and ent_emb is not None:  # inference
            relevance_score = - torch.cdist(cell_emb, ent_emb, p=2).squeeze()
        
        if output_emb is True:
            return LinkingOutput(loss=loss, relevance_score=relevance_score, cell_emb=cell_emb, ent_emb=ent_emb)
        else:
            return LinkingOutput(loss=loss, relevance_score=relevance_score)


class BiEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.pair_embedding = PairEmbedding(config)

        self.transform = None

        if config.loss == 'MSE':
            self.loss_func = nn.MSELoss()
            self.transform = nn.Linear(768 * 3, 1, bias=True)
        elif config.loss == 'BCE':
            self.loss_func = nn.BCEWithLogitsLoss()
            self.transform = nn.Linear(768 * 3, 1, bias=True)
        elif config.loss == 'Contrastive':
            self.loss_func = nn.HingeEmbeddingLoss(margin=1)
            self.transform = self._euclean_squred
        elif config.loss == 'Triplet':
            pass
        else:
            raise ValueError(f"Unknown loss function {config.loss}")
        
        self._init_weights(self.transform)
    
    def _euclean_squred(self, a, b):
        return (a - b).pow(2).sum(1)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, cell: Optional[Dict] = None, ent: Optional[Dict] = None, labels = None, output_emb=False,
                cell_emb: Optional[torch.FloatTensor]=None, ent_emb: Optional[torch.FloatTensor]=None):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if cell is not None:
            cell = {k: v.to(device) for k, v in cell.items()}
        if ent is not None:
            ent = {k: v.to(device) for k, v in ent.items()}
        if labels is not None:
            labels = labels.to(device).to(torch.float32)
        
        if cell_emb is not None and isinstance(cell_emb, (np.ndarray, np.generic)):
            cell_emb = torch.from_numpy(cell_emb).to(device)
            if cell_emb.dim() == 1:
                cell_emb = cell_emb.unsqueeze(0)
        if ent_emb is not None and isinstance(ent_emb, (np.ndarray, np.generic)):
            ent_emb = torch.from_numpy(ent_emb).to(device)
            if ent_emb.dim() == 1:
                ent_emb = ent_emb.unsqueeze(0)
        
        if cell_emb is None and ent_emb is None:
            cell_emb, ent_emb = self.pair_embedding(cell, ent)
        
        # logits basically are distances
        loss = relevance_score = None
        if cell_emb is not None and ent_emb is not None:
            if self.config.loss == 'Contrastive':
                logits = self.transform(cell_emb, ent_emb)
            elif self.config.loss in ('MSE', 'BCE'):
                logits = self.transform(torch.cat((cell_emb, ent_emb, torch.abs(cell_emb - ent_emb)), 1))

            if labels is not None:
                if self.config.loss == 'Contrastive':
                    labels[labels == 1] = -1  
                    labels[labels == 0] = 1  # matches
                    loss = self.loss_func(logits.squeeze(), labels.squeeze())
                elif self.config.loss in ('BCE', 'MSE'):
                    loss = self.loss_func(logits.squeeze(), labels.squeeze())
                
            if self.config.loss == 'BCE':
                logits = torch.sigmoid(logits.detach().squeeze())
            
            relevance_score = -logits.squeeze()
        
        if output_emb is True:
            return LinkingOutput(loss=loss, relevance_score=relevance_score, cell_emb=cell_emb, ent_emb=ent_emb)
        else:
            return LinkingOutput(loss=loss, relevance_score=relevance_score)


class PairEmbeddingWithCosHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.pair_embedding = PairEmbedding(config)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    @classmethod
    def __repr__(cls):
        return 'PairEmbeddingWithCosHead'
    
    @classmethod
    def __str__(cls):
        return repr(cls)
    
    def forward(self, cell: Optional[Dict] = None, ent: Optional[Dict] = None, labels = None):

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if cell is not None:
            cell = {k: v.to(device) for k, v in cell.items()}
        if ent is not None:
            ent = {k: v.to(device) for k, v in ent.items()}
        if labels is not None:
            labels = labels.to(device).to(torch.float32)

        cell_emb, ent_emb = self.pair_embedding(cell, ent)

        sim = self.cos(cell_emb, ent_emb)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(sim.squeeze(), labels.squeeze())

        return LinkingOutput(loss=loss, relevance_score=sim.squeeze())


class PairEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        if config.cell_embedding is not None:
            self.cell_encoder = SciBertWithAdditionalFeatures(config.cell_embedding)
            if config.cell_embedding.pool_mtd == 'cls':
                self.cell_pooler = nn.Sequential(nn.Linear(768, 768), nn.Tanh())
            
        if config.ent_embedding is not None:
            self.ent_encoder = SciBertWithAdditionalFeatures(config.ent_embedding)
            if config.ent_embedding is not None and config.ent_embedding.pool_mtd == 'cls':
                self.ent_pooler = nn.Sequential(nn.Linear(768, 768), nn.Tanh())

    def forward(self, cell: Optional[Dict] = None, ent: Optional[Dict] = None):
        """This can be used to embed cells and/or entity text."""
        
        cell_embedding = ent_embedding = None

        if cell is not None:
            cell_embedding = self.cell_encoder(**cell)
            cell_embedding = pooler(cell_embedding, self.config.cell_embedding.pool_mtd, cell['attention_mask'])
        if ent is not None:
            ent_embedding = self.ent_encoder(**ent)
            ent_embedding = pooler(ent_embedding, self.config.ent_embedding.pool_mtd, ent['attention_mask'])

        return cell_embedding, ent_embedding  # (BS, 768)
    
        
