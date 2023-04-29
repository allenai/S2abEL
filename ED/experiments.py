import sys
sys.path.append(sys.path[0] + '/..')

import pandas as pd
import numpy as np
from sklearn import metrics
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import json
import argparse


from common_utils.common_data_processing_utils import *
from common_utils.common_exp_utils import Experiment, EvalExperiment
from common_utils.common_ML_utils import *
from ED.data_loaders import load_bi_encoder_input, padding_bi_encoder_data
from ED.models import *
from ED.trainers import *
from ED.candidate_gen import compute_bi_enc_ent_candidates, mix_candidate_set
from ED.utils import replace_EL_non_train_folds, gen_test_EL_on_GT_CTC, compute_EL_perf_e2e, compute_micro_EL_perf_e2e




class ElRerank(Experiment):
    """
    For El reranking.
    :param mode: 
        'inKB' ==> inKB only cells, to compare against TURL.
        'all' ==> both inKB and outKB cells.
    """

    mode: str

    def __init__(self, args, seed, test_fold, train_file_path, test_file_path, project:str=None, use_wandb=False, wandb_dir:str=None, model_path=None, val_fold='img_class',
                 save_dir=None, lr=2e-5, num_epochs=2, mode='inKB', ctc_pred_file_path=None, BS=32, eval_BS=512):

        assert mode in ('inKB', 'all')
        
        train_df = pd.read_pickle(train_file_path)
        
        self.train_df = train_df[train_df.fold != val_fold].reset_index(drop=True).copy()
        self.valid_df = train_df[train_df.fold == val_fold].reset_index(drop=True).copy()

        self.test_df = pd.read_pickle(test_file_path)
        assert self.test_df.fold.nunique() == 1

        if mode == 'inKB':
            self.train_df = self.train_df[self.train_df.pwc_url != ''].reset_index(drop=True).copy()
            self.valid_df = self.valid_df[self.valid_df.pwc_rl != ''].reset_index(drop=True).copy()
            self.test_df = self.test_df[self.test_df.pwc_url != ''].reset_index(drop=True).copy()
            assert sum(self.train_df.pwc_url == '0') == 0
            assert sum(self.valid_df.pwc_url == '0') == 0
            assert sum(self.test_df.pwc_url == '0') == 0
        elif mode == 'all':
            pass
        
        super().__init__(seed=seed, test_fold=test_fold, model_path=model_path)

        g = set_seed(seed)

        ## effective BS = BS * grad_accum_step = 32
        config = Config(
            seed=seed,
            BS=BS,
            num_epochs=num_epochs,
            test_fold=test_fold,
            lr=lr,
            grad_accum_step=1,
            pool_mtd='avg',
            loss='BCE',
            pretrained = "allenai/scibert_scivocab_uncased",
            input_cols = ['region_type', 'row_id', 'reverse_row_id', 'col_id', 'reverse_col_id', 'cell_content', 'candidate_ent_names', 'candidate_ent_full_names', 'candidate_ent_descriptions', 'text_sentence_no_mask', 'row_context', 'col_context'],
            input_num_cols= [],
            drop_duplicates=True,
            eval_steps=300,
            use_labels=True,
            save_dir=save_dir,
            project=project,
            eval_BS=eval_BS,
            name=f"{test_fold}_{seed}"
        )

        if save_dir is not None:
            save_dir = Path(save_dir)
            if not (save_dir / project).exists():
                os.mkdir(save_dir / project)
        
        if use_wandb:
            if wandb_dir is not None:
                os.environ["WANDB_DIR"] = wandb_dir
            if project is None:
                raise ValueError("Please specify the project name for weight&bias")
            wandb.init(
                project = project,
                config = config,
                name = f"{test_fold}_{seed}"
            )      

        self.args = args
        self.g = g
        self.config = config
        self.project = project
        self.ctc_pred_file_path = ctc_pred_file_path

    def train(self):

        model = CrossEocoder(self.config)

        train_ds = load_single_DataWithTokenType(self.config, self.train_df, drop_duplicates=True)
        valid_ds = load_single_DataWithTokenType(self.config, self.valid_df, drop_duplicates=False)

        print(f"Training length {len(train_ds)}")
        print(f"Validation length {len(valid_ds)}")

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=self.config.BS, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=self.g)
        eval_dl = DataLoader(valid_ds, batch_size=self.config.eval_BS, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=self.g)

        EL_train_loop(self.args, model, train_dl, eval_dl, self.config.eval_steps)
    
    def compute_preds_on_GT_CTC(self, save_path=None):
        if save_path is None:
            model_path = os.path.join(self.config.model_dir, self.config.project, self.config.name)
            model = torch.load(model_path)
            model.eval()
            preds_on_GT_CTC = gen_test_EL_on_GT_CTC(self.test_df, model, [self.test_fold], top_n=10, BS=512, mode='cross')
            return preds_on_GT_CTC
        else:
            if not os.path.exists(save_path):
                preds_on_GT_CTC = self.compute_preds_on_GT_CTC()
                preds_on_GT_CTC.to_pickle(save_path)
            else:
                preds_on_GT_CTC = pd.read_pickle(save_path)
            return preds_on_GT_CTC
        

    def test(self, threshold, inKB_acc_at_topks):
        preds_on_GT_CTC = self.compute_preds_on_GT_CTC()

        if getattr(self, 'ctc_pred_file_path', None):
            ctc_pred = pd.read_pickle(self.ctc_pred_file_path)
            perf = compute_EL_perf_e2e(ctc_pred, preds_on_GT_CTC, threshold=threshold, acc_at_topks=inKB_acc_at_topks)
            return perf
        else:
            raise NotImplementedError()


class EDExp(ElRerank):
    def __init__(self, seed, test_fold, train_file_path, test_file_path, name, ctc_pred_file_path, mode, BS=32, eval_BS=512, lr=2e-5, epoch=2, val_fold='img_class', save_dir=None, eval_steps=300):
        train_df = pd.read_pickle(train_file_path)
        
        self.train_df = train_df[train_df.fold != val_fold].reset_index(drop=True).copy()
        self.valid_df = train_df[train_df.fold == val_fold].reset_index(drop=True).copy()

        self.test_df = pd.read_pickle(test_file_path)
        assert self.test_df.fold.nunique() == 1

        if mode == 'inKB':
            self.train_df = self.train_df[self.train_df.pwc_url != ''].reset_index(drop=True).copy()
            self.valid_df = self.valid_df[self.valid_df.pwc_rl != ''].reset_index(drop=True).copy()
            self.test_df = self.test_df[self.test_df.pwc_url != ''].reset_index(drop=True).copy()
            assert sum(self.train_df.pwc_url == '0') == 0
            assert sum(self.valid_df.pwc_url == '0') == 0
            assert sum(self.test_df.pwc_url == '0') == 0
        elif mode == 'all':
            pass

        config = Config(
            seed=seed,
            BS=BS,
            epoch=epoch,
            test_fold=test_fold,
            val_fold=val_fold,
            lr=lr,
            grad_accum_step=1,
            pool_mtd='avg',
            loss='BCE',
            pretrained = "allenai/scibert_scivocab_uncased",
            input_cols = ['region_type', 'row_pos', 'reverse_row_pos', 'col_pos', 'reverse_col_pos', 'cell_content', 'candidate_ent_names', 'candidate_ent_full_names', 'candidate_ent_descriptions', 'context_sentences', 'row_context', 'col_context'],
            input_num_cols= [],
            drop_duplicates=True,
            eval_BS=eval_BS,
            use_labels=True,
            save_dir=save_dir,
            ctc_pred_file_path=ctc_pred_file_path,
            name=name,
            eval_steps=eval_steps
        )

        self.g = set_seed(seed)
        self.config = config
    
    def train(self):

        model = CrossEocoder(self.config)

        train_ds = load_single_DataWithTokenType(self.config, self.train_df, drop_duplicates=True)
        valid_ds = load_single_DataWithTokenType(self.config, self.valid_df, drop_duplicates=False)

        print(f"Training length {len(train_ds)}")
        print(f"Validation length {len(valid_ds)}")

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=self.config.BS, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=self.g)
        eval_dl = DataLoader(valid_ds, batch_size=self.config.eval_BS, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=self.g)

        EL_train_loop_notebook(self.config, model, train_dl, eval_dl)

    def compute_preds_on_GT_CTC(self, save_path=None):
        model_path = os.path.join(self.config.save_dir, self.config.name)
        model = torch.load(model_path)
        model.eval()
        preds_on_GT_CTC = gen_test_EL_on_GT_CTC(self.test_df, model, [self.config.test_fold], top_n=10, BS=self.config.eval_BS, mode='cross')
        if save_path is not None:
            preds_on_GT_CTC.to_pickle(save_path)
        return preds_on_GT_CTC

    def test(self, threshold, inKB_acc_at_topks, save_path=None):
        preds_on_GT_CTC = self.compute_preds_on_GT_CTC(save_path=save_path)
        ctc_pred = pd.read_pickle(self.config.ctc_pred_file_path)
        perf = compute_EL_perf_e2e(ctc_pred, preds_on_GT_CTC, threshold=threshold, acc_at_topks=inKB_acc_at_topks)
        return perf
