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

        train_ds = load_single_DataWithTOkenType(self.config, self.train_df, drop_duplicates=True)
        valid_ds = load_single_DataWithTOkenType(self.config, self.valid_df, drop_duplicates=False)

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
            input_cols = ['region_type', 'row_id', 'reverse_row_id', 'col_id', 'reverse_col_id', 'cell_content', 'candidate_ent_names', 'candidate_ent_full_names', 'candidate_ent_descriptions', 'text_sentence_no_mask', 'row_context', 'col_context'],
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

        train_ds = load_single_DataWithTOkenType(self.config, self.train_df, drop_duplicates=True)
        valid_ds = load_single_DataWithTOkenType(self.config, self.valid_df, drop_duplicates=False)

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

    def test(self, threshold, inKB_acc_at_topks):
        preds_on_GT_CTC = self.compute_preds_on_GT_CTC()
        ctc_pred = pd.read_pickle(self.config.ctc_pred_file_path)
        perf = compute_EL_perf_e2e(ctc_pred, preds_on_GT_CTC, threshold=threshold, acc_at_topks=inKB_acc_at_topks)
        return perf







# if __name__ == '__main__':
#     parser=argparse.ArgumentParser(
#         description=__doc__,
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     parser.add_argument('--test_fold', help='valid fold to use', type=str, required=True)
#     parser.add_argument('--data_dir', help='dir that store data', type=str, require=True)
#     parser.add_argument('--model_dir', help='the dir to save the best model in training', type=str, default=None)
#     parser.add_argument('--project', help='weight&bias project name', type=str, default=None)
#     parser.add_argument('--output_dir', help='output dir', type=str, default=None)
#     parser.add_argument('--wandb_dir', help='weight&bias dir', type=str, default=None)
#     parser.add_argument('--CER_output_dir', help='the dir that stores CER outputs named as <fold>_<seed>', type=str, default=None)
#     parser.add_argument('--DR_output_dir', help='output dir for DR', type=str, default=None)
#     parser.add_argument('--ASR_output_dir', help='output dir for ASR', type=str, default=None)
#     parser.add_argument('--use_wandb', help='whether to use weight&bias', action='store_true')
#     parser.add_argument('--seed', help='random seed', type=int, default=42)
#     parser.add_argument('--threshold', help='ED threshold', type=float, default=0.5)
#     parser.add_argument('--top_n_to_rank', help='the number of candidates to rank in ED', type=int, default=50)
#     parser.add_argument('--BS', help='batch size', type=int, default=32)
#     parser.add_argument('--train_DR', help='train a Direct Retrieval model', action='store_true')
#     parser.add_argument('--compute_DR_outputs_all', help='compute DR outputs together', action='store_true')
#     parser.add_argument('--compute_DR_output_one', help='compute one DR output', action='store_true')
#     parser.add_argument('--mix_DR_ASR_all', help='interleave DR and ASR outputs', action='store_true')
#     parser.add_argument('--mix_DR_ASR_one', help='interleave DR and ASR outputs', action='store_true')
#     parser.add_argument('--eval_EL_e2e_per_threhold', help='evaluate e2e EL performance by different threholds', action='store_true')
#     parser.add_argument('--eval_EL_e2e_per_fold', help='evaluate e2e EL performance per fold', action='store_true')
#     parser.add_argument('--gen_e2e_ED_data', help='generate test data for e2e EL', action='store_true')
#     args = parser.parse_args()
    
#     ########################################################
#     ##### Train a DR model for one topic #####
#     if args.train_DR:
#         args.input_file = os.path.join(args.data_dir, 'DR_train.pkl')
#         exp = ElDirectCanSearchTripletTrain(
#             args=args,
#             seed=args.seed, 
#             test_fold=args.test_fold, 
#             project=args.project,
#             use_wandb=args.use_wandb,
#             wandb_dir=args.wandb_dir, 
#             input_file_path=args.input_file, 
#             save_dir=args.model_dir)
#         exp.train()
#     ########################################################
#     ##### generating all DR outputs #####
#     if args.compute_DR_outputs_all:
#         ent_file_path = os.path.join(args.data_dir, 'PwC_entities.pkl') 
#         mode = 'Triplet'
#         model_dir = os.path.join(args.model_dir, args.proj) if args.proj is not None else args.model_dir
#         top_k = 200
#         for seed in (42, ):
#             for fold in tqdm(('mt', 'img_gen', 'misc', 'speech_rec', 'qa', 'nli', 'text_class', 'object_det', 'sem_seg', 'pose_estim')):
#                 model_path=f'{model_dir}/{fold}_{seed}'
#                 ent_emb_save_path = f'{args.output_dir}/{fold}_{seed}_emb'
#                 output_path = f'{args.output_dir}/{fold}_{seed}'
#                 if not os.path.exists(model_path):
#                     print(f'<<< model at {model_path} does not exist! SKIPPING')
#                     continue
#                 exp = EvalExperiment(
#                     seed=seed, 
#                     test_fold=fold, 
#                     input_file_path=f'{args.data_dir}/EL.pkl',
#                     model_path=model_path
#                 )
#                 if not os.path.exists(ent_emb_save_path):
#                     print(f'<<< generating {ent_emb_save_path}')
#                     exp.compute_ent_enmbeddings(ent_file_path, ent_emb_save_path=ent_emb_save_path, mode=mode)
#                 else:
#                     print(f'<<< Embedding file {ent_emb_save_path} exists')
#                 if not os.path.exists(output_path):
#                     print(f'<<< generating {output_path}')
#                     exp.generate_candidates(ent_file_path, ent_emb_save_path, save_path=output_path, mode=mode, BS=args.BS, top_k=top_k,
#                         single_fold_val_test=False, mix_methods_datasets=False)
    

#     if args.compute_DR_output_one:
#         ent_file_path = os.path.join(args.data_dir, 'PwC_entities.pkl') 
#         mode = 'Triplet'
#         model_dir = os.path.join(args.model_dir, args.proj) if args.proj is not None else args.model_dir
#         model_path=f'{model_dir}/{args.test_fold}_{args.seed}'
#         top_k = 200

#         if not os.path.exists(model_path):
#             print(f'<<< model at {model_path} does not exist! SKIPPING')
#             exit(1)
#         exp = EvalExperiment(
#             seed=args.seed, 
#             test_fold=args.test_fold, 
#             input_file_path=f'{args.data_dir}/EL.pkl',
#             model_path=model_path
#         )
#         if not os.path.exists(ent_emb_save_path):
#             print(f'<<< generating {ent_emb_save_path}')
#             exp.compute_ent_enmbeddings(ent_file_path, ent_emb_save_path=ent_emb_save_path, mode=mode)
#         else:
#             print(f'<<< Embedding file {ent_emb_save_path} exists')
#         if not os.path.exists(output_path):
#             print(f'<<< generating {output_path}')
#             exp.generate_candidates(ent_file_path, f'{args.output_dir}/{args.fold}_{args.seed}_emb', save_path=f'{args.output_dir}/{args.fold}_{args.seed}', mode=mode, BS=args.BS, top_k=top_k,
#                 single_fold_val_test=False, mix_methods_datasets=False)
            
#     # #######################################################
#     # # Mix candidates from DR and ASR

#     if args.mix_DR_ASR_all:
#         seed = args.seed
#         for fold in ('mt', 'img_gen', 'misc', 'speech_rec', 'qa', 'nli', 'text_class', 'object_det', 'sem_seg', 'pose_estim'):
#             ASR = pd.read_pickle(f'{args.ASR_output_dir}/{fold}_{seed}')
#             DR = pd.read_pickle(f'{args.DR_output_dir}/{fold}_{seed}')
#             print(f'<<< generating {fold}_{seed}')
#             output = mix_candidate_set(ASR, 'RPI_preds_candidates_top_', DR, 'ent_candidates', (100, ))
#             output.to_pickle(f'{args.output_dir}/{fold}_{seed}')
    
#     if args.mix_DR_ASR_one:
#         ASR = pd.read_pickle(f'{args.ASR_output_dir}/{args.fold}_{args.seed}')
#         DR = pd.read_pickle(f'{args.DR_output_dir}/{args.fold}_{args.seed}')
#         print(f'<<< generating {args.fold}_{args.seed}')
#         output = mix_candidate_set(ASR, 'RPI_preds_candidates_top_', DR, 'ent_candidates', (100, ))
#         output.to_pickle(f'{args.output_dir}/{args.fold}_{args.seed}')


#     # #######################################################
#     # ####### generate ED e2e data ##################################
#     if args.gen_e2e_ED_data:
#         test_data_dir = f'{args.data_dir}/test'
#         os.makedirs(test_data_dir, exist_ok=True)

#         EL = pd.read_pickle(f'{args.data_dir}/EL.pkl')

#         with open(f'{args.data_dir}/methods.json') as f:
#             methods = json.load(f)

#         with open(f'{args.data_dir}/datasets.json') as f:
#             datasets = json.load(f)

#         ents = methods + datasets
#         ent_map = {}
#         for m in ents:
#             title = '' if m['paper'] is None else m.get('paper').get('title', '')
#             name = '' if m['name'] is None else m['name']
#             full_name = '' if m['full_name'] is None else m.get('full_name', '')
#             description = '' if m['description'] is None else m.get('description', '')
#             ent_map[m['url']] = (name, full_name, description, title, m['url'])
        
#         for fold in ('mt', 'img_gen', 'misc', 'speech_rec', 'qa', 'nli', 'text_class', 'object_det', 'sem_seg', 'pose_estim'):
#             CER_output_path = f'{args.CER_ouput_dir}/{fold}_{seed}'
#             data_output_path = f'{test_data_dir}/{args.top_n_to_rank}_{fold}_{seed}_e2e'
#             if os.path.exists(data_output_path):
#                 print(f'<<< existing {data_output_path}')
#                 continue
#             print(f'<<< generating {data_output_path}')
#             df = pd.read_pickle(CER_output_path)
#             df = df[df.fold == fold]

#             output = df.merge(EL[['ext_id', 'row_context', 'col_context', 'row_id', 'col_id', 'reverse_row_id', 'reverse_col_id', 'region_type', 'text_sentence_no_mask']], on='ext_id', how='inner')
#             output = convert_EL_cans_to_ML_data(output, ent_map, 'candidates_100', 'candidates_100', add_GT=False, top_n=args.top_n_to_rank)
#             output.to_pickle(data_output_path)


#     # #######################################################
#     #### Train an e2e ED model
#     if args.train_e2e_ED_one:
#         exp = ElRerank(
#                 args=args,
#                 seed=args.seed,
#                 BS=args.BS,
#                 use_wandb=args.use_wandb,
#                 wandb_dir=args.wandb_dir,
#                 test_fold=args.test_fold,
#                 project='ED',
#                 train_file_path=f'{args.data_dir}/train/{args.test_fold}_{args.seed}',
#                 test_file_path=f'{args.data_dir}/test/{args.top_n_to_rank}_{args.test_fold}_{seed}_e2e',
#                 ctc_pred_file_path=f'{args.CTC_output_dir}/{args.test_fold}_{seed}',
#                 save_dir=args.model_dir,
#                 mode='all'
#             )
#         exp.train()

#     # #######################################################
#     # #### Generate micro EL rsts per threshold  #####
#     if args.eval_EL_e2e_per_threhold:

#         thresholds = np.linspace(-1, 0, num=50, endpoint=False, retstep=False)
        
#         rsts = []
#         pbar = tqdm(total=len(thresholds) * 10)
#         for threshold in thresholds:
#             preds_on_GT_CTC = []
#             ctc_preds = []
#             for test_fold in ('mt', 'img_gen', 'misc', 'speech_rec', 'qa', 'nli', 'text_class', 'object_det', 'sem_seg', 'pose_estim'):
#                 exp = ElRerank(
#                     seed=args.seed,
#                     BS=args.BS,
#                     use_wandb=args.use_wandb,
#                     wandb_dir=args.wandb_dir,
#                     test_fold=test_fold,
#                     project='ED',
#                     train_file_path=f'{args.data_dir}/train/{test_fold}_{args.seed}',
#                     test_file_path=f'{args.data_dir}/test/{args.top_n_to_rank}_{test_fold}_{seed}_e2e',
#                     ctc_pred_file_path=f'{args.CTC_output_dir}/{test_fold}_{seed}',
#                     save_dir=args.model_dir,
#                     mode='all'
#                 )
#                 ctc_pred = pd.read_pickle(f'{args.CTC_output_dir}/{test_fold}_{seed}')
#                 ctc_pred = ctc_pred[ctc_pred.fold == test_fold]
#                 ctc_preds.append(ctc_pred)
#                 preds_on_GT_CTC.append(exp.compute_preds_on_GT_CTC())
#                 pbar.update(1)
#             preds_on_GT_CTC = pd.concat(preds_on_GT_CTC, ignore_index=True)
#             ctc_preds = pd.concat(ctc_preds, ignore_index=True)
#             rst = compute_micro_EL_perf_e2e(ctc_preds, preds_on_GT_CTC, threshold, [1])
#             rsts.append(rst)
#         rst = pd.concat(rsts, ignore_index=True)
#         pbar.close()

#         rst.to_csv(f'{args.output_dir}/{args.top_n_to_rank}_ED_{seed}_micro_e2e.csv', index=False)
#     ################################################################################################################
#     # #### Generate EL rsts per fold  #####

#     if args.eval_EL_e2e_per_fold:
#         dfs = []

#         EL_preds, ctc_preds = [], []
#         for test_fold in tqdm(('mt', 'img_gen', 'misc', 'speech_rec', 'qa', 'nli', 'text_class', 'object_det', 'sem_seg', 'pose_estim')):
#             exp = ElRerank(
#                 seed=args.seed,
#                 test_fold=test_fold,
#                 project='ED',
#                 train_file_path=f'{args.data_dir}/train/{test_fold}_{args.seed}',
#                 test_file_path=f'{args.data_dir}/test/{args.top_n_to_rank}_{test_fold}_{seed}_e2e',
#                 ctc_pred_file_path=f'{args.CTC_output_dir}/{test_fold}_{seed}',
#                 save_dir=args.model_dir,
#                 mode='all'
#             )
#             EL_preds.append(exp.compute_preds_on_GT_CTC())
#             ctc_preds.append(pd.read_pickle(exp.ctc_pred_file_path))

#         # per fold
#         dfs.append(exp.test(threshold=args.threshold, inKB_acc_at_topks=[1]))
#         df = pd.concat(dfs, ignore_index=True)
#         df.to_csv(f'{args.output_dir}/{args.top_n_to_rank}_ED_{seed}_e2e_by-fold.csv', index=False)

        