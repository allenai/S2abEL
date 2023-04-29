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
import argparse



from common_utils.common_data_processing_utils import *
from common_utils.common_exp_utils import Experiment
from common_utils.common_ML_utils import *
from ASM.generate_preds import *
from ED.candidate_gen import get_ASR_candidates_encoders
from ED.models import BiEncoderTriplet
from ED.data_loaders import load_triplet_encoder_input
from ED.trainers import EL_train_loop_notebook


class ASMEvalExp(Experiment):
    def __init__(self, seed, test_fold, model_dir=None, model_path=None, input_file_path=None, input_df=None, valid_fold='img_class', mode='Cross', epoch=2, loss='BCE'):

        if input_df is None and input_file_path is None:
            raise ValueError('Either input_df or input_file_path must be provided')
        
        if model_path is None and model_dir is None:
            raise ValueError('Either model_path or model_dir must be provided')
        if model_path is None:
            model_path = f'{model_dir}/{mode}Enc_{loss}_epoch={epoch}_seed={seed}_{test_fold}'
        if input_df is None:
            input_df = pd.read_pickle(input_file_path)

        super().__init__(seed, test_fold, input_file_path, input_df, valid_fold, model_path)        
    
    def make_predictions(self, save_path=None, BS=256):
        model = torch.load(self.model_path)
        RPI_scores = generate_scores_encoder(self.input_df, model, output_path=None, seed=self.seed, BS=BS, val_fold=self.valid_fold)
        RPI_preds = enhance_preds_encoder(RPI_scores, output_path=save_path)
        return RPI_preds
    
    def make_predictions_one_test_fold(self, save_path=None, BS=256):
        model = torch.load(self.model_path)
        input_df = self.input_df
        RPI_scores = generate_scores_encoder(input_df[input_df.fold==self.test_fold], model, output_path=None, seed=self.seed, BS=BS, single_val_test_fold=True)
        RPI_preds = enhance_preds_encoder(RPI_scores, output_path=save_path)
        return RPI_preds


class ASMEvalExpNB:
    def __init__(self, model_path):
        model = torch.load(model_path)
        model.eval()

        config = model.config
        input_df = pd.read_pickle(config.input_file)

        self.model = model
        self.config = config
        self.input_df = input_df
    
    def make_predictions(self, save_path=None, enhance=True):
        ASM_scores = generate_scores_encoder(self.input_df, self.model, output_path=None, seed=self.config.seed, BS=self.config.eval_BS, val_fold=self.config.valid_fold, use_tqdm=True)
        ASM_preds = enhance_preds_encoder(ASM_scores, output_path=save_path, enhance=enhance)
        return ASM_preds


class DirectRetrievalTripletTrain:
    def __init__(self, seed, test_fold, input_file, BS=32, valid_fold='img_class', 
                 save_dir=None, lr=2e-5, epoch=2, eval_steps=300, grad_accum_step=2, eval_BS=64,
                 name=None):

        cell_feas = cell_rep_features.copy()
        cell_feas.remove('has_reference')
        cell_embedding = Config(
            input_cols=cell_feas,
            pool_mtd='avg',
            pretrained = "allenai/scibert_scivocab_uncased",
            lr=lr
        )
        pos_ent_embedding = Config(
            input_cols=['pos_candidate_ent_names', 'pos_candidate_ent_full_names', 'pos_candidate_ent_descriptions'],
            pool_mtd='avg',
            pretrained = "allenai/scibert_scivocab_uncased",
            lr=lr
        )
        neg_ent_embedding = Config(
            input_cols=['neg_candidate_ent_names', 'neg_candidate_ent_full_names', 'neg_candidate_ent_descriptions'],
            pool_mtd='avg',
            pretrained = "allenai/scibert_scivocab_uncased",
            lr=lr
        )

        ## effective BS = BS * grad_accum_step

        assert BS % grad_accum_step == 0
        config = Config(
            seed=seed,
            cell_embedding=cell_embedding,
            pos_ent_embedding=pos_ent_embedding,
            neg_ent_embedding=neg_ent_embedding,
            ent_embedding=None,
            BS=int(BS/grad_accum_step),
            epoch=epoch,
            test_fold=test_fold,
            valid_fold=valid_fold,
            lr=lr,
            grad_accum_step=grad_accum_step,
            input_file=input_file,
            drop_duplicates=True,
            eval_steps=eval_steps,
            save_dir=save_dir,
            eval_BS=eval_BS,
            name=name
        )

        self.g = set_seed(seed)
        self.config = config
        self.input_df = pd.read_pickle(input_file)
    
    def train(self):
        model = BiEncoderTriplet(self.config)

        train_df, valid_df, _ = split_fold(self.input_df, self.config.valid_fold, self.config.test_fold)
        train_ds, valid_ds = load_triplet_encoder_input(self.config, train_df, valid_df)

        print(f"Training length {len(train_ds)}")
        print(f"Validation length {len(valid_ds)}")

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=self.config.BS, collate_fn=padding_bi_encoder_data, worker_init_fn=seed_worker, generator=self.g)
        eval_dl = DataLoader(valid_ds, batch_size=self.config.eval_BS, collate_fn=padding_bi_encoder_data, worker_init_fn=seed_worker, generator=self.g)

        EL_train_loop_notebook(self.config, model, train_dl, eval_dl)



# if __name__ == "__main__":

#     parser=argparse.ArgumentParser(
#         description=__doc__,
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     parser.add_argument('--test_fold', help='test fold to use', type=str, required=True)
#     parser.add_argument('--input_file', help='input file name', type=str, required=True)
#     parser.add_argument('--ref_extract_path', help='path to the intermediate ref_extract file', type=str, required=True)
#     parser.add_argument('--model_path', help='the path to the model', type=str, default=None)
#     parser.add_argument('--output_path', help='the path to save the output', type=str, default=None)
#     parser.add_argument('--seed', help='random seed', type=int, default=42)
#     parser.add_argument('--BS', help='batch size', type=int, default=32)
#     args = parser.parse_args()

#     #### generate ASM outputs for one model
#     exp = ASMEvalExp(args.seed, args.test_fold, model_path=args.model_path, input_file_path=args.input_file)
#     RPI_preds = exp.make_predictions(BS=args.BS)
#     print(f"<<< generating {args.output_path}")
#     ref_extract = pd.read_pickle(args.ref_extract_path)
#     RPI_can_preds = get_ASR_candidates_encoders(RPI_preds, ref_extract, ['RPI_preds'], [100])
#     RPI_can_preds.to_pickle(args.output_path)

#     ## batch generating all ASM outputs
#     ####
#     # proj = ''
#     # root_dir = '.'
#     # model_dir = f'{root_dir}/_models/{proj}'
#     # RPI_save_dir = f'{root_dir}/_outputs/{proj}'
#     # RPI_can_save_dir = f'{root_dir}/_outputs/{proj}_can'
#     # ref_extract = pd.read_pickle(args.ref_extract)

#     # for seed in (42, ):
#     #     for test_fold in ('mt', 'img_gen', 'nli', 'speech_rec', 'qa', 'misc', 'text_class', 'object_det', 'sem_seg', 'pose_estim'):
#     #         RPI_output_path = os.path.join(RPI_save_dir, f'{test_fold}_{seed}')
#     #         RPI_can_output_path = os.path.join(RPI_can_save_dir, f'{test_fold}_{seed}')
#     #         if not os.path.exists(RPI_output_path):
#     #             print(f"<<< generating {RPI_output_path}")
#     #             exp = ASMEvalExp(seed, test_fold, model_dir=model_dir, input_file_path=args.input_file)
#     #             # RPI_preds = exp.make_predictions_one_test_fold(save_path=RPI_output_path, BS=512)
#     #             RPI_preds = exp.make_predictions(save_path=RPI_output_path, BS=512)
#     #         else:
#     #             RPI_preds = pd.read_pickle(RPI_output_path)

#     #         if not os.path.exists(RPI_can_output_path):
#     #             print(f"<<< generating {RPI_can_output_path}")
#     #             RPI_can_preds = get_ASR_candidates_encoders(RPI_preds, ref_extract, ['RPI_preds'], [100])
#     #             RPI_can_preds.to_pickle(RPI_can_output_path)
    
