import sys
sys.path.append(sys.path[0] + '/..')

import pandas as pd
import numpy as np
from sklearn import metrics
from pathlib import Path
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm


from common_utils.common_data_processing_utils import padding_by_batch
from common_utils.common_exp_utils import Experiment
from CTC.TE_loaders import DataWithTokenType, load_single_DataWithTokenType
from CTC.utils import generate_CTC_preds
from CTC.models import SciBertWithAdditionalFeatures


EVAL_AXCELL = False
if EVAL_AXCELL:
    from fastai.text import *  # you would need to install fastai



def split_df(df, valid_fold, test_fold):
    is_test = df.fold == test_fold
    is_valid = df.fold == valid_fold
    test_df_all = df[is_test].copy()
    valid_df_all = df[is_valid].copy()
    train_df_all = df[(~is_test) & (~is_valid)].copy()
    return train_df_all, valid_df_all, test_df_all


class AxCellExp(Experiment):
    """
    Make sure to run these with AxCell conda activated.
    """
    @staticmethod
    def dataframes_to_databunch(base_path, train_df, valid_df, test_df, batch_size, processor):
        columns = ["label", "text", "cell_reference", "cell_styles", "cell_layout", "cell_content", "row_context", "col_context"]
        text_cols=["cell_styles", "cell_layout", "text", "cell_content", "row_context", "col_context", "cell_reference"]
        train_df, valid_df, test_df = train_df[columns], valid_df[columns], test_df[columns]
        
        label_cols = ["label"]
        train_tl = TextList.from_df(train_df, base_path, cols=text_cols, processor=processor)
        valid_tl = TextList.from_df(valid_df, base_path, cols=text_cols, processor=processor)
        test_tl  = TextList.from_df(test_df,  base_path, cols=text_cols, processor=processor)
    
        src = ItemLists(base_path, train_tl, valid_tl)\
            .label_from_df(cols=label_cols)
        src.add_test(test_tl)
    
        data_clas = src.databunch(bs=batch_size)
        return data_clas


class AxCellTrainExp(AxCellExp):
    def __init__(self, seed, BS, base_dir, test_fold, model_save_dir, input_file_path=None, input_df=None, valid_fold='img_class', model_path=None):
        super().__init__(seed, test_fold, input_file_path, input_df, valid_fold, model_path)
        self.model_save_dir = Path(model_save_dir)
        self.base_dir = Path(base_dir)
        self.BS = BS

    @staticmethod
    def align_df(df, batch_size):
        aligned_len = ( len(df) // batch_size ) * batch_size
        return df.iloc[:aligned_len]
    
    def train(self):
        from axcell.models.structure.ulmfit_experiment import ULMFiTExperiment

        processor = processor = SPProcessor(
            sp_model=self.base_dir / 'tmp' / 'spm.model',
            sp_vocab=self.base_dir / 'tmp' / 'spm.vocab',
            mark_fields=True
        )
        if self.input_df is None and self.input_file_path is None:
            raise ValueError('Either input_df or input_file_path must be provided')
        
        if self.input_df is None:
            self.input_df = pd.read_parquet(self.input_file_path)

        train_df, valid_df, test_df = split_df(self.input_df, self.valid_fold, self.test_fold)
        train_df = self.align_df(train_df, self.BS)
        data_clas = self.dataframes_to_databunch(self.base_dir, train_df, valid_df, test_df, self.BS, processor)

        experiment = ULMFiTExperiment(remove_num=False, drop_duplicates=False,
            this_paper=True, merge_fragments=True, merge_type='concat',
            evidence_source='text_highlited', split_btags=True, fixed_tokenizer=True,
            fixed_this_paper=True, mask=True, evidence_limit=None, context_tokens=None,
            lowercase=True, drop_mult=0.15, fp16=True, train_on_easy=False,
            dataset="segmented-tables",
            test_split=self.test_fold,
            valid_split=self.valid_fold,
            pretrained_lm='lm',
            seed=self.seed,
            BS=self.BS
        )
        experiment.get_trained_model(data_clas)
        experiment._model.save((self.model_save_dir / f"{experiment.valid_split}_{experiment.test_split}_{experiment.seed}").resolve())


class AxCellEvalExp(AxCellExp):
    def __init__(self, seed, base_dir, test_fold, input_file_path=None, input_df=None, valid_fold='img_class', model_path=None):
        super().__init__(seed, test_fold, input_file_path, input_df, valid_fold, model_path)

        processor = processor = SPProcessor(
            sp_model=Path(base_dir) / 'tmp' / 'spm.model',
            sp_vocab=Path(base_dir) / 'tmp' / 'spm.vocab',
            mark_fields=True
        )

        if input_df is None and input_file_path is None:
            raise ValueError('Either input_df or input_file_path must be provided')
        
        if input_df is None:
            input_df = pd.read_parquet(input_file_path)

        train_df, valid_df, test_df = split_df(input_df, valid_fold, test_fold)
        self.valid_df, self.test_df = valid_df, test_df
        data_clas = self.dataframes_to_databunch(base_dir, train_df, valid_df, test_df, 64, processor)

        from fastai.text.learner import _model_meta
        cfg = _model_meta[AWD_LSTM]['config_clas'].copy()
        cfg['n_layers'] = 3
        clas = text_classifier_learner(data_clas, AWD_LSTM, config=cfg, drop_mult=0.15, metrics=None)

        if model_path is None:
            model_path = (Path('notebooks') / 'training' / 'experiments' / 'segmentation' / f'{valid_fold}_{test_fold}_{seed}').resolve()
        
        clas.load(model_path)
        self.clas = clas
        self.label_map_reverse = {
            0: "other",
            1: "dataset",
            2: "method",
            3: "metric",
            4: "dataset_and_metric"
        }
    
    def process_probs(self, probs, df):
        preds = np.argmax(probs, axis=1)
        df['pred'] = preds
        df['pred_class'] = df['pred'].apply(lambda x: self.label_map_reverse[x])
        df['label_class'] = df['label'].apply(lambda x: self.label_map_reverse[x])
        df['is_correct'] = df['pred_class'] == df['label_class']
        return df
    
    def compute_val_cr(self, digits=4, average=None):
        valid_probs = self.clas.get_preds(ds_type=DatasetType.Valid, ordered=True)[0].cpu().numpy()
        valid_df = self.process_probs(valid_probs, self.valid_df)
        return valid_df['is_correct'].sum() / len(valid_df['is_correct']), metrics.precision_recall_fscore_support(valid_df['label_class'], valid_df['pred_class'], average=average, labels=list(self.label_map_reverse.values()))
    
    def compute_test_cr(self, digits=4, average=None):
        test_probs = self.clas.get_preds(ds_type=DatasetType.Test, ordered=True)[0].cpu().numpy()
        test_df = self.process_probs(test_probs, self.test_df)
        return test_df['is_correct'].sum() / len(test_df['is_correct']), metrics.precision_recall_fscore_support(test_df['label_class'], test_df['pred_class'], average=average, labels=list(self.label_map_reverse.values()))




class CTCEvalExp(Experiment):
    model: SciBertWithAdditionalFeatures
    valid_ds: DataWithTokenType
    test_ds: DataWithTokenType

    def __init__(self, seed, test_fold, input_file_path=None, input_df=None, valid_fold='img_class', model_path=None):
        super().__init__(seed, test_fold, input_file_path, input_df, valid_fold, model_path)
        
        model =  torch.load(model_path)
        model.eval()

        config = model.config
        assert config.test_fold == test_fold
        assert config.seed == seed

        if input_df is None:
            input_df = pd.read_pickle(input_file_path)
        
        # self.valid_ds, self.test_ds = load_DataWithTokenType(config, input_df, config.input_cols, config.input_num_cols, valid_fold, test_fold, augment=False, load_train=False)
        self.model = model
        self.config = config
        self.input_df = input_df

    def compute_cr(self, split:str, BS=512):
        """
        :param split: 'valid' or 'test'
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
        y_preds, y_trues = [], []
        if split == 'valid':
            df = self.input_df[self.input_df.fold == self.config.valid_fold].reset_index(drop=True)
        elif split == 'test':
            df = self.input_df[self.input_df.fold == self.config.test_fold].reset_index(drop=True)
        else:
            raise ValueError(f"split must be either 'valid' or 'test', got {split}")
        ds = load_single_DataWithTokenType(self.config, df, drop_duplicates=False)
        dl = DataLoader(ds, batch_size=BS, collate_fn=padding_by_batch)
    
        for batch in tqdm(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            labels = batch["labels"].view(-1)
            idx = (labels != -100).nonzero(as_tuple=True)[0]

            y_preds.append(predictions[idx].cpu())
            y_trues.append(labels[idx].cpu())
        
        y_preds = torch.cat(y_preds, dim=0).squeeze()
        y_trues = torch.cat(y_trues, dim=0).squeeze()

        mircro_f1 = sum(y_preds == y_trues) / len(y_trues)
        return mircro_f1.item(), metrics.precision_recall_fscore_support(y_trues, y_preds, average=None, labels=list(range(5)))

    def generate_preds(self, save_path=None):
        non_train_df = self.input_df[ (self.input_df.fold == self.valid_fold) | (self.input_df.fold == self.config.test_fold)].copy()
        output = generate_CTC_preds(self.config, self.model, non_train_df)
        if save_path is not None:
            output.to_pickle(save_path)
        return output


class CTCEvalExpNB(CTCEvalExp):
    def __init__(self, model_path):
        model = torch.load(model_path)
        model.eval()

        config = model.config
        input_df = pd.read_pickle(config.input_file)

        self.model = model
        self.config = config
        self.input_df = input_df
    
    def compute_cr(self, split:str):
        return super().compute_cr(split, BS=self.config.eval_BS)


