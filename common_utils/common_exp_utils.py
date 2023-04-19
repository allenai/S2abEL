from dataclasses import dataclass
import pandas as pd


from ED.utils import *
from ED.candidate_gen import compute_bi_enc_ent_candidates, test_candidate_effectiveness


@dataclass
class Experiment:
    seed: int
    test_fold: str
    input_file_path: str = None
    input_df: pd.DataFrame = None
    valid_fold: str = 'img_class'
    model_path: str = None


class EvalExperiment(Experiment):
    def __init__(self, seed, test_fold, input_file_path=None, input_df=None, valid_fold='img_class', model_path=None):
        if model_path is None:
            raise ValueError('model_path must be provided')
        
        if input_df is None:
            input_df = pd.read_pickle(input_file_path)
        
        assert str(seed) in model_path and test_fold in model_path

        super().__init__(seed, test_fold, input_file_path, input_df, valid_fold, model_path)
        if input_df is None and input_file_path is None:
            raise ValueError('Either input_df or input_file_path must be provided')
    
    def compute_ent_enmbeddings(self, entity_file_path, ent_emb_save_path=None, BS=128, mode='Contrastive'):
        model = torch.load(self.model_path)
        EL_ent = pd.read_pickle(entity_file_path)
        return compute_ent_embedding(model, EL_ent, seed=self.seed, BS=BS, output_path=ent_emb_save_path, mode=mode)

    def generate_candidates(self, entity_file_path, ent_emb_save_path, top_k=100, save_path=None, BS=128, mode='Contrastive', single_fold_val_test=False, mix_methods_datasets=False):
        model = torch.load(self.model_path)
        EL_ent = pd.read_pickle(entity_file_path)
        with open(ent_emb_save_path, 'rb') as f:
            ent_embeds = pickle.load(f)
        return compute_bi_enc_ent_candidates(model, self.input_df, ent_embeds, EL_ent, top_k=top_k, output_path=save_path, seed=self.seed, BS=BS, mode=mode, single_fold_val_test=single_fold_val_test, mix_methods_datasets=mix_methods_datasets)



class EvalExperimentNB:
    def __init__(self, model_path):
        model = torch.load(model_path)
        model.eval()
        self.config = model.config
        self.model = model
    
    def compute_ent_enmbeddings(self, ent_file_path, ent_emb_save_path=None, mode='Contrastive'):
        EL_ent = pd.read_pickle(ent_file_path)
        return compute_ent_embedding(self.model, EL_ent, seed=self.config.seed, BS=self.config.eval_BS, output_path=ent_emb_save_path, mode=mode)

    def generate_candidates(self, ent_file_path, el_df, ent_emb_save_path=None, ent_embeds=None, top_k=100, save_path=None, mode='Contrastive'):
        EL_ent = pd.read_pickle(ent_file_path)
        if ent_embeds is None:
            with open(ent_emb_save_path, 'rb') as f:
                ent_embeds = pickle.load(f)
        return compute_bi_enc_ent_candidates(self.model, el_df, ent_embeds, EL_ent, top_k=top_k, val_fold=self.config.valid_fold, output_path=save_path, seed=self.config.seed, BS=self.config.eval_BS, mode=mode)