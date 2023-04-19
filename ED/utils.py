from tqdm import tqdm
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
from sentence_transformers import util
import pickle
import re
import json

from CTC.utils import LabelsExt
from ED.models import *
from common_utils.common_ML_utils import *
from common_utils.common_data_processing_utils import *
from ED.data_loaders import load_bi_encoder_input, padding_bi_encoder_data, load_triplet_encoder_input
from CTC.TE_loaders import load_DataWithTOkenType, load_single_DataWithTOkenType


def convert_EL_can_to_TURL_data(EL_bm25f_can, EL_pred_can, pc, ent_map, ref_extract, test_fold, train_output_path=None, val_output_path=None, test_output_path=None, top_n_cans=50, val_fold='img_class'):
    val_df = EL_pred_can[(EL_pred_can.fold == val_fold) & (EL_pred_can.pwc_url != '0')]
    test_df = EL_pred_can[(EL_pred_can.fold == test_fold) & (EL_pred_can.pwc_url != '0')]
    train_df = EL_bm25f_can[(EL_bm25f_can.fold != val_fold) & (EL_bm25f_can.fold != test_fold) & (EL_bm25f_can.pwc_url != '0')]

    train_data = _convert_df_to_TURL_json(train_df, 'candidates_100', pc, ref_extract, ent_map, top_n=top_n_cans)
    val_data = _convert_df_to_TURL_json(val_df, 'candidates_100', pc, ref_extract, ent_map, top_n=top_n_cans)

    def has_GT(row):
        return row.pwc_url in row['candidates_100'][:top_n_cans]
    test_df['has_GT'] = test_df.apply(has_GT, axis=1)
    test_df = test_df[test_df.has_GT]
    del test_df['has_GT']
    test_data = _convert_df_to_TURL_json(test_df, 'candidates_100', pc, ref_extract, ent_map, top_n=top_n_cans)

    if train_output_path is not None:
        with open(train_output_path, 'w') as f:
            json.dump(train_data, f)
    if val_output_path is not None:
        with open(val_output_path, 'w') as f:
            json.dump(val_data, f)
    if test_output_path is not None:
        with open(test_output_path, 'w') as f:
            json.dump(test_data, f)
    return train_data, val_data, test_data


def _convert_df_to_TURL_json(df, can_col:str, pc, ref_extract, ent_map, top_n=50):
    df = df.copy().reset_index(drop=True)
    if 'table_id' not in df.columns:
        df['table_id'] = df.ext_id.apply(lambda x: '/'.join(x.split('/')[:2]))
    if 'row_id' not in df.columns:
        df['row_id'] = df.ext_id.apply(lambda x: x.split('/')[2])
    if 'col_id' not in df.columns:
        df['col_id'] = df.ext_id.apply(lambda x: x.split('/')[3])

    rst = []
    for table_id in df.table_id.unique():
        table_idx = int(table_id.split('_')[1].split('.')[0]) - 1
        cells_df = df[df.table_id == table_id]
        paper_id = cells_df.iloc[0].paper_id
        paper = pc.get_by_id(paper_id)
        table = paper.tables[table_idx]

        cans = set()
        for cell in cells_df.itertuples():
            # cans.update(df[df.ext_id == cell.ext_id].iloc[0][can_col][:top_n])
            cans.update(getattr(cell, can_col)[:top_n])
            cans.add(cell.pwc_url)
        cans = list(cans)

        rst.append([
            table_id,
            ref_extract[(ref_extract.paper_id == paper_id) & (ref_extract.idx == -1)].iloc[0]['title'],
            table.section_header(paper)[1],
            table.caption,
            [table.matrix.iloc[0, i] for i in range(table.shape[1])],
            [[[int(cell.row_id), int(cell.col_id)], cell.cell_content] for cell in cells_df.itertuples()],
            [ent_map[c] for c in cans],
            [cans.index(cell.pwc_url) for cell in cells_df.itertuples()],
        ])
    return rst


def convert_EL_can_to_TURL_test_fold(EL_can:pd.DataFrame, pc, ent_map, ref_extract, val_test_split=None, val_output_path=None, test_output_path=None, single_val_test_fold=True, output_path=None, top_n=50):
    """
    Convert outputs from our model to val and test splits for TURL.
    Only keeping inKB and hasGT candidates.
    """
    EL_inKB = EL_can[EL_can.pwc_url != '0']
    EL_inKB[f'candidates_top{top_n}'] = EL_inKB['candidates_100'].apply(lambda x: x[:top_n])
    def has_GT(row):
        return row.pwc_url in row.candidates_top50

    EL_inKB['has_GT'] = EL_inKB.apply(has_GT, axis=1)
    EL_inKB_has_GT = EL_inKB[EL_inKB.has_GT]
    EL_inKB_has_GT['table_id'] = EL_inKB_has_GT.ext_id.apply(lambda x: '/'.join(x.split('/')[:2]))
    EL_inKB_has_GT['row_id'] = EL_inKB_has_GT.ext_id.apply(lambda x: x.split('/')[2])
    EL_inKB_has_GT['col_id'] = EL_inKB_has_GT.ext_id.apply(lambda x: x.split('/')[3])

    rst = []
    df = EL_inKB_has_GT

    for table_id in df.table_id.unique():
        table_idx = int(table_id.split('_')[1].split('.')[0]) - 1
        cells_df = df[df.table_id == table_id]
        paper_id = cells_df.iloc[0].paper_id
        paper = pc.get_by_id(paper_id)
        table = paper.tables[table_idx]

        m_cans = set()
        for cell in cells_df.itertuples():
            m_cans.update(df[df.ext_id == cell.ext_id].iloc[0]['candidates_top50'])
        
        d_cans = set()
        for cell in cells_df.itertuples():
            d_cans.update(df[df.ext_id == cell.ext_id].iloc[0]['candidates_top50'])
        
        cans = list(m_cans) + list(d_cans)
        rst.append([
            table_id,
            ref_extract[(ref_extract.paper_id == paper_id) & (ref_extract.idx == -1)].iloc[0]['title'],
            table.section_header(paper)[1],
            table.caption,
            [table.matrix.iloc[0, i] for i in range(table.shape[1])],
            [[[int(cell.row_id), int(cell.col_id)], cell.cell_content] for cell in cells_df.itertuples()],
            [ent_map[c] for c in cans],
            [cans.index(cell.pwc_url) for cell in cells_df.itertuples()],
            cells_df.iloc[0].fold,
            paper_id
        ])
    df = pd.DataFrame(rst, columns=['table_id', 'title', 'section', 'caption', 'headers', 'cells', 'candidate_ents', 'GT_idx', 'fold', 'paper_id'])
    
    if single_val_test_fold:
        val_paper_ids = [paper for _, papers in val_test_split['val'].items() for paper in papers]    
        df['is_val'] = df.paper_id.isin(val_paper_ids)
        val_df = df[df.is_val]
        test_df = df[~df.is_val]
        with open(val_output_path, 'w') as f:
            json.dump(val_df[['table_id', 'title', 'section', 'caption', 'headers', 'cells', 'candidate_ents', 'GT_idx']].values.tolist(), f)
        with open(test_output_path, 'w') as f:
            json.dump(test_df[['table_id', 'title', 'section', 'caption', 'headers', 'cells', 'candidate_ents', 'GT_idx']].values.tolist(), f)
    else:
        pass


def transform_raw_text(s: pd.Series):
    """
    Transform the raw text containing the cell_content to data that can be fed into a LM.
    """
    s = s.replace(re.compile('\n'), '')  # replace \n
    s = s.replace(re.compile('xxref-bibbib(\d+)'), r'\1')  # transform in-text reference.
    s = s.replace(re.compile(r"xxtable-xxanchor-([\w\d-])*"), add_token_mapping['[TABLE_TITLE]'])  # handles all table titles
    s = s.replace(re.compile(r"xxanchor-bibbib[\d-]*"), add_token_mapping['[BIB_ITEM]'])  # handles all bibitems
    s = s.replace(re.compile(r"xxanchor-[\w\d-]*"), add_token_mapping['[SEC_OR_FIG_TITLE]'])  # handles all bibitems
    return s


def replace_EL_non_train_folds(EL_cans_non_learned: pd.DataFrame, EL_cans_learned: pd.DataFrame, test_fold, drop_train_folds=False):
    """
    Replace the non-train folds of EL_cans_non_learned with the learned candidates from EL_cans_learned.
    """
    assert EL_cans_learned.fold.nunique() <= 2

    if not drop_train_folds:
        output = EL_cans_non_learned.merge(EL_cans_learned, on='ext_id', how='left', suffixes=(None, '_to_drop'))
    else:
        output = EL_cans_non_learned.merge(EL_cans_learned, on='ext_id', how='inner', suffixes=(None, '_to_drop'))

    can_cols = [i for i in EL_cans_non_learned.columns if i.startswith('candidates_')]

    def _merge(row, col):
        if not hasattr(row, col + '_to_drop'):
            return getattr(row, col)
        else:
            if isinstance(getattr(row, col + '_to_drop'), list):  # otherwise is nan
                return getattr(row, col + '_to_drop')
            else:
                return getattr(row, col)

    for col in can_cols:
        output[col] = output.apply(lambda x: _merge(x, col), axis=1)
    
    to_drop_cols = [i for i in output.columns if i.endswith('_to_drop')]
    output = output.drop(columns=to_drop_cols)

    output = output[output.fold != test_fold]
    return output.reset_index(drop=True)


def convert_EL_cans_to_triplet_training_data(EL: pd.DataFrame, ent_map: Dict[str, Tuple], method_can_col:str='method_RPI_candidates', dataset_can_col:str='dataset_RPI_candidates', val_test_split=None, top_n=10000):
    attrs = ['ext_id', 'paper_id', 'fold', 'cell_type', 'cell_content', 'row_context',
       'col_context', 'row_id', 'col_id', 'reverse_row_id', 'reverse_col_id', 'region_type', 'pwc_url', 'text_sentence_no_mask']
    output = EL[ attrs ]
    output = output[output.pwc_url != '0'].copy()

    neg_candidate_ents, pos_candidate_ents = [], []

    for row in EL.itertuples():
        if row.pwc_url == '0':
            continue
        if row.cell_type == 'Method':
            tmp = set([ent_map[m_url] for m_url in getattr(row, method_can_col)[:top_n] if m_url != row.pwc_url])
        else:
            tmp = set([ent_map[d_url] for d_url in getattr(row, dataset_can_col)[:top_n] if d_url != row.pwc_url])

        neg_candidate_ents.append(tmp)
        pos_candidate_ents.append([ent_map[row.pwc_url] for _ in range(len(tmp))])
    

    pos_ent_names, pos_ent_full_names, pos_ent_descriptions, pos_ent_urls = [], [], [], []
    neg_ent_names, neg_ent_full_names, neg_ent_descriptions, neg_ent_urls = [], [], [], []

    for ent_set in pos_candidate_ents:
        pos_ent_names.append([i[0] for i in ent_set])
        pos_ent_full_names.append([i[1] for i in ent_set])
        pos_ent_descriptions.append([i[2] for i in ent_set])
        pos_ent_urls.append([i[3] for i in ent_set])
    for ent_set in neg_candidate_ents:
        neg_ent_names.append([i[0] for i in ent_set])
        neg_ent_full_names.append([i[1] for i in ent_set])
        neg_ent_descriptions.append([i[2] for i in ent_set])
        neg_ent_urls.append([i[3] for i in ent_set])
    output['pos_candidate_ent_names'] = pos_ent_names
    output['pos_candidate_ent_full_names'] = pos_ent_full_names
    output['pos_candidate_ent_descriptions'] = pos_ent_descriptions
    output['pos_candidate_ent_url'] = pos_ent_urls
    output['neg_candidate_ent_names'] = neg_ent_names
    output['neg_candidate_ent_full_names'] = neg_ent_full_names
    output['neg_candidate_ent_descriptions'] = neg_ent_descriptions
    output['neg_candidate_ent_url'] = neg_ent_urls
    output = output.explode( ['pos_candidate_ent_names', 'pos_candidate_ent_full_names', 'pos_candidate_ent_descriptions', 'pos_candidate_ent_url', 'neg_candidate_ent_names', 'neg_candidate_ent_full_names', 'neg_candidate_ent_descriptions', 'neg_candidate_ent_url'] )
    output = output.reset_index(drop=True)
    output = output[(~output.pos_candidate_ent_names.isnull()) & (~output.neg_candidate_ent_names.isnull())]
    output = output[(output.pos_candidate_ent_names != '') & (output.neg_candidate_ent_names != '')]
    output.fillna('', inplace=True)

    if val_test_split is not None:
        val_paper_ids = [paper for _, papers in val_test_split['val'].items() for paper in papers]    
        output['is_val'] = output.paper_id.isin(val_paper_ids)

    return output


def convert_EL_cans_to_ML_data(EL_RPI: pd.DataFrame, ent_map: Dict[str, Tuple], method_can_col:str='method_RPI_candidates', dataset_can_col:str='dataset_RPI_candidates', add_GT: bool=True, top_n=100000):
    """
    :return EL_ML
    """
    # attrs = ['ext_id', 'fold', 'cell_type', 'cell_content', 'row_context',
    #    'col_context', 'row_id', 'col_id', 'reverse_row_id', 'reverse_col_id', 'region_type', 'pwc_url', 'text_sentence_no_mask']
    # output = EL_RPI.copy()[ attrs ]

    output = EL_RPI.copy()
    candidate_ents = []

    for row in EL_RPI.itertuples():
        if (hasattr(row, 'cell_type') and row.cell_type == 'Method') or (hasattr(row, 'labels') and row.labels == LabelsExt.METHOD.value):
            tmp = set([ent_map[m_url] for m_url in getattr(row, method_can_col)[:top_n]])
        else:
            tmp = set([ent_map[d_url] for d_url in getattr(row, dataset_can_col)[:top_n]])

        if add_GT and row.pwc_url != '0':  # add the GT match
            tmp.add( ent_map[row.pwc_url] )
        candidate_ents.append(tmp)
    
    ent_names, ent_full_names, ent_descriptions, ent_urls = [], [], [], []

    for ent_set in candidate_ents:
        ent_names.append([i[0] for i in ent_set])
        ent_full_names.append([i[1] for i in ent_set])
        ent_descriptions.append([i[2] for i in ent_set])
        ent_urls.append([i[3] for i in ent_set])
    
    output['candidate_ent_names'] = ent_names
    output['candidate_ent_full_names'] = ent_full_names
    output['candidate_ent_descriptions'] = ent_descriptions
    output['candidate_ent_url'] = ent_urls
    output = output.explode( ['candidate_ent_names', 'candidate_ent_full_names', 'candidate_ent_descriptions', 'candidate_ent_url'] )
    output = output.reset_index(drop=True)

    if 'candidates_100' in output:
        del output['candidates_100']

    def get_label(row):
        if row.pwc_url == row.candidate_ent_url:
            return 0
        else:
            return 1
    
    output['labels'] = output.apply(lambda x: get_label(x), axis=1)
    output = output[~output.candidate_ent_names.isnull()]
    output = output[output.candidate_ent_names!='']
    output.fillna('', inplace=True)
    return output.reset_index(drop=True)


def generate_relavance_score(model, dl, use_tqdm=False):

    model.eval()

    total_loss = 0
    relevance_scores = []

    if use_tqdm:
        dl = tqdm(dl)

    for batch in dl:
        with torch.no_grad():
            outputs = model(**batch)
            if outputs.loss is not None:
                total_loss += outputs.loss
        if outputs.relevance_score is not None:
            relevance_scores += outputs.relevance_score.tolist()

    return total_loss / len(dl) if total_loss != 0 else 0, relevance_scores


def compute_ent_embedding(model, PwC_ent:pd.DataFrame, output_path=None, BS=64, seed=42, mode='Contrastive'):
    """
    Computes the embedding for each entity in the PwC_ent dataframe using bi-encoder and save.

    :param mode: 'Constrastive' or 'Triplet'
    :return output: Dict[EntType, torch.FloatTensor]. EntType = 'Method' | 'Dataset'
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    g = set_seed(seed)

    ent_embedding = Config(
        input_cols=['name', 'full_name', 'description'],
        pool_mtd='avg',
        pretrained = "allenai/scibert_scivocab_uncased"
    )
    config = Config(
        seed=seed,
        cell_embedding=None,
        ent_embedding=ent_embedding,
        use_labels=False,
    )

    output = {}
    for t in ('Method', 'Dataset'):
        if mode == 'Contrastive':
            ds = load_bi_encoder_input(config, PwC_ent[PwC_ent.type == t].reset_index(drop=True))
        elif mode == 'Triplet':
            config.pos_ent_embedding = None
            config.neg_ent_embedding = None
            ds = load_triplet_encoder_input(config, PwC_ent[PwC_ent.type == t].reset_index(drop=True))
        else:
            raise ValueError('Unknown mode')
        dl = DataLoader(ds, batch_size=BS, collate_fn=padding_bi_encoder_data, worker_init_fn=seed_worker, generator=g)
        ent_embeds = []
        for batch in tqdm(dl):
            with torch.no_grad():
                batch['output_emb'] = True
                outputs = model(**batch)
                ent_embeds.append(outputs.ent_emb.cpu())
        ent_embeds = torch.cat(ent_embeds, dim=0)
        output[t] = ent_embeds

    if output_path is not None:
        with open(output_path, 'wb') as f:
            pickle.dump(output, f)
    return output


def gen_test_EL_on_GT_CTC(EL_data, model, folds:List[str], top_n=10, BS=512, mode='cross'):
    """
    Test the entity linking model on the ground truth CTC data.
    :param mode: 'bi' or 'cross', indicating whether to use the bi-encoder or cross-encoder model.
    :param top_n: number of top ranked matches to return
    """

    config = model.config
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    config.drop_duplicates = False
    config.use_labels = False

    rst = []
    for fold in folds:
        data = EL_data[EL_data.fold == fold].reset_index(drop=True).copy()
        print(f"<<< fold = {fold}")
        if mode == 'bi':
            ds = load_bi_encoder_input(config, data)
            dl = DataLoader(ds, batch_size=BS, collate_fn=padding_bi_encoder_data)
        elif mode == 'cross':
            ds = load_single_DataWithTOkenType(config, data, drop_duplicates=False)
            dl = DataLoader(ds, batch_size=BS, collate_fn=padding_by_batch)
        else:
            raise ValueError(f"mode = {mode} is not supported.")
        
        _, relevance_score = generate_relavance_score(model, dl)
        data['relevance_score'] = relevance_score

        records = []
        for ext_id in tqdm(data.ext_id.unique()):
            ext = data[data.ext_id == ext_id]
            ext = ext.sort_values(by='relevance_score', ascending=False, ignore_index=True, axis=0)
            
            exts = ext.iloc[:top_n]
            records.append({
                'ext_id': ext_id,
                'fold': ext.iloc[0]['fold'],
                'cell_type': ext.iloc[0]['cell_type'],
                'cell_content': ext.iloc[0]['cell_content'],
                'pwc_url': ext.iloc[0]['pwc_url'],
                'returned_urls': exts['candidate_ent_url'].tolist(),
                'relevance_scores': exts['relevance_score'].tolist()
            })
        rst.append(pd.DataFrame(records))

    return pd.concat(rst, axis=0, ignore_index=True)


PRED_MISSING = 1000
PRED_FAIL = 1001  # returning a non-match entity

def _get_rank(row, threshold):
    if row.relevance_scores[0] < threshold:
        return PRED_MISSING  # if rank is PRED_MISSING, it means return 'missing from KB'
    for idx, (relevance_score, can_ent_url) in enumerate(zip(row.relevance_scores, row.returned_urls)):
        # if relevance_score < threshold:
        #     break  
        if can_ent_url == row.pwc_url:
            return idx + 1
    return PRED_FAIL


def compute_EL_perf_on_GT_CTC(EL_preds: pd.DataFrame, threshold, acc_at_topks: List[int]) -> pd.DataFrame:
    """
    Assume cells in EL_preds has correct CTC types regarding EL.

    :param: the threshold on relevance_scores
    :return fold, outKB_prec, outKB_recall, outKB_f1, inKB_acc@<X>fold in a DataFrame
    """
    assert EL_preds.fold.nunique() <= 2
    assert max(acc_at_topks) < min(PRED_MISSING, PRED_FAIL)

    EL_preds['rank'] = EL_preds.apply(lambda row: _get_rank(row, threshold), axis=1)

    records = []
    for fold in EL_preds.fold.unique():
        fold_df = EL_preds[EL_preds.fold == fold]
        GT_inKB, GT_outKB = fold_df[fold_df.pwc_url != '0'], fold_df[fold_df.pwc_url == '0']

        outKB_prec = sum(GT_outKB['rank'] == PRED_MISSING) / sum(fold_df['rank'] == PRED_MISSING) if sum(fold_df['rank'] == PRED_MISSING) != 0 else None
        outKB_recall = sum(GT_outKB['rank']== PRED_MISSING) / len(GT_outKB) if len(GT_outKB) != 0 else None
        outKB_f1 = 2*outKB_prec * outKB_recall / (outKB_prec + outKB_recall) if outKB_prec is not None and outKB_recall is not None else None

        record = {
            'fold': fold,
            'outKB prec': outKB_prec,
            'outKB recall': outKB_recall,
            'outKB f1': outKB_f1,
            # '# of cells': len(fold_df),
            # '# of cells predicted as missing': sum(fold_df['rank'] == PRED_MISSING),
            'outKB support': len(GT_outKB),
            'inKB support': len(GT_inKB),
            'gloabl acc': (sum(GT_outKB['rank'] == PRED_MISSING) + sum(GT_inKB['rank'] == 1)) / len(fold_df),
            'threshold': threshold
        }

        for acc_at_topk in acc_at_topks:
            record[f'GT_CTC inKB acc@{acc_at_topk}'] = sum(GT_inKB['rank'] <= acc_at_topk) / len(GT_inKB)
        
        records.append(record)

    return pd.DataFrame(records)


def compute_EL_perf_e2e(CTC_with_preds: pd.DataFrame, EL_preds_on_GT_CTC: pd.DataFrame, threshold, acc_at_topks: List[int]):

    assert EL_preds_on_GT_CTC.fold.nunique() <= 2

    if threshold > 0:
        threshold = threshold - 1

    EL = EL_preds_on_GT_CTC.merge(CTC_with_preds[['ext_id', 'CTC_preds']], on='ext_id', how='inner')
    EL['rank'] = EL.apply(lambda row: _get_rank(row, threshold), axis=1)
    CTC = EL
    
    correct_M_idx = (CTC.CTC_preds == CTC.cell_type) & (CTC.cell_type == 'Method')
    correct_D_idx = ((CTC.CTC_preds == 'Dataset') | (CTC.CTC_preds == 'DatasetAndMetric')) & ((CTC.cell_type == 'Dataset') | (CTC.cell_type == 'DatasetAndMetric'))
    correct_for_EL_idx = correct_M_idx | correct_D_idx
    correct_pos_idx = (CTC.cell_type != 'Other') & (CTC.CTC_preds != 'Other')


    records = []
    for fold in EL.fold.unique():
        fold_GT_inKB, fold_GT_outKB = EL[(EL.pwc_url != '0') & (EL.fold == fold)], EL[(EL.pwc_url == '0') & (EL.fold == fold)]

        outKB_TP_gt_ctc = (EL.fold == fold) & (EL.pwc_url == '0') & (EL['rank'] == PRED_MISSING)
        outKB_prec_gt_ctc = sum(outKB_TP_gt_ctc) / sum((EL.fold == fold) & (EL['rank'] == PRED_MISSING)) if sum((EL.fold == fold) & (EL['rank'] == PRED_MISSING)) != 0 else None
        outKB_recall_gt_ctc = sum(outKB_TP_gt_ctc) / sum((EL.pwc_url == '0') & (EL.fold == fold)) if sum((EL.pwc_url == '0') & (EL.fold == fold)) != 0 else None
        outKB_f1_gt_ctc = 2 * outKB_prec_gt_ctc * outKB_recall_gt_ctc / (outKB_prec_gt_ctc + outKB_recall_gt_ctc) if outKB_prec_gt_ctc is not None and outKB_recall_gt_ctc is not None else None

        outKB_TP_e2e = (EL.fold == fold) & (EL.pwc_url == '0') & (EL['rank'] == PRED_MISSING) & correct_pos_idx
        outKB_prec_e2e = sum(outKB_TP_e2e) / sum((EL.fold == fold) & (EL['rank'] == PRED_MISSING)) if sum((EL.fold == fold) & (EL['rank'] == PRED_MISSING)) != 0 else None
        outKB_recall_e2e = sum(outKB_TP_e2e) / sum((EL.pwc_url == '0') & (EL.fold == fold)) if sum((EL.pwc_url == '0') & (EL.fold == fold)) != 0 else None
        outKB_f1_e2e = 2 * outKB_prec_e2e * outKB_recall_e2e / (outKB_prec_e2e + outKB_recall_e2e) if outKB_prec_e2e is not None and outKB_recall_e2e is not None else None

        record = {
            'fold': fold,
            # 'GT_CTC outKB prec': outKB_prec_gt_ctc,
            # 'GT_CTC outKB recall': outKB_recall_gt_ctc,
            # 'GT_CTC outKB f1': outKB_f1_gt_ctc,
            'e2e outKB prec': outKB_prec_e2e,
            'e2e outKB recall': outKB_recall_e2e,
            'e2e outKB f1': outKB_f1_e2e,
            'outKB support': len(fold_GT_outKB),
            'inKB support': len(fold_GT_inKB),
            # 'GT_CTC global acc': (sum(outKB_TP_gt_ctc) + sum( (EL.fold==fold) & (EL['rank']==1) )) / sum(EL.fold == fold),
            'e2e gloabl acc': (sum(outKB_TP_e2e) + sum( (EL.fold==fold) & correct_for_EL_idx & (EL['rank']==1) )) / sum(EL.fold == fold),
            'threshold': threshold + 1
        }

        for acc_at_topk in acc_at_topks:
            # record[f'GT_CTC inKB acc@{acc_at_topk}'] = sum( (EL.fold==fold) & (EL['rank'] <= acc_at_topk) ) / sum( (EL.fold==fold) & (EL.pwc_url!='0') )
            record[f'e2e inKB acc@{acc_at_topk}'] = sum( (EL.fold==fold) & correct_for_EL_idx & (EL['rank']<=acc_at_topk) ) / sum( (EL.fold==fold) & (EL.pwc_url!='0') )
        
        records.append(record)

    return pd.DataFrame(records)


def compute_micro_EL_perf_e2e(CTC_with_preds: pd.DataFrame, EL_preds_on_GT_CTC: pd.DataFrame, threshold, acc_at_topks: List[int]):
    EL = EL_preds_on_GT_CTC.merge(CTC_with_preds[['ext_id', 'CTC_preds']], on='ext_id', how='inner')
    EL['rank'] = EL.apply(lambda row: _get_rank(row, threshold), axis=1)
    CTC = EL
    
    # TN_idx = ((CTC.CTC_preds == "Other") | (CTC.CTC_preds == 'Metric')) & ((CTC.cell_type == "Other") | (CTC.cell_type == "Metric"))
    correct_M_idx = (CTC.CTC_preds == CTC.cell_type) & (CTC.cell_type == 'Method')
    correct_D_idx = ((CTC.CTC_preds == 'Dataset') | (CTC.CTC_preds == 'DatasetAndMetric')) & ((CTC.cell_type == 'Dataset') | (CTC.cell_type == 'DatasetAndMetric'))
    correct_for_EL_idx = correct_M_idx | correct_D_idx
    correct_pos_idx = (CTC.cell_type != 'Other') & (CTC.CTC_preds != 'Other')

    GT_inKB, GT_outKB = EL[EL.pwc_url != '0'], EL[EL.pwc_url == '0']
    outKB_TP = (EL.pwc_url == '0') & (EL['rank'] == PRED_MISSING)
    e2e_outKB_prec = sum(outKB_TP & correct_pos_idx) / sum(EL['rank'] == PRED_MISSING) * 100 if sum(EL['rank'] == PRED_MISSING) != 0 else None
    e2e_outKB_recall = sum(outKB_TP & correct_pos_idx) / sum(EL.pwc_url == '0') * 100 if sum(EL.pwc_url == '0') != 0 else None
    e2e_outKB_f1 = 2 * e2e_outKB_prec * e2e_outKB_recall / (e2e_outKB_prec + e2e_outKB_recall) if e2e_outKB_prec is not None and e2e_outKB_recall is not None else None
    gt_ctc_outKB_prec = sum(outKB_TP) / sum(EL['rank'] == PRED_MISSING) * 100 if sum(EL['rank'] == PRED_MISSING) != 0 else None
    gt_ctc_outKB_recall = sum(outKB_TP) / sum(EL.pwc_url == '0') * 100 if sum(EL.pwc_url == '0') != 0 else None
    gt_ctc_outKB_f1 = 2 * gt_ctc_outKB_prec * gt_ctc_outKB_recall / (gt_ctc_outKB_prec + gt_ctc_outKB_recall) if gt_ctc_outKB_prec is not None and gt_ctc_outKB_recall is not None else None
    record = {
        'e2e outKB prec': e2e_outKB_prec,
        'e2e outKB recall': e2e_outKB_recall,
        'e2e outKB f1': e2e_outKB_f1,
        'GT_CTC outKB prec':  gt_ctc_outKB_prec,
        'GT_CTC outKB recall': gt_ctc_outKB_recall,
        'GT_CTC outKB f1': gt_ctc_outKB_f1,
        'outKB support': len(GT_outKB),
        'inKB support': len(GT_inKB),
        'GT_CTC global acc': (sum(outKB_TP) + sum( EL['rank']==1 )) / len(EL),
        'e2e gloabl acc': (sum(outKB_TP & correct_pos_idx) + sum(correct_for_EL_idx & (EL['rank']==1) )) / len(EL),
        'e2e outKB acc': ( sum(outKB_TP & correct_pos_idx) + sum((EL['rank'] != PRED_MISSING) & (EL.pwc_url != '0') & correct_pos_idx) ) / len(EL),
        'threshold': threshold
    }
    for acc_at_topk in acc_at_topks:
        record[f'GT_CTC inKB acc@{acc_at_topk}'] = sum( EL['rank'] <= acc_at_topk ) / sum( EL.pwc_url!='0' )
        record[f'e2e inKB acc@{acc_at_topk}'] = sum( correct_for_EL_idx & (EL['rank']<=acc_at_topk) ) / sum( EL.pwc_url!='0' )
    return pd.DataFrame([record])


def get_false_predictions(CTC_with_preds, EL_preds_on_GT_CTC: pd.DataFrame, cans: pd.DataFrame, top_n_to_rank, threshold):
    
    EL = EL_preds_on_GT_CTC.merge(CTC_with_preds[['ext_id', 'CTC_preds']], on='ext_id', how='inner').copy()
    EL['rank'] = EL.apply(lambda row: _get_rank(row, threshold), axis=1)
    CTC = EL

    correct_M_idx = (CTC.CTC_preds == CTC.cell_type) & (CTC.cell_type == 'Method')
    correct_D_idx = ((CTC.CTC_preds == 'Dataset') | (CTC.CTC_preds == 'DatasetAndMetric')) & ((CTC.cell_type == 'Dataset') | (CTC.cell_type == 'DatasetAndMetric'))
    correct_for_EL_idx = correct_M_idx | correct_D_idx
    correct_pos_idx = (CTC.cell_type != 'Other') & (CTC.CTC_preds != 'Other')

    outKB_TP_gt_ctc = (EL.pwc_url == '0') & (EL['rank'] == PRED_MISSING)
    inKB_TP_gt_ctc = (EL.pwc_url != '0') & (EL['rank']==1)

    # outKB_TP_e2e = outKB_TP_gt_ctc & correct_pos_idx
    # inKB_TP_e2e = inKB_TP_gt_ctc & correct_for_EL_idx

    EL['inKB_TP_gt_ctc'] = inKB_TP_gt_ctc
    EL['outKB_TP_gt_ctc'] = outKB_TP_gt_ctc
    EL['correct_pos_idx'] = correct_pos_idx
    EL['correct_for_EL_idx'] = correct_for_EL_idx

    def _GT_in_can(row):
        return row.pwc_url in row['candidates_100'][:top_n_to_rank]
    cans['GT_in_can'] = cans.apply(_GT_in_can, axis=1)
    
    return EL.merge(cans[['ext_id', 'GT_in_can']], on='ext_id', how='inner')
