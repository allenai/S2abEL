import warnings
warnings.simplefilter(action='ignore')

from polyfuzz.models import TFIDF
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
from elasticsearch import Elasticsearch
from sentence_transformers import util
import torch
import os
import sys
sys.path.insert(0, sys.path[0] + '/..')


from CTC.TE_loaders import load_single_DataWithTOkenType
from ED.data_loaders import load_bi_encoder_input, padding_bi_encoder_data
from common_utils.common_data_processing_utils import *
from common_utils.common_ML_utils import *
from ED.candidate_gen import clean_string, get_unique_top_k
from ED.utils import generate_relavance_score
from ED.models import CrossEocoder, BiEncoder


def _index(RPI_preds, item):
    if item in RPI_preds:
        return RPI_preds.index(item)
    else:
        return len(RPI_preds)


def get_RPI_accuracy(preds, only_non_missing=False):
    total = 0
    correct = 0
    for i in range(len(preds)):
        pred = preds.loc[i, 'RPI_preds']
        labels = preds.loc[i, 'labels']
        if only_non_missing is True and labels[0] == 0:
            continue
        total += 1
        if labels[0] == 0 and len(pred) == 0:
            correct += 1
        elif all(item in pred for item in labels):
            correct += 1
    return correct / total


def generate_preds_tfidf(top_n: int, n_gram: Tuple[int], BI_cells: pd.DataFrame, ref_extract: pd.DataFrame, threshold=0, at_least_n_from_prediction=8) -> pd.DataFrame:
    """
    Generate RPI predictions using TF-IDF model.
    Setting top_n to a very big value effectively includes all of the matches
    """
    tfidf = TFIDF(n_gram_range=n_gram, model_id="TF-IDF", top_n=top_n)

    ref_extract = ref_extract.fillna('')
    
    predictions = []
    BI_cells = BI_cells.copy()

    for paper_id in tqdm(BI_cells.paper_id.unique()):
        refs = ref_extract[ref_extract.arxiv_id == paper_id]

        to_list = refs['idx'].astype(str) + ' ' + refs['author'] + ' ' + ' ' + refs['year'] + ' ' + refs['title'] + ' ' + refs['abstract']
        # to_list = refs['idx'].astype(str) + ' ' + refs['title'] + ' ' + refs['abstract']
        to_list = to_list.tolist()

        cells = BI_cells[ BI_cells['paper_id'] == paper_id ]
        from_list = cells['cell_content_full'] + ' ' + cells['text']
        from_list = from_list.tolist()

         ######################################################## Create custom TF-IDF vectors
        # vectorizer = TfidfVectorizer(min_df=1, analyzer='char_wb').fit(to_list + from_list)
        # tf_idf_to = vectorizer.transform(to_list)
        # tf_idf_from = vectorizer.transform(from_list)

        # similarity_matrix = cosine_similarity(tf_idf_from, tf_idf_to)
        # indices = np.flip(np.argsort(similarity_matrix, axis=-1), axis=1)[:, :top_n]
        # similarities = np.flip(np.sort(similarity_matrix, axis=-1), axis=1)[:, :top_n]
        # similarities = [np.round(similarities[:, i], 3) for i in range(similarities.shape[1])]

        # columns = (["From"] +
        #            ["To" if i == 0 else f"To_{i+1}" for i in range(top_n)] +
        #            ["Similarity" if i ==0 else f"Similarity_{i+1}" for i in range(top_n)])
        # matches = [[to_list[idx] for idx in indices[:, i]] for i in range(indices.shape[1])]
        # matches = pd.DataFrame(np.vstack(([from_list], matches, similarities)).T, columns = columns)

        # # Update column order
        # columns = [["From", "To", "Similarity"]] + [[f"To_{i+2}", f"Similarity_{i+2}"] for i in range((top_n-1))]
        # matches = matches.loc[:, [title for column in columns for title in column]]

        # # Update types
        # for column in matches.columns:
        #     if "Similarity" in column:
        #         matches[column] = matches[column].astype(float)
        #         matches.loc[matches[column] < 0.001, column] = float(0)
        #         matches.loc[matches[column] < 0.001, column.replace("Similarity", "To")] = None
        
        # # display(matches)
        # rst = matches

        ########################################################

        rst = tfidf.match(from_list, to_list)

        ########################################################

        to_cols = ['To'] + [f"To_{i}" for i in range(2, top_n+1)]
        sim_cols = ['Similarity'] + [f"Similarity_{i}" for i in range(2, top_n+1)]
        rst = rst.apply(lambda row: get_above_threshold(row, to_cols, sim_cols, 'RPI', threshold), axis=1)
        cells['RPI_preds'] = rst.values

        predictions.append(cells)

    predictions = pd.concat(predictions, ignore_index=True)

    def _add_text_reference(row):
        rst = row['RPI_preds'][:at_least_n_from_prediction]
        rst += [i for i in row['text_reference'] if i not in rst]
        if len(row['RPI_preds']) > at_least_n_from_prediction:
            rst += [i for i in row['RPI_preds'][at_least_n_from_prediction:] if i not in rst]
        return rst[:top_n]
    predictions['RPI_preds'] = predictions.apply(_add_text_reference, axis=1)

    digists = re.compile(r'\d+')
    def _shortcut_preds(row):
        if not row.cell_reference.startswith('bibbib'):
            return row.RPI_preds
        nums = digists.findall(row.cell_reference)
        assert len(nums) == 1
        return [int(nums[0])]

    predictions['RPI_preds'] = predictions.apply(_shortcut_preds, axis=1)

    return predictions


def generate_preds_bm25f(es: Elasticsearch, BI_cells: pd.DataFrame, at_least_n_from_prediction: int, index_pofix: str, top_n=1000, clean=False) -> pd.DataFrame:
    """
    index_potfix is '' or '_n_gram'.

    Generate RPI predictions using BM25F from ES.
    Setting top_n to a very big value effectively includes all of the matches
    """
        
    predictions = []

    BI_cells = BI_cells.copy()
    BI_cells['text'] = BI_cells['text'].replace(re.compile(r"xxref-[\w\d-]*"), '')  # handles all in-text references i.e., \cite{}
    BI_cells['text'] = BI_cells['text'].replace(re.compile(r"xxtable-xxanchor-([\w\d-])*"), '')  # handles all table titles
    BI_cells['text'] = BI_cells['text'].replace(re.compile(r"xxanchor-bibbib[\d-]*"), '')  # handles all bibitems
    BI_cells['text'] = BI_cells['text'].replace(re.compile(r"xxanchor-[\w\d-]*"), '')  # handles all bibitems

    if clean:
        BI_cells['text'] = BI_cells['text'].apply(clean_string)
        BI_cells['cell_content_full'] = BI_cells['cell_content_full'].apply(clean_string)

    query = {
        "combined_fields" : {
            "query": " ",
            "fields": ["author", "year", "title", "abstract"]
        }
    }

    for paper_id in tqdm(BI_cells.paper_id.unique()):
        index = paper_id + index_pofix
        cells = BI_cells[ BI_cells['paper_id'] == paper_id ]
        cell_excerpts = cells['cell_content_full'] + ' ' + cells['text']
        
        cell_RPI_preds = []
        for cell_excerpt in cell_excerpts.values:
            query['combined_fields']['query'] = cell_excerpt
            cell_RPI_pred = [int(hit['_source']['idx']) for hit in es.search(index=index, query=query, size=top_n)['hits']['hits']]
            cell_RPI_preds.append(cell_RPI_pred)
        cells['RPI_preds'] = cell_RPI_preds
        predictions.append(cells)
    
    predictions = pd.concat(predictions, ignore_index=True)

    def _add_text_reference(row):
        rst = row['RPI_preds'][:at_least_n_from_prediction]
        rst += [i for i in row['text_reference'] if i not in rst]
        if len(row['RPI_preds']) > at_least_n_from_prediction:
            rst += [i for i in row['RPI_preds'][at_least_n_from_prediction:] if i not in rst]
        return rst[:top_n]
    predictions['RPI_preds'] = predictions.apply(_add_text_reference, axis=1)

    digists = re.compile(r'\d+')
    def _shortcut_preds(row):
        if not row.cell_reference.startswith('bibbib'):
            return row.RPI_preds
        nums = digists.findall(row.cell_reference)
        assert len(nums) == 1
        return [int(nums[0])]

    predictions['RPI_preds'] = predictions.apply(_shortcut_preds, axis=1)

    return predictions


def generate_preds_bm25f_round_robin(es: Elasticsearch, BI_cells: pd.DataFrame, top_n=1000, clean=False, robin_size=1, short_cut_mode:str='replace') -> pd.DataFrame:
    """
    :param short_cut_mode: is 'replace' or 'move_to_head'.
    Generate RPI predictions in an iterative fashion from BM25F predicions and text renferences.

    Setting top_n to a very big value effectively includes all of the matches
    """
        
    predictions = []

    BI_cells = BI_cells.copy()
    BI_cells['text'] = BI_cells['text'].replace(re.compile(r"xxref-[\w\d-]*"), '')  # handles all in-text references i.e., \cite{}
    BI_cells['text'] = BI_cells['text'].replace(re.compile(r"xxtable-xxanchor-([\w\d-])*"), '')  # handles all table titles
    BI_cells['text'] = BI_cells['text'].replace(re.compile(r"xxanchor-bibbib[\d-]*"), '')  # handles all bibitems
    BI_cells['text'] = BI_cells['text'].replace(re.compile(r"xxanchor-[\w\d-]*"), '')  # handles all bibitems

    if clean:
        BI_cells['text'] = BI_cells['text'].apply(clean_string)
        BI_cells['cell_content_full'] = BI_cells['cell_content_full'].apply(clean_string)

    query = {
        "combined_fields" : {
            "query": " ",
            "fields": ["author", "year", "title", "abstract"]
        }
    }

    for paper_id in tqdm(BI_cells.paper_id.unique()):
        index = paper_id
        cells = BI_cells[ BI_cells['paper_id'] == paper_id ]
        cell_excerpts = cells['cell_content_full'] + ' ' + cells['text']
        
        cell_RPI_preds = []
        for cell_excerpt in cell_excerpts.values:
            query['combined_fields']['query'] = cell_excerpt
            cell_RPI_pred = [int(hit['_source']['idx']) for hit in es.search(index=index, query=query, size=top_n)['hits']['hits']]
            cell_RPI_preds.append(cell_RPI_pred)
        cells['RPI_preds'] = cell_RPI_preds
        predictions.append(cells)
    
    predictions = pd.concat(predictions, ignore_index=True)

    def _add_text_reference(row):
        text_references = sorted(row['text_reference'], key=lambda x: _index(row['RPI_preds'], x))
        rst = get_unique_top_k(row['RPI_preds'], text_references, top_n, robin_size=robin_size)
        return rst

    predictions['RPI_preds'] = predictions.apply(_add_text_reference, axis=1)

    digists = re.compile(r'bibbib(\d+)')
    def _shortcut_preds(row):
        nums = digists.findall(row.cell_reference)
        if len(nums) == 0:
            return row.RPI_preds
        assert len(nums) == 1
        if short_cut_mode == 'replace':
            return [int(nums[0])]
        elif short_cut_mode == 'move_to_head':
            return [int(nums[0])] + [i for i in row.RPI_preds if i != int(nums[0])]
        else:
            raise ValueError()

    predictions['RPI_preds'] = predictions.apply(_shortcut_preds, axis=1)

    return predictions


def generate_scores_encoder(RPI_ML: pd.DataFrame, model, output_path=None, BS=64, seed=42, val_fold:str='img_class', single_val_test_fold=False, use_tqdm=False):
    """
    Using a bi-encoder or cross-encoder to generate the RPI scores.
    """
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    g = set_seed(seed)

    config = model.config
    config.seed = seed
    config.use_labels = False
    config.drop_duplicates = False
    config.drop_ones_with_cell_ref = False

    if not single_val_test_fold:
        RPI_ML = RPI_ML[(RPI_ML['fold'] == config.test_fold) | (RPI_ML['fold'] == val_fold)].reset_index(drop=True).copy()
    else:
        RPI_ML = RPI_ML.reset_index(drop=True).copy()

    if isinstance(model, CrossEocoder):
        ds = load_single_DataWithTOkenType(config, RPI_ML, drop_duplicates=False)
        dl = DataLoader(ds, batch_size=BS, collate_fn=padding_by_batch, worker_init_fn=seed_worker, generator=g)
    elif isinstance(model, BiEncoder):
        ds = load_bi_encoder_input(config, RPI_ML)
        dl = DataLoader(ds, batch_size=BS, collate_fn=padding_bi_encoder_data, worker_init_fn=seed_worker, generator=g)
    else:
        raise ValueError("model must be either a CrossEncoder or BiEncoder")
        
    _, relevance_score = generate_relavance_score(model, dl, use_tqdm)
    RPI_ML['ASM_scores'] = relevance_score
    if output_path is not None:
        RPI_ML.to_pickle(output_path)
    return RPI_ML


def enhance_preds_encoder(RPI_ML_scores: pd.DataFrame, output_path=None, enhance=True):
    """
    Generate RPI_preds from encoders RPI_scores, then enhance with cell_reference
    """

    records = []
    for ext_id in tqdm(RPI_ML_scores.ext_id.unique()):
        ext = RPI_ML_scores[RPI_ML_scores['ext_id'] == ext_id]
        ext = ext.sort_values(by='ASM_scores', ascending=False, ignore_index=True, axis=0)
        RPI_preds = ext.apply(lambda row: int(row['idx']), axis=1).tolist()
        records.append({
            'ext_id': ext_id,
            'ASM_preds': RPI_preds,
            'paper_id': ext.iloc[0]['paper_id'],
            'fold': ext.iloc[0]['fold'],
            'cell_type': ext.iloc[0]['cell_type'],
            'cell_content': ext.iloc[0]['cell_content'],
            # 'cell_content_full': ext.iloc[0]['cell_content_full'],
            'cell_reference': ext.iloc[0]['cell_reference'],
            # 'text_reference': ext.iloc[0]['text_reference'],
            'pwc_url': ext.iloc[0]['pwc_url'],
            # 'bib_entries': ext.iloc[0]['bib_entries'],
        })

    RPI_ML_preds = pd.DataFrame(records)
    assert len(RPI_ML_preds) == RPI_ML_scores.ext_id.nunique()

    def _shortcut_preds(row, col_name):
        if row.cell_reference == '':
            return getattr(row, col_name)
        else:
            return [int(row.cell_reference)]
    
    if enhance:
        # use the in-cell reference info to short cut predictions
        RPI_ML_preds['ASM_preds'] = RPI_ML_preds.apply(lambda row: _shortcut_preds(row, 'ASM_preds'), axis=1)

    if output_path is not None:
        RPI_ML_preds.to_pickle(output_path)
    return RPI_ML_preds

