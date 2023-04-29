from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict
import re
import numpy as np
from functools import partial
from itertools import chain
import pandas as pd
import re
from polyfuzz.models._utils import cosine_similarity
from tqdm import tqdm
from sentence_transformers import util
from torch.utils.data import DataLoader


from ED.data_loaders import load_bi_encoder_input, padding_bi_encoder_data, load_triplet_encoder_input
from common_utils.common_data_processing_utils import *
from common_utils.common_ML_utils import *


def mix_candidate_set(EL_RPI: pd.DataFrame, EL_RPI_col:str, EL_direct_search: pd.DataFrame, EL_direct_search_col: str, top_ks_each: List[int]):
    """ Mix candidate set from different methods.
    :param EL_RPI: should the column RPI_preds_candidates_top_<n>
    :param EL_direct_search: should have the column ent_candidates with 100 candidates
    :param EL_RPI_col: the column name prefix of EL_RPI containing candidates e.g., 'RPI_preds_candidates_top_'

    :return 
    """

    rst = EL_RPI.merge(EL_direct_search[ ['cell_id', EL_direct_search_col] ], on='cell_id', how='inner')

    # assert rst.fold.nunique() <= 2
    
    def _get_mixed(row, top_k_each):
        # return list(set(row[f'{EL_RPI_col}{top_k_each}'] + row[EL_direct_search_col][:top_k_each]))
        # can1, can2 = row[f'{EL_RPI_col}{top_k_each}'], row[EL_direct_search_col][:top_k_each]
        can1, can2 = row[EL_RPI_col], row[EL_direct_search_col][:top_k_each]
        rst = []
        c = 0
        for _ in range(min(len(can1), len(can2))):
            if can1[c] not in rst:
                rst.append(can1[c])
            if can2[c] not in rst:
                rst.append(can2[c])
            c += 1
        if c == len(can1):
            rst += [j for j in can2 if j not in rst]
        else:
            rst += [j for j in can1 if j not in rst]
        return rst

    for top_k_each in top_ks_each:
        col = f'candidates_{top_k_each}'
        rst[col] = rst.apply(lambda row: _get_mixed(row, top_k_each), axis=1)
        # cols_to_keep.append(col)
    for col in ['RPI_preds',
       'RPI_preds_candidates_top_10', 'RPI_preds_candidates_top_25',
       'RPI_preds_candidates_top_50', 'RPI_preds_candidates_top_75',
       'RPI_preds_candidates_top_100', 'ent_candidates']:
            if col in rst.columns:
                rst = rst.drop(columns=[col])
    return rst


def clean_string(string: str) -> str:
    """ Only keep alphanumerical characters """
    string = re.sub(r'[^A-Za-z0-9 ]+', '', string.lower())
    string = re.sub('\s+', ' ', string).strip()
    return string


def create_ngrams(n_gram_range: Tuple[int, int], clean_string: bool, string: str) -> List[str]:
    """ Create n_grams from a string

    Steps:
        * Extract character-level ngrams with `n_gram_range` (both ends inclusive)
        * Remove n-grams that have a whitespace in them
    """
    if clean_string:
        string = clean_string(string)

    result = []
    for n in range(n_gram_range[0], n_gram_range[1]+1):
        ngrams = zip(*[string[i:] for i in range(n)])
        ngrams = [''.join(ngram) for ngram in ngrams if ' ' not in ngram]
        result.extend(ngrams)

    return result


def extract_tf_idf(_list: List[List[str]], n_gram_range : Tuple[int, int], clean_string: bool) -> np.ndarray:
    """ Calculate distances between TF-IDF vectors of from_list and to_list """
    analyzer = partial(create_ngrams, n_gram_range, clean_string)
    vectorizer = TfidfVectorizer(min_df=1, analyzer=analyzer)
    vocab = list(chain(*_list))
    vectorizer = vectorizer.fit(vocab)
   
    return  [vectorizer] + [vectorizer.transform(_l) for _l in _list]


def generate_tfidf_candidates(EL: pd.DataFrame, vectorizer: TfidfVectorizer, ent_vecs: np.ndarray, ent_list: List[str], top_n: int=100):
    # remove special tokens used by transformers
    p = re.compile(r"\[unused\d+\]")
    p2 = re.compile(r"\[SEP\]")
    text_sentence_no_special_token = EL['text_sentence_no_mask'].apply(lambda x: re.sub(p, '', x)).apply(lambda x: re.sub(p2, '', x))

    from_list = EL['cell_content_full'] + ' ' + text_sentence_no_special_token
    from_list = from_list.tolist()
    from_vector = vectorizer.transform(from_list)

    matches = cosine_similarity(from_vector, ent_vecs, from_list, ent_list, top_n=top_n)

    to_cols = ['To'] + [f"To_{i}" for i in range(2, top_n+1)]
    sim_cols = ['Similarity'] + [f"Similarity_{i}" for i in range(2, top_n+1)]
    matches = matches.apply(lambda row: get_above_threshold(row, to_cols, sim_cols, 'ENT'), axis=1)

    return pd.DataFrame(matches.values)


def generate_BM25F_candidates(BI_cells: pd.DataFrame, EL: pd.DataFrame, method_idx:str, dataset_idx:str, es, clean=True):

    EL = EL.merge(BI_cells[['ext_id', 'text']], how='inner', on='ext_id')
    EL['text'] = EL['text'].replace(re.compile(r"xxref-[\w\d-]*"), '')  # handles all in-text references i.e., \cite{}
    EL['text'] = EL['text'].replace(re.compile(r"xxtable-xxanchor-([\w\d-])*"), '')  # handles all table titles
    EL['text'] = EL['text'].replace(re.compile(r"xxanchor-bibbib[\d-]*"), '')  # handles all bibitems
    EL['text'] = EL['text'].replace(re.compile(r"xxanchor-[\w\d-]*"), '')  # handles all bibitems

    if clean:
        EL['text'] = EL['text'].apply(clean_string)
        EL['cell_content_full'] = EL['cell_content_full'].apply(clean_string)

    cell_excerpts = EL['cell_content_full'] + ' ' + EL['text']

    query = {
        "combined_fields" : {
            "query": " ",
            "fields": ["name", "full_name", "description"]
        }
    }

    method_rst, dataset_rst = [], []
    for cell_excerpt in tqdm(cell_excerpts):
        query['combined_fields']['query'] = cell_excerpt
        dataset_resp = es.search(index=dataset_idx, query=query, size=100)
        method_resp = es.search(index=method_idx, query=query, size=100)
        dataset_rst.append([hit['_source']['url'] for hit in dataset_resp['hits']['hits']])
        method_rst.append([hit['_source']['url'] for hit in method_resp['hits']['hits']])

    return method_rst, dataset_rst


def get_unique_top_k(l1:List[str], l2:List[str], top_n, robin_size=1):
    """
    Evenly get candidates from l1 and l2
    """

    pointer = 0
    rst = []
    while len(rst) < top_n and pointer < len(l1) and pointer < len(l2):
        rst += [i for i in l1[pointer: pointer + robin_size] if i not in rst]
        rst += [i for i in l2[pointer: pointer + robin_size] if i not in rst]
        pointer += robin_size
    
    if len(rst) >= top_n:
        return rst[:top_n]
    
    if pointer >= len(l1) and pointer >= len(l2):
        return rst
    elif pointer >= len(l1):
        rst += [i for i in l2[pointer:] if i not in rst]
    elif pointer >= len(l2):
        rst += [i for i in l1[pointer:] if i not in rst]
    else:
        print(len(l1))
        print(len(l2))
        print(len(rst))
        print(pointer)
        raise ValueError("bug???")    
    return rst[:top_n]


def generate_BM25F_candidates_combined_idx(EL: pd.DataFrame, es, top_n=100):
    """
    Generate entities from both word-gram index and n-gram index, half from each.
    """
    # remove special tokens used by transformers
    p = re.compile(r"\[unused\d+\]")
    p2 = re.compile(r"\[SEP\]")
    text_sentence_no_special_token = EL['text_sentence_no_mask'].apply(lambda x: re.sub(p, '', x)).apply(lambda x: re.sub(p2, '', x))

    cell_excerpts = EL['cell_content_full'] + ' ' + text_sentence_no_special_token

    query = {
        "combined_fields" : {
            "query": " ",
            "fields": ["name", "full_name", "description"]
        }
    }

    method_rst, dataset_rst = [], []
    for cell_excerpt in tqdm(cell_excerpts):
        query['combined_fields']['query'] = cell_excerpt
        dataset_resp_1 = [hit['_source']['url'] for hit in es.search(index='datasets', query=query, size=2*top_n)['hits']['hits']]
        dataset_resp_2 = [hit['_source']['url'] for hit in es.search(index='datasets_n_gram', query=query, size=2*top_n)['hits']['hits']]
        method_resp_1 = [hit['_source']['url'] for hit in es.search(index='methods', query=query, size=2*top_n)['hits']['hits']]
        method_resp_2 = [hit['_source']['url'] for hit in es.search(index='methods_n_gram', query=query, size=2*top_n)['hits']['hits']]

        dataset_rst.append(get_unique_top_k(dataset_resp_1, dataset_resp_2, top_n))
        method_rst.append(get_unique_top_k(method_resp_1, method_resp_2, top_n))

    return method_rst, dataset_rst


def get_ASR_candidates_encoders(EL: pd.DataFrame, ref_extract:pd.DataFrame, ASM_pred_cols: List[str], top_ns: List[int], output_cols:List[str]=None):
    """
    Extend with candidates using ASM predicitions from encoders.
    EL has columns: RPI_preds
    """
    EL = EL.copy()
    EL['arxiv_id'] = EL['cell_id'].apply(lambda x: x.split('/')[0])
    ref_extract_arxiv_id = ref_extract['ref_id'].apply(lambda x: x.split('/')[0]).unique()
    # somehow 5 rows are missing.
    EL = EL[ (EL.cell_id!='1612.04211v1/table_02.csv/14/1') & (EL.cell_id!='1809.06309v2/table_03.csv/0/1') &
        (EL.cell_id!='1809.06309v2/table_03.csv/0/2') & (EL.cell_id!='1904.03288v1/table_02.csv/11/1') & (EL.cell_id!='1507.06228v2/table_01.csv/2/0')]
    EL = EL[EL['arxiv_id'].isin(ref_extract_arxiv_id)]
    EL = EL.reset_index(drop=True)

    for top_n in tqdm(top_ns):
        for idx, col in enumerate(ASM_pred_cols):
            def _get_ent(row):
                if row.cell_type.lower() == 'method':
                    attibute = 'related_methods'
                elif row.cell_type.lower() == 'dataset' or row.cell_type == 'dataset&metric':
                    attibute = 'related_datasets'
                else:
                    assert 0
                RPI_preds = getattr(row, col)
                if len(RPI_preds) == 1 and RPI_preds[0] == 0:
                    return []
                rst = []
                for pred in RPI_preds:
                    ref_id = f"{row.paper_id}/{pred}"
                    rst += [i for i in ref_extract[ref_extract.ref_id == ref_id][attibute].item() if i not in rst]
                    if top_n is not None and len(rst) > top_n and len(rst) > 0:
                        break
                return rst
            if output_cols is None:
                EL[f'{col}_candidates_top_{top_n}'] = EL.apply(lambda x: _get_ent(x), axis=1)
            else:
                EL[output_cols[idx]] = EL.apply(lambda x: _get_ent(x), axis=1)

    return EL


# def extend_with_RPI_candidates(EL: pd.DataFrame, ref_extract:pd.DataFrame, BI_cells_preds: pd.DataFrame, top_ns: List[int]):
#     """
#     DEPRECATED!!

#     Setting top_n to None means including all candidates from the RPI.
#     :param BI_cells_preds: has columns  'RPI_preds'.
#     """
#     EL = EL.copy()
#     # somehow 5 rows are missing.
#     EL = EL[ (EL.ext_id!='1612.04211v1/table_02.csv/14/1') & (EL.ext_id!='1809.06309v2/table_03.csv/0/1') &
#         (EL.ext_id!='1809.06309v2/table_03.csv/0/2') & (EL.ext_id!='1904.03288v1/table_02.csv/11/1') & (EL.ext_id!='1507.06228v2/table_01.csv/2/0')]
#     EL = EL.reset_index(drop=True)

#     EL = EL.merge(BI_cells_preds[ ['ext_id', 'RPI_preds'] ], on='ext_id', how='left')

#     ## Get methods
#     for top_n in tqdm(top_ns):

#         def _get_ent(row):
#             if row.cell_type == 'Method':
#                 return _get_RPI_method(row)
#             else:
#                 return _get_RPI_dataset(row)

#         def _get_RPI_method(row):
#             rst = set()
#             for pred in row.RPI_preds:
#                 ref_id = f"{row.paper_id}/{pred}"
#                 tmp = rst.copy()
#                 tmp.update(ref_extract[ref_extract.ref_id == ref_id].related_methods.item())
#                 if top_n is not None and len(tmp) > top_n and len(rst) > 0:
#                     break
#                 else:
#                     rst = tmp
#             return list(rst)
    
#         def _get_RPI_dataset(row):
#             rst = set()
#             for pred in row.RPI_preds:
#                 ref_id = f"{row.paper_id}/{pred}"
#                 tmp = rst.copy()
#                 tmp.update(ref_extract[ref_extract.ref_id == ref_id].related_datasets.item())
#                 if top_n is not None and len(tmp) > top_n and len(rst) > 0:
#                     break
#                 else:
#                     rst = tmp
#             return list(rst)

#         EL[f'RPI_preds_candidates_top_{top_n}'] = EL.apply(lambda x: _get_ent(x), axis=1)

#     return EL


def compute_bi_enc_ent_candidates(model, EL:pd.DataFrame, ent_embeds: Dict[str, torch.FloatTensor], pwc_ents: pd.DataFrame, val_fold:str='img_class', BS=64, seed=42, top_k=100, output_path=None,
                                    mode:str='Contrastive', single_fold_val_test=False, mix_methods_datasets=False):
    """
    Use a bi-encoder model to generate entity candidates for each cell in the EL dataframe.

    :param mode: 'Contrastive' or 'Triplet'
    :parm ent_embeds: dict with keys 'Method' and 'Dataset' and values the entity embeddings for each type.
    :return EL: EL.ent_candidates is a Series of List[url]
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    model.to(device)
    set_seed(seed)

    config = model.config
    config.seed = seed
    config.ent_embedding = None
    config.use_labels = False
    config.drop_duplicates = False

    if not single_fold_val_test:
        EL = EL[(EL.fold == val_fold) | (EL.fold == config.test_fold)].copy().reset_index(drop=True)
    else:
        EL = EL[EL.fold == config.test_fold].copy().reset_index(drop=True)
    if mode =='Contrastive':
        ds = load_bi_encoder_input(config, EL)
    elif mode == 'Triplet':
        config.pos_ent_embedding = None
        config.neg_ent_embedding = None
        ds = load_triplet_encoder_input(config, EL)
    else:
        raise ValueError(f"mode must be 'Contrastive' or 'Triplet', got {mode}")
    dl = DataLoader(ds, batch_size=BS, collate_fn=padding_bi_encoder_data)

    cell_embeds = []
    for batch in tqdm(dl):
        with torch.no_grad():
            batch['output_emb'] = True
            outputs = model(**batch)
            cell_embeds.append(outputs.cell_emb.cpu())
    cell_embeds = torch.cat(cell_embeds, dim=0)
    EL['cell_embeds'] = cell_embeds.tolist()

    def _get_hits(row):
        cell_embed = np.asarray(row.cell_embeds).astype('float32')
        if not mix_methods_datasets:
            if row.cell_type.lower() == 'method':
                hits = util.semantic_search(cell_embed, ent_embeds['method'], top_k=top_k, score_function=lambda x, y: sim_func(model, x, y))[0]
            else:
                hits = util.semantic_search(cell_embed, ent_embeds['dataset'], top_k=top_k, score_function=lambda x, y: sim_func(model, x, y))[0]
        else:
            raise NotImplementedError('mix_methods_datasets not implemented yet')
            # hits = util.semantic_search(cell_embed, ent_embeds['Method'] + ent_embeds['Dataset'], top_k=top_k, score_function=lambda x, y: sim_func(model, x, y))[0]
        pbar.update(1)
        return hits
    
    pbar = tqdm(total=len(EL))
    EL['hits'] = EL.apply(_get_hits, axis=1)
    pbar.close()
    EL = EL[['cell_id', 'fold', 'hits', 'cell_type', 'pwc_url', 'cell_content']]

    def _convert_to_ent_url(row):
        if not mix_methods_datasets:
            ent_set = pwc_ents[pwc_ents.type == 'method'] if row.cell_type == 'method' else pwc_ents[pwc_ents.type == 'dataset']
        else:
            ent_set = pwc_ents
        return [ent_set.iloc[h['corpus_id']]['url'] for h in row.hits]

    EL['DR_candidates'] = EL.apply(_convert_to_ent_url, axis=1)

    if output_path is not None:
        EL.to_pickle(output_path)

    return EL


def test_candidate_effectiveness(EL: pd.DataFrame, method_can_col: str, dataset_can_col: str, top_n=None, verbose=False):
    """
    :param top_n: top n candidates to consider, None to consider all candidates in that column
    """
    precision_sum = total_non_emp_non_missing_count = valid_candidate_ent_count = in_effective_count = out_count = 0
    
    for _, row in EL.iterrows():
        if row.pwc_url == '0':
            out_count += 1
        
        dataset_can = row[dataset_can_col]
        method_can = row[method_can_col]
        if top_n is not None:
            dataset_can = dataset_can[:top_n]
            method_can = method_can[:top_n]

        if row.cell_type.lower() == 'method':
            valid_candidate_ent_count += len(method_can)
            if len(method_can) != 0 and row.pwc_url != '0':
                total_non_emp_non_missing_count += 1
            if row.pwc_url in method_can:
                in_effective_count += 1
                precision_sum += 1 / len(method_can)
            elif verbose and row.pwc_url != '0':
                print(row.cell_content, row.pwc_url, method_can)
        else:
            valid_candidate_ent_count += len(dataset_can)
            if len(dataset_can) != 0 and row.pwc_url != '0':
                total_non_emp_non_missing_count += 1
            if row.pwc_url in dataset_can:
                in_effective_count += 1
                precision_sum += 1 / len(dataset_can)
            elif verbose and row.pwc_url != '0':
                print(row.cell_content, row.pwc_url, dataset_can)
    
    # inKB_precision@k, inKB_recall@k, avg_can_size, _, _, inKB_count
    return precision_sum / len(EL[EL.pwc_url != '0'])*100, in_effective_count / len(EL[EL.pwc_url != '0'])*100, valid_candidate_ent_count / len(EL), in_effective_count / total_non_emp_non_missing_count, total_non_emp_non_missing_count, len(EL[EL.pwc_url == '0'])