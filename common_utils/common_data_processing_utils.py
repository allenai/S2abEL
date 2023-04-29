import re
from unidecode import unidecode
from common_utils.common_ML_utils import add_token_mapping, LabelsExt


cell_rep_features = ['region_type', 'row_pos', 'reverse_row_pos', 
                    'col_pos', 'reverse_col_pos', 'has_reference', 
                    'cell_content', 'row_context', 'col_context', 
                    'context_sentences']
paper_rep_features = ['idx', 'year', 'author', 'title', 'abstract']




def _get_table(x, paper_map):
    paper_id, table_name = x['cell_id'].split('/')[0:2]
    tables = paper_map[paper_id]['tables']
    table = tables[ f'{paper_id}/{table_name}' ]
    return table

idx_re = re.compile(r'\d+')
def get_reverse_row(x, paper_map):
    table = _get_table(x, paper_map)
    row = len(table)
    return str(row - int(x['row_pos']) - 1)


def get_reverse_col(x, paper_map):
    table = _get_table(x, paper_map)
    col = len(table[0])
    return str(col - int(x['col_pos']) - 1)


style_tags_re = re.compile(r"</?(bold|italic|red|green|blue)>")
def remove_text_styles(s):
    return style_tags_re.sub("", s)


empty_paren_re = re.compile(r"\(\s*\)|\[\s*\]")
reference_re = re.compile(r"<ref id='([^']*)'>(.*?)</ref>")
def remove_references(s):
    s = reference_re.sub("", s)
    return empty_paren_re.sub("", s)


def cleanup_str(string):
    return remove_references(remove_text_styles(string)).replace("\xa0", " ")


def get_cell_content(x, paper_map):
    table = _get_table(x, paper_map)
    return cleanup_str(table[int(x['row_pos'])][int(x['col_pos'])])


ref_pattern = r'bibbib\d+'
def get_refs(x, paper_map):
    table = _get_table(x, paper_map)
    raw_cell_content = table[int(x['row_pos'])][int(x['col_pos'])]
    parts = reference_re.split(raw_cell_content)
    refs = [r.replace('-', '') for r in parts[1::3]]
    refs = [int(r[6:]) for r in refs if re.match(ref_pattern, r) is not None]
    if len(refs) == 0:
        return ''
    else:
        return refs[0]


def get_has_reference(x, paper_map):
    table = _get_table(x, paper_map)
    return int('bib-bib' in table[int(x['row_pos'])][int(x['col_pos'])])


def get_row_context(x, paper_map):
    table = _get_table(x, paper_map)
    row = table[int(x['row_pos'])]
    row = [cleanup_str(i) for i in row]
    row = [add_token_mapping['[EMPTY]'] if i == '' else i for i in row] 
    return f" {add_token_mapping['[BORDER]']} ".join(row)


def get_col_context(x, paper_map):
    table = _get_table(x, paper_map)
    col = [row[int(x['col_pos'])] for row in table]
    col = [cleanup_str(i) for i in col]
    col = [add_token_mapping['[EMPTY]'] if i == '' else i for i in col] 
    return f" {add_token_mapping['[BORDER]']} ".join(col)


def get_context_sentences(x, paper_map):
    paper_id = x['cell_id'].split('/')[0]
    context_sentences = [paper_map[paper_id]['raw_text'][s:e] for s, e in x['context_spans']]
    context_sentences = " [SEP] ".join(context_sentences)
    context_sentences = re.compile(r"(\</?b\>)").sub(r" \1 ", context_sentences)
    context_sentences = re.compile(r"xxref-[\w\d-]*").sub(add_token_mapping['[REF]'], context_sentences)  # handles all in-text references i.e., \cite{}
    context_sentences = re.compile(r"xxtable-xxanchor-([\w\d-])*").sub(add_token_mapping['[TABLE_TITLE]'], context_sentences)  # handles all table titles
    context_sentences = re.compile(r"xxanchor-bibbib[\d-]*").sub(add_token_mapping['[BIB_ITEM]'], context_sentences)  # handles all bibitems
    context_sentences = re.compile(r"xxanchor-[\w\d-]*").sub(add_token_mapping['[SEC_OR_FIG_TITLE]'], context_sentences)  # handles all bibitems
    context_sentences = re.compile(r"\bdata set\b").sub(" dataset ", context_sentences)
    return context_sentences


labels_ext_map = {
    'other': LabelsExt.OTHER.value,
    'dataset': LabelsExt.DATASET.value,
    'method': LabelsExt.METHOD.value,
    'metric': LabelsExt.METRIC.value,
    'dataset&metric': LabelsExt.DATASET_AND_METRIC.value,
}


def assemble_ctc_data(cell_types, papers):
    ctc = cell_types.copy()
    ctc['row_pos'] = ctc['cell_id'].apply(lambda x: x.split('/')[-2])
    ctc['col_pos'] = ctc['cell_id'].apply(lambda x: x.split('/')[-1])
    paper_map = {}
    for _, row in papers.iterrows():
        paper_map[row['arxiv_id']] = row
    ctc['reverse_row_pos'] = ctc.apply(lambda x:  get_reverse_row(x, paper_map), axis=1)
    ctc['reverse_col_pos'] = ctc.apply(lambda x:  get_reverse_col(x, paper_map), axis=1)
    ctc['fold'] = ctc['cell_id'].apply(lambda x: paper_map[x.split('/')[0]]['topic'])
    ctc['cell_content'] = ctc.apply(lambda x: get_cell_content(x, paper_map), axis=1)
    ctc['has_reference'] = ctc.apply(lambda x: get_has_reference(x, paper_map), axis=1)
    ctc['row_context'] = ctc.apply(lambda x: get_row_context(x, paper_map), axis=1)
    ctc['col_context'] = ctc.apply(lambda x: get_col_context(x, paper_map), axis=1)
    ctc['cell_reference'] = ctc.apply(lambda x: get_refs(x, paper_map), axis=1)
    ctc['context_sentences'] = ctc.apply(lambda x: get_context_sentences(x, paper_map), axis=1)
    del ctc['context_spans']

    ctc['labels'] = ctc['cell_type'].apply(lambda x: labels_ext_map[x])
    # print(paper_map['1606.02270v2']['tables']['1606.02270v2/table_02.csv'][0])

    return ctc





def get_above_threshold(row, to_cols, sim_cols, mode, threshold=None):
    """
    Function used to get rst from polyfuzzy matches.
    mode = 'RPI' or 'ENT'
    """
    rst = []
    for to, sim in zip(to_cols, sim_cols):
        if sim not in row:
            break
        if row[sim] is not None and (threshold is None or row[sim] > threshold):
            if row[to] is not None:
                if mode == 'RPI':
                    rst.append(int(row[to].split(' ')[0]))
                elif mode == 'ENT':
                    # rst.append(row[to].split('/')[-1])
                    rst.append(row[to])
                else:
                    raise ValueError('mode must be RPI or ENT')
    return rst


def split_fold(df, valid_split, test_split):
    is_test = df.fold == test_split
    is_valid = df.fold == valid_split
    test_df = df[is_test].copy()
    valid_df = df[is_valid].copy()
    train_df = df[(~is_test) & (~is_valid)].copy()
    return train_df, valid_df, test_df


def split_EL_data(df, test_fold):
    "Half of the papers in the test fold is used for validation. The other half is used for testing."
    test_df = df[(df.fold == test_fold) & (df.is_val == False)].copy()
    valid_df = df[(df.fold == test_fold) & (df.is_val == True)].copy()
    train_df = df[(df.fold != test_fold)].copy()
    return train_df, valid_df, test_df


def padding_by_batch(features, pad_labels=False, special_attn=False):
    from torch.nn.utils.rnn import pad_sequence
    import torch
    import torch.nn.functional as F

    first = features[0]
    batch = {}

    no_pad_keys = ['num_rows', 'num_cols', 'labels', 'spans_or_locations', 'row_id', 'col_id', 'has_reference_ids', 'reverse_row_id', 'reverse_col_id', 'region_type']
    pad_keys_1 = ['is_empty_ids', 'input_text_sentence_embeds', 'input_type_ids', 'input_ids']
    pad_keys_2 = ['input_cell_content_ids', 'input_text_sentence_ids']
   
    for k, v in first.items():
        if k not in no_pad_keys: continue
        if k == 'labels' and pad_labels: continue
        if isinstance(v, torch.Tensor):
            batch[k] = torch.stack([f[k] for f in features])
        else:
            batch[k] = torch.tensor([f[k] for f in features])
    
    input_key = 'input_ids' if 'input_ids' in first.keys() else 'input_embeds'
    if special_attn:
        input_key = 'input_cell_content_ids'
    attention_masks = [torch.ones(len(f[input_key])) for f in features]
    batch['attention_mask'] = pad_sequence(attention_masks, batch_first=True)

    if pad_labels:
        batch['labels'] = pad_sequence([f['labels'] for f in features], batch_first=True, padding_value=-100)

    for k, _ in first.items():
        if k not in pad_keys_1: continue
        batch[k] = pad_sequence([f[k] for f in features], batch_first=True)
    
    for k, _ in first.items():
        if k not in pad_keys_2: continue

        # max_cell_count = max([len(f[k]) for f in features])
        max_seq_length = max([len(l) for f in features for l in f[k]])

        seq_padded = [torch.stack([F.pad(l, [0, max_seq_length - l.size(0)]) for l in f[k]]) for f in features]
        batch[k] = pad_sequence(seq_padded, batch_first=True)
        
        attn_mask = [[torch.ones_like(l) for l in f[k]] for f in features]
        attn_mask = [torch.stack([F.pad(k, [0, max_seq_length - k.size(0)]) for k in f]) for f in attn_mask]
        batch[k+"_attn_masks"] = pad_sequence(attn_mask, batch_first=True)

    return batch