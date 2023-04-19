

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