import re
from enum import Enum
from functools import partial
from tqdm import tqdm


from common_utils.common_data_processing_utils import *
from common_utils.common_ML_utils import *
from CTC.TE_loaders import load_single_DataWithTOkenType



class LabelsExt(Enum):
    OTHER=0
    DATASET=1
    METHOD=2
    METRIC=3
    DATASET_AND_METRIC=4


labels_ext_map = {
    'Other': LabelsExt.OTHER.value,
    'Dataset': LabelsExt.DATASET.value,
    'Method': LabelsExt.METHOD.value,
    'Metric': LabelsExt.METRIC.value,
    'DatasetAndMetric': LabelsExt.DATASET_AND_METRIC.value,
    'MethodAndMetric': LabelsExt.METHOD.value,
    '': LabelsExt.OTHER.value
}


class Labels(Enum):
    OTHER=0
    DATASET=1
    METHOD=2
    METRIC=3

label_map_reverse = {
    0: "Other",
    1: "Dataset",
    2: "Method",
    3: "Metric",
    4: "DatasetAndMetric"
}

label_map = {
    "dataset": Labels.DATASET.value,
    "dataset-sub": Labels.DATASET.value,
    "model-paper": Labels.METHOD.value,
    "model-best": Labels.METHOD.value,
    "model-ensemble": Labels.METHOD.value,
    "model-competing": Labels.METHOD.value,
    "dataset-metric": Labels.METRIC.value,
    'Metric': Labels.METRIC.value,
    'Dataset': Labels.DATASET.value,
    'Method': Labels.METHOD.value
}


def transform_df(config, *dfs):
    transformed = [_transform_df(config, df) for df in dfs]
    if len(transformed) == 1:
        return transformed[0]
    return transformed


def _transform_df(config, df):
        df = df.copy(True)
        df = df[ (df['cell_content'] != '') | (df['cell_reference'] != '')]
        # df.row_context.replace(re.compile("border"), add_token_mapping['[ROW_BORDER]'], inplace=True)
        # df.col_context.replace(re.compile("border"), add_token_mapping['[COL_BORDER]'], inplace=True)

        # df['has_reference'] = (df.cell_reference != '')
        # df.cell_styles = df.cell_styles.astype(str)
        if config.merge_type not in ["concat", "vote_maj", "vote_avg", "vote_max"]:
            raise Exception(f"merge_type must be one of concat, vote_maj, vote_avg, vote_max, but {config.merge_type} was given")
        if config.mark_this_paper and (config.merge_type != "concat" or config.this_paper):
            raise Exception("merge_type must be 'concat' and this_paper must be false")
        if config.evidence_limit is not None:
            df = df.groupby(by=["ext_id", "this_paper"]).head(config.evidence_limit)
        if config.context_tokens is not None:
            df.loc["text_highlited"] = df["text_highlited"].apply(config._limit_context)
            df.loc["text"] = df["text_highlited"].str.replace("<b>", " ").replace("</b>", " ")
        

        duplicates_columns = ["text", "cell_content", "cell_type", "row_context", "col_context", "cell_reference", "cell_layout", "cell_styles"]
        columns_to_keep = ["ext_id", "cell_content", "cell_type", "row_context", "col_context", "cell_reference", "cell_layout", "cell_styles", "fold"]

        if config.this_paper:
            df = df[df.this_paper]
        if config.merge_fragments and config.merge_type == "concat":
            df1 = df.groupby(by=columns_to_keep)['text'].apply(
                lambda x: " [SEP] ".join(x.values)).reset_index()
            df2 = df.groupby(by=columns_to_keep)['text_highlited'].apply(
                lambda x: " [SEP] ".join(x.values)).reset_index()
            df1['text_highlited'] = df2['text_highlited']
            df = df1
            
        if config.drop_duplicates:
            df = df.drop_duplicates(duplicates_columns).fillna("")
        
        df['text_sentence_masked'] = df['text_highlited'].replace(re.compile("<b>.*?</b>"), " [MASK] ")
        df['text_sentence_no_mask'] = df['text']
        # <b>a</b> -> <b> a </b>
        df['text_sentence_no_mask'].replace(re.compile(r"(\</?b\>)"), r" \1 ", inplace=True)

        # Replace references with generic ref
        # df = df.replace(re.compile(r"(xxref|xxanchor)-[\w\d-]*"), r"\1")

        df = df.replace(re.compile(r"xxref-[\w\d-]*"), add_token_mapping['[REF]'])  # handles all in-text references i.e., \cite{}
        df = df.replace(re.compile(r"xxtable-xxanchor-([\w\d-])*"), add_token_mapping['[TABLE_TITLE]'])  # handles all table titles
        df = df.replace(re.compile(r"xxanchor-bibbib[\d-]*"), add_token_mapping['[BIB_ITEM]'])  # handles all bibitems
        df = df.replace(re.compile(r"xxanchor-[\w\d-]*"), add_token_mapping['[SEC_OR_FIG_TITLE]'])  # handles all bibitems
        
        
        if config.remove_num:
            raise ValueError("Should not be used")
            df = df.replace(re.compile(r"(^|[ ])\d+\.\d+(\b|%)"), " xxnum ")
            df = df.replace(re.compile(r"(^|[ ])\d+(\b|%)"), " xxnum ")

        df = df.replace(re.compile(r"\bdata set\b"), " dataset ")

        return df


with_letters_re = re.compile(r"(?:^\s*[a-zA-Z])|(?:[a-zA-Z]{2,})")
def keep_alphacells(df):
    which = df.cell_content.str.contains(with_letters_re)
    return df[which], df[~which]


def align_df(df, batch_size):
    aligned_len = ( len(df) // batch_size ) * batch_size
    return df.iloc[:aligned_len]


def split_df(df, config):
    # whether or not get rif of pure numeric cells
    if config.drop_numeric:
        df, _ = keep_alphacells(df)
    
    train_df = align_df(train_df, config.BS)

    return split_fold(df, config.valid_split, config.test_split)


def prepare_df(df, config):
    df = transform_df(config, df)
    train_df, valid_df, test_df = split_df(df, config)
    return concate_cols(add_token_mapping['[FIELD]'], config.cols_to_use, train_df, valid_df, test_df)


def df_names_to_idx(names, df):
    "Return the column indexes of `names` in `df`."
    if isinstance(names[0], int): return names
    return [df.columns.get_loc(c) for c in names]


def concate_cols(sep_token, cols_to_use, *dfs):
    for df in dfs:
        df[','.join(cols_to_use)] = df.iloc[:, df_names_to_idx(cols_to_use, df)].apply(lambda x: f" {sep_token} ".join(x), axis=1)


def tokenize_function(tokenizer, col, example):
    return tokenizer(example[col], truncation=True, max_length=512)


def get_tokenized_ds(tokenizer, config, *dfs):
    import datasets
    
    rst = []
    for df in dfs:
        # features = Features({config.input_col: Value('string'), 'labels': ClassLabel(names=['Other', 'Dataset', 'Method', 'Metric', 'DatasetAndMetric'])})
        try:
            # ds = datasets.Dataset.from_pandas(df, features=features)
            ds = datasets.Dataset.from_pandas(df)
        except:
            assert 0
        
        tokenized_ds = ds.map(partial(tokenize_function, tokenizer, config.input_col), batched=True)
        tokenized_ds = tokenized_ds.remove_columns([i for i in list(df.columns) if i != 'labels'] + ['__index_level_0__'])
        tokenized_ds.set_format('torch')
        rst.append(tokenized_ds)
    if len(rst) == 1:
        return rst[0]
    return rst


def generate_CTC_preds(config, model, df, BS=512):
    """
    Generate CTC predictions for a given fold.
    """
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    test_ds = load_single_DataWithTOkenType(config, df, drop_duplicates=False)
    test_dl = DataLoader(test_ds, batch_size=BS, collate_fn=padding_by_batch)

    predictions_all = []
    for batch in tqdm(test_dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        labels = batch["labels"].view(-1)
        idx = (labels != -100).nonzero(as_tuple=True)[0]
        predictions = predictions[idx]

        predictions_all += predictions.tolist()
    
    df['CTC_preds'] = [label_map_reverse[p] for p in predictions_all]
    df['cell_type'] = df['labels'].apply(lambda x: label_map_reverse[x])
    return df[ ['ext_id', 'CTC_preds', 'fold', 'cell_type'] ]
