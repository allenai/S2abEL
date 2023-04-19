from typing import List
from CTC.utils import additional_special_tokens, set_seed, seed_worker
import pandas as pd

from torch.utils.data import DataLoader
import re


reference_re = re.compile(r"<ref id='([^']*)'>(.*?)</ref>")
empty_paren_re = re.compile(r"\(\s*\)|\[\s*\]")
def remove_references(s):
    s = reference_re.sub("", s)
    return empty_paren_re.sub("", s)


style_tags_re = re.compile(r"</?(bold|italic|red|green|blue)>")
def remove_text_styles(s):
    return style_tags_re.sub("", s)


def remove_xa0(s):
    return s.replace("\xa0", " ")


def flatten_single_table_text_and_cell_separated(cell_excerpts: pd.DataFrame, paper_id: str, table, orient: str):
    """This function flattens a single table with 'text_sentences' and 'cell_content' seperated. 
    No special tokens like [EMPTY] added.
    Specital tokens [REF], [MASK] are added to 'text_sentences'."""
    
    if orient != 'col':
        raise NotImplementedError()
    
    row_position_ids = []
    col_position_ids = []
    has_reference_ids = []
    ext_ids = []
    is_empty_ids = []
    text_sentences = []
    cell_contents = []
    table_id = f"{paper_id}/{table.name}"

    for col_id in table.df.columns:
        for row_id, cell in enumerate(table.df[col_id]):
            cell_content = remove_xa0(remove_text_styles(remove_references(cell.raw_value))).strip()
            cell_contents.append(cell_content)
            raw_value = remove_xa0(cell.raw_value).strip()
            row_position_ids.append(row_id)
            col_position_ids.append(col_id)
            has_reference_ids.append(1 if cell.refs else 0)
            is_empty_ids.append(0 if raw_value else 1)
            ext_id = f"{table_id}/{row_id}/{col_id}"
            ext_ids.append(ext_id)
            tmp = cell_excerpts[cell_excerpts['ext_id']==ext_id]
            if tmp.shape[0] == 0:  # the cell not in the df
                text_sentence = ''
            else:
                text_sentence = cell_excerpts[cell_excerpts['ext_id']==ext_id].squeeze()['text']
            text_sentences.append(text_sentence)
    
    return pd.DataFrame(data={
        "ext_ids": ext_ids,
        "has_reference_ids": has_reference_ids,
        "row_position_ids": row_position_ids,
        "col_position_ids": col_position_ids,
        "is_empty_ids": is_empty_ids,
        "text_sentences": text_sentences,
        "cell_contents": cell_contents
    })



def flatten_single_table(cell_excerpts: pd.DataFrame, paper_id: str, table, orient: str):
    """Flatten a table where the text_excerpts contain either the text_sentences if not empty, otherwise the cell_contents"""
    input_cells : List[str]  = []
    row_position_ids = []
    col_position_ids = []
    has_reference_ids = []
    ext_ids = []
    is_empty_ids = []
    text_sentences = []
    table_id = f"{paper_id}/{table.name}"

    if orient == 'col':
        for col_id in table.df.columns:
            for row_id, cell in enumerate(table.df[col_id]):
                cell_content = remove_xa0(remove_text_styles(remove_references(cell.raw_value))).strip()
                raw_value = remove_xa0(cell.raw_value).strip()
                input_cells.append(cell_content)
                row_position_ids.append(row_id)
                col_position_ids.append(col_id)
                has_reference_ids.append(1 if cell.refs else 0)
                is_empty_ids.append(0 if raw_value else 1)
                ext_id = f"{table_id}/{row_id}/{col_id}"
                ext_ids.append(ext_id)
                tmp = cell_excerpts[cell_excerpts['ext_id']==ext_id]
                if tmp.shape[0] == 0:  # the cell not in the df
                    text_sentence = cell_content
                else:
                    text_sentence = cell_excerpts[cell_excerpts['ext_id']==ext_id].squeeze()['text']
                if not text_sentence:
                    text_sentence = cell_content
                text_sentences.append(text_sentence)

    elif orient == 'row':
        raise NotImplementedError
    elif orient == 'random':
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    return pd.DataFrame(data={
        "ext_ids": ext_ids,
        "has_reference_ids": has_reference_ids,
        "row_position_ids": row_position_ids,
        "col_position_ids": col_position_ids,
        "input_cells": input_cells,
        "is_empty_ids": is_empty_ids,
        "text_excerpts": text_sentences
    })
