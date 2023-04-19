import re
import pandas as pd
from tqdm import tqdm

from common_utils.common_ML_utils import *


def generate_RPI_ML(ref_extract: pd.DataFrame, EL: pd.DataFrame):
    """
    :returns RPI_ML
    """
    ref_extract = ref_extract.fillna('')

    attrs = ['ext_id', 'paper_id', 'fold', 'bib_entries', 'cell_type', 'cell_content', 'has_reference', 'row_context', 'cell_reference',
       'text_reference', 'col_context', 'row_id', 'col_id', 'reverse_row_id', 'reverse_col_id', 'region_type', 'pwc_url', 'text']

    output = EL[ attrs ].copy()
    indices, authors, years, titles, abstracts = [], [], [], [], []

    for row in tqdm(EL.itertuples(), total=len(EL)):
        paper_id = row.paper_id
        refs = ref_extract[ref_extract.paper_id == paper_id]

        indices.append(refs.idx.values)
        authors.append(refs.author.values)
        years.append(refs.year.values)
        titles.append(refs.title.values)
        abstracts.append(refs.abstract.values)
    
    output['idx'], output['author'], output['year'], output['abstract'], output['title'] = indices, authors, years, abstracts, titles
    output = output.explode( ['idx', 'author', 'year', 'abstract', 'title'] )
    output = output.reset_index(drop=True)

    def get_label(row):
        pbar.update(1)
        if row.idx in row.bib_entries:
            return 0
        else:
            return 1
      
    pbar = tqdm(total=len(output))
    output['labels'] = output.apply(lambda x: get_label(x), axis=1)
    pbar.close()
    output.fillna('', inplace=True)
    return output