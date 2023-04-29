import pandas as pd
from tqdm import tqdm


def get_ref_extract(papers: pd.DataFrame):
    records = []
    for row in papers.itertuples():
        for ref in row.references:
            records.append({
                'paper_id': row.arxiv_id,
                'idx': ref['idx'],
                'author': ref['author'],
                'year': ref['year'],
                'title': ref['title'],
                'abstract': ref['abstract'],
            })
        records.append({
            'paper_id': row.arxiv_id,
            'idx': 0,
            'author': row.author,
            'year': row.year,
            'title': row.title,
            'abstract': row.abstract,
        })
    return pd.DataFrame(records)


def generate_RPI_ML(ref_extract: pd.DataFrame, asm: pd.DataFrame):
    """
    :returns RPI_ML
    """
    ref_extract = ref_extract.fillna('')

    attrs = ['cell_id', 'fold', 'cell_type', 'cell_content', 'has_reference', 'row_context', 'cell_reference',
             'col_context', 'row_pos', 'col_pos', 'reverse_row_pos', 'reverse_col_pos', 'region_type', 'context_sentences',
             'attributed_source']

    output = asm[ attrs ].copy()
    indices, authors, years, titles, abstracts = [], [], [], [], []

    for row in tqdm(asm.itertuples(), total=len(asm)):
        paper_id = row.cell_id.split('/')[0]
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
        if row.idx in row.attributed_source:
            return 0
        else:
            return 1
      
    pbar = tqdm(total=len(output))
    output['labels'] = output.apply(lambda x: get_label(x), axis=1)
    pbar.close()
    output.fillna('', inplace=True)
    # del output['attributed_source']
    return output