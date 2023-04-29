This file contains the schema and description for individual datasets. The dataset can be downloaded here:.

# papers.jsonl
This file contains extracted tables and the papers containing them.

Each line represents the data for one paper, in json format.
```python
{
    "arxiv_id": str,  # the arxiv id of a paper
    "topic": str,  # the topic fold this paper is about, e.g., question answering
    "tables": Dict[TableID, List[List[str]]],  # each table is a 2D array of raw cell content
    "year": str, # the year
    "author": str, # the last name of the first author of this paper
    "title": str,  # title of a paper
    "abstract": str,  # abstract of a paper
    "references": List[Reference],  # list of papers in the reference section
}

TableID: str = arxiv_id + '/' + 'table_<table_idx>.csv' 
Reference = {
    "idx": int,  # the 1-based index of this source as in the reference section, or 0 if this is the current paper
    'title': str,  # title of this source
    'abstract': str,
    's2_corpusId': str,  # s2_corpus_id of this source
    'author': str,  # the last name of the first author of this source
    'year': str,
    'pwc_url': str,  # PwC paper url of this source, if any, otherwise empty
}
```

# cell_type.jsonl
This file contains the ground truth data for the cell type classification task, one line per cell in json format.
```python
{
    "cell_id": TableID + '/' + RowPos + '/' + ColPos,
    "cell_type": Union["dataset", "metric", "method", "dataset&metric", "other"],
}

RowPos = int  # row position of the cell in the table
ColPos = int  # column position of the cell in the table
```


# attributed_source.jsonl
This file contains the ground truth data for the attributed source matching task, one line per cell, in json format.
```python
{
    "cell_id": TableID + '/' + RowPos + '/' + ColPos,
    "attributed_source": List[Union[int, Missing, Current]],
}
Missing = -1  # the cell is not attibuted to a source
Current = 0  # the attrubited source is the current paper
```

# entity_linking.jsonl
This file contains the ground truth data for the end-to-end entity linking including both inKB and outKB mentions, one line per source, in json format.
```python
{
    "cell_id": TableID + '/' + RowPos + '/' + ColPos,
    "pwc_url": Union[Missing, PwCURL],
}
PwCURL = str
Missing = 0
```


# cell_type_additional_data.jsonl
This file contains additional data we derived for each cell, one line per cell, in json format.
```python
{
    "cell_id": TableID + '/' + RowPos + '/' + ColPos,
    "context_spans": List[Tuple[int, int]],  # Spans indexing into context sentences for the cell from the raw paper text
    "region_type": Union[0, 1, 2, 3]  # Indicates whether the cell is in the top-left, top-right, bottom-left, or bottom-right region of the table
}

```
