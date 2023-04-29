# S2abEL: A Dataset and Baseline for Entity Linking from Scientific Tables with Out-of-Knowledge-Base Mentions

This repository provides access to the dataset S2abEL (can be downloaded here: https://github.com/allenai/S2abEL/blob/main/data/release_data.tar.gz) and is the official implementation of [S2abEL: A Dataset and Baseline for Entity Linking from Scientific Tables with Out-of-Knowledge-Base Mentions](https:google.com) by Yuze Lou, Bailey Kuehl, Erin Bransom, Aakanksha Naik, Sergey Feldman, and Doug Downey.

## Installation
To create a [conda](https://www.anaconda.com/distribution/) environment named `s2abel` and install requirements run:

```python
git clone https://github.com/allenai/S2abEL.git
cd S2abEL
conda create -y --name s2abel python==3.9.5
conda activate s2abel
chmod +x install.sh
./install.sh
```
Note that you might need to change torch version in `install.sh` depending on your CUDA version.

## Datasets
We release the following S2abel datasets on Zenodo: [ExtractedTables & Papers](), [CellTypeClassification](), [AttributedSourceMatching](), and [EntityLinking](), generated from source data under arXiv.org’s license. Access to the data is granted on request provided the user's intent is in accordance with license terms. [Detailed data schema is here](data_schema.md).

## Training and Evaluation
We offer notebooks for each of the sub-tasks mentioned in the paper, namely: [cell type classification](notebooks/ctc.ipynb), [attributed source mathing & candidate generation](notebooks/asm.ipynb), [entity disambiguation](notebooks/el.ipynb), which include training and evaluating corresponding models. Please make sure to modify the data_dir in the notebooks appropriately to the directory where the training data has been downloaded.

## Reproducibility
The experiments in the paper were run with the python (3.9.5) package versions in `install.sh`. Rerunning provided notebooks should produce the same numbers as in the paper.

## License
S2abEL is released under the [Apache 2.0 license](LICENSE).

## Citation
The dataset and method is described in the following paper:
