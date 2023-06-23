# S2abEL: A Dataset for Entity Linking from Scientific Tables

This repository provides access to the dataset `S2abEL` and is the official implementation of ```S2abEL: A Dataset for Entity Linking from Scientific Tables```.

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
We release the following S2abel datasets: `ExtractedTables & Papers`, `CellTypeClassification`, `AttributedSourceMatching`, and `EntityLinking` here: https://github.com/allenai/S2abEL/blob/main/data/release_data.tar.gz. 

[Detailed data schema is here](data_schema.md).

## Training and Evaluation
We offer notebooks for training and evaluating each of the sub-tasks mentioned in the paper, namely: [cell type classification](notebooks/ctc.ipynb), [attributed source mathing & candidate generation](notebooks/asm.ipynb), and [entity disambiguation](notebooks/el.ipynb).


## License
S2abEL is released under the [Apache 2.0 license](LICENSE).

