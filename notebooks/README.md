# Data preparation and analysis notebooks

The loader notebooks prepare raw datasets in ignored `data-local/` folders and build reusable graph corpora under `data-local/graph_corpus_cache/`. They can also report dataset statistics, so they serve as the data setup step for downstream workflows.

## Preparation order

1. [Prepare a PubChem assay](data_preparation/prepare_pubchem_assay.ipynb) downloads one AID as active and inactive SDF files to `data-local/PUBCHEM/`. Choose the same AID in the analysis notebook.
2. [ZINC loader statistics](examples/example_zinc_loader_statistics.ipynb) downloads `zinc_250k.csv` to `data-local/ZINC/` when needed and prepares its graph cache.
3. [QM9 loader statistics](examples/example_qm9_loader_statistics.ipynb) downloads `qm9.csv` to `data-local/QM9/` when needed and prepares its graph cache.
4. [GDB loader statistics](examples/example_gdb_loader_statistics.ipynb) downloads the selected GDB mode to `data-local/GDB/` and prepares its graph cache.
5. [PubChem loader statistics](examples/example_pubchem_loader_statistics.ipynb) inspects assays already prepared in step 1.

Run the preparation notebook for a dataset only when its raw files or cache are missing, or when you want to rebuild them. Downstream notebooks in `abstractgraph-ml` and `abstractgraph-generative` link to the required preparation notebook near the top.

## Data storage

Raw downloads and generated graph caches are excluded from Git. The expected paths are:

- `data-local/PUBCHEM/`
- `data-local/ZINC/`
- `data-local/QM9/`
- `data-local/GDB/`
- `data-local/graph_corpus_cache/`

Set the corresponding `ABSTRACTGRAPH_*_ROOT` environment variable or pass an explicit root to use a dataset stored elsewhere. PubChem source and assay links are written into a per-assay manifest for provenance.
