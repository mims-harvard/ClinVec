# Knowledge graph embeddings capture portable medical knowledge to power clinical AI 

## Overview
xxx

## Installation and setup

Depending on what level of analysis you're performing, you may not need to install the full list of packages.

### "I just want the KG and/or embeddings"
No need to install any dependencies. Please download the embeddings from Harvard Dataverse here.

### "I just want the scripts to construct the KG from the source files"

This only requires the following standard libraries. For this step, the exact version is not typically a strict requirement. 

```
pandas
numpy
...
```

Navigate to `/kg/...` where `xxx.ipynb` will walk through downloading all of the source files and constructing the knowledge graph from scratch. Note that due to licensing, users will be required to register and download certain source files (e.g. LOINC codes). 

### "I want to train and create embeddings from scratch"

This requires installing ML graph-specific libraries. We've provided the conda environment below. Note that the version requirement for DGL (Deep Graph Library) is very specific (v1.1)

Follow the previous step above to construct the KG or download the KG csv from Harvard Dataverse here. 

Navigate to `../` to begin training. 

### "I want to recreate the entire paper because I liked it so much"

Thank you for the flattery. We've provided individual Jupyter notebooks for each of the main analyses presented in the paper. You will need to download the embeddings and associated key file (mapping indices to node names) here and change the file location at the top of each notebook.

Note that we provide a modified version of the phenotype risk score analysis using synthetic data since this analysis requires individual-level patient data. 
