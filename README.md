# Deleterious mutations

This repository containes the code and data which were used for the analysis of deleterious mutations classification. Comparison between performing
classification with and without Transfer Learning is provided in the notebook `main.ipynb`. Inintial data tables are placed in `data/`. The directory
`classifiers/` containes 2 directories: `all` with the results of all training repetitions and tables with accuracies; `best` which keeps outperformed
classifiers.

## Description

There are several section in the notebook:

1. Preparation
2. Import Datasets
3. Computation of Weghts
4. Grid Search for Hyperparameters
5. Classifiers' Training
6. Search of the Best Classifiers
7. Import of the Best Classifiers
8. Resulting Tables
9. ROC-curves and Metrics for A. thaliana
10. ROC-curves and Metrics for O. sativa
11. ROC-curves and Metrics for P. sativum

Sections **4** and **5** are time demanding. For the given data they lasts almost 10 and 40 minutes, respectively.

The notebook utilizes methods from `functions.py` besides standart libraries.

## Implemented methods

* `make_dir_if_not_exists` - the creation of a directory if one does not exist
* `get_data` - loading a feature table with further distribution over several variables
* `copy_clfs` - copying the best classifiers to a separate directory
* `result_table` - the creation of a resulting table with predictions 
* `mets_mats_arab` - the creation of confusion matrixes and the calculation of the metrics in interest related with A. thaliana
* `mets_others` - the calculation of the metrics in interest related with the others organisms
* `roc_curves` - the creation of ROC-curves

## Requirements

To run content you need Python 3.6.5 or later. A list of required Python packages that the code depends on, are in `requirements.txt`.

## Installation

```
git clone https://github.com/kovmax/DelMut.git
```

## References

Kovalev et al., *A pipeline for classifying deleterious coding mutations in agricultural plants*.
