# Deleterious mutations

This repository contains the code and data which were used for the analysis of deleterious mutations classification.

## Description

The comparison between performing classification with and without Transfer Learning and the figures from the Supplement Materials are provided in the notebook `delmut_main.ipynb`. The code related to the figures from the manuscript is placed in `delmut_plot_figures.ipynb`. The analysis of *C. arietinum* predictions is performed in `delmut_snp_freq_analysis.ipynb`.

The initial data tables are placed in `init_data/`. The data tables related to the analysis of *C. arietinum* are located in `freq_cicer/`. The directory `classifiers/` contains 2 directories: `all` with the results of all training repetitions and additional tables for the analysis; `best` which keeps outperformed classifiers.

There are several sections in the `delmut_main.ipynb`:

1. Preparation
2. Import Datasets
3. Computation of Weights
4. Grid Search for Hyperparameters
5. Classifiers' Training
6. Search of the Best Classifiers
7. Import of the Best Classifiers
8. Resulting Tables
9. ROC-curves and Metrics for *A. thaliana*
10. ROC-curves and Metrics for *O. sativa*
11. ROC-curves and Metrics for *P. sativum*
12. Resulting Table for *C. arietinum*

Sections **4** and **5** are time demanding. For the given data they last almost 10 and 40 minutes, respectively.

The notebook utilizes methods from `functions.py` besides standard libraries.

## Implemented methods

* `make_dir_if_not_exists` - the creation of a directory if one does not exist
* `get_data` - loading a feature table with further distribution over several variables
* `copy_clfs` - copying the best classifiers to a separate directory
* `result_table` - the creation of a resulting table with predictions
* `arab_plot_accs` - overall information about classifiers' fitting. 
* `mets_mats_arab` - the creation of confusion matrixes and the calculation of the metrics in interest related with *A. thaliana*
* `mets_others` - the calculation of the metrics in interest related with the others organisms
* `roc_curves` - the creation of ROC-curves
* `cicer_table` - the creation of a resulting table for C. arietinum with predictions

## Requirements

To run content from `delmut_main.ipynb` you need Python 3.6.5 or later. The list of required Python packages is in `requirements_python.txt`. To run content from `delmut_plot_figures.ipynb` and `delmut_snp_freq_analysis.ipynb` you need R and the libraries which listed in `requirements_R.txt`.

## Installation

```
git clone https://github.com/kovmax/DelMut.git
```

## Authors

**Maxim Kovalev**  contributed in `delmut_main.ipynb` and `functions.py`.

**Anna Igolkina** contributed in `delmut_plot_figures.ipynb` and `delmut_snp_freq_analysis.ipynb`, [e-mail](mailto:igolkinaanna11@gmail.com).

**SergeySalikhov** rewrite some code fragments and delete features.

## References

Kovalev et al., *A pipeline for classifying deleterious coding mutations in agricultural plants*.
