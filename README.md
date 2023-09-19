SOLSTICE: Site-of-metabolism prediction with active learning
==============================

Active learning for sites of metabolism prediction  
Demonstrated with [Zaretzki dataset](https://doi.org/10.1021/ci400518g). This data set was motified from the original data set to get the format of molecular structure (preprocessed using [RDKit](https://www.rdkit.org/) and [ChEMBL Structue Pipeline](https://github.com/chembl/ChEMBL_Structure_Pipeline)) with all annotated sites of metabolism in one .sdf file (data/zaretzki_preprocessed.sdf). 

## Dependencies
* [Numpy](https://numpy.org/)
* [Pandas](https://github.com/pandas-dev/pandas)
* [scikit-learn](https://scikit-learn.org/stable/)
* [Matplotlib](https://matplotlib.org/)
* [CDPKit](https://cdpkit.org/)

## Files 
#### Calculating CDPKit FAME descriptor
- Code to calculate CDPKit FAME descriptor.
    - **src/features/cdpkit_calculate_fame_descriptors.py** 
    - Example: 
```commandline
python3 src/features/cdpkit_calculate_fame_descriptors.py -i data/zaretzki_preprocessed.sdf -o output/ -r 5 -m
```

#### Active learning for site of metabolism prediction
- Code to run active learning with random forest classifier. 
    - **src/models/AL_for_SoM_pred.py** 
    - Examples: 

```commandline
# active learning in 5-fold cross validation
python3 src/models/AL_for_SoM_pred.py -i output/zaretzki_r5_5folds_random_split.csv -o output/active_learning/01.random_sampling_vs_AL/ -ct 0.3 -af -n 5 
```  

```commandline
# random selection in 5-fold cross validation
python3 src/models/AL_for_SoM_pred.py -i output/zaretzki_r5_5folds_random_split.csv -o output/active_learning/01.random_sampling_vs_AL/ -ct 0.3 -af -n 5 -rs
```  

```commandline
# repeat on one validation set for 5 times
python3 src/models/AL_for_SoM_pred.py -i output/zaretzki_r5_5folds_random_split.csv -o output/active_learning/01.random_sampling_vs_AL/ -ct 0.3 -tf 1 -n 5 -rfs
```

 - Visualize results
    - **notebooks/zaretzki_active_learning_result.ipynb** 

