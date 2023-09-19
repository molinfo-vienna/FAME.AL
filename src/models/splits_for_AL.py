import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

import random
random.seed(42)


if __name__ == '__main__':
    # load the file with descriptors to use for active lerning
    print('Loading descriptors file......')
    df_descriptor_r5 = 'output\zaretzki_preprocessed_5_descriptors.csv'
    df = pd.read_csv(df_descriptor_r5,low_memory=False)
    print('Number of atoms: ' + str(len(df)))

    # remove dupliates by descriptors
    df.sort_values('som_label',ascending=False,inplace=True)
    df.drop_duplicates(subset=df.columns[3:],inplace=True)
    print('After removing duplicate atoms: '+ str(len(df)))
    df = df.reset_index() # to give an index for each row
    
    folds = 5
    
    ####### Stratified random split
    skf = StratifiedKFold(n_splits=folds, random_state=42, shuffle=True)
    for fold, (train_index, test_index) in enumerate(skf.split(df[df.columns[3:-1]],df['som_label'])):
        df.loc[test_index,'fold_id'] = fold+1 # fold_id from 1 to 5
        
    print('wrting down file with folds assigned by StratifiedKFold: '+str(len(df)))
    df.to_csv('output/zaretzki_r5_5folds_random_split.csv',index=False)

    
    