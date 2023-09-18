###################
# import libraries
import numpy as np
from sklearn.cluster import KMeans
import time
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# adapt from https://github.com/RekerLab/Active-Subsampling

###################
# define functions to perform active learing pipeline
def al_subsampling(model, dataset, metric, num_repeats,\
    threshold=0.5, all_folds=False, test_fold=1, \
        mini_batch=False, samples_number=10, from_samples=0, \
        random_first_samples=False, rand=False):
    '''
         Args: 
        model: the machine learning model used.
        dataset: pandas dataframe
        metric: selected model evaluation metric
        num_repeats(int): Number of repeats to run the pipeline.
        threshold: the threshold for the classifier
        all_folds: run a cross validation manner, each time a different fold used as the validation set
        test_fold: using certain fold as validation set
        mini_batch: if adding more than one most inforamtive sample to the training set every iteration
        samples_number: the number of samples added to the training set every iteration
        from_samples: if this is not 0, diverse mini-batch will be performed. 
            The number of samples added to the training set will be selected by kmeans clustering from those most diverse samples
        random_first_samples: if the first samples are picked randomly when repeating with the same pooling set
        rand(default False): if set to True, the selection strategy would be set to passive learning which
             correspond to randomly selecting the data. If False, uncertainty based active learning is used


    Returns: Four "lists of lists" of results, where every list contains a lists of specific type of result for every iteration and every repeat 
    performances: Lists of repeated machine Learning predictive performance values for every iteration with the specified evaluation metric
    positive_selection_ratios: Lists of positive label ratios at every iteration and for every repeat
    ids: Lists of lists of atom ids selected at every learning iteration for every repeat
    labels: Lists of lists of atom labels selcted at every learning iteration for every repeat
       '''
    
    # create empty lists to store results
    performances = []  
    positive_selection_ratios = []
    ids = []
    labels = []
    feature_names = dataset.columns[4:-1] # first three columns are ['index','som_label','mol_id','atom_idx'], the last column is 'fold_id'
    
    # repeat results
    for i in range(num_repeats):
        
        # split data into active learning set and validation (test) set
        if all_folds:
            #if run all folds, each repeats will use a new fold until all folds served as test once.
            df_test = dataset[dataset['fold_id']==i%5+1] # becauese have 5 folds input, from 1 to 5 
            # print('fold '+str(i%5+1) +' is the validation set')
        else:
            df_test = dataset[dataset['fold_id']==test_fold]
            # print('fold '+str(test_fold)+' as val')
 
        df_train = dataset[~dataset['index'].isin(df_test['index'])]
        
        learning_x = df_train[feature_names].to_numpy()
        learning_y = df_train.som_label.to_numpy()             
        learning_s = df_train['index'].to_numpy()
        
        validate_x = df_test[feature_names].to_numpy()
        validate_y = df_test.som_label.to_numpy()

        # create container to collect performances for this repeat
        temp_perf = []
        
        if random_first_samples:
            # print('different random seed to pick the first samples')
            np.random.seed(int(time.time()))

        # select two random samples from active learning pool into training data
        training_mask = np.array([False for i in learning_y])
        training_mask[np.random.choice(np.where(learning_y == 0)[0])] = True
        training_mask[np.random.choice(np.where(learning_y == 1)[0])] = True
        
        training_x = learning_x[training_mask]
        training_y = learning_y[training_mask]
        training_s = learning_s[training_mask]
        
        ps_num = np.sum(training_y)
        atom_num = len(training_y)
        temp_ps_num = [ps_num]
        temp_atom_num = [atom_num]
        
        learning_x = learning_x[np.invert(training_mask)]
        learning_y = learning_y[np.invert(training_mask)]
        learning_s = learning_s[np.invert(training_mask)]
        
        # start active learning process
        for i in range(len(learning_s)+1):
            _ = model.fit(training_x, training_y) # fit the model
            
            preds = model.predict(validate_x) # predict test data
            
            # calculate performance on test data
            temp_perf += [metric(validate_y, preds)]

            if len(learning_x) == 0: 
                break
            else: 
                # pick new datapoint
                if rand == True: # Switching between active learning and random sampling
                    new_pick = np.random.randint(len(learning_x))
                else:
                    probas = model.predict_proba(learning_x)
                    new_pick = np.argmin(np.abs(probas[:,1] - threshold))
                    
                    if mini_batch:
                        # print('using batch samples')
                        if from_samples:
                            # print('first sample the '+str(from_samples)+' most uncertain samples')
                            if len(probas[:,1]) > from_samples:
                                new_picks = np.argpartition(np.abs(probas[:,1] - threshold),from_samples)[:from_samples]
                            else:
                                new_picks = np.array(range(len(probas[:,1])))
                                
                            # print('then pick '+str(samples_number)+' the most diverse samples')
                            # Kmeans clustering
                            if len(new_picks) > samples_number:
                                num_clusters = samples_number
                                kmeans = KMeans(n_clusters=num_clusters)
                                kmeans.fit(learning_x[new_picks])
                                
                                # Get the cluster assignments for each sample
                                cluster_labels = kmeans.labels_
                                
                                # Calculate the distances between samples and cluster centers
                                distances = kmeans.transform(learning_x[new_picks])
                                
                                # Create a list to store the indices of representative samples
                                representative_indices = []

                                # Iterate through the clusters and select the closest sample to the cluster center
                                for cluster_idx in range(num_clusters):
                                    # Find the samples belonging to the current cluster
                                    samples_in_cluster = np.where(cluster_labels == cluster_idx)[0]
                                    
                                    # Calculate the distances to the cluster center for samples in the current cluster
                                    cluster_distances = distances[samples_in_cluster, cluster_idx]
                                    
                                    # Find the index of the closest sample to the cluster center
                                    closest_index = samples_in_cluster[np.argmin(cluster_distances)]
                                    
                                    # Add the index of the representative sample to the list
                                    representative_indices.append(closest_index)
                            
                                new_pick = [new_picks[i] for i in representative_indices]
                            else:
                                new_pick = new_picks
                        else:
                            # print('add '+str(samples_number)+' most uncertain samples each time')
                            if len(probas[:,1]) > samples_number:
                                new_pick = np.argpartition(np.abs(probas[:,1] - threshold),samples_number)[:samples_number]
                            else:
                                new_pick = np.array(range(len(probas[:,1])))
                
                # add new selection to training data
                training_x = np.vstack((training_x, learning_x[new_pick]))
                training_y = np.append(training_y, learning_y[new_pick])
                training_s = np.append(training_s, learning_s[new_pick])
                
                ps_num = np.sum(training_y)
                atom_num = len(training_y)
                temp_ps_num.append(ps_num)
                temp_atom_num.append(atom_num)
                
                # remove new selection from pool data
                learning_x = np.delete(learning_x, new_pick, 0)
                learning_y = np.delete(learning_y, new_pick)
                learning_s = np.delete(learning_s, new_pick)
            
            
        ids += [training_s]	# collect ids of selected data
        performances += [temp_perf]  # collect performance on test data
        labels += [training_y]
        positive_selection_ratios += [np.array(temp_ps_num)/np.array(temp_atom_num)] # collect percentage of selected positive data


    return performances, positive_selection_ratios, ids, labels


