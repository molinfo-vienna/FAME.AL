import pandas as pd
from sklearn.metrics import matthews_corrcoef as mcc
import argparse
import pickle
import os
from src.utils.classifiers import ThresholdRandomForestClassifier
from src.utils.ALSubsampling_adapt import al_subsampling


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='run active learning with ThresholdRandomForestClassifier and MCC as performance matrix')

    parser.add_argument('-i',
                        dest='in_file',
                        required=True,
                        metavar='<input file>',
                        help='Input file with calculated descriptors')
    parser.add_argument('-o',
                        dest='out_folder',
                        required=True,
                        metavar='<output folder>',
                        help='Output location')
    
    # classifier related
    parser.add_argument('-ct',
                        dest='classifier_threshold',
                        required=True,
                        metavar='<classifier threshold>',
                        default=0.5,
                        help='threshold of the classifier',
                        type=float)
    parser.add_argument('-cd',
                        dest='classifier_max_depth',
                        required=False,
                        metavar='<classifier max_depth>',
                        default=None,
                        help='max_depth of the classifier',
                        type=int)
    
    # sampling    
    parser.add_argument('-rs',
                        dest='random_sampling',
                        required=False,
                        metavar='<switch off ative learning>',
                        help='Random sampling instead of active learning',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-rfs',
                        dest='random_first_samples',
                        required=False,
                        metavar='<pick the first examples randomly in each run and repeat>',
                        help='pick the first examples randomly in each run and repeat',
                        action=argparse.BooleanOptionalAction)   
    
    # mini-batch and diverse mini-batch
    parser.add_argument('-b',
                        dest='mini_batch',
                        required=False,
                        metavar='<pick more samples>',
                        help='pick more than one sample each time',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-sn',
                        dest='samples_number',
                        required=False,
                        metavar='<samples number>',
                        help='add this number of samples, only use combined with -b mini_batch',
                        type=int)
    parser.add_argument('-fn',
                        dest='from_samples',
                        required=False,
                        metavar='<from samples>',
                        help='from those number of samples, only use combined with -sn, -b',
                        type=int)

    # which fold as validation set and if repeats
    parser.add_argument('-af',
                        dest='all_folds',
                        required=False,
                        metavar='<all folds>',
                        default=False,
                        help='which fold use as test',
                        action=argparse.BooleanOptionalAction) 
    parser.add_argument('-tf',
                        dest='test_fold',
                        required=False,
                        metavar='<test fold>',
                        default=1,
                        choices=[1,2,3,4,5],
                        help='which fold use as test',
                        type=int)    
    parser.add_argument('-n',
                        dest='repeat_times',
                        required=True,
                        metavar='<repeat times>',
                        default=1,
                        help='repeat how many times',
                        type=int)

    parse_args = parser.parse_args()

    return parse_args


if __name__ == '__main__':
    args = parseArgs()

    # create output folder if it does not exist
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
        print('The new output folder is created.')

    # read data set
    df = pd.read_csv(args.in_file)
    
    # TODO: to remove, for test only
    # df = df.head(100)
        
    # initialize model and performance metric
    model=ThresholdRandomForestClassifier(n_estimators=250, 
                                          random_state=42,
                                          class_weight='balanced_subsample',
                                          decision_threshold=args.classifier_threshold,
                                          max_depth=args.classifier_max_depth, 
                                          n_jobs=8)
    metric = mcc

    prefix = args.out_folder+'/'+args.in_file.split('/')[-1].split('.')[0]+'_result_' + str(args.classifier_threshold)+'_TH_'

    # different max_depth for RFC
    if args.classifier_max_depth:
        prefix += str(args.classifier_max_depth)+'_max_depth_'
    
    # random sampling or active learning
    if args.random_sampling:
        prefix += 'RandomSampling_'
    else:
        prefix += "AL_"
        
    # random first samples     
    if args.random_first_samples:
        prefix += 'diffFirstSamples_'
    
    # mini-batch and diverse mini-batch    
    if args.mini_batch:
        prefix += 'mini_batch_'    
    if args.samples_number:
        prefix += str(args.samples_number) +'_samples_'
    if args.from_samples:
        prefix = prefix.replace('mini_batch_','diverse_mini_batch_')
        prefix += 'from_'+str(args.from_samples) +'_samples_'
       
    if args.all_folds:
        prefix += 'AllFolds_' + str(args.repeat_times/5) +'repeats'
    elif args.test_fold:  # only works when not using all_folds 
        # print('using fold '+str(args.test_fold) +' as validation set')
        prefix += 'Fold' + str(args.test_fold) +'_'+str(args.repeat_times) +'repeats'
        
    # print(prefix)
    
    result = al_subsampling(model, df, metric, args.repeat_times, threshold=args.classifier_threshold, all_folds = args.all_folds, \
        test_fold = args.test_fold, mini_batch=args.mini_batch,samples_number=args.samples_number,from_samples=args.from_samples, \
            random_first_samples = args.random_first_samples,rand=args.random_sampling)
    
    with open(prefix+'.pickle', 'wb') as f:
        pickle.dump(result, f)
