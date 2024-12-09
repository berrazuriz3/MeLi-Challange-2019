import argparse
import os
import shutil
import json
import csv

with open('metadata.json', 'r') as f:
    metadata = json.load(f)

with open('./data/item_popularity.json', 'r') as f:
    item_popularity = json.load(f)


class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Train or load a GRU4Rec model & measure recall and MRR on the specified test set(s).')
parser.add_argument('path', metavar='PATH', type=str, help='Path to the training data (TAB separated file (.tsv or .txt) or pickled pandas.DataFrame object (.pickle)) (if the --load_model parameter is NOT provided) or to the serialized model (if the --load_model parameter is provided).')
parser.add_argument('-ps', '--parameter_string', metavar='PARAM_STRING', type=str, help='Training parameters provided as a single parameter string. The format of the string is `param_name1=param_value1,param_name2=param_value2...`, e.g.: `loss=bpr-max,layers=100,constrained_embedding=True`. Boolean training parameters should be either True or False; parameters that can take a list should use / as the separator (e.g. layers=200/200). Mutually exclusive with the -pf (--parameter_file) and the -l (--load_model) arguments and one of the three must be provided.')
parser.add_argument('-pf', '--parameter_file', metavar='PARAM_PATH', type=str, help='Alternatively, training parameters can be set using a config file specified in this argument. The config file must contain a single OrderedDict named `gru4rec_params`. The parameters must have the appropriate type (e.g. layers = [100]). Mutually exclusive with the -ps (--parameter_string) and the -l (--load_model) arguments and one of the three must be provided.')
parser.add_argument('-l', '--load_model', action='store_true', help='Load an already trained model instead of training a model. Mutually exclusive with the -ps (--parameter_string) and the -pf (--parameter_file) arguments and one of the three must be provided.')
parser.add_argument('-s', '--save_model', metavar='MODEL_PATH', type=str, help='Save the trained model to the MODEL_PATH. (Default: don\'t save model)')
parser.add_argument('-t', '--test', metavar='TEST_PATH', type=str, nargs='+', help='Path to the test data set(s) located at TEST_PATH. Multiple test sets can be provided (separate with spaces). (Default: don\'t evaluate the model)')
parser.add_argument('-m', '--measure', metavar='AT', type=int, nargs='+', default=[20], help='Measure recall & MRR at the defined recommendation list length(s). Multiple values can be provided. (Default: 20)')
parser.add_argument('-e', '--eval_type', metavar='EVAL_TYPE', choices=['standard', 'conservative', 'median'], default='standard', help='Sets how to handle if multiple items in the ranked list have the same prediction score (which is usually due to saturation or an error). See the documentation of batch_eval() in evaluation.py for further details. (Default: standard)')
parser.add_argument('-ss', '--sample_store_size', metavar='SS', type=int, default=10000000, help='GRU4Rec uses a buffer for negative samples during training to maximize GPU utilization. This parameter sets the buffer length. Lower values require more frequent recomputation, higher values use more (GPU) memory. Unless you know what you are doing, you shouldn\'t mess with this parameter. (Default: 10000000)')
parser.add_argument('-g', '--gru4rec_model', metavar='GRFILE', type=str, default='gru4rec_pytorch', help='Name of the file containing the GRU4Rec class. Can be used to select different varaiants. (Default: gru4rec_pytorch)')
parser.add_argument('-d', '--device', metavar='D', type=str, default='cuda:0', help='Device used for computations (default: cuda:0).')
parser.add_argument('-ik', '--item_key', metavar='IK', type=str, default='ItemId', help='Column name corresponding to the item IDs (detault: ItemId).')
parser.add_argument('-sk', '--session_key', metavar='SK', type=str, default='SessionId', help='Column name corresponding to the session IDs (default: SessionId).')
parser.add_argument('-tk', '--time_key', metavar='TK', type=str, default='Time', help='Column name corresponding to the timestamp (default: Time).')
parser.add_argument('-pm', '--primary_metric', metavar='METRIC', choices=['recall', 'mrr'], default='recall', help='Set primary metric, recall or mrr (e.g. for paropt). (Default: recall)')
parser.add_argument('-lpm', '--log_primary_metric', action='store_true', help='If provided, evaluation will log the value of the primary metric at the end of the run. Only works with one test file and list length.')
args = parser.parse_args()



def ndcg(y_pred, y_true):
    dcg = 0
    idcg = 22.42461597 * len(y_true)
    contador = 0
    contador2 = 0
    contador3 = 0
    for pred, true in zip(y_pred, y_true):
        contador3 +=1
        position = 1
        for item in pred:
            if item == true:
                dcg += 12 / (np.log(1 + position))
                contador += 1
            elif item in metadata and true in metadata:
                if metadata[item]['domain_id'] == metadata[true]['domain_id']:
                    dcg += 1 / (np.log(1 + position))
                    contador2 +=1
            position += 1
    print(contador)
    print(contador2)
    print(contador3)
    
    score = dcg / idcg if idcg != 0 else 0
    return score

def calculate_novelty(recommendations):
    # Evitar división por cero
    novelty = 0
    for item in recommendations:
        popularity = item_popularity.get(item, 1)
        novelty += 1 / np.log(1 + popularity)
    if len(recommendations) > 0:
        return novelty / len(recommendations)
    else:
      return 0

def calculate_diversity(recommendations, item_df):
    categories = []
    no_category = 0
    for item in recommendations:
        if item in item_df.index:
            categories.append(item_df.loc[item, 'category_id'])
        else:
            no_category += 1
    if not categories:
        return 0
    diversity = 1 - (len(set(categories)) / len(categories))
    return diversity


import os.path
orig_cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import datetime as dt
import sys
import time
from collections import OrderedDict
import importlib
GRU4Rec = importlib.import_module(args.gru4rec_model).GRU4Rec
import evaluation
import importlib.util
import joblib
os.chdir(orig_cwd)

def load_data(fname, args):
    if fname.endswith('.pickle'):
        print('Loading data from pickle file: {}'.format(fname))
        data = joblib.load(fname)
        if args.session_key not in data.columns:
            print('ERROR. The column specified for session IDs "{}" is not in the data file ({})'.format(args.session_key, fname))
            print('The default column name is "SessionId", but you can specify otherwise by setting the `session_key` parameter of the model.')
            sys.exit(1)
        if args.item_key not in data.columns:
            print('ERROR. The column specified for item IDs "{}" is not in the data file ({})'.format(args.item_key, fname))
            print('The default column name is "ItemId", but you can specify otherwise by setting the `item_key` parameter of the model.')
            sys.exit(1)
        if args.time_key not in data.columns:
            print('ERROR. The column specified for time "{}" is not in the data file ({})'.format(args.time_key, fname))
            print('The default column name is "Time", but you can specify otherwise by setting the `time_key` parameter of the model.')
            sys.exit(1)
    else:
        with open(fname, 'rt') as f:
            header = f.readline().strip().split('\t')
        if args.session_key not in header:
            print('ERROR. The column specified for session IDs "{}" is not in the data file ({})'.format(args.session_key, fname))
            print('The default column name is "SessionId", but you can specify otherwise by setting the `session_key` parameter of the model.')
            sys.exit(1)
        if args.item_key not in header:
            print('ERROR. The colmn specified for item IDs "{}" is not in the data file ({})'.format(args.item_key, fname))
            print('The default column name is "ItemId", but you can specify otherwise by setting the `item_key` parameter of the model.')
            sys.exit(1)
        if args.time_key not in header:
            print('ERROR. The column specified for time "{}" is not in the data file ({})'.format(args.time_key, fname))
            print('The default column name is "Time", but you can specify otherwise by setting the `time_key` parameter of the model.')
            sys.exit(1)
        print('Loading data from TAB separated file: {}'.format(fname))
        data = pd.read_csv(fname, sep='\t', usecols=[args.session_key, args.item_key, args.time_key], dtype={args.session_key:'int32', args.item_key:'str'})
    return data

if (args.parameter_string is not None) + (args.parameter_file is not None) + (args.load_model) != 1:
    print('ERROR. Exactly one of the following parameters must be provided: --parameter_string, --parameter_file, --load_model')
    sys.exit(1)

if args.load_model:
    print('Loading trained model from file: {} (to device "{}")'.format(args.path, args.device))
    gru = GRU4Rec.loadmodel(args.path, device=args.device)
else:
    if args.parameter_file:
        param_file_path = os.path.abspath(args.parameter_file)
        param_dir, param_file = os.path.split(param_file_path)
        spec = importlib.util.spec_from_file_location(param_file.split('.py')[0], os.path.abspath(args.parameter_file))
        params = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params)
        gru4rec_params = params.gru4rec_params
        print('Loaded parameters from file: {}'.format(param_file_path))
    if args.parameter_string:
        gru4rec_params = OrderedDict([x.split('=') for x in args.parameter_string.split(',')])
    print('Creating GRU4Rec model on device "{}"'.format(args.device))
    gru = GRU4Rec(device=args.device)
    gru.set_params(**gru4rec_params)
    print('Loading training data...')
    data = load_data(args.path, args)
    print('Started training')
    t0 = time.time()
    gru.fit(data, sample_cache_max_size=args.sample_store_size, item_key=args.item_key, session_key=args.session_key, time_key=args.time_key)
    t1 = time.time()
    print('Total training time: {:.2f}s'.format(t1 - t0))
    if args.save_model is not None:
        print('Saving trained model to: {}'.format(args.save_model))
        gru.savemodel(args.save_model)
    
if args.test is not None:
    if args.primary_metric.lower() == 'recall':
        pm_index = 0
    elif args.primary_metric.lower() == 'mrr':
        pm_index = 1
    else:
        raise RuntimeError('Invalid value `{}` for `primary_metric` parameter'.format(args.primary_metric))

    y_pred = []
    y_true = []
    for test_file in args.test:
        print('Loading test data...')
        test_data = load_data(test_file, args)
        print('Starting evaluation (cut-off={}, using {} mode for tiebreaking)'.format(args.measure, args.eval_type))
        t0 = time.time()
        res = evaluation.batch_eval(gru, test_data, batch_size=512, cutoff=args.measure, mode=args.eval_type, item_key=args.item_key, session_key=args.session_key, time_key=args.time_key, return_predictions=True)
        print("paso aca bien")
        predictions = res[2]
        y_pred = [pred[1] for pred in predictions]
        y_true = [pred[2] for pred in predictions]

        recommendations = []
        for session_id, pred_items, actual_item in predictions:
            recommendations.append([session_id, pred_items, actual_item])

        with open('recommendations_gru4rec.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["session_id", "predicted_items", "actual_item"])
            writer.writerows(recommendations)

        for session_id, pred_items, actual_item in predictions:
            y_pred.append(pred_items)
            y_true.append(actual_item)
        
        t1 = time.time()
        print('Evaluation took {:.2f}s'.format(t1 - t0))
        for c in args.measure:
            print('Recall@{}: {:.6f} MRR@{}: {:.6f}'.format(c, res[0][c], c, res[1][c]))

        if args.log_primary_metric:
            print('PRIMARY METRIC: {}'.format([x for x in res[pm_index].values()][0]))

    
    ndcg_score = ndcg(y_pred, y_true)
    print("NDCG Score:", ndcg_score)

    diversity_score = calculate_diversity(y_pred)
    novelty_score = calculate_novelty(y_pred)
    print(f"Diversity: {diversity_score}")
    print(f"Novelty: {novelty_score}")

    
    
    

