import argparse
import pickle
from itertools import product
from os import listdir
from time import time

from communityDetection import load_network
from models import *
from utils.util import eval


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, default=1)
    args = args.parse_args()
    data_num = args.data

    scores = dict()
    grid = {
        'data_path': [f'./TC2/tc1{data_num}.dat'],
        'epsilon': [0.05, 0.5],
        'strategy': ['freq'],
        'use_coreness': [True]
    }

    for data_path, epsilon, strategy, use_coreness in product(*grid.values()):
        args = argparse.Namespace(
            networkFile=f'{data_path}',
            communitySavePath=f'{data_path}.cmty',
        )
        graph = load_network(args.networkFile)
        
        start = time()
        labels = my_nlta(graph, epsilon, strategy, use_coreness, None, max_epochs=20)
        end = time()
    
        scores[f'{data_path.split('/')[-1][:-4]}-{epsilon}-{strategy}-{use_coreness}'] = end-start
    
    with open(f'./checkpoints/ours_with_coreness_tc2_{data_num}.pickle', 'wb') as f:
        pickle.dump(scores, f)
        
    return


if __name__ == '__main__':
    main()