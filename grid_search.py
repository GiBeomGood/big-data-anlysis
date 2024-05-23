import argparse
import pickle
from itertools import product

from communityDetection import load_network
from models import *
from utils.util import eval


def main():
    scores = dict()
    grid = {
        'data_path': [
            'TC1/TC1-1/1-1',
            'TC1/TC1-6/1-6',
        ],
        'epsilon': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
        'strategy': ['prob', 'freq'],
    }

    for data_path, epsilon, strategy in product(*grid.values()):
        args = argparse.Namespace(
            networkFile=f'{data_path}.dat',
            communityFile=f'{data_path}-c.dat',
            communitySavePath=f'{data_path}.cmty',
        )
        graph = load_network(args.networkFile)
        labels = my_nlta_wich_coreness(graph, epsilon, strategy, args.communityFile, max_epochs=20)
        score = eval(labels, args.communityFile)
    
        scores[f'{data_path.split('/')[-1]}-{epsilon}-{strategy}'] = score
    
    with open('./checkpoints/ours_with_coreness.pickle', 'wb') as f:
        pickle.dump(scores, f)
    return


if __name__ == '__main__':
    main()